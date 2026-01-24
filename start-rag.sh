#!/bin/bash

# ============================================================
# RAG 시스템 시작 스크립트
# CUDA 자동 감지 및 LLM 백엔드 선택 (vLLM 또는 Ollama)
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}RAG 시스템 시작 스크립트${NC}"
echo -e "${BLUE}============================================================${NC}"

# ============================================================
# 1. CUDA 감지
# ============================================================
echo -e "\n${YELLOW}[1/3] CUDA 감지 중...${NC}"

HAS_CUDA=false
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}✓ NVIDIA GPU 감지됨${NC}"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
    HAS_CUDA=true
else
    echo -e "${YELLOW}⚠ NVIDIA GPU 미감지${NC}"
    echo -e "${YELLOW}  CPU 모드로 실행됩니다.${NC}"
fi

# ============================================================
# 2. LLM 백엔드 결정
# ============================================================
echo -e "\n${YELLOW}[2/3] LLM 백엔드 선택${NC}"

# .env에서 LLM_BACKEND 읽기
if [ -f .env ]; then
    LLM_BACKEND_ENV=$(grep -E "^LLM_BACKEND=" .env | cut -d'=' -f2 | tr -d ' ' || echo "auto")
else
    LLM_BACKEND_ENV="auto"
fi

# 최종 선택 결정
if [ "$LLM_BACKEND_ENV" = "auto" ]; then
    # 자동 선택
    if [ "$HAS_CUDA" = true ]; then
        LLM_BACKEND="vllm"
    else
        LLM_BACKEND="ollama"
    fi
    echo -e "${BLUE}  자동 선택 모드${NC}"
else
    # 수동 선택
    LLM_BACKEND="$LLM_BACKEND_ENV"
    echo -e "${BLUE}  수동 선택 모드${NC}"
fi

# 선택된 백엔드 확인
if [ "$LLM_BACKEND" = "vllm" ]; then
    echo -e "${GREEN}✓ vLLM (Qwen2.5-Coder-14B with CUDA)${NC}"
    PROFILES="vllm"
else
    echo -e "${GREEN}✓ Ollama (CPU 최적화, 온프레미스)${NC}"
    PROFILES="ollama"
fi

# ============================================================
# 3. Docker Compose 실행
# ============================================================
echo -e "\n${YELLOW}[3/3] Docker 컨테이너 시작${NC}"

# 먼저 orphaned 컨테이너 제거
echo -e "${BLUE}  orphaned 컨테이너 제거 중...${NC}"
docker-compose down --remove-orphans 2>/dev/null || true

# 선택된 백엔드로 컨테이너 시작
echo -e "${BLUE}  Docker 컨테이너 시작 중 (profiles: $PROFILES)...${NC}"
docker-compose --profile $PROFILES up -d

# 컨테이너 시작 대기 (15초)
echo -e "${BLUE}  컨테이너 초기화 중 (15초 대기)...${NC}"
sleep 15

# ============================================================
# 4. 상태 확인
# ============================================================
echo -e "\n${YELLOW}[4/3] 시스템 상태 확인${NC}"

echo -e "\n${BLUE}실행 중인 컨테이너:${NC}"
docker-compose --profile $PROFILES ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# ============================================================
# 5. 서비스 URL 정보
# ============================================================
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}RAG 시스템 시작 완료!${NC}"
echo -e "${GREEN}============================================================${NC}"

echo -e "\n${BLUE}서비스 URL:${NC}"
echo -e "  • Open WebUI:      ${GREEN}http://localhost:3000${NC}"
echo -e "  • RAG API Server:  ${GREEN}http://localhost:9000${NC}"

if [ "$LLM_BACKEND" = "vllm" ]; then
    echo -e "  • vLLM Server:     ${GREEN}http://localhost:8000${NC}"
    echo -e "\n${BLUE}LLM 정보:${NC}"
    echo -e "  • 모델:            Qwen2.5-Coder-14B"
    echo -e "  • 방식:            GPU 가속 (CUDA)"
    echo -e "  • 데이터:          외부 로그 없음 (온프레미스)"
else
    echo -e "  • Ollama Server:   ${YELLOW}localhost:11434 (로컬 전용)${NC}"
    echo -e "\n${BLUE}LLM 정보:${NC}"
    echo -e "  • 모델:            Qwen2.5"
    echo -e "  • 방식:            CPU 최적화"
    echo -e "  • 보안:            ${GREEN}온프레미스 (완전 로컬)${NC}"
    echo -e "  • 데이터:          외부 전송 없음"
fi

echo -e "\n${BLUE}주요 명령어:${NC}"
echo -e "  • 로그 확인:       docker-compose logs -f sync-worker"
echo -e "  • 시스템 중지:     docker-compose down"
echo -e "  • LLM 모델 변경:   LLM_BACKEND=vllm 또는 ollama로 설정 후 다시 시작"

echo -e "\n${BLUE}온프레미스 설정 안내:${NC}"
echo -e "  • 모든 데이터:     로컬 ./data 디렉토리에 저장"
echo -e "  • 네트워크:        내부 네트워크만 사용 (172.28.0.0/16)"
echo -e "  • 외부 API:        Notion, GitHub, Gitea만 사용"
echo -e "  • 인증:            내부 전용 (비활성화됨)"

echo ""
