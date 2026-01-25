#!/bin/bash

# ============================================================
# RAG 시스템 통합 실행 스크립트 (DeepSeek-R1 & AWS Optimized)
# ============================================================

set -e # 에러 발생 시 즉시 중단

# 스크립트 위치 기준 경로 설정
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   Enterprise RAG System Deployment (Chief Architect)${NC}"
echo -e "${BLUE}============================================================${NC}"

# 1. GPU 및 CUDA 감지
echo -e "\n${YELLOW}[1/4] GPU 인프라 감지 중...${NC}"
HAS_CUDA=false
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader)
    echo -e "${GREEN}✓ NVIDIA GPU 감지됨: $GPU_NAME${NC}"
    HAS_CUDA=true
else
    echo -e "${RED}⚠ NVIDIA GPU 미감지 (CPU 모드로 실행됩니다)${NC}"
fi

# 2. 백엔드 결정 (vLLM vs Ollama)
echo -e "\n${YELLOW}[2/4] LLM 백엔드 프로파일 선택${NC}"
if [ "$HAS_CUDA" = true ]; then
    PROFILES="vllm"
    MODEL_INFO="DeepSeek-R1-Distill-Qwen-14B (AWQ)"
    echo -e "${GREEN}✓ 선택된 모드: vLLM (GPU 가속 추론)${NC}"
else
    PROFILES="ollama"
    MODEL_INFO="DeepSeek-R1-Distill-Qwen-7B (Ollama)"
    echo -e "${BLUE}✓ 선택된 모드: Ollama (CPU 최적화)${NC}"
fi

# 3. Docker 빌드 및 실행
echo -e "\n${YELLOW}[3/4] Docker 컨테이너 빌드 및 시작${NC}"
echo -e "${BLUE}  기존 컨테이너 정리 중...${NC}"
docker compose --profile $PROFILES down --remove-orphans 2>/dev/null || true

echo -e "${BLUE}  이미지 빌드 및 실행 중 (이 과정은 소요시간이 걸릴 수 있습니다)...${NC}"
# ✅ [핵심] --build 옵션을 통해 'rag-backend' 이미지를 로컬에서 직접 생성합니다.
docker compose --profile $PROFILES up -d --build

# 4. 상태 확인 및 모델 로딩 대기 안내
echo -e "\n${YELLOW}[4/4] 시스템 배포 상태 확인${NC}"
docker compose --profile $PROFILES ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}🚀 RAG 시스템이 성공적으로 배포되었습니다!${NC}"
echo -e "${GREEN}============================================================${NC}"

echo -e "\n${BLUE}중요 안내:${NC}"
echo -e "  • ${YELLOW}모델 로딩:${NC} DeepSeek-R1(14B) 모델은 약 9GB이며 로딩에 3~5분이 소요됩니다."
echo -e "  • ${YELLOW}진행 확인:${NC} 다음 명령어로 모델 로딩 로그를 확인하세요:"
echo -e "    ${GREEN}docker compose logs -f vllm${NC}"

echo -e "\n${BLUE}서비스 URL:${NC}"
echo -e "  • Open WebUI:      ${GREEN}http://localhost:3000${NC}"
echo -e "  • RAG API Server:  ${GREEN}http://localhost:9000${NC}"
echo -e "  • vLLM Backend:    ${GREEN}http://localhost:8000${NC}"

echo -e "\n${BLUE}시스템 정보:${NC}"
echo -e "  • LLM 모델:        $MODEL_INFO"
echo -e "  • 백엔드:          $LLM_BACKEND"
echo -e "  • 데이터 저장:     ./data 디렉토리 (Persistent)"

echo -e "\n${YELLOW}Chef's Pro-tip:${NC} API가 404를 뱉는다면 vLLM이 모델을 아직 로드 중인 것입니다."
echo ""