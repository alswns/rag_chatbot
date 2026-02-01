#!/bin/bash
# =============================================================================
# 리눅스 LLM 서버 시작 스크립트
# 환경: Ubuntu + NVIDIA GPU (AWS g5.xlarge 등)
# 용도: vLLM + ChromaDB + RAG API 서버 시작
# =============================================================================

set -e

# 스크립트 위치 기준 경로 설정 및 git pull
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 최신 코드 자동 pull
if [ -d .git ]; then
    echo -e "${YELLOW}[0/4] git pull로 최신 코드 동기화...${NC}"
    git pull origin main --rebase --autostash || true
fi

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   🚀 RAG LLM Server 시작 (Ubuntu + GPU)${NC}"
echo -e "${BLUE}============================================================${NC}"

# 1. GPU 확인
echo -e "\n${YELLOW}[1/4] GPU 확인 중...${NC}"
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    echo -e "${GREEN}✓ GPU 감지: ${GPU_NAME} (${GPU_MEMORY})${NC}"
else
    echo -e "${RED}✗ NVIDIA GPU 미감지${NC}"
    echo -e "${YELLOW}  GPU 없이 실행하려면 docker-compose-server.yml에서 vLLM 서비스를 수정하세요${NC}"
    exit 1
fi

# 2. 환경변수 확인 (.env가 git 루트에 위치해야 함)
echo -e "\n${YELLOW}[2/4] 환경변수 확인 중...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${RED}✗ .env 파일이 없습니다 (git 루트에 위치해야 함)${NC}"
    echo -e "${YELLOW}  .env.example을 복사하여 설정하세요${NC}"
    exit 1
fi
source .env
echo -e "${GREEN}✓ .env 로드 완료${NC}"

# 3. 데이터 디렉토리 확인 (git 루트 하위 data)
echo -e "\n${YELLOW}[3/4] 데이터 디렉토리 확인 중...${NC}"
mkdir -p data/chroma
if [ -f "data/graph.pkl" ]; then
    echo -e "${GREEN}✓ 그래프 파일 존재: data/graph.pkl${NC}"
else
    echo -e "${YELLOW}⚠ 그래프 파일 없음 (첫 동기화 필요)${NC}"
fi

if [ -d "data/chroma" ] && [ "$(ls -A data/chroma 2>/dev/null)" ]; then
    echo -e "${GREEN}✓ ChromaDB 데이터 존재${NC}"
else
    echo -e "${YELLOW}⚠ ChromaDB 데이터 없음 (첫 동기화 필요)${NC}"
fi

# 4. Docker Compose 시작
echo -e "\n${YELLOW}[4/4] Docker Compose 시작 중...${NC}"

# 기존 컨테이너 정리
echo -e "${BLUE}  기존 컨테이너 정리...${NC}"
docker compose -f docker-compose-server.yml down --remove-orphans 2>/dev/null || true

# 이미지 빌드 및 시작
echo -e "${BLUE}  이미지 빌드 및 시작...${NC}"
docker compose -f docker-compose-server.yml build --parallel
docker compose -f docker-compose-server.yml up -d

# 5. 상태 확인
echo -e "\n${BLUE}============================================================${NC}"
echo -e "${GREEN}✅ RAG 서버 시작 완료!${NC}"
echo -e "${BLUE}============================================================${NC}"

echo -e "\n${YELLOW}📊 서비스 상태:${NC}"
docker compose -f docker-compose-server.yml ps

echo -e "\n${YELLOW}🔗 접속 URL:${NC}"
echo -e "  • vLLM API:     http://localhost:8000"
echo -e "  • ChromaDB:     http://localhost:8001"
echo -e "  • RAG API:      http://localhost:8010"
echo -e "  • Open WebUI:   http://localhost:3000 (설정된 경우)"

echo -e "\n${YELLOW}📋 로그 확인:${NC}"
echo -e "  docker compose -f docker-compose-server.yml logs -f"

echo -e "\n${YELLOW}🛑 서버 중지:${NC}"
echo -e "  docker compose -f docker-compose-server.yml down"
