#!/bin/bash
# =============================================================================
# RAG API 서버 업데이트 스크립트
# 용도: vLLM, ChromaDB는 유지하고 rag-api 코드만 갱신 후 재시작
# =============================================================================

set -e

# 스크립트 위치 기준 경로 설정 (프로젝트 루트로 이동)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   🔄 RAG API 컨테이너 업데이트 (Zero Downtime for LLM)${NC}"
echo -e "${BLUE}============================================================${NC}"

# 1. 최신 코드 가져오기
echo -e "\n${YELLOW}[1/4] Git Pull (최신 코드 동기화)...${NC}"
if [ -d .git ]; then
    git pull origin main --rebase --autostash || true
    echo -e "${GREEN}✓ 코드 동기화 완료${NC}"
else
    echo -e "${RED}✗ .git 디렉토리가 없습니다. 코드 갱신을 건너뜁니다.${NC}"
fi

# 2. 환경변수 확인 (API 실행에 필수)
echo -e "\n${YELLOW}[2/4] 환경변수 확인 중...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${RED}✗ .env 파일이 없습니다!${NC}"
    exit 1
fi
echo -e "${GREEN}✓ .env 확인 완료${NC}"

# 3. Docker 컨테이너 재빌드 및 재시작 (핵심 단계)
# --build: 이미지 새로 빌드 (pip install 등 반영)
# --no-deps: vLLM, ChromaDB 등 연관 컨테이너는 재시작하지 않음 (시간 절약)
# rag-api: 특정 서비스만 지정
echo -e "\n${YELLOW}[3/4] rag-api 컨테이너 재빌드 및 교체 중...${NC}"
docker compose -f docker-compose-server.yml up -d --build --no-deps rag-api

# 불필요한 댕글링 이미지 정리 (디스크 공간 확보)
docker image prune -f > /dev/null 2>&1

# 4. 상태 확인 및 로그 모니터링
echo -e "\n${YELLOW}[4/4] 서비스 상태 확인...${NC}"

# 잠시 대기 후 상태 확인
sleep 2
CONTAINER_STATE=$(docker compose -f docker-compose-server.yml ps -q rag-api | xargs docker inspect -f '{{.State.Status}}' 2>/dev/null)

if [ "$CONTAINER_STATE" == "running" ]; then
    echo -e "${GREEN}✅ rag-api 업데이트 및 재시작 완료!${NC}"
    echo -e "${BLUE}============================================================${NC}"
    
    echo -e "\n${YELLOW}📋 최근 로그 (5초간 모니터링, Ctrl+C로 종료 가능):${NC}"
    # 5초 동안 로그를 보여주고 자동으로 빠져나옴 (에러 확인용)
    timeout 5s docker compose -f docker-compose-server.yml logs -f rag-api || true
    
    echo -e "\n${GREEN}🚀 서버가 정상적으로 실행 중입니다.${NC}"
else
    echo -e "${RED}❌ rag-api 시작 실패! 로그를 확인하세요.${NC}"
    docker compose -f docker-compose-server.yml logs --tail=20 rag-api
    exit 1
fi