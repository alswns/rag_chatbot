#!/bin/bash
# =============================================================================
# Notion 데이터 동기화 스크립트
# 환경: Mac (Apple Silicon) 또는 Linux
# 용도: docker-compose-sync.yml로 Notion → ChromaDB 동기화
# =============================================================================

set -e

# 스크립트 위치 기준 경로 설정
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 옵션 파싱
DETACH=false
BUILD=false
DASHBOARD_ONLY=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -d|--detach) DETACH=true ;;
        -b|--build) BUILD=true ;;
        --dashboard) DASHBOARD_ONLY=true ;;
        -h|--help)
            echo "사용법: $0 [옵션]"
            echo ""
            echo "옵션:"
            echo "  -d, --detach     백그라운드 실행"
            echo "  -b, --build      이미지 재빌드"
            echo "  --dashboard      대시보드만 시작"
            echo "  -h, --help       도움말 표시"
            exit 0
            ;;
        *) echo "알 수 없는 옵션: $1"; exit 1 ;;
    esac
    shift
done

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   📥 Notion 데이터 동기화${NC}"
echo -e "${BLUE}============================================================${NC}"

# 1. 환경변수 확인
echo -e "\n${YELLOW}[1/4] 환경변수 확인 중...${NC}"
if [ ! -f ".env" ]; then
    echo -e "${RED}✗ .env 파일이 없습니다${NC}"
    exit 1
fi
source .env

if [ -z "$NOTION_TOKEN" ]; then
    echo -e "${RED}✗ NOTION_TOKEN이 설정되지 않았습니다${NC}"
    exit 1
fi
echo -e "${GREEN}✓ NOTION_TOKEN 확인됨${NC}"

if [ -n "$NOTION_DATABASE_ID" ]; then
    echo -e "${GREEN}✓ NOTION_DATABASE_ID: ${NOTION_DATABASE_ID:0:8}...${NC}"
else
    echo -e "${YELLOW}⚠ NOTION_DATABASE_ID 미설정 (전체 워크스페이스 검색)${NC}"
fi

# 2. 데이터 디렉토리 준비
echo -e "\n${YELLOW}[2/4] 데이터 디렉토리 준비 중...${NC}"
mkdir -p data/chroma
echo -e "${GREEN}✓ data/chroma 디렉토리 준비 완료${NC}"

# 3. Docker Compose 실행
echo -e "\n${YELLOW}[3/4] Docker Compose 실행 중...${NC}"

COMPOSE_CMD="docker compose -f docker-compose-sync.yml"

if [ "$DASHBOARD_ONLY" = true ]; then
    echo -e "${BLUE}  대시보드만 시작...${NC}"
    $COMPOSE_CMD up -d dashboard
    echo -e "\n${GREEN}✅ 대시보드 시작 완료!${NC}"
    echo -e "  • 대시보드: http://localhost:8080"
    exit 0
fi

# 기존 컨테이너 정리
echo -e "${BLUE}  기존 컨테이너 정리...${NC}"
$COMPOSE_CMD down --remove-orphans 2>/dev/null || true

# 빌드 옵션
if [ "$BUILD" = true ]; then
    echo -e "${BLUE}  이미지 빌드 중...${NC}"
    $COMPOSE_CMD build --parallel
fi

# 실행
if [ "$DETACH" = true ]; then
    echo -e "${BLUE}  백그라운드 실행...${NC}"
    $COMPOSE_CMD up -d
else
    echo -e "${BLUE}  포그라운드 실행 (Ctrl+C로 중지)...${NC}"
    $COMPOSE_CMD up
fi

# 4. 결과 확인 (백그라운드 모드일 때만)
if [ "$DETACH" = true ]; then
    echo -e "\n${YELLOW}[4/4] 서비스 상태 확인 중...${NC}"
    sleep 3
    $COMPOSE_CMD ps

    echo -e "\n${BLUE}============================================================${NC}"
    echo -e "${GREEN}✅ 동기화 시작 완료!${NC}"
    echo -e "${BLUE}============================================================${NC}"

    echo -e "\n${YELLOW}🔗 접속 URL:${NC}"
    echo -e "  • ChromaDB:   http://localhost:8001"
    echo -e "  • 대시보드:   http://localhost:8080"

    echo -e "\n${YELLOW}📋 로그 확인:${NC}"
    echo -e "  docker compose -f docker-compose-sync.yml logs -f sync-worker"

    echo -e "\n${YELLOW}📊 동기화 진행 확인:${NC}"
    echo -e "  cat data/sync_progress.json"

    echo -e "\n${YELLOW}🛑 중지:${NC}"
    echo -e "  docker compose -f docker-compose-sync.yml down"
fi
