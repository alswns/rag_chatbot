#!/bin/bash
# =============================================================================
# 로컬 데이터를 원격 서버로 업로드하는 스크립트
# 용도: Mac에서 수집한 데이터를 Ubuntu 서버로 전송
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

# =============================================================================
# 설정 (환경에 맞게 수정하세요)
# =============================================================================
REMOTE_USER="${REMOTE_USER:-ubuntu}"
REMOTE_HOST="${REMOTE_HOST:-your-server.com}"
REMOTE_PORT="${REMOTE_PORT:-22}"
REMOTE_PATH="${REMOTE_PATH:-/home/ubuntu/rag}"
SSH_KEY="${SSH_KEY:-~/.ssh/id_rsa}"

# 옵션 파싱
DRY_RUN=false
VERBOSE=false

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -n|--dry-run) DRY_RUN=true ;;
        -v|--verbose) VERBOSE=true ;;
        -u|--user) REMOTE_USER="$2"; shift ;;
        -h|--host) REMOTE_HOST="$2"; shift ;;
        -p|--port) REMOTE_PORT="$2"; shift ;;
        -r|--remote-path) REMOTE_PATH="$2"; shift ;;
        -k|--key) SSH_KEY="$2"; shift ;;
        --help)
            echo "사용법: $0 [옵션]"
            echo ""
            echo "옵션:"
            echo "  -u, --user USER        원격 사용자명 (기본: ubuntu)"
            echo "  -h, --host HOST        원격 호스트 (기본: your-server.com)"
            echo "  -p, --port PORT        SSH 포트 (기본: 22)"
            echo "  -r, --remote-path PATH 원격 경로 (기본: /home/ubuntu/rag)"
            echo "  -k, --key KEY          SSH 키 경로 (기본: ~/.ssh/id_rsa)"
            echo "  -n, --dry-run          실제 전송 없이 테스트"
            echo "  -v, --verbose          상세 출력"
            echo "  --help                 도움말 표시"
            echo ""
            echo "환경변수:"
            echo "  REMOTE_USER, REMOTE_HOST, REMOTE_PORT, REMOTE_PATH, SSH_KEY"
            exit 0
            ;;
        *) echo "알 수 없는 옵션: $1"; exit 1 ;;
    esac
    shift
done

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}   📤 서버로 데이터 업로드${NC}"
echo -e "${BLUE}============================================================${NC}"

# 설정 확인
echo -e "\n${YELLOW}📋 업로드 설정:${NC}"
echo -e "  • 원격 서버:  ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
echo -e "  • 원격 경로:  ${REMOTE_PATH}"
echo -e "  • SSH 키:     ${SSH_KEY}"

if [ "$REMOTE_HOST" = "your-server.com" ]; then
    echo -e "\n${RED}✗ 원격 호스트가 설정되지 않았습니다${NC}"
    echo -e "${YELLOW}  스크립트 상단의 REMOTE_HOST를 수정하거나 --host 옵션을 사용하세요${NC}"
    exit 1
fi

# SSH 키 확인
if [ ! -f "${SSH_KEY/#\~/$HOME}" ]; then
    echo -e "\n${RED}✗ SSH 키를 찾을 수 없습니다: ${SSH_KEY}${NC}"
    exit 1
fi

# 1. 로컬 데이터 확인
echo -e "\n${YELLOW}[1/4] 로컬 데이터 확인 중...${NC}"

DATA_SIZE=$(du -sh data 2>/dev/null | cut -f1 || echo "N/A")
echo -e "  • 데이터 크기: ${DATA_SIZE}"

if [ -f "data/graph.pkl" ]; then
    GRAPH_SIZE=$(du -h data/graph.pkl | cut -f1)
    echo -e "  ${GREEN}✓ graph.pkl: ${GRAPH_SIZE}${NC}"
else
    echo -e "  ${RED}✗ graph.pkl 없음${NC}"
fi

if [ -d "data/chroma" ] && [ "$(ls -A data/chroma 2>/dev/null)" ]; then
    CHROMA_SIZE=$(du -sh data/chroma | cut -f1)
    echo -e "  ${GREEN}✓ chroma/: ${CHROMA_SIZE}${NC}"
else
    echo -e "  ${RED}✗ chroma/ 비어있음${NC}"
fi

# 2. SSH 연결 테스트
echo -e "\n${YELLOW}[2/4] SSH 연결 테스트 중...${NC}"
SSH_CMD="ssh -i ${SSH_KEY/#\~/$HOME} -p ${REMOTE_PORT} -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new"

if $SSH_CMD ${REMOTE_USER}@${REMOTE_HOST} "echo 'SSH 연결 성공'" 2>/dev/null; then
    echo -e "${GREEN}✓ SSH 연결 성공${NC}"
else
    echo -e "${RED}✗ SSH 연결 실패${NC}"
    exit 1
fi

# 3. 원격 디렉토리 준비
echo -e "\n${YELLOW}[3/4] 원격 디렉토리 준비 중...${NC}"
$SSH_CMD ${REMOTE_USER}@${REMOTE_HOST} "mkdir -p ${REMOTE_PATH}/data/chroma"
echo -e "${GREEN}✓ 원격 디렉토리 생성 완료${NC}"

# 4. 데이터 전송
echo -e "\n${YELLOW}[4/5] 데이터 전송 중...${NC}"

RSYNC_OPTS="-avz --progress"
if [ "$VERBOSE" = true ]; then
    RSYNC_OPTS="$RSYNC_OPTS -v"
fi
if [ "$DRY_RUN" = true ]; then
    RSYNC_OPTS="$RSYNC_OPTS --dry-run"
    echo -e "${YELLOW}  (Dry-run 모드 - 실제 전송 없음)${NC}"
fi

echo -e "${BLUE}  graph.pkl 전송 중...${NC}"
if [ -f "data/graph.pkl" ]; then
    rsync $RSYNC_OPTS \
        -e "ssh -i ${SSH_KEY/#\~/$HOME} -p ${REMOTE_PORT}" \
        data/graph.pkl \
        ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/data/
fi

echo -e "${BLUE}  sync_progress.json 전송 중...${NC}"
if [ -f "data/sync_progress.json" ]; then
    rsync $RSYNC_OPTS \
        -e "ssh -i ${SSH_KEY/#\~/$HOME} -p ${REMOTE_PORT}" \
        data/sync_progress.json \
        ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/data/
fi

echo -e "${BLUE}  chroma/ 전송 중...${NC}"
if [ -d "data/chroma" ]; then
    rsync $RSYNC_OPTS --delete \
        -e "ssh -i ${SSH_KEY/#\~/$HOME} -p ${REMOTE_PORT}" \
        data/chroma/ \
        ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/data/chroma/
fi

# 5. .env 파일 전송 (별도)
echo -e "\n${YELLOW}[5/5] .env 파일 전송 중...${NC}"
if [ -f ".env" ]; then
    # scp로 .env 파일 별도 전송 (rsync 대신)
    if [ "$DRY_RUN" = true ]; then
        echo -e "${YELLOW}  (Dry-run) scp .env → ${REMOTE_PATH}/.env${NC}"
    else
        scp -i ${SSH_KEY/#\~/$HOME} -P ${REMOTE_PORT} \
            .env \
            ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PATH}/.env
        echo -e "${GREEN}✓ .env 전송 완료${NC}"
    fi
else
    echo -e "${RED}✗ .env 파일 없음 (스킵)${NC}"
fi

# 완료
echo -e "\n${BLUE}============================================================${NC}"
if [ "$DRY_RUN" = true ]; then
    echo -e "${YELLOW}🔍 Dry-run 완료 (실제 전송 없음)${NC}"
else
    echo -e "${GREEN}✅ 데이터 업로드 완료!${NC}"
fi
echo -e "${BLUE}============================================================${NC}"

echo -e "\n${YELLOW}📋 다음 단계:${NC}"
echo -e "  1. 서버에 SSH 접속:"
echo -e "     ssh -i ${SSH_KEY} -p ${REMOTE_PORT} ${REMOTE_USER}@${REMOTE_HOST}"
echo -e ""
echo -e "  2. 서버에서 RAG 시작:"
echo -e "     cd ${REMOTE_PATH}"
echo -e "     ./scripts/start-server.sh"
