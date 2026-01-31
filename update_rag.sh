#!/bin/bash
# =============================================================================
# RAG 무중단 데이터 갱신 스크립트
# 용도: Crontab에서 매시간 실행하여 Notion 데이터 동기화
# 사용법: 0 * * * * /path/to/update_rag.sh
# =============================================================================

set -e

# 설정
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
COMPOSE_FILE="${SCRIPT_DIR}/docker-compose-server.yml"
LOG_DIR="${SCRIPT_DIR}/logs"
LOG_FILE="${LOG_DIR}/sync.log"
LOCK_FILE="/tmp/rag_sync.lock"

# 로그 디렉토리 생성
mkdir -p "$LOG_DIR"

# 타임스탬프 함수
timestamp() {
    date "+%Y-%m-%d %H:%M:%S"
}

# 로깅 함수
log() {
    echo "[$(timestamp)] $1" | tee -a "$LOG_FILE"
}

log_error() {
    echo "[$(timestamp)] [ERROR] $1" | tee -a "$LOG_FILE" >&2
}

# 중복 실행 방지
if [ -f "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)
    if kill -0 "$LOCK_PID" 2>/dev/null; then
        log_error "이전 동기화 작업이 아직 실행 중입니다 (PID: $LOCK_PID)"
        exit 1
    else
        log "고아 락 파일 제거"
        rm -f "$LOCK_FILE"
    fi
fi

# 락 파일 생성
echo $$ > "$LOCK_FILE"
trap "rm -f $LOCK_FILE" EXIT

# =============================================================================
# 메인 로직
# =============================================================================

log "=========================================="
log "RAG 데이터 동기화 시작"
log "=========================================="

# 1. Sync Worker 실행 (1회성)
log "[1/3] Sync Worker 실행 중..."

SYNC_START=$(date +%s)

if docker compose -f "$COMPOSE_FILE" run --rm sync-worker 2>&1 | tee -a "$LOG_FILE"; then
    SYNC_END=$(date +%s)
    SYNC_DURATION=$((SYNC_END - SYNC_START))
    log "[1/3] ✓ Sync Worker 완료 (소요시간: ${SYNC_DURATION}초)"
else
    log_error "[1/3] ✗ Sync Worker 실패"
    exit 1
fi

# 2. graph.pkl 존재 확인
GRAPH_FILE="${SCRIPT_DIR}/data/graph.pkl"
if [ -f "$GRAPH_FILE" ]; then
    GRAPH_SIZE=$(du -h "$GRAPH_FILE" | cut -f1)
    log "[2/3] ✓ 그래프 파일 확인 (크기: $GRAPH_SIZE)"
else
    log_error "[2/3] ✗ 그래프 파일 없음: $GRAPH_FILE"
    exit 1
fi

# 3. RAG API만 재시작 (vLLM, WebUI는 유지)
log "[3/3] RAG API 재시작 중..."

RESTART_START=$(date +%s)

if docker compose -f "$COMPOSE_FILE" restart rag-api 2>&1 | tee -a "$LOG_FILE"; then
    RESTART_END=$(date +%s)
    RESTART_DURATION=$((RESTART_END - RESTART_START))
    log "[3/3] ✓ RAG API 재시작 완료 (소요시간: ${RESTART_DURATION}초)"
else
    log_error "[3/3] ✗ RAG API 재시작 실패"
    exit 1
fi

# 4. 헬스체크
log "[검증] RAG API 헬스체크 중..."
sleep 3

MAX_RETRIES=10
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -sf http://localhost:8010/health > /dev/null 2>&1; then
        log "[검증] ✓ RAG API 정상 작동 중"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    log "[검증] 재시도 중... ($RETRY_COUNT/$MAX_RETRIES)"
    sleep 2
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    log_error "[검증] ✗ RAG API 헬스체크 실패"
    exit 1
fi

# 완료
TOTAL_END=$(date +%s)
TOTAL_DURATION=$((TOTAL_END - SYNC_START))

log "=========================================="
log "✓ RAG 데이터 동기화 완료"
log "  - 총 소요시간: ${TOTAL_DURATION}초"
log "  - 그래프 크기: $GRAPH_SIZE"
log "  - 다음 동기화: 1시간 후"
log "=========================================="

exit 0
