#!/bin/bash

# ============================================================
# RAG 시스템 운영 관리 스크립트
# 시작, 중지, 재시작, 백업, 모니터링 등의 기능 제공
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 색상 정의
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# 설정
BACKUP_DIR="./backups"
LOG_DIR="./logs"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# ============================================================
# 유틸리티 함수
# ============================================================

print_header() {
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================================${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# ============================================================
# 메인 명령어
# ============================================================

case "${1:-help}" in

# ============================================================
# START - 서비스 시작
# ============================================================
start)
    print_header "RAG 시스템 시작"
    
    # CUDA 감지
    echo -e "\n${YELLOW}CUDA 확인 중...${NC}"
    if command -v nvidia-smi &> /dev/null; then
        print_success "NVIDIA GPU 감지됨"
        nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
        LLM_BACKEND=${LLM_BACKEND:-auto}
    else
        print_warning "NVIDIA GPU 미감지"
        LLM_BACKEND=${LLM_BACKEND:-ollama}
    fi
    
    # 로그 디렉토리 생성
    mkdir -p "$LOG_DIR"
    
    # 컨테이너 시작
    echo -e "\n${YELLOW}Docker 컨테이너 시작 중...${NC}"
    docker-compose up -d 2>&1 | tee "$LOG_DIR/start_$TIMESTAMP.log"
    
    # 대기 (초기화)
    echo -e "\n${YELLOW}초기화 중... (15초 대기)${NC}"
    sleep 15
    
    # 상태 확인
    echo -e "\n${YELLOW}컨테이너 상태:${NC}"
    docker-compose ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    # 헬스체크
    echo -e "\n${YELLOW}헬스체크:${NC}"
    
    if curl -s http://localhost:3000 > /dev/null 2>&1; then
        print_success "Open WebUI (http://localhost:3000)"
    else
        print_warning "Open WebUI 아직 준비 중..."
    fi
    
    if curl -s http://localhost:9000/health > /dev/null 2>&1; then
        print_success "RAG API Server (http://localhost:9000)"
    else
        print_warning "RAG API Server 아직 준비 중..."
    fi
    
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_success "Ollama (localhost:11434 - 로컬 전용)"
    else
        print_warning "Ollama 아직 준비 중..."
    fi
    
    print_success "RAG 시스템이 시작되었습니다!"
    echo -e "\n${BLUE}다음 주소에서 접속하세요:${NC}"
    echo "  • Open WebUI: http://localhost:3000"
    echo "  • RAG API:    http://localhost:9000"
    ;;

# ============================================================
# STOP - 서비스 중지
# ============================================================
stop)
    print_header "RAG 시스템 중지"
    
    echo -e "${YELLOW}컨테이너 중지 중...${NC}"
    docker-compose down 2>&1 | tee "$LOG_DIR/stop_$TIMESTAMP.log"
    
    print_success "RAG 시스템이 중지되었습니다!"
    ;;

# ============================================================
# RESTART - 서비스 재시작
# ============================================================
restart)
    print_header "RAG 시스템 재시작"
    
    $0 stop
    sleep 3
    $0 start
    ;;

# ============================================================
# STATUS - 서비스 상태 확인
# ============================================================
status)
    print_header "RAG 시스템 상태"
    
    echo -e "${YELLOW}실행 중인 컨테이너:${NC}"
    docker-compose ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    
    echo -e "\n${YELLOW}리소스 사용량:${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
    
    echo -e "\n${YELLOW}디스크 사용량:${NC}"
    docker system df
    
    echo -e "\n${YELLOW}API 상태:${NC}"
    
    if curl -s http://localhost:9000/health > /dev/null 2>&1; then
        print_success "RAG API Server"
    else
        print_error "RAG API Server"
    fi
    
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_success "Ollama Server"
    else
        print_error "Ollama Server"
    fi
    
    if curl -s http://localhost:8001 > /dev/null 2>&1; then
        print_success "ChromaDB"
    else
        print_error "ChromaDB"
    fi
    ;;

# ============================================================
# LOGS - 로그 확인
# ============================================================
logs)
    print_header "RAG 시스템 로그"
    
    SERVICE=${2:-""}
    
    if [ -z "$SERVICE" ]; then
        echo -e "${YELLOW}전체 로그 보기:${NC}"
        docker-compose logs -f
    else
        echo -e "${YELLOW}[$SERVICE] 로그 보기:${NC}"
        docker-compose logs -f "$SERVICE"
    fi
    ;;

# ============================================================
# BACKUP - 데이터 백업
# ============================================================
backup)
    print_header "RAG 시스템 백업"
    
    mkdir -p "$BACKUP_DIR"
    BACKUP_PATH="$BACKUP_DIR/backup_$TIMESTAMP"
    mkdir -p "$BACKUP_PATH"
    
    echo -e "${YELLOW}백업 경로: $BACKUP_PATH${NC}"
    
    # 데이터 백업
    echo -e "\n${YELLOW}데이터 백업 중...${NC}"
    cp -r data/chromadb "$BACKUP_PATH/" 2>/dev/null || true
    cp -r data/open-webui "$BACKUP_PATH/" 2>/dev/null || true
    cp -r data/ollama "$BACKUP_PATH/" 2>/dev/null || true
    
    # 설정 백업
    echo -e "${YELLOW}설정 백업 중...${NC}"
    cp .env "$BACKUP_PATH/.env.backup" 2>/dev/null || true
    cp docker-compose.yml "$BACKUP_PATH/" 2>/dev/null || true
    
    # 압축
    echo -e "${YELLOW}압축 중...${NC}"
    tar -czf "$BACKUP_PATH.tar.gz" -C "$BACKUP_DIR" "backup_$TIMESTAMP" 2>/dev/null
    rm -rf "$BACKUP_PATH"
    
    # 백업 크기
    SIZE=$(du -h "$BACKUP_PATH.tar.gz" | cut -f1)
    print_success "백업 완료: $BACKUP_PATH.tar.gz ($SIZE)"
    ;;

# ============================================================
# RESTORE - 백업 복구
# ============================================================
restore)
    print_header "RAG 시스템 복구"
    
    BACKUP_FILE=${2:-""}
    
    if [ -z "$BACKUP_FILE" ]; then
        print_error "백업 파일을 지정해주세요"
        echo "사용법: $0 restore <backup_file>"
        echo ""
        echo "사용 가능한 백업:"
        ls -lh "$BACKUP_DIR"/*.tar.gz 2>/dev/null || echo "백업 파일이 없습니다"
        exit 1
    fi
    
    if [ ! -f "$BACKUP_FILE" ]; then
        print_error "백업 파일을 찾을 수 없습니다: $BACKUP_FILE"
        exit 1
    fi
    
    # 현재 데이터 백업
    print_warning "현재 데이터를 백업 중..."
    cp -r data "data_backup_$TIMESTAMP"
    
    # 복구
    echo -e "${YELLOW}복구 중...${NC}"
    tar -xzf "$BACKUP_FILE" -C "$BACKUP_DIR"
    RESTORE_DIR=$(tar -tzf "$BACKUP_FILE" | head -1 | cut -d/ -f1)
    
    # 데이터 복구
    cp -r "$BACKUP_DIR/$RESTORE_DIR/chromadb" data/ 2>/dev/null || true
    cp -r "$BACKUP_DIR/$RESTORE_DIR/open-webui" data/ 2>/dev/null || true
    cp -r "$BACKUP_DIR/$RESTORE_DIR/ollama" data/ 2>/dev/null || true
    
    # 정리
    rm -rf "$BACKUP_DIR/$RESTORE_DIR"
    
    print_success "복구 완료!"
    print_warning "서비스를 재시작해주세요: $0 restart"
    ;;

# ============================================================
# CLEAN - 불필요한 파일 정리
# ============================================================
clean)
    print_header "RAG 시스템 정리"
    
    echo -e "${YELLOW}Docker 리소스 정리 중...${NC}"
    docker system prune -f
    
    echo -e "${YELLOW}불필요한 이미지 삭제...${NC}"
    docker image prune -af
    
    echo -e "${YELLOW}불필요한 볼륨 정리...${NC}"
    docker volume prune -f
    
    print_success "정리 완료!"
    ;;

# ============================================================
# UPDATE - 이미지 업데이트
# ============================================================
update)
    print_header "RAG 시스템 업데이트"
    
    # 백업
    print_warning "업데이트 전 백업을 실행합니다..."
    $0 backup
    
    # 서비스 중지
    echo -e "\n${YELLOW}서비스 중지 중...${NC}"
    docker-compose down
    
    # 이미지 업데이트
    echo -e "${YELLOW}이미지 업데이트 중...${NC}"
    docker-compose pull
    docker-compose build
    
    # 서비스 재시작
    echo -e "${YELLOW}서비스 재시작 중...${NC}"
    docker-compose up -d
    
    print_success "업데이트 완료!"
    sleep 5
    $0 status
    ;;

# ============================================================
# PRUNE-LOGS - 로그 정리
# ============================================================
prune-logs)
    print_header "로그 정리"
    
    DAYS=${2:-30}
    
    echo -e "${YELLOW}$DAYS일 이상 된 로그 파일 삭제...${NC}"
    find "$LOG_DIR" -type f -mtime +"$DAYS" -delete
    
    print_success "로그 정리 완료!"
    ;;

# ============================================================
# HEALTH-CHECK - 상세 헬스체크
# ============================================================
health-check)
    print_header "상세 헬스체크"
    
    HEALTHY=0
    UNHEALTHY=0
    
    # Open WebUI
    echo -e "\n${YELLOW}Open WebUI 확인 중...${NC}"
    if curl -s -o /dev/null -w "%{http_code}" http://localhost:3000 | grep -q "200"; then
        print_success "Open WebUI"
        ((HEALTHY++))
    else
        print_error "Open WebUI"
        ((UNHEALTHY++))
    fi
    
    # RAG API
    echo -e "\n${YELLOW}RAG API 확인 중...${NC}"
    if curl -s http://localhost:9000/health > /dev/null 2>&1; then
        print_success "RAG API Server"
        ((HEALTHY++))
    else
        print_error "RAG API Server"
        ((UNHEALTHY++))
    fi
    
    # Ollama
    echo -e "\n${YELLOW}Ollama 확인 중...${NC}"
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        print_success "Ollama"
        MODELS=$(curl -s http://localhost:11434/api/tags | grep -o '"name":"[^"]*"' | wc -l)
        echo "  설치된 모델: $MODELS개"
        ((HEALTHY++))
    else
        print_error "Ollama"
        ((UNHEALTHY++))
    fi
    
    # ChromaDB
    echo -e "\n${YELLOW}ChromaDB 확인 중...${NC}"
    if curl -s http://localhost:8001/api/v1/heartbeat > /dev/null 2>&1; then
        print_success "ChromaDB"
        ((HEALTHY++))
    else
        print_error "ChromaDB"
        ((UNHEALTHY++))
    fi
    
    # 요약
    echo -e "\n${BLUE}요약: ${GREEN}$HEALTHY 정상${NC} / ${RED}$UNHEALTHY 문제${NC}"
    ;;

# ============================================================
# HELP - 도움말
# ============================================================
help)
    cat << EOF
${BLUE}RAG 시스템 운영 관리 스크립트${NC}

${BLUE}사용법: $0 <명령어> [옵션]${NC}

${GREEN}서비스 관리:${NC}
  start           서비스 시작
  stop            서비스 중지
  restart         서비스 재시작
  status          서비스 상태 확인

${GREEN}로깅 & 모니터링:${NC}
  logs [service]  로그 확인 (서비스 지정 가능)
  health-check    상세 헬스체크

${GREEN}백업 & 복구:${NC}
  backup          데이터 백업
  restore <file>  백업 복구

${GREEN}유지보수:${NC}
  update          이미지 업데이트
  clean           불필요한 파일 정리
  prune-logs [일수] 오래된 로그 삭제 (기본값: 30일)

${GREEN}기타:${NC}
  help            도움말 표시

${BLUE}예시:${NC}
  $0 start                    # 서비스 시작
  $0 logs sync-worker         # sync-worker 로그 보기
  $0 backup                   # 데이터 백업
  $0 restore backups/backup_20260125_120000.tar.gz
  $0 prune-logs 7             # 7일 이상 된 로그 삭제

EOF
    ;;

*)
    print_error "알 수 없는 명령어: $1"
    echo "도움말: $0 help"
    exit 1
    ;;
esac
