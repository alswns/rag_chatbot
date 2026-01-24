# 📦 RAG 시스템 배포 & 운영 가이드

## 🚀 배포 체크리스트

### 사전 준비 (Pre-Deployment)

- [ ] Docker & Docker Compose 설치
- [ ] 시스템 요구사항 확인
- [ ] `.env` 파일 준비
- [ ] 데이터 디렉토리 준비
- [ ] 방화벽/포트 설정 확인

---

## 📋 시스템 요구사항

### 최소 사양

```
OS: Linux, macOS, Windows (Docker Desktop)
RAM: 8GB
CPU: 4코어
디스크: 50GB (모델 + 데이터)
네트워크: 안정적인 인터넷 (모델 다운로드용)
```

### 권장 사양

```
OS: Linux (Ubuntu 20.04+)
RAM: 16GB
CPU: 8코어
디스크: 100GB
GPU: NVIDIA GPU (vLLM 사용시)
```

---

## 🔧 배포 절차

### 1단계: 저장소 준비

```bash
# 프로젝트 디렉토리로 이동
cd /path/to/rag

# Git에서 최신 버전 가져오기 (선택사항)
git pull origin main

# 디렉토리 구조 확인
ls -la
```

### 2단계: 환경 설정

```bash
# .env 파일 복사
cp .env.example .env  # 또는 기존 .env 파일 사용

# .env 파일 편집 (필수 항목 입력)
# - NOTION_TOKEN
# - GITHUB_TOKEN
# - GITEA_URL / GITEA_TOKEN
# - HF_TOKEN (vLLM 사용시)
# - LLM_BACKEND=auto 또는 ollama/vllm
```

**주요 환경변수:**

```env
# LLM 백엔드 선택
LLM_BACKEND=auto

# 외부 API 토큰
NOTION_TOKEN=your_notion_token
GITHUB_TOKEN=your_github_token
GITEA_URL=https://your-gitea.com
GITEA_TOKEN=your_gitea_token

# 모델 (vLLM 사용시)
HF_TOKEN=your_huggingface_token

# 시스템 설정
SYNC_INTERVAL=600
LOG_LEVEL=INFO
SEARCH_TOP_K=5
```

### 3단계: 데이터 디렉토리 생성

```bash
# 데이터 저장 디렉토리 생성
mkdir -p data/{ollama,chromadb,vllm-models,open-webui}

# 권한 설정
chmod 755 data
```

### 4단계: Docker 이미지 빌드

```bash
# Sync Worker 이미지 빌드
docker-compose build sync-worker

# 또는 전체 이미지 빌드
docker-compose build
```

### 5단계: 서비스 시작

**개발/테스트 환경:**

```bash
./start-rag.sh
```

**프로덕션 환경:**

```bash
# CUDA 감지 무시하고 Ollama로 실행
LLM_BACKEND=ollama docker-compose up -d

# 또는 vLLM 강제 실행 (GPU 있는 환경)
LLM_BACKEND=vllm docker-compose up -d
```

### 6단계: 상태 확인

```bash
# 컨테이너 상태 확인
docker-compose ps

# 로그 확인
docker-compose logs -f

# 서비스 헬스체크
curl http://localhost:3000        # Open WebUI
curl http://localhost:9000/health # RAG API
curl http://localhost:11434/api/tags # Ollama (로컬만)
```

---

## 🎯 운영 (Operations)

### 일일 운영

#### 시작

```bash
cd /Users/bagminjun/Desktop/rag

# 방법 1: 자동 스크립트 (추천)
./start-rag.sh

# 방법 2: 수동 시작
docker-compose up -d
```

#### 중지

```bash
# 우아한 종료 (graceful shutdown)
docker-compose down

# 강제 종료
docker-compose kill
```

#### 재시작

```bash
# 특정 서비스만 재시작
docker-compose restart sync-worker

# 전체 재시작
docker-compose down && docker-compose up -d
```

#### 로그 확인

```bash
# 실시간 로그
docker-compose logs -f sync-worker

# 최근 100줄 로그
docker-compose logs --tail=100

# 특정 시간 이후 로그
docker-compose logs --since 1h
```

---

### 정기 유지보수

#### 주 1회: 데이터 백업

```bash
#!/bin/bash
# backup-rag.sh

BACKUP_DIR="/backup/rag-$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# 데이터 백업
cp -r data/chromadb "$BACKUP_DIR/"
cp -r data/open-webui "$BACKUP_DIR/"

# 환경 변수 백업
cp .env "$BACKUP_DIR/.env.backup"

echo "Backup completed: $BACKUP_DIR"
```

#### 월 1회: 로그 정리

```bash
# 30일 이상 된 로그 삭제
docker-compose logs --since 30d | wc -l

# 또는 수동으로
docker system prune -a  # 사용하지 않는 이미지 삭제
```

#### 월 1회: 이미지 업데이트

```bash
# 최신 이미지로 업데이트
docker-compose pull

# 업데이트된 이미지로 재시작
docker-compose up -d
```

---

## 🔍 모니터링

### CPU & 메모리 사용량

```bash
# 컨테이너별 리소스 사용량
docker stats

# 특정 컨테이너만
docker stats rag-ollama rag-chromadb
```

### 디스크 사용량

```bash
# 전체 Docker 디스크 사용량
docker system df

# 데이터 디렉토리 사용량
du -sh data/
du -sh data/*
```

### API 상태 확인

```bash
# RAG API 헬스체크
curl http://localhost:9000/health

# Ollama 모델 확인
curl http://localhost:11434/api/tags

# ChromaDB 상태
curl http://localhost:8001/api/v1/heartbeat
```

---

## 🆘 문제 해결

### 컨테이너가 시작되지 않음

```bash
# 로그 확인
docker-compose logs <service-name>

# 컨테이너 상태 확인
docker inspect <container-name>

# 강제 재시작
docker-compose down
docker-compose up -d --force-recreate
```

### 메모리 부족

**증상**: "llama runner process has terminated: signal: killed" 에러

```bash
# 1️⃣ 시스템 메모리 확인
free -h

# 2️⃣ 더 가벼운 모델로 변경
# docker-compose.yml의 ollama entrypoint 수정:
# qwen2.5 대신 mistral, neural-chat, orca-mini 사용

# 3️⃣ 불필요한 컨테이너 정리
docker-compose down
docker image prune -a
docker volume prune

# 4️⃣ 환경변수로 메모리 제한
# docker-compose.yml의 ollama environment에 추가:
# - OLLAMA_MEMORY_FRACTION=0.8  # 80% 메모리만 사용
```

**권장 모델 (메모리 요구사항 기준)**:

```
Qwen2.5 (7B)     → 4.3GB RAM (현재 설치됨) - 가장 좋음
Mistral (7B)     → 3.5GB RAM (권장) - 빠르고 가벼움
Neural-Chat (7B) → 3.0GB RAM - 더 가벼움
Orca-Mini (3B)   → 1.5GB RAM - 매우 가벼움
```

**모델 변경 방법**:

```bash
# 1. docker-compose.yml 수정
nano docker-compose.yml

# 2. ollama entrypoint 변경
# entrypoint: /bin/bash -c "ollama serve & sleep 5 && ollama pull mistral && wait"

# 3. 재시작
docker-compose down
rm -rf data/ollama/*
docker-compose up -d ollama

# 4. 모델 확인
curl http://localhost:11434/api/tags
```

**메모리 최적화 팁**:

```bash
# macOS에서 Docker 메모리 증설
# Docker Desktop → Preferences → Resources → Memory 증가

# 또는 Ollama 메모리 제한 설정
export OLLAMA_MEMORY_FRACTION=0.8  # 80% 메모리만 사용
```

### 모델 관련 문제

```bash
# Ollama 모델 재설치
docker exec rag-ollama ollama pull qwen2.5

# 모델 목록 확인
docker exec rag-ollama ollama list

# 모델 삭제 (필요시)
docker exec rag-ollama ollama rm qwen2.5
```

### 네트워크 문제

```bash
# 네트워크 확인
docker network ls

# 특정 네트워크 상세 정보
docker network inspect rag_rag-network

# 네트워크 재구성
docker-compose down
docker network prune
docker-compose up -d
```

---

## 🔄 업그레이드 절차

### 이미지 업그레이드

```bash
# 1. 현재 데이터 백업
cp -r data data-backup-$(date +%Y%m%d)

# 2. 서비스 중지
docker-compose down

# 3. 이미지 업데이트
docker-compose pull

# 4. 서비스 재시작
docker-compose up -d

# 5. 상태 확인
docker-compose logs -f
```

### 설정 업데이트

```bash
# 1. .env 파일 수정
nano .env

# 2. 설정이 반영되는 서비스 재시작
docker-compose restart sync-worker

# 또는 전체 재시작
docker-compose down && docker-compose up -d
```

---

## 📊 성능 최적화

### Ollama 최적화

```env
# docker-compose.yml에서 ollama 섹션 수정
OLLAMA_NUM_PARALLEL=2    # CPU 코어 수에 맞게 조정
OLLAMA_NUM_THREADS=4     # 스레드 수 조정
OLLAMA_KEEP_ALIVE=300    # 모델 캐시 시간 (초)
```

### ChromaDB 최적화

```bash
# 벡터 DB 최적화 (선택사항)
# 정기적으로 데이터 정리

# 또는 PostgreSQL 백엔드 사용 (고급)
# docker-compose.postgres.yml 사용
```

### vLLM 최적화 (GPU 있는 경우)

```bash
# docker-compose.yml에서 vllm 섹션 수정
command: vllm serve Qwen/Qwen2.5-Coder-14B \
  --port 8000 \
  --dtype float16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.9
```

---

## 🔐 보안 체크리스트

- [ ] `.env` 파일에 민감 정보 보관 (Git 무시)
- [ ] 포트는 로컬호스트 전용 (127.0.0.1)
- [ ] 방화벽에서 불필요한 포트 차단
- [ ] 정기적인 로그 검토
- [ ] 토큰/키 주기적 갱신
- [ ] SSL/TLS 설정 (프로덕션)

---

## 🚨 재해 복구 (Disaster Recovery)

### 전체 데이터 손실 시

```bash
# 1. 백업 데이터 복원
cp -r /backup/rag-20260125/chromadb data/

# 2. 서비스 재시작
docker-compose restart chromadb

# 3. 상태 확인
docker-compose logs chromadb
```

### 특정 벡터 DB 손상 시

```bash
# 1. ChromaDB 컨테이너 중지
docker-compose stop chromadb

# 2. 데이터 삭제
rm -rf data/chromadb/*

# 3. 컨테이너 재시작 (데이터 다시 생성)
docker-compose up -d chromadb

# 4. Sync Worker 재시작하여 문서 다시 동기화
docker-compose restart sync-worker
```

---

## 📞 지원

### 로그에서 오류 찾기

```bash
# 에러 로그만 필터링
docker-compose logs | grep -i error

# 특정 시간대 로그
docker-compose logs --since 2026-01-25T01:00:00
```

### 디버그 모드 활성화

```bash
# .env에 추가
LOG_LEVEL=DEBUG

# 또는 명령어로
LOG_LEVEL=DEBUG docker-compose up -d
```

### 시스템 정보 수집

```bash
#!/bin/bash
# collect-debug-info.sh

echo "=== Docker Version ==="
docker --version

echo "=== Docker Compose Version ==="
docker-compose --version

echo "=== System Info ==="
uname -a

echo "=== Running Containers ==="
docker ps

echo "=== Docker Logs ==="
docker-compose logs --tail=50

echo "=== Disk Usage ==="
df -h
```

---

## 📋 운영 체크리스트 (주간)

```
[ ] 컨테이너 상태 확인
[ ] 디스크 공간 확인
[ ] 로그 오류 확인
[ ] API 헬스체크
[ ] 성능 메트릭 확인
[ ] 백업 상태 확인
```

---

**배포 및 운영에 문제가 있으면 로그를 확인하고 위의 문제 해결 섹션을 참고하세요!**
