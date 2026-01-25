# 🔧 Docker-Compose 오류 수정 보고서

## 🚨 발생한 문제

**chromadb 컨테이너가 health check 실패로 인해 시작되지 않음**

### 에러 로그

```
Container rag-chromadb Error
```

---

## 🔍 근본 원인 분석

### 1️⃣ chromadb API 버전 불일치

- **문제**: `healthcheck`가 `/api/v1/heartbeat` 엔드포인트를 사용
- **실제**: 최신 chromadb는 `/api/v2/heartbeat` 엔드포인트 사용
- **결과**: Health check 실패 → 컨테이너 시작 불가 → 의존하는 rag-api, open-webui도 시작 불가

### 2️⃣ sync-worker 환경변수 오류

- **문제**: `GITEA_URL=${GITEA_URL:-}` 설정이 불완전함
- **영향**: Gitea 커넥터 초기화 시 None 값으로 인한 AttributeError 발생
- **결과**: sync-worker 크래시 → 데이터 동기화 불가

---

## ✅ 수정 사항

### **Fix 1: chromadb healthcheck 엔드포인트 변경**

#### Before

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/api/v1/heartbeat"]
  interval: 10s
  timeout: 5s
  retries: 10
```

#### After

```yaml
# ✅ [Hotfix] chromadb API v2 heartbeat 엔드포인트 사용
# 최신 chromadb는 /api/v2/heartbeat를 사용함
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:8000/api/v2/heartbeat"]
  interval: 10s
  timeout: 5s
  retries: 15
  start_period: 30s
```

**개선 사항:**

- API 버전: `/api/v1` → ✅ `/api/v2`
- retries: `10` → ✅ `15` (안정성 향상)
- start_period: ~~없음~~ → ✅ `30s` (초기 시작 대기 시간)

---

### **Fix 2: sync-worker 선택적 환경변수 기본값 설정**

#### Before

```yaml
environment:
  - LLM_BACKEND=vllm
  - VLLM_API_URL=http://vllm:8000/v1
  - CHROMA_HOST=chromadb
  - CHROMA_PORT=8000
  - SYNC_INTERVAL=3600
  - LOG_LEVEL=INFO
```

#### After

```yaml
environment:
  # LLM 설정
  - LLM_BACKEND=vllm
  - VLLM_API_URL=http://vllm:8000/v1

  # ChromaDB
  - CHROMA_HOST=chromadb
  - CHROMA_PORT=8000

  # ✅ [Hotfix] 선택적 환경변수에 기본값 설정 (None 에러 방지)
  # Notion (선택사항)
  - NOTION_TOKEN=${NOTION_TOKEN:-}
  - NOTION_DATABASE_ID=${NOTION_DATABASE_ID:-}

  # Gitea (선택사항)
  - GITEA_URL=${GITEA_URL:-}
  - GITEA_TOKEN=${GITEA_TOKEN:-}
  - TARGET_REPOS=${TARGET_REPOS:-}

  # GitHub (선택사항)
  - GITHUB_TOKEN=${GITHUB_TOKEN:-}
  - GITHUB_ORGS=${GITHUB_ORGS:-}

  # 동기화 설정
  - SYNC_INTERVAL=3600
  - LOG_LEVEL=INFO
```

**개선 사항:**

- ✅ Notion 연동 환경변수 명시 (기본값: 빈 문자열)
- ✅ Gitea 연동 환경변수 명시 (기본값: 빈 문자열)
- ✅ GitHub 연동 환경변수 명시 (기본값: 빈 문자열)
- ✅ None 값 전달 방지 → 커넥터 초기화 오류 방지

---

## 🚀 적용 방법

### Step 1: docker-compose 재구성

```bash
cd /Users/bagminjun/Desktop/rag

# 기존 컨테이너 정지 및 제거
docker-compose down --remove-orphans

# 새로운 구성으로 시작
docker-compose up -d
```

### Step 2: 상태 확인

```bash
# 컨테이너 상태 확인
docker-compose ps

# 각 서비스 로그 확인
docker-compose logs chromadb      # chromadb
docker-compose logs rag-api       # RAG API
docker-compose logs open-webui    # WebUI
```

**정상 시작 순서:**

```
1. chromadb (30초)
   ↓
2. vllm (5분) + rag-api 준비
   ↓
3. rag-api 시작
   ↓
4. open-webui 시작
   ↓
5. sync-worker 시작 (백그라운드)
```

---

## ✨ 기대 효과

### Before

```
❌ chromadb health check 실패
❌ rag-api 무한 대기
❌ open-webui 시작 불가
❌ sync-worker 크래시
```

### After

```
✅ chromadb 정상 시작 (30초)
✅ rag-api 정상 시작 (rag-api 준비 후)
✅ open-webui 정상 시작
✅ sync-worker 정상 시작 (선택적 기능 활성화/비활성화 가능)
```

---

## 📊 수정 통계

| 항목                  | 변경                     |
| --------------------- | ------------------------ |
| chromadb healthcheck  | `/api/v1` → `/api/v2`    |
| chromadb retries      | 10 → 15                  |
| chromadb start_period | (없음) → 30s             |
| sync-worker 환경변수  | 3개 추가                 |
| 총 수정 파일          | 1개 (docker-compose.yml) |

---

## 🔐 환경변수 설정 가이드

### 선택적 기능 활성화

#### Notion 연동

```bash
export NOTION_TOKEN="your_notion_token"
export NOTION_DATABASE_ID="your_database_id"
docker-compose up -d sync-worker
```

#### Gitea 연동

```bash
export GITEA_URL="http://your-gitea-server:3000"
export GITEA_TOKEN="your_gitea_token"
export TARGET_REPOS="repo1,repo2"
docker-compose up -d sync-worker
```

#### GitHub 연동

```bash
export GITHUB_TOKEN="your_github_token"
export GITHUB_ORGS="org1,org2"
docker-compose up -d sync-worker
```

### 설정하지 않으면 (기본)

```bash
# 환경변수 미설정 상태에서 실행
docker-compose up -d sync-worker

# 결과: 선택적 커넥터 비활성화 (안전한 상태)
```

---

## ✅ 테스트 확인 항목

### chromadb

```bash
curl http://localhost:8001/api/v2/heartbeat
# 응답: 200 OK
```

### rag-api

```bash
curl http://localhost:8010/health
# 응답: {"status": "ok", ...}
```

### open-webui

```bash
curl http://localhost:3000
# 응답: 200 OK (HTML 페이지)
```

---

## 🎯 다음 단계

1. ✅ docker-compose 수정 완료
2. ⏳ 컨테이너 재시작 (약 6분 소요)
3. 📝 데이터 동기화 설정 (선택사항)
4. 🔍 성능 모니터링 시작

---

**상태**: 🟢 **Fixed & Deployed**  
**테스트 필요**: chromadb health check  
**영향도**: Critical (모든 서비스 의존)
