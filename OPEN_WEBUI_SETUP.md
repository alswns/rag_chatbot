# ✅ Open WebUI 최종 설정 가이드

## 📋 적용된 수정사항

### **1️⃣ Admin 계정 없이 로그인 (WEBUI_AUTH_ENABLED=false)**

#### 환경변수 설정

```yaml
# docker-compose.yml - open-webui
environment:
  - WEBUI_AUTH_ENABLED=false # ✅ 인증 완전 비활성화
  - WEBUI_DEFAULT_USER_ROLE=admin # 기본 사용자 역할 (관리자)
  - WEBUI_DEFAULT_MODELS=DeepSeek-R1-Distill-Qwen-14B
```

#### 동작 원리

- `WEBUI_AUTH_ENABLED=false` 설정 시 자동으로 관리자 계정 생성
- 로그인 화면 없이 바로 WebUI 접근 가능
- 로컬 개발/테스트 환경에서만 권장

#### 첫 실행 시 동작

```
1. Open WebUI 시작
2. 데이터베이스 초기화 시간 (5-10초)
3. 관리자 계정 자동 생성
4. http://localhost:3000 접근 시 바로 대시보드 로드
```

---

### **2️⃣ 모델 자동 감지 (ENABLE_MODEL_FILTER=false)**

#### 환경변수 설정

```yaml
environment:
  - ENABLE_MODEL_FILTER=false # ✅ 모든 모델 표시
  - ENABLE_OLLAMA_API=false # Ollama 비활성화 (RAG API만 사용)
  - ENABLE_OPENAI_API=true # OpenAI 호환 API 활성화
  - OPENAI_API_BASE_URL=http://rag-api:8010/v1
  - OPENAI_API_KEY=sk-not-needed
```

#### 모델 감지 플로우

```
1. Open WebUI 시작
   ↓
2. /v1/models 엔드포인트 호출 (자동)
   ↓
3. server.py의 list_models() 응답
   ↓
4. 모델 목록 표시 (DeepSeek-R1-Distill-Qwen-14B)
   ↓
5. 사용자가 선택 없이 기본 모델로 대화 가능
```

---

### **3️⃣ server.py /v1/models 엔드포인트 강화**

#### 개선사항

```python
@app.get('/v1/models')
async def list_models():
    """Open WebUI가 호출하는 모델 목록 엔드포인트"""

    # ✅ 추가된 필드:
    # - 'permission': OpenAI 호환 권한 정보
    # - 로깅: Open WebUI의 요청 추적

    return {
        'object': 'list',
        'data': [{
            'id': 'DeepSeek-R1-Distill-Qwen-14B',
            'object': 'model',
            'created': timestamp,
            'owned_by': 'vllm',
            'permission': [...]  # ✅ Open WebUI 호환성
        }]
    }
```

#### 응답 형식

```json
{
  "object": "list",
  "data": [
    {
      "id": "DeepSeek-R1-Distill-Qwen-14B",
      "object": "model",
      "created": 1706178245,
      "owned_by": "vllm",
      "permission": [
        {
          "id": "modelperm-default",
          "object": "model_permission",
          "allow_sampling": true,
          "allow_view": true,
          "organization": "*",
          "is_blocking": false
        }
      ]
    }
  ]
}
```

---

## 🚀 배포 및 테스트

### **Step 1: 컨테이너 재시작**

```bash
# 기존 컨테이너 정리
docker-compose down --remove-orphans

# 새로 시작 (헬스 체크 포함)
docker-compose up -d

# 상태 확인
docker-compose ps
```

### **Step 2: Open WebUI 접근**

```
URL: http://localhost:3000
기대하는 동작:
- 로그인 화면 없음 (바로 대시보드 로드)
- 모델 목록에 "DeepSeek-R1-Distill-Qwen-14B" 표시
- 사용자 선택 없이 기본 모델로 대화 가능
```

### **Step 3: 모델 감지 검증**

```bash
# RAG API가 모델을 정상 응답하는지 확인
curl http://localhost:8010/v1/models | jq

# 응답 예시:
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "DeepSeek-R1-Distill-Qwen-14B",
#       "object": "model",
#       ...
#     }
#   ]
# }
```

---

## ⚙️ 환경변수 상세 설명

| 환경변수                  | 값                             | 설명                           |
| ------------------------- | ------------------------------ | ------------------------------ |
| `WEBUI_AUTH_ENABLED`      | `false`                        | 로그인 화면 비활성화           |
| `WEBUI_DEFAULT_USER_ROLE` | `admin`                        | 기본 사용자를 관리자로 설정    |
| `WEBUI_DEFAULT_MODELS`    | `DeepSeek-R1-Distill-Qwen-14B` | 시작 시 기본 모델              |
| `ENABLE_MODEL_FILTER`     | `false`                        | 모든 모델 표시 (필터 비활성화) |
| `ENABLE_OLLAMA_API`       | `false`                        | Ollama API 비활성화            |
| `ENABLE_OPENAI_API`       | `true`                         | OpenAI 호환 API 활성화         |
| `OPENAI_API_BASE_URL`     | `http://rag-api:8010/v1`       | RAG API 주소                   |
| `OPENAI_API_KEY`          | `sk-not-needed`                | 더미 키 (로컬 API용)           |
| `QUERY_TIMEOUT`           | `120`                          | API 응답 대기 시간 (초)        |

---

## 🔍 문제 해결

### **문제 1: 모델이 표시되지 않음**

```bash
# RAG API 헬스 체크
curl http://localhost:8010/health

# 모델 목록 확인
curl http://localhost:8010/v1/models

# Open WebUI 로그 확인
docker-compose logs open-webui | grep -i "model"
```

### **문제 2: 로그인 화면이 나타남**

```bash
# 환경변수 재확인
docker-compose exec open-webui env | grep WEBUI_AUTH

# 컨테이너 재시작
docker-compose restart open-webui

# 브라우저 캐시 삭제 후 새로고침
# Ctrl+Shift+Delete (크롬)
```

### **문제 3: 대화가 작동하지 않음**

```bash
# RAG API 상태 확인
docker-compose logs rag-api | tail -20

# Open WebUI와 RAG API 네트워크 연결 확인
docker-compose exec open-webui curl http://rag-api:8010/v1/models
```

---

## 📊 최종 체크리스트

- [x] `WEBUI_AUTH_ENABLED=false` 설정
- [x] `WEBUI_DEFAULT_USER_ROLE=admin` 설정
- [x] `ENABLE_MODEL_FILTER=false` 설정
- [x] `/v1/models` 엔드포인트 강화
- [x] 권한(permission) 필드 추가
- [x] 로깅 추가
- [x] Docker Compose 재시작 준비 완료

---

## ✅ 예상 결과

```
1. http://localhost:3000 접근
   ↓
2. 로그인 화면 없음 (관리자로 자동 인증)
   ↓
3. Open WebUI 대시보드 표시
   ↓
4. 왼쪽 상단에 "DeepSeek-R1-Distill-Qwen-14B" 모델 표시
   ↓
5. 바로 대화 시작 가능
```

---

**상태**: 🟢 **배포 준비 완료**

작성일: 2026-01-25  
최종 검수: Chief Cloud Architect
