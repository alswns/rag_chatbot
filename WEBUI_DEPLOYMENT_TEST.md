# 🚀 Open WebUI + RAG API 배포 테스트 가이드

## ✅ 적용된 모든 수정사항

### **1. docker-compose.yml 수정**
- ✅ `WEBUI_AUTH_ENABLED=false` 추가
- ✅ `WEBUI_DEFAULT_USER_ROLE=admin` 추가  
- ✅ `WEBUI_DEFAULT_MODELS=DeepSeek-R1-Distill-Qwen-14B` 추가
- ✅ `ENABLE_MODEL_FILTER=false` 추가
- ✅ `ENABLE_OLLAMA_API=false` 추가
- ✅ `ENABLE_OPENAI_API=true` 추가
- ✅ `QUERY_TIMEOUT=120` 추가
- ✅ `DATABASE_URL=sqlite:////app/backend/data/webui.db` 추가

### **2. server.py /v1/models 엔드포인트 강화**
- ✅ OpenAI 호환 `permission` 필드 추가
- ✅ 로깅 강화 (요청 추적)
- ✅ 빈 모델 목록 시 기본값 반환

---

## 🔧 배포 절차

### **Step 1: 기존 컨테이너 정리**

```bash
cd /Users/bagminjun/Desktop/rag

# 모든 컨테이너 제거
docker-compose down --remove-orphans

# 볼륨도 함께 제거 (신규 설정 적용)
docker-compose down -v
```

### **Step 2: 새로 시작**

```bash
# 백그라운드에서 시작
docker-compose up -d

# 상태 모니터링 (30초 정도 대기)
docker-compose ps

# 예상 출력:
# NAME               STATUS
# rag-vllm           Up 20s (health: starting)
# rag-chromadb       Up 25s (health: starting)
# rag-api            Up 15s
# rag-webui          Up 10s
# rag-sync           Up 5s
```

### **Step 3: 각 서비스 헬스 체크**

```bash
# vLLM 상태 (모델 로딩 중)
curl http://localhost:8000/health

# ChromaDB 상태
curl http://localhost:8001/api/v2/heartbeat

# RAG API 상태
curl http://localhost:8010/health

# Open WebUI 상태
curl http://localhost:3000
```

### **Step 4: 모델 감지 검증**

```bash
# RAG API의 모델 목록 확인
curl http://localhost:8010/v1/models | jq

# 예상 응답:
# {
#   "object": "list",
#   "data": [
#     {
#       "id": "DeepSeek-R1-Distill-Qwen-14B",
#       "object": "model",
#       "created": 1706178245,
#       "owned_by": "vllm",
#       "permission": [...]
#     }
#   ]
# }
```

---

## 🌐 Open WebUI 접근

### **URL**
```
http://localhost:3000
```

### **기대하는 동작**

| 항목 | 예상 동작 |
|-----|---------|
| **로그인 화면** | ❌ 없음 (바로 대시보드 로드) |
| **기본 모델** | DeepSeek-R1-Distill-Qwen-14B (자동 선택) |
| **모델 드롭다운** | 1개 모델만 표시 |
| **대화 시작** | 즉시 가능 (로그인 없음) |

### **첫 접근 시 진행 상황**
```
1. http://localhost:3000 로드
   ├─ Open WebUI 초기화 (2-3초)
   ├─ 관리자 계정 자동 생성
   └─ 대시보드 로드 완료

2. 대시보드 표시
   ├─ 왼쪽 상단: "DeepSeek-R1-Distill-Qwen-14B" 모델
   ├─ 채팅 입력창: 활성화
   └─ 즉시 대화 가능

3. 첫 메시지 전송
   ├─ RAG API 요청 (Context 검색)
   ├─ vLLM 모델 추론
   └─ 응답 표시
```

---

## 📊 실시간 로그 확인

### **모든 서비스 로그**
```bash
docker-compose logs -f
```

### **특정 서비스 로그**
```bash
# RAG API
docker-compose logs -f rag-api

# Open WebUI
docker-compose logs -f open-webui

# vLLM (모델 로딩 상태)
docker-compose logs -f vllm
```

### **모델 감지 로그 확인**
```bash
# Open WebUI가 모델을 감지했는지 확인
docker-compose logs open-webui | grep -i "model"

# RAG API의 모델 요청 로그
docker-compose logs rag-api | grep "/v1/models"
```

---

## 🧪 기능 테스트

### **Test 1: 로그인 없이 접근**
```
1. 브라우저에서 http://localhost:3000 접근
2. 예상: 로그인 화면 없음, 바로 대시보드 로드
3. 확인: ✅ 대시보드 보임
```

### **Test 2: 모델 확인**
```
1. 왼쪽 상단 모델 드롭다운 클릭
2. 예상: "DeepSeek-R1-Distill-Qwen-14B" 1개만 표시
3. 확인: ✅ 모델이 목록에 표시됨
```

### **Test 3: 대화 시작**
```
1. 채팅 입력창에 메시지 입력: "안녕하세요"
2. 예상: RAG API를 통해 vLLM이 응답
3. 확인: ✅ 응답이 나타남
```

### **Test 4: RAG 검색**
```
1. 입력: "당신의 프로젝트 구조는?"
2. 예상: 
   - ChromaDB에서 관련 문서 검색
   - XML 포맷의 Context 사용
   - CoT 방식의 추론 답변
3. 확인: ✅ 프로젝트 관련 정보 포함된 답변
```

---

## 🚨 문제 해결

### **문제 1: Open WebUI에 로그인 화면이 나타남**

**원인**: 환경변수가 적용되지 않음

**해결책**:
```bash
# 1. 컨테이너 재시작
docker-compose restart open-webui

# 2. 환경변수 확인
docker-compose exec open-webui env | grep WEBUI_AUTH

# 3. 여전히 안 되면 전체 삭제 후 재시작
docker-compose down -v
docker-compose up -d
```

### **문제 2: 모델이 표시되지 않음**

**원인**: RAG API의 /v1/models 엔드포인트 미응답

**해결책**:
```bash
# 1. RAG API 응답 확인
curl http://localhost:8010/v1/models

# 2. 응답이 없으면 RAG API 재시작
docker-compose restart rag-api

# 3. 로그 확인
docker-compose logs rag-api | grep -i "model"
```

### **문제 3: 대화가 작동하지 않음**

**원인**: vLLM 모델이 아직 로딩 중

**해결책**:
```bash
# 1. vLLM 상태 확인
docker-compose logs vllm | tail -20

# 2. 예상 로그:
#    "vLLM server ready at 0.0.0.0:8000"

# 3. 10분 정도 더 대기 (모델 로딩 15GB)

# 4. 강제 재시작
docker-compose restart vllm
```

### **문제 4: "Connection refused" 에러**

**원인**: Open WebUI와 RAG API의 네트워크 연결 실패

**해결책**:
```bash
# 1. 네트워크 확인
docker network ls | grep rag-network

# 2. Open WebUI에서 RAG API로 접근 가능한지 확인
docker-compose exec open-webui curl http://rag-api:8010/health

# 3. 네트워크 재설정
docker-compose down
docker network rm rag-network
docker-compose up -d
```

---

## ✅ 최종 체크리스트

배포 후 다음을 확인하세요:

- [ ] `docker-compose ps` 에서 모든 서비스가 "Up" 상태
- [ ] http://localhost:3000 접근 가능 (로그인 화면 없음)
- [ ] 왼쪽 상단에 "DeepSeek-R1-Distill-Qwen-14B" 모델 표시
- [ ] 채팅 입력창이 활성화됨
- [ ] "안녕하세요" 테스트 메시지 정상 응답
- [ ] `curl http://localhost:8010/v1/models` 정상 응답

---

## 🎯 성공 신호

```
✅ Open WebUI 로그인 없음
✅ 모델 자동 감지
✅ CoT 기반 추론 답변
✅ XML 포맷 Context 사용
✅ RAG 검색 통합
✅ <think> 태그 필터링
```

---

**상태**: 🟢 **배포 완료**  
**테스트**: ✅ **준비 완료**  
**예상 시간**: 5-15분 (vLLM 모델 로딩 포함)

문제 발생 시 위의 문제 해결 섹션을 참고하세요!
