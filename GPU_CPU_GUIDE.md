# 🚀 GPU/CPU 자동 감지 및 배포 가이드

## 현재 설정

### 1️⃣ GPU 자동 감지 (start-rag.sh)

```bash
# GPU 감지 로직
if command -v nvidia-smi &> /dev/null; then
    LLM_BACKEND="vllm"  # vLLM (GPU 가속)
else
    LLM_BACKEND="ollama"  # Ollama (CPU 최적화)
fi
```

✅ **GPU 있음 → vLLM (Qwen2.5-Coder-14B with CUDA)**
✅ **GPU 없음 → Ollama (CPU 최적화, 온프레미스)**

---

## 📋 배포 시나리오

### 시나리오 1: GPU 서버에 배포 (고성능)

```bash
# GPU가 있는 서버에서
cd /path/to/rag
./start-rag.sh

# 자동으로:
# ✓ CUDA 감지
# ✓ vLLM 실행 (GPU 사용)
# ✓ Qwen2.5-Coder-14B 모델 로드
# ✓ 성능 최적화

# 외부에서 접근:
http://<서버-IP>:3000  # Open WebUI
http://<서버-IP>:8000  # vLLM API (선택사항)
```

### 시나리오 2: CPU 서버에 배포 (온프레미스)

```bash
# CPU만 있는 서버에서
cd /path/to/rag
./start-rag.sh

# 자동으로:
# ✓ GPU 미감지
# ✓ Ollama 실행 (CPU 모드)
# ✓ Mistral 7B 모델 로드
# ✓ 메모리 최적화

# 외부에서 접근:
http://<서버-IP>:3000  # Open WebUI
```

---

## 🌐 외부 접근 설정

### Open WebUI (웹 인터페이스)

```yaml
# docker-compose.yml
ports:
  - "3000:8080" # 모든 인터페이스에서 접근 가능
```

✅ **외부 접근 가능**: `http://<IP>:3000`

### vLLM API (GPU 모드)

```yaml
# docker-compose.yml
ports:
  - "8000:8000" # 모든 인터페이스에서 접근 가능
```

✅ **외부 접근 가능**: `http://<IP>:8000/v1` (GPU 있을 때만)

### Ollama API (CPU 모드)

```yaml
# docker-compose.yml
ports:
  - "127.0.0.1:11434:11434" # 로컬호스트만
```

⚠️ **외부 접근 불가** (보안 정책)

---

## 📊 성능 비교

| 항목       | GPU (vLLM)        | CPU (Ollama)    |
| ---------- | ----------------- | --------------- |
| 모델       | Qwen2.5-Coder-14B | Mistral 7B      |
| 추론 속도  | 빠름 (GPU 가속)   | 느림 (CPU 기반) |
| 메모리     | 높음 (14GB+)      | 낮음 (4GB+)     |
| 비용       | 높음 (GPU 필요)   | 낮음 (CPU만)    |
| 온프레미스 | ✓ 가능            | ✓ 가능          |
| 자동 선택  | ✓ 있음            | ✓ 있음          |

---

## 🔧 수동 설정 (선택사항)

`.env` 파일에서 LLM 백엔드를 강제로 선택할 수 있습니다:

```env
# 자동 감지 (권장)
LLM_BACKEND=auto

# 또는 강제 선택
LLM_BACKEND=vllm    # 항상 vLLM 사용 (CUDA 필수)
LLM_BACKEND=ollama  # 항상 Ollama 사용
```

---

## ✅ 배포 체크리스트

### GPU 서버 배포

```
[ ] NVIDIA GPU 설치 & 드라이버 확인
    nvidia-smi 커맨드 실행

[ ] Docker & Docker Compose 설치

[ ] CUDA 환경 설정
    # GPU 메모리 할당 (docker-compose.yml)
    # gpu_memory_fraction 설정 (필요시)

[ ] .env 파일 설정
    HF_TOKEN=your_huggingface_token (선택사항)

[ ] 실행
    ./start-rag.sh

[ ] 접근 테스트
    http://<IP>:3000  # Open WebUI
    http://<IP>:8000  # vLLM API
```

### CPU 서버 배포

```
[ ] Docker & Docker Compose 설치

[ ] .env 파일 설정
    LLM_BACKEND=auto (또는 ollama)

[ ] 메모리 확인 (최소 8GB 권장)
    free -h

[ ] 실행
    ./start-rag.sh

[ ] 접근 테스트
    http://<IP>:3000  # Open WebUI
```

---

## 🚨 문제 해결

### GPU가 감지되지 않음

```bash
# 1. nvidia-smi 확인
nvidia-smi

# 2. Docker GPU 지원 확인
docker run --rm --gpus all nvidia/cuda:11.0-runtime nvidia-smi

# 3. 강제로 Ollama 사용
nano .env
# LLM_BACKEND=ollama
```

### 외부에서 접근 불가

```bash
# 1. 방화벽 확인
sudo ufw status
sudo ufw allow 3000  # Open WebUI 포트 개방
sudo ufw allow 8000  # vLLM 포트 개방

# 2. 포트 바인딩 확인
netstat -tuln | grep 3000

# 3. Docker 네트워크 확인
docker-compose ps
docker inspect rag-webui
```

### 메모리 부족

```bash
# 1. Ollama로 변경 (가벼움)
LLM_BACKEND=ollama

# 2. 더 작은 모델 사용
# docker-compose.yml 수정:
# ollama pull mistral (3.5GB)
# ollama pull neural-chat (3GB)
```

---

## 📈 성능 최적화

### GPU 모드 (vLLM)

```yaml
# docker-compose.yml
environment:
  - CUDA_VISIBLE_DEVICES=0 # GPU 선택
  - VLLM_GPU_MEMORY_UTILIZATION=0.9 # 90% 사용
```

### CPU 모드 (Ollama)

```yaml
# docker-compose.yml
environment:
  - OLLAMA_NUM_PARALLEL=2 # CPU 코어 수
  - OLLAMA_NUM_THREADS=4 # 스레드 수
  - OLLAMA_MEMORY_FRACTION=0.8 # 80% 메모리 사용
```

---

## 📞 지원

### 로그 확인

```bash
# 전체 로그
./manage-rag.sh logs

# 특정 서비스
./manage-rag.sh logs sync-worker
./manage-rag.sh logs open-webui

# 최신 로그만
docker logs rag-webui --tail 50
```

### 상태 확인

```bash
# 전체 상태
./manage-rag.sh status

# 헬스체크
./manage-rag.sh health-check

# 컨테이너 상태
docker ps
```

---

**작성일**: 2026-01-25  
**상태**: ✅ 배포 준비 완료  
**자동 감지**: ✅ GPU/CPU 자동 선택  
**외부 접근**: ✅ 모든 IP에서 접근 가능
