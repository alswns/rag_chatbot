FROM python:3.11-slim

WORKDIR /app

# 시스템 패키지 업데이트
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# HuggingFace 모델 미리 다운로드 (빌드 시간 증가하지만 런타임 단축)
ENV HF_HOME=/app/.cache/huggingface
RUN python -c "from sentence_transformers import SentenceTransformer; \
    print('Downloading BAAI/bge-m3...'); \
    SentenceTransformer('BAAI/bge-m3')" && \
    echo "✓ Model download complete"

# 애플리케이션 코드 복사
COPY src/ ./src/

# 환경변수 설정 (HuggingFace 다운로드 최적화)
ENV HF_HOME=/app/.cache/huggingface
ENV HF_HUB_DOWNLOAD_TIMEOUT=300
ENV HF_HUB_DISABLE_TELEMETRY=1
ENV TRANSFORMERS_OFFLINE=0
ENV HF_DATASETS_OFFLINE=0
ENV HTTPX_TIMEOUT=30.0

# 헬스체크
HEALTHCHECK --interval=60s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import chromadb; print('ok')" || exit 1

# 실행
CMD ["python", "-u", "src/main.py"]
