# 📚 RAG Chatbot System

Graph-Enhanced RAG 시스템 - Notion 데이터 기반 지식 검색 및 LLM 질의응답

## 📁 프로젝트 구조

```
rag/
├── docker-compose-sync.yml    # Mac 데이터 수집용
├── docker-compose-server.yml  # Ubuntu 서버 추론용
├── docker-compose.yml         # 통합 구성 (개발용)
├── start.sh                   # 시스템 시작 스크립트
├── update_rag.sh              # 데이터 갱신 스크립트 (Cron용)
├── .env                       # 환경변수 설정
│
├── data/                      # 영속 데이터
│   ├── chroma/                # ChromaDB 벡터 저장소
│   ├── graph.pkl              # NetworkX 그래프
│   └── sync_progress.json     # 동기화 상태
│
└── sync-worker/               # 핵심 애플리케이션
    ├── Dockerfile
    ├── requirements.txt
    └── src/
        ├── main.py            # 데이터 동기화 Worker
        ├── server.py          # RAG API 서버 (FastAPI)
        ├── dashboard.py       # 관리 대시보드 (웹)
        │
        ├── connectors/        # 외부 데이터 소스
        │   ├── notion.py      # Notion API 연동
        │   └── git_connector.py  # GitHub/Gitea 저장소 분석
        │
        ├── processors/        # 데이터 처리
        │   ├── chunking.py    # 텍스트 분할 (Context Injection)
        │   ├── graph_rag.py   # 그래프 구축 (NetworkX)
        │   └── pipeline.py    # 전체 파이프라인 (Delta Sync)
        │
        ├── db/                # 데이터베이스
        │   └── vector_store.py  # ChromaDB + Hybrid Search + Cross-Encoder
        │
        └── utils/             # 유틸리티
            ├── embedding_service.py  # BGE-M3 임베딩 (싱글톤)
            └── sync_state.py  # 동기화 상태 관리
```

## 🔑 핵심 기능

### 1. 데이터 수집 (Delta Sync)

- Notion 페이지 수정 시 변경분만 동기화
- 그래프 연결성 유지 (Ghost Node, Virtual Root)

### 2. 검색 파이프라인

```
Query → Vector Search → Graph Expansion → BM25 Hybrid → Cross-Encoder Rerank → Top-K
```

### 3. 그래프 활용

- `hierarchy`: 부모-자식 관계
- `references`: 명시적 관계 속성
- `mention`: 본문 내 링크

## 🚀 실행 방법

### Mac (데이터 수집)

```bash
docker compose -f docker-compose-sync.yml up -d
```

### Ubuntu (추론 서버)

```bash
docker compose -f docker-compose-server.yml up -d
```

## 📊 대시보드

- URL: `http://localhost:8080`
- 기능: 문서 조회, 그래프 통계, 검색 테스트

## 🔧 환경변수

| 변수                  | 설명             | 기본값                 |
| --------------------- | ---------------- | ---------------------- |
| `NOTION_TOKEN`        | Notion API 토큰  | -                      |
| `NOTION_DATABASE_ID`  | Notion DB ID     | -                      |
| `CHROMA_HOST`         | ChromaDB 호스트  | localhost              |
| `CHROMA_PORT`         | ChromaDB 포트    | 8000                   |
| `GRAPH_PERSIST_PATH`  | 그래프 저장 경로 | ./data/graph.pkl       |
| `CROSS_ENCODER_MODEL` | Reranker 모델    | ms-marco-MiniLM-L-6-v2 |
