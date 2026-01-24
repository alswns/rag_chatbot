#!/bin/bash

# ============================================================
# ChromaDB 테스트 데이터 생성 스크립트
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 색상 정의
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}ChromaDB 테스트 데이터 생성${NC}"
echo -e "${BLUE}============================================================${NC}"

# Python 스크립트를 통해 데이터 추가
python3 << 'EOF'
import requests
import json
from datetime import datetime

# ChromaDB 설정
CHROMA_URL = "http://localhost:8001"
COLLECTION_NAME = "rag-documents"

print(f"\n{YELLOW}[1/3] ChromaDB에 연결 중...{NC}")

# v2 API로 컬렉션 생성/조회
headers = {"Content-Type": "application/json"}

# 데이터 예시
test_documents = [
    {
        "id": "doc-001",
        "title": "RAG 시스템 개요",
        "content": """
RAG (Retrieval-Augmented Generation) 시스템은 대규모 언어 모델과 벡터 데이터베이스를 결합한 시스템입니다.

주요 기능:
1. 문서 검색: ChromaDB에서 유사한 문서를 찾음
2. 컨텍스트 생성: 검색된 문서를 바탕으로 프롬프트 구성
3. 응답 생성: LLM이 컨텍스트와 질문을 바탕으로 답변 생성

이를 통해 최신 정보를 반영한 정확한 답변을 제공할 수 있습니다.
        """,
        "source": "system",
        "created_at": datetime.now().isoformat()
    },
    {
        "id": "doc-002",
        "title": "기업 정보 보호 정책",
        "content": """
우리 기업의 정보 보호 정책:

1. 온프레미스 운영 - 모든 데이터는 내부 서버에만 저장
2. 접근 제어 - 인증된 사용자만 접근 가능
3. 암호화 - 모든 민감 정보는 암호화됨
4. 감사 로그 - 모든 접근은 기록됨
5. 정기 백업 - 매일 자동 백업 실행

이러한 정책을 통해 기업 정보의 안전성을 보장합니다.
        """,
        "source": "policy",
        "created_at": datetime.now().isoformat()
    },
    {
        "id": "doc-003",
        "title": "기술 스택",
        "content": """
우리 RAG 시스템의 기술 스택:

Frontend:
- Open WebUI: 사용자 인터페이스
- React/TypeScript: 웹 애플리케이션 프레임워크

Backend:
- FastAPI: API 서버
- Python 3.10+: 메인 언어
- LangChain: RAG 프레임워크

LLM:
- Ollama: CPU 최적화 로컬 LLM
- Mistral 7B: 기본 모델

Vector Database:
- ChromaDB: 벡터 데이터베이스
- BAAI/bge-m3: 임베딩 모델

Infrastructure:
- Docker/Docker Compose: 컨테이너화
- GitHub/Gitea: 저장소 관리
- Notion: 문서 관리
        """,
        "source": "technical",
        "created_at": datetime.now().isoformat()
    },
    {
        "id": "doc-004",
        "title": "배포 가이드",
        "content": """
RAG 시스템 배포 절차:

1. 사전 준비
   - Docker & Docker Compose 설치
   - .env 파일 설정
   - 필요한 API 토큰 준비

2. 서비스 시작
   ./start-rag.sh

3. 상태 확인
   ./manage-rag.sh health-check

4. 데이터 동기화
   - Notion: DATABASE_ID 설정
   - GitHub: TOKEN과 ORG 설정
   - Gitea: URL과 TOKEN 설정

5. 모니터링
   ./manage-rag.sh logs
        """,
        "source": "documentation",
        "created_at": datetime.now().isoformat()
    },
    {
        "id": "doc-005",
        "title": "운영 체크리스트",
        "content": """
일일 운영 체크리스트:

매일 아침:
✓ 서비스 시작: ./start-rag.sh
✓ 상태 확인: ./manage-rag.sh status
✓ 에러 확인: docker-compose logs

주간 (금요일):
✓ 전체 헬스체크: ./manage-rag.sh health-check
✓ 데이터 백업: ./manage-rag.sh backup
✓ 로그 검토

월간 (말일):
✓ 시스템 정리: ./manage-rag.sh clean
✓ 로그 정리: ./manage-rag.sh prune-logs 30
✓ 이미지 업데이트: ./manage-rag.sh update

매일 저녁:
✓ 서비스 중지: ./manage-rag.sh stop
✓ 종료 확인
        """,
        "source": "operations",
        "created_at": datetime.now().isoformat()
    }
]

print(f"\n{YELLOW}[2/3] 테스트 데이터 준비 중...{NC}")
print(f"총 {len(test_documents)}개의 문서를 추가합니다.\n")

# ChromaDB에 데이터 추가 (직접 API 호출 대신 Python으로)
try:
    # langchain으로 직접 추가
    from langchain_chroma import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # 임베딩 모델 로드
    print(f"임베딩 모델 로드 중... (BAAI/bge-m3)")
    embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-m3")
    
    # 문서 텍스트 추출
    texts = [doc["content"].strip() for doc in test_documents]
    metadatas = [
        {
            "source": doc["source"],
            "title": doc["title"],
            "created_at": doc["created_at"],
            "id": doc["id"]
        }
        for doc in test_documents
    ]
    
    # ChromaDB에 추가
    print(f"ChromaDB에 데이터 추가 중...")
    vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embeddings,
        metadatas=metadatas,
        collection_name=COLLECTION_NAME,
        persist_directory="./data/chromadb",
        client_kwargs={"host": "localhost", "port": 8001}
    )
    
    print(f"\n{GREEN}✓ {len(test_documents)}개의 문서가 ChromaDB에 추가되었습니다!{NC}")
    
    # 검색 테스트
    print(f"\n{YELLOW}[3/3] 검색 테스트 중...${NC}")
    
    results = vector_store.similarity_search("RAG 시스템은 무엇인가?", k=3)
    
    print(f"\n{GREEN}✓ 검색 테스트 성공!${NC}")
    print(f"\n검색 결과 (상위 3개):")
    for i, doc in enumerate(results, 1):
        print(f"\n{i}. {doc.metadata.get('title', 'Unknown')}")
        print(f"   출처: {doc.metadata.get('source', 'Unknown')}")
        print(f"   내용 미리보기: {doc.page_content[:80]}...")
    
except Exception as e:
    print(f"\n{RED}✗ 에러 발생: {str(e)}${NC}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n{BLUE}============================================================${NC}")
print(f"{GREEN}테스트 데이터 생성 완료!${NC}")
print(f"{BLUE}이제 Open WebUI에서 아래 질문을 해보세요:${NC}")
print(f"  • 'RAG 시스템이란?'")
print(f"  • '배포는 어떻게 하나?'")
print(f"  • '기술 스택이 뭐지?'")
print(f"{BLUE}============================================================${NC}")

EOF

exit 0
