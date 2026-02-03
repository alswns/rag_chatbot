"""Startup and initialization logic"""
import logging
import os
import time
import openai
from core.config import (
    VLLM_API_URL, CHROMA_HOST, CHROMA_PORT, GRAPH_PERSIST_PATH,
    ENABLE_RERANKING, RERANKER_MODEL, MODEL_NAME,
    ENABLE_QUERY_EXPANSION, ENABLE_MULTI_PATH, FUSION_METHOD,
    ENABLE_WEB_SEARCH, ENABLE_INTELLIGENT_CHUNKING, CHUNK_SIZE, CHUNK_OVERLAP
)
import core.dependencies as deps

logger = logging.getLogger(__name__)


def wait_for_vllm(max_retries: int = 30, retry_interval: int = 10) -> bool:
    """vLLM 서버 대기"""
    logger.info(f'⏳ vLLM 서버 연결 대기 중...')
    
    for attempt in range(1, max_retries + 1):
        try:
            client = openai.OpenAI(api_key='sk-not-needed', base_url=VLLM_API_URL, timeout=5.0)
            client.models.list()
            logger.info(f'✅ vLLM 연결 성공 ({attempt}회 시도)')
            deps.vllm_client = client
            return True
        except Exception as e:
            logger.debug(f'vLLM 대기 중... ({attempt}/{max_retries})')
            if attempt < max_retries:
                time.sleep(retry_interval)
    
    return False


def preload_embedding_service():
    """임베딩 서비스 사전 로드"""
    try:
        from utils.embedding_service import get_embedding_service
        logger.info('🔄 임베딩 서비스 사전 로드 중...')
        embedding_service = get_embedding_service()
        model_name = getattr(embedding_service, 'model_name', 'BAAI/bge-m3')
        logger.info(f'✅ 임베딩 서비스 로드 완료: {model_name}')
    except Exception as e:
        logger.warning(f'⚠️ 임베딩 서비스 사전 로드 실패: {str(e)}')


def preload_reranker():
    """Reranker 사전 로드"""
    if not ENABLE_RERANKING:
        logger.info('ℹ️  Reranking 비활성화')
        return
    
    try:
        from db.vector_store import get_reranker
        logger.info(f'🔄 Reranker 사전 로드 중...')
        reranker = get_reranker()
        if reranker:
            logger.info(f'✅ Reranker 로드 완료')
    except Exception as e:
        logger.warning(f'⚠️ Reranker 사전 로드 실패: {str(e)}')


async def initialize_app():
    """애플리케이션 초기화"""
    try:
        from db.vector_store import VectorStoreManager
        from db.drill_down_retriever import GraphDrillDownRetriever
        from utils.intent_router import get_intent_router
        from processors.graph_rag import GraphRAGProcessor
        from utils.semantic_router import SemanticIntentRouter
        from utils.embedding_service import get_embedding_service
    except ImportError as e:
        logger.error(f'모듈 임포트 실패: {str(e)}')
        raise
    
    logger.info('=' * 70)
    logger.info('🚀 RAG API 시작 중...')
    logger.info('=' * 70)
    
    # 임베딩 서비스 사전 로드
    preload_embedding_service()
    if ENABLE_RERANKING:
        preload_reranker()
    
    # 벡터 DB 초기화
    logger.info('1️⃣  벡터 DB 초기화...')
    deps.vector_store = VectorStoreManager(chroma_host=CHROMA_HOST, chroma_port=CHROMA_PORT)
    stats = deps.vector_store.get_collection_stats()
    logger.info(f'✅ 벡터 DB 준비: {stats.get("document_count", 0)}개 문서')
    
    # 그래프 로드
    logger.info('2️⃣  그래프 로드...')
    graph = None
    if GraphRAGProcessor and os.path.exists(GRAPH_PERSIST_PATH):
        deps.graph_processor = GraphRAGProcessor.from_file(GRAPH_PERSIST_PATH)
        if deps.graph_processor:
            graph = deps.graph_processor.graph
            logger.info(f'✅ 그래프 로드 완료: {graph.number_of_nodes()}개 노드')
    
    # Intent Router 초기화
    logger.info('3️⃣  Intent Router 초기화...')
    if get_intent_router:
        deps.intent_router = get_intent_router(active_domains=['notion'])
        logger.info('✅ Intent Router 준비')
    
    # Drill-Down Retriever 초기화
    logger.info('4️⃣  Drill-Down Retriever 초기화...')
    if GraphDrillDownRetriever and graph:
        deps.drill_down_retriever = GraphDrillDownRetriever(
            vector_store=deps.vector_store,
            graph=graph,
            hub_types=['document', 'virtual_root'],  # ✅ 실제 그래프 노드 타입에 맞춤
            hub_score_threshold=0.3
        )
        logger.info('✅ Drill-Down Retriever 준비')
    
    # Semantic Intent Router
    logger.info('5️⃣  Semantic Intent Router 초기화...')
    try:
        embedding_service = get_embedding_service()
        deps.semantic_router = SemanticIntentRouter(embedding_service)
        logger.info('✅ Semantic Intent Router 준비')
    except Exception as e:
        logger.warning(f'⚠️  Semantic Intent Router 초기화 실패: {str(e)}')
    
    # vLLM 대기
    logger.info('6️⃣  vLLM 서버 연결 대기...')
    vllm_ready = wait_for_vllm()
    
    logger.info('=' * 70)
    logger.info(f'   - vLLM: {"✅" if vllm_ready else "❌"}')
    logger.info('=' * 70)
    
    # ✅ 지능형 검색 기능 요약
    logger.info('=' * 70)
    logger.info('📊 활성화된 고급 기능:')
    logger.info('=' * 70)
    
    # Query Expansion & Multi-path
    logger.info('🔍 Intelligent Retrieval:')
    logger.info(f'   - Query Expansion: {"✅" if ENABLE_QUERY_EXPANSION else "❌"}')
    logger.info(f'   - Multi-path Drill-down: {"✅" if ENABLE_MULTI_PATH else "❌"}')
    logger.info(f'   - Result Fusion: {FUSION_METHOD.upper()}')
    
    # Web Search
    logger.info('🌐 Web Search:')
    logger.info(f'   - Enabled: {"✅" if ENABLE_WEB_SEARCH else "❌"}')
    
    # Intelligent Chunking
    logger.info('🔬 Data Processing:')
    logger.info(f'   - Intelligent Chunking: {"✅" if ENABLE_INTELLIGENT_CHUNKING else "❌"}')
    if ENABLE_INTELLIGENT_CHUNKING:
        logger.info(f'      • Chunk Size: {CHUNK_SIZE}자 (Overlap: {CHUNK_OVERLAP}자)')
        logger.info(f'      • Features: Metadata Extraction + Keyword + Metadata Header')
    
    # Reranking
    logger.info(f'   - Reranking: {"✅" if ENABLE_RERANKING else "❌"}')
    
    logger.info('=' * 70)


async def cleanup_app():
    """애플리케이션 종료 시 정리"""
    logger.info('🛑 서버 종료 중...')
    
    if deps.vllm_client:
        try:
            deps.vllm_client.close()
            logger.info('✅ vLLM Client 정리 완료')
        except Exception as e:
            logger.warning(f'⚠️ vLLM Client 정리 실패: {str(e)}')
    
    deps.executor.shutdown(wait=True)
    logger.info('✅ ThreadPool 정리 완료')
