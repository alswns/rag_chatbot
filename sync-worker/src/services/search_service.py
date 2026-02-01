"""Vector search service with RAG integration"""
import logging
import asyncio
from typing import List, Dict, Any
from core.config import SEARCH_TOP_K, ENABLE_RERANKING, RERANKER_TOP_K
import core.dependencies as deps

logger = logging.getLogger(__name__)


class VectorSearchManager:
    """벡터 DB 검색 관리"""
    
    _last_search_results: List[Dict[str, Any]] = []
    
    @staticmethod
    def _search_sync(query: str, top_k: int = SEARCH_TOP_K) -> str:
        """동기 검색 로직"""
        if deps.vector_store is None:
            logger.warning('❌ Vector Store 미초기화')
            return ""
        
        try:
            # Intent 분류
            if deps.intent_router is not None:
                intent_result = deps.intent_router.route(query)
                logger.info(f'🎯 Intent: {intent_result.intent} | Domain: {intent_result.domain}')
                
                if intent_result.intent == 'chat':
                    logger.info('💬 일상 대화 감지 → 검색 스킵')
                    return ""
            
            # Drill-Down 검색
            if deps.drill_down_retriever is not None:
                logger.info(f'🔍 드릴다운 검색 시작: "{query[:50]}..."')
                
                search_k = RERANKER_TOP_K if ENABLE_RERANKING else top_k
                
                documents = deps.drill_down_retriever.retrieve(
                    query=query,
                    k=search_k,
                    use_reranking=ENABLE_RERANKING
                )
                
                if documents:
                    documents = documents[:top_k]
                    
                    # 웹 검색 판단용 결과 저장
                    VectorSearchManager._last_search_results = [
                        {
                            'content': doc.content,
                            'metadata': doc.metadata,
                            'score': doc.score
                        } for doc in documents
                    ]
                    
                    context_xml = deps.drill_down_retriever._format_as_xml(documents)
                    logger.info(f'✅ 드릴다운 검색 완료: {len(documents)}개 문서')
                    return context_xml
                else:
                    VectorSearchManager._last_search_results = []
            
            # Fallback 검색
            logger.info(f'🔍 Fallback 검색: "{query[:50]}..."')
            
            context = deps.vector_store.retrieve_context(
                query, 
                top_k=top_k, 
                use_hybrid=True,
                use_reranking=ENABLE_RERANKING
            )
            
            if not context:
                logger.info('⚠️ 검색 결과 없음')
                return ""
            
            logger.info(f'✅ 컨텍스트 생성 완료: {len(context)}자')
            return context
            
        except Exception as e:
            logger.error(f'❌ 검색 실패: {str(e)}', exc_info=True)
            return ""
    
    @staticmethod
    async def search(query: str, top_k: int = SEARCH_TOP_K) -> str:
        """비동기 검색"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            deps.executor,
            VectorSearchManager._search_sync,
            query,
            top_k
        )
