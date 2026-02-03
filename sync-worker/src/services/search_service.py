"""Vector search service with RAG integration + Query Expansion + Multi-path Retrieval"""
import logging
import asyncio
import os
from typing import List, Dict, Any, Optional
from core.config import SEARCH_TOP_K, ENABLE_RERANKING, RERANKER_TOP_K
import core.dependencies as deps
from utils.query_expansion import MultiQueryExpander
from utils.result_fusion import ResultFusionManager

logger = logging.getLogger(__name__)


class VectorSearchManager:
    """벡터 DB 검색 관리 + Query Expansion + Multi-path + Result Fusion"""
    
    _last_search_results: List[Dict[str, Any]] = []
    
    # Query Expansion 설정
    ENABLE_QUERY_EXPANSION = os.getenv('ENABLE_QUERY_EXPANSION', 'true').lower() == 'true'
    ENABLE_MULTI_PATH = os.getenv('ENABLE_MULTI_PATH', 'true').lower() == 'true'
    FUSION_METHOD = os.getenv('FUSION_METHOD', 'rrf')  # 'rrf' 또는 'score_sum'
    
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
            
            # Drill-Down 검색 (Query Expansion + Multi-path + Result Fusion)
            if deps.drill_down_retriever is not None:
                logger.info(f'🔍 지능형 검색 시작: "{query[:50]}..." (expansion={VectorSearchManager.ENABLE_QUERY_EXPANSION}, multi_path={VectorSearchManager.ENABLE_MULTI_PATH})')
                
                search_k = RERANKER_TOP_K if ENABLE_RERANKING else top_k
                
                # 🔄 Query Expansion (선택적)
                if VectorSearchManager.ENABLE_QUERY_EXPANSION:
                    documents = VectorSearchManager._search_with_query_expansion(
                        query=query,
                        k=search_k,
                        use_reranking=ENABLE_RERANKING,
                        use_multi_path=VectorSearchManager.ENABLE_MULTI_PATH
                    )
                else:
                    # Query Expansion 없이 기존 드릴다운 검색
                    documents = deps.drill_down_retriever.retrieve(
                        query=query,
                        k=search_k,
                        use_reranking=ENABLE_RERANKING,
                        use_multi_path=VectorSearchManager.ENABLE_MULTI_PATH
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
                    logger.info(f'✅ 지능형 검색 완료: {len(documents)}개 문서')
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
    def _search_with_query_expansion(
        query: str,
        k: int = SEARCH_TOP_K,
        use_reranking: bool = False,
        use_multi_path: bool = True
    ) -> List:
        """
        Query Expansion + 다중 검색 + Result Fusion
        
        1. 원본 쿼리를 3가지 버전으로 확장
        2. 각 쿼리에 대해 멀티패스 드릴다운 검색 수행
        3. 모든 결과를 RRF/점수 합산으로 통합
        4. 최상위 k개 반환
        """
        try:
            # ✅ Step 1: Query Expansion (동기 실행)
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            expansion_result = loop.run_until_complete(
                MultiQueryExpander.expand_query(query)
            )
            
            expanded_queries = expansion_result.get('queries', [query])
            logger.info(f'📝 Query Expansion: {len(expanded_queries)}개 쿼리 생성')
            for i, q in enumerate(expanded_queries, 1):
                logger.debug(f'   [{i}] {q}')
            
            # ✅ Step 2: 각 쿼리에 대해 멀티패스 검색 수행
            all_result_sets = []
            
            for i, expanded_query in enumerate(expanded_queries, 1):
                logger.info(f'[검색 {i}/{len(expanded_queries)}] "{expanded_query[:50]}..."')
                
                try:
                    results = deps.drill_down_retriever.retrieve(
                        query=expanded_query,
                        k=k,
                        use_reranking=use_reranking,
                        use_multi_path=use_multi_path
                    )
                    
                    # RetrievedDocument를 Dict로 변환 (Result Fusion용)
                    result_dicts = []
                    for doc in results:
                        result_dicts.append({
                            'id': doc.id,
                            'document_id': doc.metadata.get('document_id', doc.id),
                            'content': doc.content,
                            'metadata': doc.metadata,
                            'score': doc.score,
                            'source_step': doc.source_step,
                            'query_source': expanded_query
                        })
                    
                    all_result_sets.append(result_dicts)
                    logger.info(f'   ✓ {len(result_dicts)}개 결과')
                    
                except Exception as e:
                    logger.warning(f'[검색 {i}] 실패: {str(e)}')
                    all_result_sets.append([])
            
            # ✅ Step 3: Result Fusion (중복 제거 + 점수 통합)
            fused_results = ResultFusionManager.fuse_results(
                result_sets=all_result_sets,
                method=VectorSearchManager.FUSION_METHOD,
                k=k,
                remove_duplicates=True
            )
            
            # Dict를 RetrievedDocument로 변환
            from db.drill_down_retriever import RetrievedDocument
            
            final_documents = []
            for result in fused_results:
                doc = RetrievedDocument(
                    id=result.get('id', ''),
                    content=result.get('content', ''),
                    metadata=result.get('metadata', {}),
                    score=result.get('score', 0.0),
                    source_step=result.get('source_step', 'fused')
                )
                final_documents.append(doc)
            
            logger.info(f'🔀 Result Fusion 완료: {len(final_documents)}개 최종 결과 (통합 방식: {VectorSearchManager.FUSION_METHOD})')
            
            return final_documents
            
        except Exception as e:
            logger.error(f'❌ Query Expansion 검색 실패: {str(e)}', exc_info=True)
            # Fallback: 원본 쿼리로 검색
            logger.info('📌 Query Expansion 실패 → 원본 쿼리로 단일 검색')
            return deps.drill_down_retriever.retrieve(
                query=query,
                k=k,
                use_reranking=use_reranking,
                use_multi_path=use_multi_path
            )
    
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
    
    @staticmethod
    async def search_with_web_integration(
        query: str,
        top_k: int = SEARCH_TOP_K,
        enable_web_search: bool = True,
        enable_query_expansion: bool = True
    ) -> str:
        """
        RAG 검색 + 웹 검색 통합 (선택적)
        
        Args:
            query: 사용자 질문
            top_k: 최종 반환 문서 개수
            enable_web_search: 웹 검색 활성화
            enable_query_expansion: Query Expansion 활성화
        
        Returns:
            포맷된 컨텍스트 (RAG + 웹 통합)
        """
        try:
            # Step 1: RAG 검색 수행
            logger.info(f'🔍 RAG 검색 시작: "{query[:50]}..."')
            
            loop = asyncio.get_event_loop()
            rag_context = await loop.run_in_executor(
                deps.executor,
                VectorSearchManager._search_sync,
                query,
                top_k
            )
            
            # Step 2: 웹 검색 (선택적)
            if not enable_web_search:
                logger.info('⚠️ 웹 검색 비활성화')
                return rag_context
            
            # 웹 검색 서비스 초기화
            from utils.web_search import get_web_search_service
            web_search_service = get_web_search_service()
            
            # 웹 검색 필요성 판단 및 실행
            web_context = await web_search_service.search_if_needed(
                user_query=query,
                internal_context=rag_context,
                force_search=False
            )
            
            # Step 3: 결과 통합
            if web_context:
                logger.info('🔀 RAG + Web 결과 통합')
                integrated = f"{rag_context}\n\n---\n\n{web_context}"
                return integrated
            else:
                logger.info('ℹ️  웹 검색 결과 없음 → RAG 결과만 반환')
                return rag_context
            
        except Exception as e:
            logger.error(f'❌ 웹 검색 통합 실패: {str(e)}', exc_info=True)
            # Fallback: RAG 결과만 반환
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                deps.executor,
                VectorSearchManager._search_sync,
                query,
                top_k
            )

