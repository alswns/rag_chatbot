"""
GraphDrillDownRetriever - 3단계 드릴다운 검색 클래스

검색 전략:
1. [Coarse Search] Hub Node 식별 - 페이지/루트 단위의 앵커 노드 탐색
2. [Scope Expansion] 그래프 확장 - Hub의 하위 노드 + 멘션 관계 수집
3. [Precise Search] 범위 내 정밀 검색 - 범위 내 벡터 검색 + Reranking

장점:
- 대규모 문서에서 효율적인 검색 (범위 축소)
- 그래프 관계 활용으로 문맥 유지
- 3단계 필터링으로 정밀도 향상
"""

import os
import logging
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import networkx as nx

logger = logging.getLogger(__name__)


@dataclass
class RetrievedDocument:
    """검색된 문서 데이터 클래스"""
    id: str
    content: str
    metadata: Dict[str, Any]
    score: float
    source_step: str  # 'hub', 'scoped', 'global'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'content': self.content,
            'metadata': self.metadata,
            'score': self.score,
            'source_step': self.source_step
        }


class GraphDrillDownRetriever:
    """
    3단계 드릴다운 검색 클래스
    
    Step 1: Hub Node 식별 (Coarse Search)
        - 페이지/루트 단위의 앵커 노드를 찾음
        
    Step 2: 그래프 확장 (Scope Expansion)
        - Hub의 하위 노드(descendants) 수집
        - 멘션(mention) 관계로 연결된 노드도 포함
        
    Step 3: 범위 내 정밀 검색 (Scoped Vector Search)
        - 수집된 범위 내에서만 벡터 검색
        - Cross-Encoder Reranking (기본적으로 비활성화)
    """
    
    def __init__(
        self,
        vector_store,
        graph: nx.DiGraph,
        hub_types: List[str] = None,
        hub_score_threshold: float = 0.3,
        include_mention_depth: int = 1
    ):
        """
        Args:
            vector_store: VectorStoreManager 인스턴스
            graph: NetworkX DiGraph (지식 그래프)
            hub_types: Hub로 간주할 노드 타입 ['page', 'root']
            hub_score_threshold: Hub 검색 최소 점수 임계값
            include_mention_depth: mention 엣지 탐색 깊이
        """
        self.vector_store = vector_store
        self.graph = graph
        self.hub_types = hub_types or ['page', 'root']
        self.hub_score_threshold = hub_score_threshold
        self.mention_depth = include_mention_depth
        
        # ✅ 그래프 노드 타입 분석 (디버깅용)
        node_types = set()
        for _, data in graph.nodes(data=True):
            node_type = data.get('node_type', data.get('type', 'unknown'))
            node_types.add(node_type)
        
        logger.info(f'✅ GraphDrillDownRetriever 초기화')
        logger.info(f'   - Hub 타입: {self.hub_types}')
        logger.info(f'   - 그래프 노드: {graph.number_of_nodes()}개')
        logger.info(f'   - 그래프 엣지: {graph.number_of_edges()}개')
        logger.info(f'   - 탐지된 노드 타입: {node_types}')
    
    def retrieve(
        self,
        query: str,
        k: int = 5,
        hub_k: int = 3,
        use_reranking: bool = False
    ) -> List[RetrievedDocument]:
        """
        3단계 드릴다운 검색 수행
        
        Args:
            query: 검색 쿼리
            k: 최종 반환할 문서 개수 (원래 top_k)
            hub_k: Step 1에서 검색할 Hub 개수
            use_reranking: Cross-Encoder Reranking 사용 여부
        
        Returns:
            검색된 문서 리스트
        """
        # ✅ Reranking 사용시 후보 개수 확장
        reranker_top_k = int(os.getenv('RERANKER_TOP_K', '50'))
        candidate_k = reranker_top_k if use_reranking else k
        
        # 디버그 로그
        logger.info(f'🔧 DEBUG: RERANKER_TOP_K env = {os.getenv("RERANKER_TOP_K")}, parsed = {reranker_top_k}')
        logger.info(f'🔧 DEBUG: use_reranking = {use_reranking}, k = {k}')
        logger.info(f'🔧 DEBUG: candidate_k calculation: {reranker_top_k} if {use_reranking} else {k} = {candidate_k}')
        
        logger.info(f'🔍 드릴다운 검색 시작: "{query[:50]}..." (k={k}, candidate_k={candidate_k}, rerank={use_reranking})')
        
        # =====================================================
        # Step 1: Hub Node 식별 (Coarse Search)
        # =====================================================
        hub_nodes = self._find_hub_nodes(query, top_k=hub_k)
        
        if not hub_nodes:
            logger.warning('[Step 1] Hub 노드 없음 → 전역 검색으로 전환')
            return self._global_search(query, k, candidate_k, use_reranking)
        
        best_hub = hub_nodes[0]
        logger.info(f'[Step 1] ✓ Hub 식별: "{best_hub["title"]}" (score={best_hub["score"]:.3f})')
        
        # 점수가 임계값 미달이면 전역 검색
        if best_hub['score'] < self.hub_score_threshold:
            logger.warning(f'[Step 1] Hub 점수 미달 ({best_hub["score"]:.3f} < {self.hub_score_threshold})')
            logger.warning(f'[Step 1] 발견된 Hub 목록: {[(h["title"], h["score"]) for h in hub_nodes]}')
            logger.warning(f'[Step 1] → 전역 검색으로 전환')
            return self._global_search(query, k, candidate_k, use_reranking)
        
        # =====================================================
        # Step 2: 그래프 확장 및 범위 설정 (Scope Expansion)
        # =====================================================
        allowed_doc_ids = self._expand_scope(best_hub['node_id'])
        
        if not allowed_doc_ids:
            logger.warning('[Step 2] 확장된 범위 없음 → 전역 검색으로 전환')
            return self._global_search(query, k, candidate_k, use_reranking)
        
        logger.info(f'[Step 2] ✓ 범위 설정: {len(allowed_doc_ids)}개 노드')
        
        # =====================================================
        # Step 3: 범위 내 정밀 검색 (Scoped Vector Search)
        # =====================================================
        results = self._scoped_search(
            query=query,
            allowed_doc_ids=allowed_doc_ids,
            k=candidate_k,  # ✅ 후보 개수 사용
            use_reranking=use_reranking
        )
        
        # ✅ 최종 k개로 자르기
        final_results = results[:k]
        logger.info(f'[Step 3] ✓ 정밀 검색 완료: {len(final_results)}개 문서 반환 (후보 {len(results)}개에서 선택)')
        
        return final_results
    
    def _find_hub_nodes(
        self,
        query: str,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Step 1: Hub Node 식별
        
        페이지/루트 단위의 노드만 검색하여 앵커 포인트 찾기
        """
        try:
            # 그래프에서 Hub 타입 노드 ID 추출
            hub_node_ids = []
            all_node_types = {}  # 디버깅: 모든 노드 타입 수집
            
            for node_id, data in self.graph.nodes(data=True):
                node_type = data.get('node_type', data.get('type', ''))
                all_node_types[node_type] = all_node_types.get(node_type, 0) + 1
                
                if node_type in self.hub_types:
                    hub_node_ids.append(node_id)
            
            # ✅ 디버깅: Hub 타입 노드 찾기 실패 원인 분석
            if not hub_node_ids:
                logger.warning(f'[Step 1] Hub 타입 노드 없음! 기대: {self.hub_types}')
                logger.warning(f'[Step 1] 그래프 내 실제 노드 타입: {dict(all_node_types)}')
                
                # ✅ Fallback: 모든 노드를 후보로 사용 (더 유연한 검색)
                if self.graph.number_of_nodes() > 0:
                    logger.info('[Step 1] Fallback: 모든 노드를 Hub 후보로 사용')
                    hub_node_ids = list(self.graph.nodes())[:100]  # 최대 100개
                else:
                    logger.error('[Step 1] 그래프가 비어있음!')
                    return []
            
            logger.info(f'[Step 1] Hub 후보: {len(hub_node_ids)}개')
            
            # 쿼리 임베딩 생성
            query_embedding = self.vector_store.embedding_service.encode([query])[0].tolist()
            
            # ChromaDB에서 Hub 노드만 필터링하여 검색
            # document_id 필터 사용 (청크가 아닌 원본 문서 기준)
            try:
                results = self.vector_store.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=min(top_k * 5, 50),  # 넉넉하게 검색
                    where={"document_id": {"$in": hub_node_ids[:100]}}  # ChromaDB $in 제한
                )
            except Exception as filter_err:
                # 필터 실패 시 전체 검색 후 필터링
                logger.debug(f'[Step 1] 필터 검색 실패, 수동 필터링: {str(filter_err)}')
                results = self.vector_store.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=100
                )
            
            if not results['ids'][0]:
                logger.warning('[Step 1] ChromaDB 검색 결과 없음')
                logger.warning(f'[Step 1] Hub 후보 IDs: {hub_node_ids[:10]}...')  # 처음 10개
                return []
            
            # Hub 노드만 필터링 및 정렬
            hub_results = []
            seen_docs = set()
            skipped_count = 0  # 디버깅: 건너뛄 문서 수
            
            for i, doc_id in enumerate(results['ids'][0]):
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                document_id = metadata.get('document_id', doc_id.split('_chunk_')[0])
                
                # 중복 문서 제거 (같은 문서의 여러 청크)
                if document_id in seen_docs:
                    continue
                
                # Hub 타입 확인
                if document_id in hub_node_ids or self._is_hub_type(document_id):
                    seen_docs.add(document_id)
                    
                    # 거리를 점수로 변환
                    distance = results['distances'][0][i] if results['distances'] else 1.0
                    score = 1 / (1 + distance)
                    
                    hub_results.append({
                        'node_id': document_id,
                        'title': metadata.get('title', 'Untitled'),
                        'score': score,
                        'metadata': metadata
                    })
                    
                    if len(hub_results) >= top_k:
                        break
                else:
                    skipped_count += 1
            
            # ✅ 디버꺅: 필터링 결과
            if not hub_results:
                logger.warning(f'[Step 1] Hub 필터링 후 결과 없음 (ChromaDB {len(results["ids"][0])}개 검색, {skipped_count}개 건너뛜)')
                logger.warning(f'[Step 1] Hub IDs 예시: {hub_node_ids[:5]}')
            else:
                logger.info(f'[Step 1] Hub 필터링 성공: {len(hub_results)}개 발견 ({skipped_count}개 건너뛜)')
            
            # 점수 기준 정렬
            hub_results.sort(key=lambda x: x['score'], reverse=True)
            
            return hub_results[:top_k]
            
        except Exception as e:
            logger.error(f'[Step 1] Hub 검색 실패: {str(e)}')
            return []
    
    def _is_hub_type(self, node_id: str) -> bool:
        """노드가 Hub 타입인지 확인"""
        if node_id not in self.graph:
            return False
        
        node_data = self.graph.nodes[node_id]
        node_type = node_data.get('node_type', node_data.get('type', ''))
        return node_type in self.hub_types
    
    def _expand_scope(self, hub_node_id: str) -> List[str]:
        """
        Step 2: 그래프 확장 및 범위 설정
        
        Hub 노드의 하위 노드(descendants) + 멘션 관계 노드 수집
        """
        if hub_node_id not in self.graph:
            logger.warning(f'[Step 2] Hub 노드가 그래프에 없음: {hub_node_id}')
            return []
        
        allowed_ids: Set[str] = set()
        
        # 1. Hub 노드 자체 포함
        allowed_ids.add(hub_node_id)
        
        # 2. 하위 노드(Descendants) 수집
        try:
            descendants = nx.descendants(self.graph, hub_node_id)
            allowed_ids.update(descendants)
            logger.debug(f'[Step 2] Descendants: {len(descendants)}개')
        except nx.NetworkXError as e:
            logger.debug(f'[Step 2] Descendants 탐색 실패: {str(e)}')
        
        # 3. Mention 엣지로 연결된 노드 수집 (depth=1)
        if self.mention_depth > 0:
            mention_nodes = self._get_mention_neighbors(hub_node_id, depth=self.mention_depth)
            allowed_ids.update(mention_nodes)
            logger.debug(f'[Step 2] Mention 관계: {len(mention_nodes)}개')
        
        # 4. 모든 수집된 노드의 청크 ID도 포함
        chunk_ids = self._get_chunk_ids(list(allowed_ids))
        allowed_ids.update(chunk_ids)
        
        return list(allowed_ids)
    
    def _get_mention_neighbors(self, node_id: str, depth: int = 1) -> Set[str]:
        """Mention 엣지로 연결된 이웃 노드 수집"""
        neighbors: Set[str] = set()
        
        try:
            # BFS로 mention 관계 탐색
            current_level = {node_id}
            
            for _ in range(depth):
                next_level = set()
                
                for nid in current_level:
                    if nid not in self.graph:
                        continue
                    
                    # 나가는 엣지 (outgoing)
                    for _, target, data in self.graph.out_edges(nid, data=True):
                        edge_type = data.get('edge_type', data.get('type', ''))
                        if edge_type in ['mention', 'references', 'link']:
                            next_level.add(target)
                    
                    # 들어오는 엣지 (incoming)
                    for source, _, data in self.graph.in_edges(nid, data=True):
                        edge_type = data.get('edge_type', data.get('type', ''))
                        if edge_type in ['mention', 'references', 'link']:
                            next_level.add(source)
                
                neighbors.update(next_level)
                current_level = next_level
                
        except Exception as e:
            logger.debug(f'Mention 탐색 실패: {str(e)}')
        
        return neighbors
    
    def _get_chunk_ids(self, doc_ids: List[str]) -> Set[str]:
        """문서 ID에 해당하는 청크 ID 수집"""
        chunk_ids: Set[str] = set()
        
        try:
            # ChromaDB에서 document_id로 청크 조회
            for doc_id in doc_ids[:50]:  # 제한
                try:
                    results = self.vector_store.collection.get(
                        where={"document_id": doc_id},
                        include=[]  # ID만 필요
                    )
                    if results and results.get('ids'):
                        chunk_ids.update(results['ids'])
                except:
                    continue
                    
        except Exception as e:
            logger.debug(f'청크 ID 수집 실패: {str(e)}')
        
        return chunk_ids
    
    def _scoped_search(
        self,
        query: str,
        allowed_doc_ids: List[str],
        k: int,
        use_reranking: bool
    ) -> List[RetrievedDocument]:
        """
        Step 3: 범위 내 정밀 검색
        
        수집된 범위 내에서만 벡터 검색 수행
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.vector_store.embedding_service.encode([query])[0].tolist()
            
            # 범위 내 검색 (ChromaDB $in 필터)
            # ✅ RERANKER_TOP_K 환경변수 활용
            reranker_top_k = int(os.getenv('RERANKER_TOP_K', '50'))
            if use_reranking:
                search_k = min(reranker_top_k, len(allowed_doc_ids))
            else:
                search_k = min(k * 4, len(allowed_doc_ids), 50)
            
            try:
                # 청크 ID로 직접 필터링 시도
                results = self.vector_store.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=search_k,
                    where={"$or": [
                        {"document_id": {"$in": allowed_doc_ids[:50]}},
                    ]}
                )
            except Exception as filter_err:
                logger.debug(f'[Step 3] $in 필터 실패, ID 직접 조회: {str(filter_err)}')
                # Fallback: ID로 직접 조회
                results = self.vector_store.collection.get(
                    ids=allowed_doc_ids[:search_k],
                    include=['documents', 'metadatas', 'embeddings']
                )
                # 유사도 계산 필요
                results = self._compute_similarity_for_get_results(
                    results, query_embedding, search_k
                )
            
            if not results or not results.get('ids') or not results['ids'][0]:
                logger.debug('[Step 3] 범위 내 검색 결과 없음')
                return []
            
            # 결과 정리
            documents = []
            for i, doc_id in enumerate(results['ids'][0]):
                content = results['documents'][0][i] if results['documents'] else ''
                metadata = results['metadatas'][0][i] if results['metadatas'] else {}
                distance = results['distances'][0][i] if results.get('distances') else 1.0
                score = 1 / (1 + distance)
                
                documents.append(RetrievedDocument(
                    id=doc_id,
                    content=content,
                    metadata=metadata,
                    score=score,
                    source_step='scoped'
                ))
            
            # Cross-Encoder Reranking
            if use_reranking and documents:
                logger.info(f'🔍 Reranking 시작: {len(documents)}개 문서')
                reranked = self._rerank_documents(query, documents, k)
                if reranked:
                    logger.info(f'✅ Reranking 완료: {len(reranked)}개 문서')
                    documents = reranked
                else:
                    logger.warning('⚠️ Reranking 실패 - 원본 순서 유지')
            
            # 상위 k개 반환
            documents.sort(key=lambda x: x.score, reverse=True)
            return documents[:k]
            
        except Exception as e:
            logger.error(f'[Step 3] 범위 내 검색 실패: {str(e)}')
            return []
    
    def _compute_similarity_for_get_results(
        self,
        results: Dict,
        query_embedding: List[float],
        k: int
    ) -> Dict:
        """get() 결과에 대해 유사도 계산"""
        import numpy as np
        
        if not results.get('ids'):
            return {'ids': [[]], 'documents': [[]], 'metadatas': [[]], 'distances': [[]]}
        
        # Fix: 임베딩이 없으면 거리 계산 불가 (배열 비교 에러 방지)
        embeddings = results.get('embeddings')
        if embeddings is None or len(embeddings) == 0:
            return {
                'ids': [results['ids'][:k]],
                'documents': [results.get('documents', [''] * len(results['ids']))[:k]],
                'metadatas': [results.get('metadatas', [{}] * len(results['ids']))[:k]],
                'distances': [[0.5] * min(k, len(results['ids']))]
            }
        
        # 코사인 유사도 계산
        query_vec = np.array(query_embedding)
        similarities = []
        
        for emb in results['embeddings']:
            doc_vec = np.array(emb)
            cos_sim = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
            similarities.append(float(cos_sim))
        
        # 유사도 기준 정렬
        indices = np.argsort(similarities)[::-1][:k]
        
        return {
            'ids': [[results['ids'][i] for i in indices]],
            'documents': [[results['documents'][i] for i in indices]] if results.get('documents') else [[]],
            'metadatas': [[results['metadatas'][i] for i in indices]] if results.get('metadatas') else [[]],
            'distances': [[1 - similarities[i] for i in indices]]  # 거리로 변환
        }
    
    def _rerank_documents(
        self,
        query: str,
        documents: List[RetrievedDocument],
        k: int
    ) -> List[RetrievedDocument]:
        """Cross-Encoder로 문서 재정렬"""
        try:
            from db.vector_store import get_reranker
            reranker = get_reranker()
            
            if not reranker:
                logger.warning('⚠️ Reranker 미로드 - Reranking 스킵')
                return documents
            
            logger.debug(f'🔍 Reranker 모델 로드 완료, {len(documents)}개 문서 재정렬 시작')
            
            # Query-Document 쌍 생성
            pairs = [(query, doc.content[:2000]) for doc in documents]
            
            # Reranking 점수 계산
            scores = reranker.predict(pairs)
            
            # Min-Max 정규화
            min_score = min(scores)
            max_score = max(scores)
            score_range = max_score - min_score if max_score > min_score else 1
            
            # 점수 업데이트
            for i, doc in enumerate(documents):
                normalized_score = (scores[i] - min_score) / score_range
                # 기존 점수와 Rerank 점수 결합
                doc.score = 0.4 * doc.score + 0.6 * normalized_score
            
            logger.debug(f'[Rerank] {len(documents)}개 문서 재정렬 완료')
            return documents
            
        except Exception as e:
            logger.warning(f'Reranking 실패: {str(e)}')
            return documents
    
    def _global_search(
        self,
        query: str,
        k: int,
        candidate_k: int,
        use_reranking: bool
    ) -> List[RetrievedDocument]:
        """
        Fallback: 전역 검색
        
        Hub 식별 실패 시 전체 범위에서 검색
        
        Args:
            k: 최종 반환할 문서 개수
            candidate_k: 검색할 후보 문서 개수
        """
        logger.info('[Fallback] 전역 검색 수행')
        
        try:
            logger.info(f'[Fallback] 파라미터: k={k}, candidate_k={candidate_k}, use_reranking={use_reranking}')
            
            # 기존 vector_store의 search 메서드 활용
            results = self.vector_store.search(
                query=query,
                top_k=candidate_k,  # ✅ 후보 개수 사용
                use_hybrid=True,
                use_graph=True,
                use_reranking=use_reranking
            )
            
            # RetrievedDocument로 변환
            documents = []
            for r in results:
                documents.append(RetrievedDocument(
                    id=r.get('id', ''),
                    content=r.get('content', ''),
                    metadata=r.get('metadata', {}),
                    score=r.get('final_score', r.get('score', 0.0)),
                    source_step='global'
                ))
            
            # ✅ 최종 k개로 자르기
            final_documents = documents[:k]
            logger.info(f'[Fallback] ✅ 전역 검색 완료: {len(final_documents)}개 반환 (후보 {len(documents)}개에서 선택)')
            
            return final_documents
            
        except Exception as e:
            logger.error(f'전역 검색 실패: {str(e)}')
            return []
    
    def retrieve_with_context(
        self,
        query: str,
        k: int = 5,
        context_format: str = 'xml'
    ) -> Tuple[List[RetrievedDocument], str]:
        """
        검색 + 컨텍스트 포맷팅
        
        Args:
            query: 검색 쿼리
            k: 반환할 문서 개수
            context_format: 'xml' 또는 'markdown'
        
        Returns:
            (문서 리스트, 포맷된 컨텍스트 문자열)
        """
        documents = self.retrieve(query, k)
        
        if context_format == 'xml':
            context = self._format_as_xml(documents)
        else:
            context = self._format_as_markdown(documents)
        
        return documents, context
    
    def _format_as_xml(self, documents: List[RetrievedDocument]) -> str:
        """
        XML 형식으로 컨텍스트 포맷
        
        Fix: Prevent OOM on L4 GPU - Context Explosion 방지
        - 부모 노드가 3000자 초과시 검색된 청크 주변 ±1000자만 포함
        """
        if not documents:
            return "<context>검색된 문서가 없습니다.</context>"
        
        MAX_PARENT_LENGTH = 3000  # ~5k tokens
        WINDOW_SIZE = 1000  # ±1000자 윈도우
        
        lines = ["<context>"]
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get('title', 'Untitled')
            source = doc.metadata.get('source_url', '')
            content = doc.content
            
            # Fix: Prevent context explosion - 부모 노드 텍스트 길이 체크
            if len(content) > MAX_PARENT_LENGTH:
                # 청크 주변 ±WINDOW_SIZE 추출 (간소화: 중간 부분만)
                start = max(0, len(content) // 2 - WINDOW_SIZE)
                end = min(len(content), start + (WINDOW_SIZE * 2))
                trimmed_content = f"[...전략...]\n{content[start:end]}\n[...후략...]"
                
                logger.debug(f'📏 문서 "{title}" 길이 제한: {len(content)} → {len(trimmed_content)}자')
                content = trimmed_content
            
            lines.append(f'  <document id="{i}" title="{title}">')
            lines.append(f'    <content>{content}</content>')
            if source:
                lines.append(f'    <source>{source}</source>')
            lines.append(f'  </document>')
        
        lines.append("</context>")
        return '\n'.join(lines)
    
    def _format_as_markdown(self, documents: List[RetrievedDocument]) -> str:
        """Markdown 형식으로 컨텍스트 포맷"""
        if not documents:
            return "## 검색된 문서 없음"
        
        lines = ["## 검색된 문서\n"]
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get('title', 'Untitled')
            source = doc.metadata.get('source_url', '')
            
            lines.append(f"### {i}. {title}")
            lines.append(f"\n{doc.content}\n")
            if source:
                lines.append(f"*출처: {source}*\n")
            lines.append("---\n")
        
        return '\n'.join(lines)


# 편의 함수
def create_drill_down_retriever(
    vector_store,
    graph_path: str = None
) -> GraphDrillDownRetriever:
    """
    GraphDrillDownRetriever 생성 헬퍼 함수
    
    Args:
        vector_store: VectorStoreManager 인스턴스
        graph_path: 그래프 파일 경로 (None이면 환경변수 사용)
    
    Returns:
        GraphDrillDownRetriever 인스턴스
    """
    import os
    import pickle
    
    # 그래프 로드
    if graph_path is None:
        graph_path = os.getenv('GRAPH_PERSIST_PATH', './data/graph.pkl')
    
    if os.path.exists(graph_path):
        with open(graph_path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, dict):
                graph = data.get('graph', nx.DiGraph())
            else:
                graph = data
        logger.info(f'✅ 그래프 로드: {graph.number_of_nodes()}개 노드')
    else:
        graph = nx.DiGraph()
        logger.warning(f'⚠️ 그래프 파일 없음: {graph_path}')
    
    return GraphDrillDownRetriever(
        vector_store=vector_store,
        graph=graph
    )
