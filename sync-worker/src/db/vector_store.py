"""
VectorStoreManager - ChromaDB를 통한 벡터 저장소 관리

싱글톤 임베딩 서비스를 사용하여 중복 로드를 방지합니다.
Cross-Encoder Reranking과 Graph Traversal을 지원합니다.
"""

import logging
import os
import pickle
from typing import List, Dict, Any, Optional, Set
import chromadb
from utils.embedding_service import get_embedding_service
import time
import torch
from rank_bm25 import BM25Okapi
import re
import networkx as nx

logger = logging.getLogger(__name__)

# BGE Reranker 싱글톤 (한국어 성능 우수)
_reranker = None

def get_reranker():
    """
    ✅ BGE Reranker 싱글톤 반환
    
    - 기본 모델: BAAI/bge-reranker-v2-m3 (다국어, 한국어 성능 우수)
    - 대안 모델: cross-encoder/ms-marco-MiniLM-L-6-v2 (영어 전용)
    """
    global _reranker
    if _reranker is None:
        try:
            from sentence_transformers import CrossEncoder
            # ✅ BGE Reranker - 한국어/다국어 성능 우수
            model_name = os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')
            logger.info(f'🔄 BGE Reranker 로딩: {model_name}')
            # _reranker = CrossEncoder(model_name, max_length=1024)  # BGE는 1024 지원
            _reranker = CrossEncoder(
                model_name, 
                max_length=512, 
                device='cuda',
                automodel_args={"torch_dtype": torch.float16} # FP16 적용
            )
            logger.info(f'✅ BGE Reranker 로드 완료')
        except Exception as e:
            logger.warning(f'⚠️ Reranker 로드 실패: {str(e)}')
            _reranker = None
    return _reranker

# 하위 호환성 유지
def get_cross_encoder():
    """[DEPRECATED] get_reranker() 사용 권장"""
    return get_reranker()


class VectorStoreManager:
    """ChromaDB를 통한 벡터 저장소 관리 클래스"""
    
    def __init__(self, chroma_host: str = 'localhost', chroma_port: int = 8000, model_name: str = "BAAI/bge-m3"):
        """
        Args:
            chroma_host: ChromaDB 호스트
            chroma_port: ChromaDB 포트
            model_name: HuggingFace 임베딩 모델 이름
        """
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        self.model_name = model_name
        
        # ChromaDB 클라이언트 초기화 (재시도 로직)
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                logger.info(f'ChromaDB 클라이언트 연결 시도: {chroma_host}:{chroma_port}')
                self.client = chromadb.HttpClient(
                    host=chroma_host,
                    port=chroma_port
                )
                # 연결 테스트
                self.client.heartbeat()
                logger.info(f'ChromaDB 클라이언트 연결 성공: {chroma_host}:{chroma_port}')
                break
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # 지수 백오프
                    logger.warning(f'ChromaDB 연결 실패 (시도 {retry_count}/{max_retries}): {str(e)}')
                    logger.info(f'{wait_time}초 후 재시도...')
                    time.sleep(wait_time)
                else:
                    logger.error(f'ChromaDB 연결 최종 실패: {str(e)}')
                    raise
        
        # 싱글톤 임베딩 서비스 (중복 로드 방지)
        try:
            logger.info(f'🔄 싱글톤 임베딩 서비스 초기화: {model_name}')
            self.embedding_service = get_embedding_service(model_name)
            logger.info(f'✅ 싱글톤 임베딩 서비스 준비 완료: {model_name}')
        except Exception as e:
            logger.error(f'임베딩 서비스 초기화 실패: {str(e)}', exc_info=True)
            raise
        
        # 기본 컬렉션 이름
        self.collection_name = 'rag_documents'
        
        # 컬렉션 초기화 또는 기존 컬렉션 로드
        try:
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f'컬렉션 준비 완료: {self.collection_name}')
        except Exception as e:
            logger.error(f'컬렉션 초기화 실패: {str(e)}')
            raise
        
        # 그래프 (검색 시 활용)
        self.graph: Optional[nx.DiGraph] = None
        self._load_graph()
    
    def _load_graph(self) -> None:
        """그래프 파일 로드 (검색 시 활용)"""
        graph_path = os.getenv('GRAPH_PERSIST_PATH', './data/graph.pkl')
        try:
            if os.path.exists(graph_path):
                with open(graph_path, 'rb') as f:
                    data = pickle.load(f)
                    # dict 형식인 경우 graph 키에서 추출
                    if isinstance(data, dict):
                        self.graph = data.get('graph', nx.DiGraph())
                    else:
                        self.graph = data
                logger.info(f'✅ 그래프 로드 완료: {self.graph.number_of_nodes()}개 노드, {self.graph.number_of_edges()}개 엣지')
            else:
                self.graph = nx.DiGraph()
                logger.info('📊 그래프 파일 없음 - 빈 그래프 생성')
        except Exception as e:
            logger.warning(f'⚠️ 그래프 로드 실패: {str(e)}')
            self.graph = nx.DiGraph()
        
        # BM25 인덱스 초기화 (지연 로딩)
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_corpus: List[List[str]] = []
        self.bm25_doc_ids: List[str] = []
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        배치 단위로 문서를 벡터 저장소에 추가
        
        Args:
            documents: 추가할 문서 리스트. 각 문서는 다음 구조:
                {
                    'id': str,
                    'content': str,
                    'metadata': dict,
                    'embedding': Optional[List[float]]  # 사전 생성된 임베딩 (선택사항)
                }
            batch_size: 배치 크기
        
        Returns:
            추가된 문서 개수
        """
        if not documents:
            logger.warning('추가할 문서가 없습니다.')
            return 0
        
        logger.info(f'{len(documents)}개 문서 추가 시작...')
        
        added_count = 0
        failed_count = 0
        
        try:
            # 배치 단위로 처리
            for batch_idx, i in enumerate(range(0, len(documents), batch_size)):
                batch = documents[i:i + batch_size]
                batch_num = batch_idx + 1
                
                ids = []
                documents_text = []
                metadatas = []
                embeddings_list = []
                use_precomputed = False
                
                for doc in batch:
                    # 필수 필드 검증
                    if not doc.get('id') or not doc.get('content'):
                        logger.warning(f'문서 필드 누락: {doc.get("id", "unknown")}')
                        continue
                    
                    ids.append(str(doc['id']))  # ID는 문자열로 변환
                    documents_text.append(str(doc['content']))  # Content는 문자열로 변환
                    
                    # 메타데이터 정제 (ChromaDB는 None 값 불허)
                    metadata = doc.get('metadata', {})
                    cleaned_metadata = {k: v for k, v in metadata.items() if v is not None}
                    metadatas.append(cleaned_metadata)
                    
                    # 사전 생성된 임베딩이 있으면 사용 (numpy array 체크)
                    if 'embedding' in doc:
                        embedding = doc['embedding']
                        if embedding is not None and (isinstance(embedding, list) or hasattr(embedding, '__len__')):
                            embeddings_list.append(embedding)
                            use_precomputed = True
                
                if not ids:
                    logger.warning(f'배치 {batch_num}: 유효한 문서 없음')
                    continue
                
                # 임베딩 생성 또는 사용
                try:
                    if use_precomputed and len(embeddings_list) == len(ids):
                        # 사전 생성된 임베딩 사용
                        logger.info(f'배치 {batch_num}: {len(embeddings_list)}개 문서 (사전 생성 임베딩)')
                    else:
                        # 임베딩 생성 (싱글톤 서비스 사용)
                        logger.debug(f'배치 {batch_num}: {len(ids)}개 문서 임베딩 생성 중...')
                        embeddings_array = self.embedding_service.encode(documents_text)
                        embeddings_list = embeddings_array.tolist()  # numpy array → list
                        logger.debug(f'배치 {batch_num}: {len(documents_text)}개 문서 임베딩 완료')
                except Exception as e:
                    logger.error(f'임베딩 생성 실패 (배치 {batch_num}): {str(e)}')
                    failed_count += len(batch)
                    continue
                
                # ChromaDB에 추가
                try:
                    logger.debug(f'배치 {batch_num}: ChromaDB에 저장 중... (ID: {ids[0:3]}...)')
                    
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings_list,
                        documents=documents_text,
                        metadatas=metadatas
                    )
                    
                    added_count += len(batch)
                    logger.info(f'배치 {batch_num}: {len(batch)}개 문서 저장 완료 (누적: {added_count}/{len(documents)})')
                    
                except Exception as e:
                    logger.error(f'ChromaDB 저장 실패 (배치 {batch_num}): {str(e)}', exc_info=True)
                    failed_count += len(batch)
                    continue
            
            logger.info(f'총 {added_count}개 문서 추가 완료 (실패: {failed_count}개)')
            
            # 최종 통계
            if added_count > 0:
                stats = self.get_collection_stats()
                logger.info(f'✓ 현재 컬렉션 상태: {stats.get("document_count", 0)}개 문서')
            
            return added_count
            
        except Exception as e:
            logger.error(f'문서 추가 중 오류: {str(e)}', exc_info=True)
            return added_count
    
    def delete_by_source(self, source: str) -> int:
        """
        특정 소스(Notion 페이지 또는 Gitea 저장소)의 모든 데이터 삭제
        
        Delta Sync 시 중복을 방지하기 위해 업데이트 전에 기존 데이터를 삭제합니다.
        
        Args:
            source: 소스 식별자 (Notion page_id 또는 Gitea repo URL)
        
        Returns:
            삭제된 문서 개수
        """
        logger.info(f'소스 "{source}"의 데이터 삭제 시작...')
        
        try:
            # 소스에 해당하는 모든 문서 검색
            results = self.collection.get(
                where={"source": source}
            )
            
            if not results or not results.get('ids'):
                logger.info(f'삭제할 데이터 없음: {source}')
                return 0
            
            delete_ids = results['ids']
            
            # 삭제 수행
            self.collection.delete(ids=delete_ids)
            
            logger.info(f'소스 "{source}"에서 {len(delete_ids)}개 문서 삭제 완료')
            return len(delete_ids)
            
        except Exception as e:
            logger.error(f'데이터 삭제 실패 ({source}): {str(e)}')
            return 0
    
    def delete_by_document_id(self, document_id: str) -> int:
        """
        ✅ [Delta Sync] 특정 문서 ID의 모든 청크 삭제
        
        Notion 페이지가 수정되면 해당 페이지의 모든 청크를 삭제하고
        새로 생성된 청크로 교체합니다.
        
        Args:
            document_id: Notion 페이지 ID
        
        Returns:
            삭제된 청크 개수
        """
        try:
            # document_id에 해당하는 모든 청크 검색
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if not results or not results.get('ids'):
                return 0
            
            delete_ids = results['ids']
            
            # 삭제 수행
            self.collection.delete(ids=delete_ids)
            
            logger.debug(f'문서 "{document_id}"에서 {len(delete_ids)}개 청크 삭제')
            return len(delete_ids)
            
        except Exception as e:
            logger.error(f'청크 삭제 실패 ({document_id}): {str(e)}')
            return 0
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        use_graph: bool = True,
        use_reranking: bool = False,
        graph_depth: int = 1
    ) -> List[Dict[str, Any]]:
        """
        ✅ [Graph-Enhanced Hybrid Search + Cross-Encoder Reranking]
        
        5단계 검색 파이프라인:
        1. [Broad] 벡터 + BM25로 후보 추출
        2. [Graph Expansion] 후보의 연결된 문서도 포함
        3. [Hybrid Score] 벡터 0.7 + BM25 0.3 + Graph 보너스
        4. [Cross-Encoder Reranking] 정밀 재정렬
        5. [Select] 상위 top_k개 반환
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 문서 개수
            use_hybrid: 하이브리드 검색 사용 여부
            use_graph: 그래프 확장 사용 여부
            use_reranking: Cross-Encoder Reranking 사용 여부
            graph_depth: 그래프 탐색 깊이
        
        Returns:
            정밀하게 재정렬된 검색 결과
        """
        try:
            # ✅ RERANKER_TOP_K 환경변수 가져오기 (기본값 50)
            reranker_top_k = int(os.getenv('RERANKER_TOP_K', '50'))
            
            logger.info(f'🔍 검색 시작: "{query[:50]}..." (top_k={top_k}, graph={use_graph}, rerank={use_reranking})')
            
            # ========================================
            # [Step 1] 1차 검색: 벡터 검색
            # ========================================
            # ✅ Reranking 사용시 RERANKER_TOP_K만큼, 아니면 기존 로직
            if use_reranking:
                search_k = reranker_top_k
            else:
                search_k = min(top_k * 4, 30)  # 후보 풀
            
            logger.debug(f'[Step 1] 검색 후보 개수: {search_k} (reranking={use_reranking})')
            
            # 쿼리 임베딩 생성
            query_embedding_array = self.embedding_service.encode([query])
            query_embedding = query_embedding_array[0].tolist()
            
            vector_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=search_k
            )
            
            if not vector_results['ids'][0]:
                logger.warning('검색 결과 없음')
                return []
            
            # 결과 초기화
            results_map: Dict[str, Dict[str, Any]] = {}
            
            for i, doc_id in enumerate(vector_results['ids'][0]):
                similarity = 1 / (1 + vector_results['distances'][0][i])
                results_map[doc_id] = {
                    'id': doc_id,
                    'content': vector_results['documents'][0][i],
                    'metadata': vector_results['metadatas'][0][i],
                    'vector_score': similarity,
                    'bm25_score': 0.0,
                    'graph_score': 0.0,
                    'rerank_score': 0.0,
                    'final_score': similarity,
                    'source': 'vector'
                }
            
            logger.debug(f'[Step 1] 벡터 검색: {len(results_map)}개 후보')
            
            # ========================================
            # [Step 2] 그래프 확장 (Graph Traversal)
            # ========================================
            if use_graph and self.graph and self.graph.number_of_nodes() > 0:
                expanded_doc_ids = self._expand_with_graph(
                    seed_doc_ids=list(results_map.keys()),
                    depth=graph_depth
                )
                
                # 확장된 문서 중 새로운 것만 추가
                new_doc_ids = [d for d in expanded_doc_ids if d not in results_map]
                
                if new_doc_ids:
                    # ChromaDB에서 확장된 문서 조회
                    try:
                        expanded_docs = self.collection.get(
                            ids=new_doc_ids[:20],  # 최대 20개
                            include=['documents', 'metadatas']
                        )
                        
                        for i, doc_id in enumerate(expanded_docs['ids']):
                            if doc_id not in results_map:
                                results_map[doc_id] = {
                                    'id': doc_id,
                                    'content': expanded_docs['documents'][i] if expanded_docs['documents'] else '',
                                    'metadata': expanded_docs['metadatas'][i] if expanded_docs['metadatas'] else {},
                                    'vector_score': 0.3,  # 그래프로 발견된 문서는 낮은 벡터 점수
                                    'bm25_score': 0.0,
                                    'graph_score': 0.5,  # 그래프 보너스
                                    'rerank_score': 0.0,
                                    'final_score': 0.4,
                                    'source': 'graph'
                                }
                        
                        logger.debug(f'[Step 2] 그래프 확장: +{len(new_doc_ids)}개 문서')
                    except Exception as e:
                        logger.warning(f'그래프 확장 문서 조회 실패: {str(e)}')
            
            # ========================================
            # [Step 3] BM25 + Hybrid 스코어 계산 (✅ Min-Max 정규화)
            # ========================================
            if use_hybrid:
                doc_ids = list(results_map.keys())
                documents = [results_map[d]['content'] for d in doc_ids]
                
                bm25_scores = self._calculate_bm25_scores(
                    query=query,
                    doc_ids=doc_ids,
                    documents=documents
                )
                
                # ✅ [Min-Max Normalization] BM25 점수를 0~1로 정규화
                bm25_values = list(bm25_scores.values())
                bm25_min = min(bm25_values) if bm25_values else 0
                bm25_max = max(bm25_values) if bm25_values else 1
                bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1
                
                # ✅ [Min-Max Normalization] Vector 점수도 정규화
                vector_values = [r['vector_score'] for r in results_map.values()]
                vector_min = min(vector_values) if vector_values else 0
                vector_max = max(vector_values) if vector_values else 1
                vector_range = vector_max - vector_min if vector_max > vector_min else 1
                
                # Hybrid 스코어 계산 (정규화된 점수 사용)
                for doc_id, result in results_map.items():
                    bm25_score = bm25_scores.get(doc_id, 0.0)
                    result['bm25_score'] = bm25_score
                    
                    # ✅ Min-Max 정규화 (0~1 범위로 압축)
                    normalized_vector = (result['vector_score'] - vector_min) / vector_range
                    normalized_bm25 = (bm25_score - bm25_min) / bm25_range
                    graph_bonus = result.get('graph_score', 0.0)  # 이미 0~1
                    
                    # ✅ Hybrid = 0.6 * Vector + 0.25 * BM25 + 0.15 * Graph
                    # 모든 점수가 0~1 범위이므로 공정한 가중치 적용
                    result['final_score'] = (
                        0.6 * normalized_vector +
                        0.25 * normalized_bm25 +
                        0.15 * graph_bonus
                    )
                
                logger.debug(f'[Step 3] Hybrid 스코어 계산 완료 (BM25 범위: {bm25_min:.2f}~{bm25_max:.2f})')
            
            # ========================================
            # [Step 4] BGE Reranker (✅ Min-Max 정규화)
            # ========================================
            if use_reranking:
                reranker = get_reranker()
                
                if reranker:
                    # Reranking할 후보 (상위 15개)
                    sorted_results = sorted(
                        results_map.values(),
                        key=lambda x: x['final_score'],
                        reverse=True
                    )[:15]
                    
                    # Query-Document 쌍 생성 (BGE는 1024자 지원)
                    pairs = [(query, r['content'][:2000]) for r in sorted_results]
                    
                    try:
                        rerank_scores = reranker.predict(pairs)
                        
                        # ✅ [Min-Max Normalization] Rerank 점수 정규화
                        rerank_min = min(rerank_scores)
                        rerank_max = max(rerank_scores)
                        rerank_range = rerank_max - rerank_min if rerank_max > rerank_min else 1
                        
                        for i, result in enumerate(sorted_results):
                            raw_score = float(rerank_scores[i])
                            result['rerank_score'] = raw_score
                            
                            # ✅ Min-Max 정규화 (0~1)
                            normalized_rerank = (raw_score - rerank_min) / rerank_range
                            
                            # ✅ Rerank 점수 반영 (0.4 * 기존 + 0.6 * Rerank)
                            result['final_score'] = (
                                0.4 * result['final_score'] +
                                0.6 * normalized_rerank
                            )
                        
                        logger.debug(f'[Step 4] BGE Reranker 완료 (Rerank 범위: {rerank_min:.2f}~{rerank_max:.2f})')
                    except Exception as e:
                        logger.warning(f'Reranking 실패: {str(e)}')
                else:
                    logger.debug('[Step 4] Reranker 미사용')
            
            # ========================================
            # [Step 5] 최종 정렬 및 반환
            # ========================================
            final_results = sorted(
                results_map.values(),
                key=lambda x: x['final_score'],
                reverse=True
            )[:top_k]
            
            # 로그
            if use_reranking:
                logger.info(f'✅ 검색 완료: {len(final_results)}개 반환 (후보 {search_k}개에서 Reranking 적용)')
            else:
                logger.info(f'✅ 검색 완료: {len(final_results)}개 반환 (후보 {search_k}개)')
            for i, r in enumerate(final_results[:3]):
                title = r['metadata'].get('title', 'Untitled')[:30]
                logger.debug(f'   #{i+1}: {title}... (score={r["final_score"]:.3f}, src={r["source"]})')
            
            return final_results
            
        except Exception as e:
            logger.error(f'❌ 검색 실패: {str(e)}', exc_info=True)
            return []
    
    def _expand_with_graph(
        self,
        seed_doc_ids: List[str],
        depth: int = 1
    ) -> List[str]:
        """
        ✅ [Graph Traversal] 시드 문서에서 연결된 문서 확장
        
        Args:
            seed_doc_ids: 시드 문서 ID 리스트
            depth: 탐색 깊이
        
        Returns:
            확장된 문서 ID 리스트 (시드 포함)
        """
        if not self.graph or self.graph.number_of_nodes() == 0:
            return seed_doc_ids
        
        expanded: Set[str] = set()
        
        for doc_id in seed_doc_ids[:5]:  # 상위 5개만 확장
            # document_id 추출 (chunk_id에서)
            base_doc_id = doc_id.split('_chunk_')[0] if '_chunk_' in doc_id else doc_id
            
            if base_doc_id in self.graph:
                try:
                    # ego_graph로 depth 범위 내 노드 추출
                    subgraph = nx.ego_graph(
                        self.graph,
                        base_doc_id,
                        radius=depth,
                        undirected=True
                    )
                    
                    for node_id in subgraph.nodes():
                        # 가상 노드 제외
                        node_data = self.graph.nodes.get(node_id, {})
                        if node_data.get('node_type') not in ['virtual_root', 'ghost']:
                            expanded.add(node_id)
                            
                except Exception as e:
                    logger.debug(f'그래프 탐색 실패 ({base_doc_id}): {str(e)}')
        
        logger.debug(f'🔗 그래프 확장: {len(seed_doc_ids)} → {len(expanded)}개 문서')
        return list(expanded)
    
    def _calculate_bm25_scores(
        self,
        query: str,
        doc_ids: List[str],
        documents: List[str]
    ) -> Dict[str, float]:
        """
        쿼리에 대한 각 문서의 BM25 점수 계산
        
        Args:
            query: 검색 쿼리
            doc_ids: 문서 ID 리스트
            documents: 문서 텍스트 리스트
        
        Returns:
            {doc_id: bm25_score} 딕셔너리
        """
        try:
            # 텍스트 토큰화
            corpus = [self._tokenize(doc) for doc in documents]
            query_tokens = self._tokenize(query)
            
            if not corpus or not query_tokens:
                return {doc_id: 0.0 for doc_id in doc_ids}
            
            # BM25 객체 생성 및 점수 계산
            bm25 = BM25Okapi(corpus)
            scores = bm25.get_scores(query_tokens)
            
            # 문서 ID와 점수 매핑
            bm25_scores = {
                doc_ids[i]: float(scores[i])
                for i in range(len(doc_ids))
            }
            
            return bm25_scores
            
        except Exception as e:
            logger.warning(f'BM25 계산 실패: {str(e)}')
            return {doc_id: 0.0 for doc_id in doc_ids}
    
    def _tokenize(self, text: str) -> List[str]:
        """
        간단한 텍스트 토큰화 (한글 + 영문 지원)
        
        Args:
            text: 토큰화할 텍스트
        
        Returns:
            토큰 리스트
        """
        # 소문자 변환
        text = text.lower()
        
        # 특수 문자 제거, 공백으로 분리
        tokens = re.findall(r'\b\w+\b', text)
        
        return tokens
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        컬렉션 통계 조회
        
        Returns:
            컬렉션 통계
        """
        try:
            count = self.collection.count()
            
            return {
                'collection_name': 'rag_documents',
                'document_count': count,
                'host': self.chroma_host,
                'port': self.chroma_port
            }
        except Exception as e:
            logger.error(f'통계 조회 실패: {str(e)}')
            return {}
    
    