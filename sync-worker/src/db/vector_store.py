"""
VectorStoreManager - ChromaDB를 통한 벡터 저장소 관리

HuggingFace의 BAAI/bge-m3 임베딩 모델을 사용하여 문서를 저장하고 조회합니다.
Hybrid Search (BM25 + Vector) + Reranking으로 검색 정확도를 극대화합니다.
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
import torch
from rank_bm25 import BM25Okapi
import re

# ✅ [Reranking] FlashRank 임포트
try:
    from flashrank import Ranker
    FLASHRANK_AVAILABLE = True
except ImportError:
    FLASHRANK_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning('⚠️  FlashRank 미설치: Reranking 기능 비활성화')

logger = logging.getLogger(__name__)


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
        
        # HuggingFace 임베딩 모델 초기화 (GPU 자동 감지)
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f'임베딩 디바이스: {device}')
            
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={"device": device},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info(f'HuggingFace 임베딩 모델 로드 완료: {model_name} ({device})')
        except Exception as e:
            logger.error(f'임베딩 모델 로드 실패: {str(e)}')
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
        
        # BM25 인덱스 초기화 (지연 로딩)
        self.bm25_index: Optional[BM25Okapi] = None
        self.bm25_corpus: List[List[str]] = []
        self.bm25_doc_ids: List[str] = []
        
        # ✅ [Reranking] FlashRank Reranker 초기화
        self.ranker: Optional[Ranker] = None
        if FLASHRANK_AVAILABLE:
            try:
                logger.info('⏳ FlashRank Reranker 초기화 중...')
                self.ranker = Ranker(
                    model_name="ms-marco-MiniLM-L-12-v2",
                    cache_dir="/app/models"
                )
                logger.info('✅ FlashRank Reranker 초기화 완료 (가벼운 모델: ~40MB)')
            except Exception as e:
                logger.warning(f'⚠️  FlashRank 초기화 실패: {str(e)}. Reranking 없이 진행합니다.')
                self.ranker = None
        else:
            logger.warning('⚠️  FlashRank 미설치: Reranking 기능 비활성화')
    
    def add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100) -> int:
        """
        배치 단위로 문서를 벡터 저장소에 추가
        
        Args:
            documents: 추가할 문서 리스트. 각 문서는 다음 구조:
                {
                    'id': str,
                    'content': str,
                    'metadata': dict
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
                
                for doc in batch:
                    # 필수 필드 검증
                    if not doc.get('id') or not doc.get('content'):
                        logger.warning(f'문서 필드 누락: {doc.get("id", "unknown")}')
                        continue
                    
                    ids.append(str(doc['id']))  # ID는 문자열로 변환
                    documents_text.append(str(doc['content']))  # Content는 문자열로 변환
                    metadatas.append(doc.get('metadata', {}))
                
                if not ids:
                    logger.warning(f'배치 {batch_num}: 유효한 문서 없음')
                    continue
                
                logger.debug(f'배치 {batch_num}: {len(ids)}개 문서 임베딩 생성 중...')
                
                # 임베딩 생성
                try:
                    embeddings_list = self.embeddings.embed_documents(documents_text)
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
    
    def search(self, query: str, top_k: int = 5, use_hybrid: bool = True) -> List[Dict[str, Any]]:
        """
        ✅ [하이브리드 + Reranking] Hybrid Search + 정밀 재정렬
        
        3단계 검색:
        1. [Broad] 벡터 + BM25로 20개 후보 추출 (빠르지만 부정확)
        2. [Reranking] 정밀 Reranker가 20개를 꼼꼼히 재평가 (정확함)
        3. [Select] Reranking 상위 top_k개 반환
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 문서 개수
            use_hybrid: 하이브리드 검색 사용 여부
        
        Returns:
            정밀하게 재정렬된 검색 결과
        """
        try:
            # [Step 1] 1차 검색: 벡터 + BM25 (넉넉하게 20개 추출)
            search_k = min(top_k * 4, 20)  # top_k=5면 20개, top_k=2면 8개
            
            query_embedding = self.embeddings.embed_query(query)
            vector_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=search_k
            )
            
            vector_scores = []
            for i, doc_id in enumerate(vector_results['ids'][0]):
                # 거리를 유사도 점수로 변환 (코사인 거리: 0~2, 유사도: 1~(-1))
                similarity = 1 / (1 + vector_results['distances'][0][i])
                vector_scores.append({
                    'id': doc_id,
                    'content': vector_results['documents'][0][i],
                    'metadata': vector_results['metadatas'][0][i],
                    'vector_score': similarity,
                    'bm25_score': 0.0,
                    'hybrid_score': similarity
                })
            
            if not use_hybrid or not vector_scores:
                return vector_scores[:top_k]
            
            # [Step 2] BM25 점수 계산
            bm25_scores = self._calculate_bm25_scores(
                query=query,
                doc_ids=[r['id'] for r in vector_scores],
                documents=[r['content'] for r in vector_scores]
            )
            
            # [Step 3] Hybrid 스코어 계산 (0.7 * 벡터 + 0.3 * BM25)
            for result in vector_scores:
                doc_id = result['id']
                bm25_score = bm25_scores.get(doc_id, 0.0)
                result['bm25_score'] = bm25_score
                
                # 정규화
                normalized_vector = max(0, min(1, result['vector_score']))
                normalized_bm25 = max(0, min(1, bm25_score))
                
                result['hybrid_score'] = (0.7 * normalized_vector) + (0.3 * normalized_bm25)
            
            # [Step 4] Hybrid 점수로 정렬
            vector_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
            logger.info(f'✅ Hybrid Search: {search_k}개 후보 추출')
            
            # [Step 5] ✅ Reranking (선택사항)
            if self.ranker is not None and len(vector_scores) > 0:
                logger.info(f'⏳ Reranking 시작: {query} ({len(vector_scores)}개 후보)')
                
                try:
                    # FlashRank 포맷: List[{"id": "...", "text": "..."}]
                    passages = [
                        {
                            "id": r['id'],
                            "text": r['content']
                        }
                        for r in vector_scores
                    ]
                    
                    # ✅ FlashRank rerank() 메서드 호출 (올바른 API)
                    ranked = self.ranker.rerank(query, passages, top_k=top_k)
                    
                    # Reranked 결과를 원본 포맷으로 변환
                    reranked_results = []
                    for rank_idx, ranked_item in enumerate(ranked):
                        # 원본 결과에서 찾기
                        for orig in vector_scores:
                            if orig['id'] == ranked_item['id']:
                                reranked_results.append({
                                    **orig,
                                    'reranker_score': float(ranked_item.get('score', 0.0)),
                                    'reranker_rank': rank_idx + 1
                                })
                                break
                    
                    if reranked_results:
                        logger.info(f'✅ Reranking 완료: 상위 {len(reranked_results)}개 (1위 점수: {reranked_results[0].get("reranker_score", 0.0):.4f})')
                        return reranked_results
                    else:
                        logger.warning('⚠️  Reranking 결과 없음 (Hybrid 결과로 폴백)')
                        return vector_scores[:top_k]
                    
                except AttributeError as e:
                    logger.warning(f'⚠️  FlashRank API 오류: {str(e)}. Hybrid 결과로 폴백합니다.')
                    return vector_scores[:top_k]
                except Exception as e:
                    logger.warning(f'⚠️  Reranking 오류 (Hybrid 결과로 폴백): {str(e)}')
                    return vector_scores[:top_k]
            else:
                # Reranker 없으면 Hybrid 결과만 반환
                logger.info(f'ℹ️  Reranker 미사용: Hybrid 결과 직접 반환')
                return vector_scores[:top_k]
            
        except Exception as e:
            logger.error(f'❌ 검색 실패: {str(e)}')
            return []
    
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
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 2,
        use_hybrid: bool = True
    ) -> str:
        """
        ✅ [업그레이드] 페이지 전체 문맥 검색 (Full Page Retrieval)
        
        조각만 던져주는 게 아니라, 검색된 조각이 포함된 '원본 페이지 전체'를 복원하여 제공합니다.
        
        1단계: 질문과 관련된 가장 유사한 '조각' top_k개를 찾습니다.
        2단계: 그 조각들이 어느 '원본 페이지(Source URL)'에서 왔는지 확인합니다.
        3단계: 해당 페이지의 모든 조각을 DB에서 가져와서 원본 순서대로 합칩니다.
        4단계: 합쳐진 '전체 페이지 내용'을 AI에게 던져줍니다.
        
        Args:
            query: 사용자 질문
            top_k: 힌트가 될 조각 개수 (2-3개로 충분함. 전체 페이지를 다 가져오므로)
            use_hybrid: 하이브리드 검색 사용 여부
        
        Returns:
            마크다운 형식의 전체 페이지 컨텍스트
        """
        logger.info(f'✅ 페이지 전체 검색 시작 (Full Page Mode): {query}')
        
        # Step 1: 일단 가장 관련성 높은 '조각'들을 찾습니다.
        results = self.search(query=query, top_k=top_k, use_hybrid=use_hybrid)
        
        if not results:
            logger.warning(f'검색 결과 없음: {query}')
            return ''
        
        logger.info(f'Step 1: {len(results)}개 힌트 조각 발견')
        
        # Step 2: 검색된 결과에서 유니크한 'Source(출처)'를 추출합니다.
        # 예: ['notion.so/page_A', 'notion.so/page_B']
        unique_sources = []
        seen_sources = set()
        
        for r in results:
            src = r['metadata'].get('source')
            if src and src not in seen_sources:
                unique_sources.append(src)
                seen_sources.add(src)
        
        logger.info(f'Step 2: 관련된 원본 페이지 발견: {len(unique_sources)}개')
        for i, src in enumerate(unique_sources, 1):
            logger.debug(f'  - 페이지 {i}: {src}')

        # Step 3: 각 소스별로 전체 내용을 다시 긁어옵니다.
        context_parts = []
        
        for page_idx, source in enumerate(unique_sources, 1):
            try:
                logger.debug(f'Step 3-{page_idx}: "{source}"의 모든 조각 조회 중...')
                
                # 해당 소스를 가진 모든 청크를 가져옵니다.
                page_data = self.collection.get(
                    where={"source": source},
                    include=['documents', 'metadatas']
                )
                
                if not page_data['ids']:
                    logger.warning(f'페이지 조회 실패: {source} (조각 없음)')
                    continue

                logger.info(f'  ✓ {len(page_data["ids"])}개 조각 발견')

                # ID 순서대로 정렬하여 원본 순서 유지
                # (보통 chunk_0, chunk_1, chunk_2 순으로 ID가 생성되므로 문자열 정렬로 커버됨)
                combined = sorted(
                    zip(page_data['ids'], page_data['documents'], page_data['metadatas']),
                    key=lambda x: x[0]  # ID 기준으로 정렬
                )
                
                # 메타데이터에서 제목 추출 (첫 번째 조각의 제목 사용)
                title = combined[0][2].get('title', '제목 없음')
                
                # 전체 텍스트를 원본 순서대로 합치기
                full_text = "\n\n".join([item[1] for item in combined])
                
                # Context 구성
                if page_idx > 1:
                    context_parts.append('\n' + '='*60 + '\n')
                
                context_parts.append(f'### 📄 페이지 {page_idx}: {title}')
                context_parts.append(f'**출처**: {source}')
                context_parts.append(f'**조각 수**: {len(combined)}개 (전체 복원됨)')
                context_parts.append('-' * 60)
                context_parts.append(f'{full_text}')
                context_parts.append('-' * 60)
                
                logger.info(f'  ✓ 페이지 복원 완료: {len(full_text)}자')
                
            except Exception as e:
                logger.error(f'페이지 복원 실패 ({source}): {str(e)}', exc_info=True)
                continue
        
        # Step 4: 최종 컨텍스트 생성
        final_context = ''.join(context_parts)
        logger.info(f'✅ 전체 페이지 컨텍스트 생성 완료: {len(unique_sources)}개 페이지, 총 {len(final_context)}자')
        
        return final_context
