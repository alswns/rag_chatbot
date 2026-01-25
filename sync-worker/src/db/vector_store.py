"""
VectorStoreManager - ChromaDB를 통한 벡터 저장소 관리

HuggingFace의 BAAI/bge-m3 임베딩 모델을 사용하여 문서를 저장하고 조회합니다.
Hybrid Search (BM25 + Vector)를 지원하여 검색 정확도를 높입니다.
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
import time
import torch
from rank_bm25 import BM25Okapi
import re

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
        하이브리드 검색 (Vector + BM25)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 문서 개수
            use_hybrid: 하이브리드 검색 사용 여부 (False면 벡터 검색만)
        
        Returns:
            재정렬된 검색 결과 리스트
        """
        try:
            # 1단계: 벡터 검색 (더 많은 문서를 검색한 후 재정렬)
            search_k = min(top_k * 3, 50)  # 상위 50개까지 검색
            
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
            
            # 2단계: BM25 점수 계산
            bm25_scores = self._calculate_bm25_scores(
                query=query,
                doc_ids=[r['id'] for r in vector_scores],
                documents=[r['content'] for r in vector_scores]
            )
            
            # 3단계: 재정렬 (가중치 합산: 0.7 * 벡터 + 0.3 * BM25)
            for result in vector_scores:
                doc_id = result['id']
                bm25_score = bm25_scores.get(doc_id, 0.0)
                result['bm25_score'] = bm25_score
                
                # 정규화 (벡터 점수: 0~1, BM25 점수: 0~1)
                normalized_vector = max(0, min(1, result['vector_score']))
                normalized_bm25 = max(0, min(1, bm25_score))
                
                result['hybrid_score'] = (0.7 * normalized_vector) + (0.3 * normalized_bm25)
            
            # 하이브리드 점수 기준 정렬
            vector_scores.sort(key=lambda x: x['hybrid_score'], reverse=True)
            
            logger.debug(f'하이브리드 검색 완료: {query} (상위 {top_k}개)')
            return vector_scores[:top_k]
            
        except Exception as e:
            logger.error(f'하이브리드 검색 실패: {str(e)}')
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
        top_k: int = 5,
        use_hybrid: bool = True
    ) -> str:
        """
        쿼리에 대한 관련 문서를 검색하고 컨텍스트 문자열 생성
        
        RAG 모델에 입력할 컨텍스트를 생성합니다.
        하이브리드 검색으로 정확도를 높입니다.
        
        Args:
            query: 사용자 질문
            top_k: 반환할 상위 문서 개수
            use_hybrid: 하이브리드 검색 사용 여부
        
        Returns:
            마크다운 형식의 컨텍스트 문자열
        """
        logger.info(f'컨텍스트 검색 시작: {query}')
        
        # 하이브리드 검색 수행
        results = self.search(query=query, top_k=top_k, use_hybrid=use_hybrid)
        
        if not results:
            logger.warning(f'검색 결과 없음: {query}')
            return ''
        
        # 마크다운 형식의 컨텍스트 생성
        context_parts = ['## 검색 결과\n']
        
        for i, result in enumerate(results, 1):
            title = result['metadata'].get('title', '제목 없음')
            source = result['metadata'].get('source', '소스 없음')
            content = result['content']
            
            score_info = f"Score: {result['hybrid_score']:.3f}"
            if result.get('vector_score'):
                score_info += f" (Vector: {result['vector_score']:.3f}, BM25: {result['bm25_score']:.3f})"
            
            context_parts.append(f"\n### {i}. {title}")
            context_parts.append(f"**Source**: {source}")
            context_parts.append(f"**{score_info}**\n")
            context_parts.append(f"{content}\n")
        
        context = ''.join(context_parts)
        logger.info(f'컨텍스트 생성 완료: {len(results)}개 문서')
        
        return context
