"""
VectorStoreManager - ChromaDB를 통한 벡터 저장소 관리

HuggingFace의 BAAI/bge-m3 임베딩 모델을 사용하여 문서를 저장하고 조회합니다.
"""

import logging
from typing import List, Dict, Any, Optional
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
import time

logger = logging.getLogger(__name__)


class VectorStoreManager:
    """ChromaDB를 통한 벡터 저장소 관리 클래스"""
    
    def __init__(self, chroma_host: str = 'localhost', chroma_port: int = 8000):
        """
        Args:
            chroma_host: ChromaDB 호스트
            chroma_port: ChromaDB 포트
        """
        self.chroma_host = chroma_host
        self.chroma_port = chroma_port
        
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
        
        # HuggingFace 임베딩 모델 초기화
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name="BAAI/bge-m3",
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True}
            )
            logger.info('HuggingFace 임베딩 모델 로드 완료: BAAI/bge-m3')
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
        
        # 컬렉션 초기화 (존재하지 않으면 생성)
        try:
            self.collection = self.client.get_or_create_collection(
                name='rag_documents',
                metadata={"hnsw:space": "cosine"}
            )
            logger.info('컬렉션 "rag_documents" 준비 완료')
        except Exception as e:
            logger.error(f'컬렉션 생성 실패: {str(e)}')
            raise
    
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
        
        try:
            # 배치 단위로 처리
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                
                ids = []
                documents_text = []
                metadatas = []
                
                for doc in batch:
                    ids.append(doc['id'])
                    documents_text.append(doc['content'])
                    metadatas.append(doc.get('metadata', {}))
                
                # 임베딩 생성
                try:
                    embeddings_list = self.embeddings.embed_documents(documents_text)
                    logger.debug(f'배치 {i//batch_size + 1}: {len(documents_text)}개 문서 임베딩 완료')
                except Exception as e:
                    logger.error(f'임베딩 생성 실패 (배치 {i//batch_size + 1}): {str(e)}')
                    continue
                
                # ChromaDB에 추가
                try:
                    self.collection.add(
                        ids=ids,
                        embeddings=embeddings_list,
                        documents=documents_text,
                        metadatas=metadatas
                    )
                    added_count += len(batch)
                    logger.info(f'배치 {i//batch_size + 1}: {len(batch)}개 문서 저장 완료 (누적: {added_count})')
                except Exception as e:
                    logger.error(f'ChromaDB 저장 실패 (배치 {i//batch_size + 1}): {str(e)}')
                    continue
            
            logger.info(f'총 {added_count}개 문서 추가 완료')
            return added_count
            
        except Exception as e:
            logger.error(f'문서 추가 중 오류: {str(e)}')
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
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        유사도 검색
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 상위 문서 개수
        
        Returns:
            검색 결과 리스트
        """
        try:
            # 쿼리 임베딩 생성
            query_embedding = self.embeddings.embed_query(query)
            
            # 유사도 검색
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k
            )
            
            # 결과 정렬
            search_results = []
            for i, doc_id in enumerate(results['ids'][0]):
                search_results.append({
                    'id': doc_id,
                    'content': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            return search_results
            
        except Exception as e:
            logger.error(f'검색 실패: {str(e)}')
            return []
    
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
