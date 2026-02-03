"""
GraphRAG 통합 파이프라인 - Notion 연동

Notion 데이터 → Node → 그래프 → Chroma DB → 질문 처리

의존성:
- NotionConnector: Notion API 데이터 수집
- GraphRAGProcessor: 그래프 구축 및 탐색
- VectorStoreManager: Chroma DB 관리
"""

import os
import json
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional

from connectors.notion import NotionConnector
from processors.graph_rag import GraphRAGProcessor, GraphNode
from db.vector_store import VectorStoreManager

logger = logging.getLogger(__name__)


class GraphRAGPipeline:
    """
    Notion → GraphRAG → Chroma DB 완전 파이프라인
    """
    
    def __init__(
        self,
        notion_token: str,
        chroma_host: str = 'localhost',
        chroma_port: int = 8000,
        notion_database_id: Optional[str] = None,
        max_chunk_tokens: int = 512,
        traversal_depth: int = 2
    ) -> None:
        """
        초기화
        
        Args:
            notion_token: Notion Integration Token
            chroma_host: Chroma DB 호스트
            chroma_port: Chroma DB 포트
            notion_database_id: 특정 Database ID (None이면 모든 DB)
            max_chunk_tokens: 청크 최대 토큰
            traversal_depth: 그래프 탐색 깊이
        """
        logger.info('GraphRAG 파이프라인 초기화 중')
        
        # 컴포넌트 초기화
        self.notion_connector: NotionConnector = NotionConnector(
            token=notion_token,
            database_id=notion_database_id
        )
        
        self.vector_store: VectorStoreManager = VectorStoreManager(
            chroma_host=chroma_host,
            chroma_port=chroma_port
        )
        
        # ✅ ChunkingProcessor 추가 (문서 → 청크 분할)
        from processors.chunking import ChunkingProcessor
        from processors.intelligent_chunking_adapter import IntelligentChunkingAdapter
        from core.config import ENABLE_INTELLIGENT_CHUNKING
        
        self.chunking_processor: ChunkingProcessor = ChunkingProcessor(
            recursive_chunk_size=os.getenv('CHUNK_SIZE', 900),
            recursive_chunk_overlap=os.getenv('CHUNK_OVERLAP', 200)
        )
        
        # ✅ Intelligent Chunking 어댑터 (메타데이터 추출 + 키워드)
        self.intelligent_chunking_adapter: Optional[IntelligentChunkingAdapter] = None
        if ENABLE_INTELLIGENT_CHUNKING:
            self.intelligent_chunking_adapter = IntelligentChunkingAdapter(
                chunk_size=int(os.getenv('CHUNK_SIZE', '900')),
                chunk_overlap=int(os.getenv('CHUNK_OVERLAP', '200'))
            )
            logger.info('✅ Intelligent Chunking 활성화')
        else:
            logger.info('⚠️ 기본 Chunking 사용 (Intelligent Chunking 비활성화)')
        
        self.processor: GraphRAGProcessor = GraphRAGProcessor(
            embedding_model=os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3'),
            max_chunk_tokens=max_chunk_tokens,
            traversal_depth=traversal_depth,
            relation_column_name=os.getenv('NOTION_RELATION_COLUMN', '작업')
        )
        logger.info('✓ 파이프라인 초기화 완료')
    
    def run_full_pipeline(
        self,
        collection_name: str = 'notion_graph',
        progress_file: str = '/app/data/sync_progress.json',
        batch_size: int = 16,
        last_sync_time: Optional[str] = None  # ✅ Delta Sync용 마지막 동기화 시간
    ) -> Dict[str, Any]:
        """
        ✅ [리팩토링] 배치 처리 파이프라인
        
        흐름:
        1. Notion에서 페이지 목록 조회 (메타데이터만)
        2. 32개씩 배치로 처리: 청크 분할 → 일괄 임베딩 → 일괄 DB 저장
        3. 모든 문서 처리 후: 그래프 구축 (metadata 기반)
        
        Args:
            collection_name: Chroma DB 컬렉션명
            progress_file: 진행률 저장 파일 경로
            batch_size: 배치 처리 크기 (기본 32)
        
        Returns:
            처리 결과 통계
        """
        logger.info('=' * 80)
        logger.info(f'GraphRAG 파이프라인 실행 시작 (배치 크기: {batch_size})')
        logger.info('=' * 80)
        
        try:
            # 초기화
            os.makedirs(os.path.dirname(progress_file), exist_ok=True)
            processed_docs = 0
            processed_chunks = 0
            failed_documents = 0
            all_nodes: List[GraphNode] = []
            graph_metadata_list: List[Dict[str, Any]] = []  # ✅ 그래프 엣지용 메타데이터
            total_documents = 0
            
            # 1단계: Notion에서 페이지 목록 조회 (마지막 동기화 이후 수정된 페이지만)
            if last_sync_time:
                logger.info(f'[1/4] Delta Sync 모드 - {last_sync_time} 이후 수정된 페이지만 조회')
            else:
                logger.info('[1/4] Full Sync 모드 - 모든 페이지 조회')
            
            pages: List[Dict[str, Any]] = self.notion_connector.get_updated_pages(
                last_sync_time=last_sync_time
            )
            
            # ✅ [개선] 조회 실패해도 기존 그래프 유지
            graph_path = os.getenv('GRAPH_PERSIST_PATH', './data/graph.pkl')
            
            if not pages:
                logger.warning('[1/4] Notion에서 조회된 페이지가 없습니다')
                
                # 기존 그래프가 있으면 로드하여 유지
                if os.path.exists(graph_path):
                    logger.info('[1/4] 기존 그래프 유지 (변경 없음)')
                    try:
                        self.processor.load_graph(graph_path)
                        node_count = self.processor.graph.number_of_nodes()
                        edge_count = self.processor.graph.number_of_edges()
                        logger.info(f'[1/4] ✓ 기존 그래프 로드 완료 (노드: {node_count}, 엣지: {edge_count})')
                    except Exception as e:
                        logger.warning(f'기존 그래프 로드 실패: {str(e)}')
                
                self._save_progress(progress_file, 0, 0, 'completed')
                return {
                    'status': 'success',  # 실패가 아닌 성공 (변경 없음)
                    'sync_mode': 'no_changes',
                    'message': '조회된 페이지가 없습니다 (변경 없음)',
                    'documents_count': 0,
                    'nodes_count': self.processor.graph.number_of_nodes() if self.processor.graph else 0,
                    'edges_count': self.processor.graph.number_of_edges() if self.processor.graph else 0
                }
            
            total_documents = len(pages)
            logger.info(f'[1/4] ✓ {total_documents}개 페이지 목록 조회 완료')
            
            # ✅ [개선] 전체 페이지 맵 구축 (부모 경로 추적용 - 캐싱 방식)
            # API 호출 없이 모든 페이지의 부모 경로를 추적할 수 있음
            logger.info('[1/4] 전체 페이지 맵 구축 중 (부모 경로 추적용)...')
            page_map = self.notion_connector.build_full_page_map()
            logger.info(f'[1/4] ✓ 전체 페이지 맵 구축 완료: {len(page_map)}개 페이지')
            
            # 초기 진행률 저장 (시작됨)
            self._save_progress(progress_file, 0, total_documents, 'processing')
            
            # 2단계: 배치 처리 초기화
            is_delta_sync = last_sync_time is not None
            logger.info(f'[2/4] 배치 처리 시작 (배치 크기: {batch_size}, Delta Sync: {is_delta_sync})')
            
            batch_buffer: List[Dict[str, Any]] = []  # 청크 버퍼
            batch_nodes: List[GraphNode] = []
            batch_documents: List[Dict[str, Any]] = []
            batch_count = 0
            updated_page_ids: List[str] = []  # Delta Sync로 업데이트된 페이지 ID
            
            # 제너레이터로 문서 순차 처리
            for doc_idx, document in enumerate(self._fetch_notion_documents_generator(pages)):
                try:
                    page_id = document.get('id')
                    
                    # ✅ [Delta Sync] 수정된 페이지의 기존 청크 삭제
                    if is_delta_sync and page_id:
                        deleted_count = self.vector_store.delete_by_document_id(page_id)
                        if deleted_count > 0:
                            logger.info(f'🔄 [Delta] 기존 청크 삭제: {document.get("title")[:30]} ({deleted_count}개)')
                        updated_page_ids.append(page_id)
                    
                    # ✅ [추가] 부모 경로 생성 (루트까지 추적)
                    breadcrumb_path = self._get_breadcrumb_path(
                        page_id=page_id,
                        page_map=page_map
                    )
                    
                    # 2-1. 문서를 청크로 분할 (Intelligent Chunking 사용 또는 기본)
                    if self.intelligent_chunking_adapter is not None:
                        # ✅ Intelligent Chunking: 메타데이터 보강 + Metadata Header 삽입
                        logger.info(f'🔬 Intelligent Chunking: {document.get("title")[:30]}...')
                        
                        intelligent_chunks = self.intelligent_chunking_adapter.process_notion_page_with_enrichment(
                            page_id=page_id,
                            title=document.get('title', 'Untitled'),
                            content=document.get('content', ''),
                            last_edited_time=document.get('updated_at', ''),
                            breadcrumb_path=breadcrumb_path,
                            parent_id=document.get('parent_id'),
                            url=document.get('url', '')
                        )
                        chunks = intelligent_chunks
                    else:
                        # 기본 청킹: 기존 로직 사용
                        chunks = self.chunking_processor.process_notion_page(
                            page_id=page_id,
                            title=document.get('title', 'Untitled'),
                            content=document.get('content', ''),
                            last_edited_time=document.get('updated_at', ''),
                            breadcrumb_path=breadcrumb_path
                        )
                    
                    if not chunks:
                        logger.debug(f'청크 없음 (빈 문서): {document.get("title")}')
                        continue
                    
                    # 2-2. Node 생성 (문서당 1개)
                    nodes = self.processor.process_document_to_nodes([document])
                    if not nodes:
                        continue
                    
                    node = nodes[0]
                    all_nodes.append(node)
                    batch_nodes.append(node)
                    batch_documents.append(document)
                    
                    # ✅ [핵심] 그래프 엣지용 메타데이터 저장 (content 포함 - Mention 파싱용)
                    graph_metadata_list.append({
                        'id': document.get('id'),
                        'title': document.get('title'),
                        'parent_id': document.get('parent_id'),
                        'properties': document.get('properties', {}),
                        'content': document.get('content', '')  # Mention 링크 추출용
                    })
                    
                    # 2-3. 각 청크를 버퍼에 추가
                    for chunk_idx, chunk in enumerate(chunks):
                        # Intelligent Chunking 결과인지 확인
                        if isinstance(chunk, dict) and 'content' in chunk:
                            # Intelligent Chunking 형식 (formatted_content + metadata)
                            chunk_content = chunk.get('content', '')
                            chunk_metadata = chunk.get('metadata', {})
                            
                            # Metadata Header가 포함된 포맷된 콘텐츠 사용 (임베딩 시 메타데이터도 함께)
                            chunk_text = chunk.get('formatted_content', chunk_content)
                        else:
                            # 기본 청킹 형식
                            chunk_content = chunk.get('content', '')
                            chunk_metadata = chunk.get('metadata', {})
                            chunk_text = chunk_content
                        
                        chroma_doc = {
                            'id': f"{document.get('id')}_chunk_{chunk_idx}",
                            'content': chunk_text,  # Metadata Header 포함
                            'metadata': self._sanitize_metadata({
                                'title': chunk_metadata.get('title', document.get('title', 'Untitled')),
                                'source_url': document.get('url', ''),
                                'document_id': document.get('id'),
                                'chunk_index': chunk_idx,
                                'chunk_count': len(chunks),
                                'created_at': document.get('created_at'),
                                'updated_at': document.get('updated_at'),
                                'source': 'notion',
                                'parent_id': document.get('parent_id'),
                                'header_1': chunk_metadata.get('header_1', ''),
                                'header_2': chunk_metadata.get('header_2', ''),
                                
                                # ✅ Intelligent Chunking 메타데이터
                                'breadcrumb_path': chunk_metadata.get('breadcrumb_path', ''),
                                'document_type': chunk_metadata.get('document_type', ''),
                                'subject': chunk_metadata.get('subject', ''),
                                'keywords': chunk_metadata.get('keywords', ''),
                                'temporal_info': chunk_metadata.get('temporal_info', ''),
                                'entities': chunk_metadata.get('entities', ''),
                                'local_keywords': str(chunk_metadata.get('local_keywords', []))
                            })
                        }
                        batch_buffer.append(chroma_doc)
                    
                    processed_docs += 1
                    logger.info(f'📝 문서 처리 완료: {document.get("title")[:30]} → {len(chunks)}개 청크')
                    
                    # ✅ 2-4. 배치 크기 도달 시 일괄 처리
                    if len(batch_nodes) >= batch_size:
                        batch_count += 1
                        logger.info(f'🔄 배치 #{batch_count} 처리 시작 ({len(batch_buffer)}개 청크)')
                        self._process_batch(
                            batch_count=batch_count,
                            batch_nodes=batch_nodes,
                            batch_documents=batch_documents,
                            batch_buffer=batch_buffer
                        )
                        processed_chunks += len(batch_buffer)
                        
                        # 버퍼 초기화
                        batch_nodes = []
                        batch_documents = []
                        batch_buffer = []
                        
                        # 진행률 저장
                        self._save_progress(progress_file, processed_docs, total_documents, 'processing')
                        
                        pct = int((processed_docs / total_documents) * 100)
                        logger.info(f'[2/4] 진행: {processed_docs}/{total_documents} ({pct}%) - 청크: {processed_chunks}개')
                    
                except Exception as e:
                    failed_documents += 1
                    logger.warning(f'문서 처리 실패 ({document.get("id")}): {str(e)}')
                    continue
            
            # 남은 배치 처리
            if batch_buffer:
                batch_count += 1
                self._process_batch(
                    batch_count=batch_count,
                    batch_nodes=batch_nodes,
                    batch_documents=batch_documents,
                    batch_buffer=batch_buffer
                )
                processed_chunks += len(batch_buffer)
            
            logger.info(f'[2/4] ✓ 총 {processed_docs}개 문서, {processed_chunks}개 청크 처리 완료')
            
            # 3단계: 그래프 구축/업데이트 (graph_path는 이미 위에서 정의됨)
            if is_delta_sync and os.path.exists(graph_path):
                # ✅ [Delta Sync] 기존 그래프 로드 후 수정된 노드만 업데이트
                logger.info('[3/4] Delta Sync - 기존 그래프 업데이트 중...')
                try:
                    # 기존 그래프 로드
                    self.processor.load_graph(graph_path)
                    existing_node_count = self.processor.graph.number_of_nodes()
                    
                    # ✅ 새 build_graph: 기존 엣지 정리 + 새 엣지 연결
                    self.processor.build_graph(
                        documents=graph_metadata_list,
                        nodes=all_nodes,
                        clear_existing_edges=True  # Delta Sync 시 Stale Edge 제거
                    )
                    
                    new_node_count = self.processor.graph.number_of_nodes()
                    logger.info(f'[3/4] ✓ 그래프 업데이트 완료 (기존: {existing_node_count} → 현재: {new_node_count})')
                    
                except Exception as e:
                    logger.warning(f'기존 그래프 로드 실패, 새로 구축: {str(e)}')
                    self.processor.build_graph(
                        documents=graph_metadata_list,
                        nodes=all_nodes,
                        clear_existing_edges=False
                    )
            else:
                # ✅ [Full Sync] 새 그래프 구축
                logger.info('[3/4] Full Sync - 새 그래프 구축 중...')
                try:
                    self.processor.build_graph(
                        documents=graph_metadata_list,
                        nodes=all_nodes,
                        clear_existing_edges=False  # Full Sync는 새로 구축
                    )
                except Exception as e:
                    logger.warning(f'그래프 구축 중 오류: {str(e)}')
            
            edge_count = self.processor.graph.number_of_edges()
            node_count = self.processor.graph.number_of_nodes()
            logger.info(f'[3/4] ✓ 그래프 완료 (노드: {node_count}, 엣지: {edge_count}개)')
            
            # 4단계: 그래프 영속화
            if self.processor.save_graph(graph_path):
                logger.info(f'[4/4] ✓ 그래프 파일 저장: {graph_path}')
            
            # 완료
            self._save_progress(progress_file, processed_docs, total_documents, 'completed')
            
            logger.info('=' * 80)
            logger.info('✓ 파이프라인 실행 완료')
            logger.info('=' * 80)
            
            return {
                'status': 'success',
                'sync_mode': 'delta' if is_delta_sync else 'full',
                'documents_count': total_documents,
                'processed_count': processed_docs,
                'updated_pages': len(updated_page_ids) if is_delta_sync else 0,
                'failed_count': failed_documents,
                'chunks_count': processed_chunks,
                'nodes_count': node_count,
                'edges_count': edge_count,
                'graph_path': graph_path
            }
            
        except Exception as e:
            logger.error(f'파이프라인 실행 실패: {str(e)}', exc_info=True)
            
            # ✅ [개선] 실패해도 기존 그래프 유지
            graph_path = os.getenv('GRAPH_PERSIST_PATH', './data/graph.pkl')
            if os.path.exists(graph_path):
                logger.info('⚠️ 파이프라인 실패 - 기존 그래프 유지')
                try:
                    self.processor.load_graph(graph_path)
                    logger.info(f'✓ 기존 그래프 로드 (노드: {self.processor.graph.number_of_nodes()}, 엣지: {self.processor.graph.number_of_edges()})')
                except Exception as load_err:
                    logger.warning(f'기존 그래프 로드 실패: {str(load_err)}')
            
            self._save_progress(progress_file, 0, 0, 'failed')
            return {
                'status': 'failed',
                'message': str(e),
                'nodes_count': self.processor.graph.number_of_nodes() if self.processor.graph else 0,
                'edges_count': self.processor.graph.number_of_edges() if self.processor.graph else 0
            }
    
    def _process_batch(
        self,
        batch_count: int,
        batch_nodes: List[GraphNode],
        batch_documents: List[Dict[str, Any]],
        batch_buffer: List[Dict[str, Any]]
    ) -> None:
        """
        ✅ [신규] 배치 단위 처리: 임베딩 생성 → ChromaDB 저장
        
        Args:
            batch_count: 현재 배치 번호
            batch_nodes: 노드 리스트 (임베딩 대상)
            batch_documents: 원본 문서 리스트 (content 포함)
            batch_buffer: ChromaDB에 저장할 청크 리스트
        """
        try:
            logger.info(f'[배치 #{batch_count}] 임베딩 생성 중... ({len(batch_nodes)}개 노드)')
            
            # 1. 배치 임베딩 생성 (노드 기준)
            embedding_dict = self.processor.generate_embeddings(batch_nodes, batch_documents)
            
            # 2. 임베딩을 청크에 매핑
            if embedding_dict:
                for chroma_doc in batch_buffer:
                    doc_id = chroma_doc['metadata'].get('document_id')
                    if doc_id in embedding_dict:
                        chroma_doc['embedding'] = embedding_dict[doc_id]
            
            logger.info(f'[배치 #{batch_count}] ChromaDB 저장 중... ({len(batch_buffer)}개 청크)')
            
            # 3. ChromaDB에 일괄 저장
            self.vector_store.add_documents(batch_buffer, batch_size=len(batch_buffer))
            
            logger.info(f'[배치 #{batch_count}] ✓ 저장 완료')
            
        except Exception as e:
            logger.error(f'[배치 #{batch_count}] 처리 실패: {str(e)}', exc_info=True)
    
    def _fetch_notion_documents_generator(
        self,
        pages: List[Dict[str, Any]]
    ):
        """
        ✅ [리팩토링] 제너레이터 방식으로 Notion 문서 스트리밍 추출
        
        Rate Limit 방어를 위해 각 요청 사이 sleep 추가
        
        Args:
            pages: 페이지 메타데이터 리스트
        
        Yields:
            document: 완전한 document 객체
        """
        import time
        
        logger.info(f'📄 {len(pages)}개 페이지 콘텐츠 추출 시작 (제너레이터 모드)')
        
        successful_count = 0
        failed_count = 0
        
        for idx, page in enumerate(pages):
            try:
                page_id: str = page.get('id') or page.get('page_id')
                title: str = page.get('title', 'Untitled')
                
                # ✅ 진행 상황 로그 (매 페이지마다)
                logger.info(f'📥 [{idx+1}/{len(pages)}] 페이지 추출 중: {title[:30]}...')
                
                # ✅ Rate Limit 방어 (각 요청 사이 150ms 대기)
                time.sleep(0.15)
                
                # 페이지 내용 추출 (Notion API 호출)
                content: str = self.notion_connector.fetch_page_content(page_id)
                
                if not content:
                    logger.debug(f'빈 페이지 건너뜀: {title}')
                    continue
                
                # Notion URL
                notion_url: str = f"https://www.notion.so/{page_id.replace('-', '')}"
                
                # parent_id (계층 구조)
                parent_id: Optional[str] = page.get('parent_id')
                if not parent_id:
                    parent = page.get('parent', {})
                    parent_id = parent.get('page_id') or parent.get('database_id')
                
                # properties ("작업" 관계 추출용)
                properties: Dict[str, Any] = page.get('properties', {})
                
                document: Dict[str, Any] = {
                    'id': page_id,
                    'title': title,
                    'content': content,
                    'url': notion_url,
                    'created_at': page.get('created_time'),
                    'updated_at': page.get('last_edited_time'),
                    'properties': properties,
                    'parent_id': parent_id
                }
                
                successful_count += 1
                
                # ✅ 매 페이지 yield 시 로그 출력
                logger.info(f'✅ [{idx+1}/{len(pages)}] 페이지 추출 완료: {title[:30]} ({len(content)}자)')
                
                yield document
                
            except Exception as e:
                failed_count += 1
                logger.warning(f'❌ [{idx+1}/{len(pages)}] 페이지 처리 실패 ({page.get("id")}): {str(e)}')
                continue
        
        logger.info(f'🎉 페이지 콘텐츠 추출 완료: {successful_count}개 성공, {failed_count}개 실패')
    
    def _save_progress(
        self,
        progress_file: str,
        processed: int,
        total: int,
        status: str
    ) -> None:
        """
        진행률을 JSON 파일에 저장
        
        Args:
            progress_file: 진행률 저장 파일 경로
            processed: 처리된 문서 수
            total: 전체 문서 수
            status: 상태 ('processing', 'completed', 'failed')
        """
        try:
            progress_data = {
                'timestamp': datetime.now().isoformat(),
                'status': status,
                'processed': processed,
                'total': total,
                'percentage': int((processed / total * 100) if total > 0 else 0)
            }
            
            os.makedirs(os.path.dirname(progress_file), exist_ok=True)
            with open(progress_file, 'w') as f:
                json.dump(progress_data, f, indent=2)
            
            logger.debug(f'진행률 저장: {processed}/{total} ({progress_data["percentage"]}%)')
            
        except Exception as e:
            logger.warning(f'진행률 저장 실패: {str(e)}')
    
    def _sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        ✅ [신규] ChromaDB 저장을 위한 메타데이터 정제
        
        ChromaDB는 None 값을 허용하지 않으므로 빈 문자열로 변환합니다.
        
        Args:
            metadata: 원본 메타데이터
            
        Returns:
            정제된 메타데이터
        """
        sanitized = {}
        for key, value in metadata.items():
            if value is None:
                sanitized[key] = ''
            elif isinstance(value, (list, dict)):
                sanitized[key] = str(value)
            elif isinstance(value, bool):
                sanitized[key] = str(value).lower()
            else:
                sanitized[key] = value
        return sanitized
    
    def _build_page_map(self, pages: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """
        ✅ [신규] 페이지 ID -> 정보 매핑 생성 (부모 경로 추적용)
        
        Args:
            pages: 페이지 메타데이터 리스트
            
        Returns:
            {page_id: {'title': ..., 'parent_id': ...}} 딕셔너리
        """
        page_map = {}
        for page in pages:
            page_id = page.get('id') or page.get('page_id')
            if page_id:
                # parent_id 추출
                parent_id = page.get('parent_id')
                if not parent_id:
                    parent = page.get('parent', {})
                    parent_id = parent.get('page_id') or parent.get('database_id')
                
                page_map[page_id] = {
                    'title': page.get('title', 'Untitled'),
                    'parent_id': parent_id
                }
        
        logger.debug(f'페이지 맵 생성 완료: {len(page_map)}개')
        return page_map
    
    def _get_breadcrumb_path(
        self,
        page_id: str,
        page_map: Dict[str, Dict[str, Any]],
        max_depth: int = 5
    ) -> str:
        """
        ✅ [신규] 부모 경로를 루트까지 추적하여 경로 문자열 생성
        
        예: "학사팀 매뉴얼 > 2024년 > 근로장학생 업무"
        
        Args:
            page_id: 현재 페이지 ID
            page_map: 페이지 ID -> 정보 매핑
            max_depth: 최대 탐색 깊이 (무한 루프 방지)
            
        Returns:
            부모 경로 문자열 (루트 > 부모 > ... > 현재)
        """
        path_parts = []
        current_id = page_id
        visited = set()
        
        for _ in range(max_depth):
            if not current_id or current_id in visited:
                break
            
            visited.add(current_id)
            page_info = page_map.get(current_id)
            
            if not page_info:
                break
            
            path_parts.append(page_info['title'])
            current_id = page_info.get('parent_id')
        
        # 역순으로 정렬 (루트 > 부모 > ... > 현재)
        path_parts.reverse()
        
        breadcrumb = " > ".join(path_parts)
        logger.debug(f'부모 경로 생성: {breadcrumb}')
        return breadcrumb
    
    # 확장성을 위한 메서드
    
    def add_custom_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: str = 'custom',
        weight: float = 1.0
    ) -> None:
        """
        커스텀 엣지 추가
        
        Args:
            source_id: 출발 노드 ID
            target_id: 도착 노드 ID
            edge_type: 엣지 타입
            weight: 가중치
        """
        if source_id in self.processor.graph and target_id in self.processor.graph:
            self.processor.graph.add_edge(
                source_id,
                target_id,
                edge_type=edge_type,
                weight=weight
            )
            logger.info(f'커스텀 엣지 추가: {source_id} → {target_id}')
        else:
            logger.warning(f'엣지 추가 실패: 노드 없음 ({source_id}, {target_id})')
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """
        그래프 통계 조회
        
        Returns:
            {
                'nodes_count': 노드 개수,
                'edges_count': 엣지 개수,
                'avg_degree': 평균 차수,
                ...
            }
        """
        return {
            'nodes_count': self.processor.graph.number_of_nodes(),
            'edges_count': self.processor.graph.number_of_edges(),
            'avg_degree': 2 * self.processor.graph.number_of_edges() / max(1, self.processor.graph.number_of_nodes())
        }
    
    def export_graph(self, filepath: str) -> None:
        """
        그래프를 파일로 내보내기
        
        Args:
            filepath: 내보낼 파일 경로 (.graphml 형식)
        """
        import networkx as nx
        
        try:
            nx.write_graphml(self.processor.graph, filepath)
            logger.info(f'그래프 내보내기 완료: {filepath}')
        except Exception as e:
            logger.error(f'그래프 내보내기 실패: {str(e)}')
