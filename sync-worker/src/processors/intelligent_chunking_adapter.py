"""
Intelligent Chunking Integration Module

기존 ChunkingProcessor와 intelligent_chunking을 통합하는 어댑터 레이어.
Notion 파이프라인에서 자동으로 메타데이터 기반 청킹을 수행합니다.
"""

import logging
from typing import List, Dict, Any, Optional
from processors.intelligent_chunking import (
    IntelligentChunkingEngine,
    MetadataExtractor,
    KeywordExtractor,
    EnrichedChunk
)

logger = logging.getLogger(__name__)


class IntelligentChunkingAdapter:
    """기존 파이프라인과 intelligent chunking을 연결하는 어댑터"""
    
    def __init__(
        self,
        chunk_size: int = 900,
        chunk_overlap: int = 200,
        enable_metadata_enrichment: bool = True,
        enable_keyword_extraction: bool = True
    ):
        """
        Args:
            chunk_size: 청크 크기 (기본 900자)
            chunk_overlap: 청크 오버랩 (기본 200자)
            enable_metadata_enrichment: 메타데이터 보강 활성화
            enable_keyword_extraction: 키워드 추출 활성화
        """
        self.chunking_engine = IntelligentChunkingEngine(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        
        self.enable_metadata_enrichment = enable_metadata_enrichment
        self.enable_keyword_extraction = enable_keyword_extraction
        
        logger.info(f'IntelligentChunkingAdapter 초기화 (metadata={enable_metadata_enrichment}, keywords={enable_keyword_extraction})')
    
    def process_notion_page_with_enrichment(
        self,
        page_id: str,
        title: str,
        content: str,
        last_edited_time: str,
        breadcrumb_path: str = "",
        parent_id: Optional[str] = None,
        url: str = ""
    ) -> List[Dict[str, Any]]:
        """
        Notion 페이지를 지능형 청킹으로 처리
        
        Args:
            page_id: 페이지 ID
            title: 페이지 제목
            content: 페이지 내용 (마크다운)
            last_edited_time: 마지막 수정 시간
            breadcrumb_path: 계층 경로 (예: "학사팀 > 근로장학생")
            parent_id: 부모 페이지 ID
            url: 페이지 URL
        
        Returns:
            청크 리스트 (Chroma DB 호환 형식)
        """
        logger.info(f'🔬 Intelligent Chunking 시작: "{title}" (breadcrumb: {breadcrumb_path})')
        
        if not content:
            logger.warning(f'  ⚠️ 빈 콘텐츠: {title}')
            return []
        
        try:
            # 메타데이터 보강을 포함한 청킹
            enriched_chunks = self.chunking_engine.create_enriched_chunks(
                text=content,
                title=title,
                document_id=page_id,
                breadcrumb_path=breadcrumb_path,
                metadata={
                    'parent_id': parent_id,
                    'url': url,
                    'updated_at': last_edited_time
                }
            )
            
            # Chroma DB 호환 형식으로 변환
            chroma_compatible_chunks = []
            
            for chunk_idx, enriched in enumerate(enriched_chunks):
                # Metadata Header + 원본 콘텐츠 결합
                # (Embedding 시 메타데이터도 임베딩되도록)
                full_content = enriched.formatted_content
                
                # 메타데이터 준비
                chunk_metadata = {
                    'title': title,
                    'source_url': url,
                    'document_id': page_id,
                    'chunk_index': chunk_idx,
                    'chunk_count': len(enriched_chunks),
                    'created_at': '',  # Notion에서 전달받아야 함
                    'updated_at': last_edited_time,
                    'source': 'notion',
                    'parent_id': parent_id,
                    
                    # ✅ 지능형 메타데이터
                    'breadcrumb_path': breadcrumb_path,
                    'document_type': enriched.metadata.get('document_type', 'general'),
                    'subject': enriched.metadata.get('subject', ''),
                    'keywords': ','.join(enriched.keywords),  # 문자열로 변환
                    'temporal_info': self._serialize_temporal(enriched.metadata.get('temporal', {})),
                    'entities': self._serialize_entities(enriched.metadata.get('entities', {}))
                }
                
                chroma_doc = {
                    'id': f"{page_id}_chunk_{chunk_idx}",
                    'content': full_content,  # Metadata Header + 원본 콘텐츠
                    'metadata': self._sanitize_metadata(chunk_metadata)
                }
                
                chroma_compatible_chunks.append(chroma_doc)
            
            logger.info(f'✅ Intelligent Chunking 완료: {len(chroma_compatible_chunks)}개 청크')
            logger.info(f'   📊 메타데이터:')
            if chroma_compatible_chunks:
                sample = chroma_compatible_chunks[0]['metadata']
                logger.info(f'      - Type: {sample.get("document_type")}')
                logger.info(f'      - Keywords: {sample.get("keywords", "")[:50]}')
                logger.info(f'      - Path: {sample.get("breadcrumb_path")}')
            
            return chroma_compatible_chunks
            
        except Exception as e:
            logger.error(f'❌ Intelligent Chunking 실패: {str(e)}', exc_info=True)
            # Fallback: 빈 청크 반환 (기존 처리 로직이 처리)
            return []
    
    @staticmethod
    def _serialize_temporal(temporal_info: Dict[str, Any]) -> str:
        """Temporal 정보를 문자열로 변환 (메타데이터 저장용)"""
        if not temporal_info:
            return ""
        
        items = []
        for key, values in temporal_info.items():
            if isinstance(values, list):
                items.append(f"{key}:{','.join(values)}")
            else:
                items.append(f"{key}:{values}")
        
        return ";".join(items)[:200]  # 길이 제한
    
    @staticmethod
    def _serialize_entities(entities: Dict[str, Any]) -> str:
        """엔티티를 문자열로 변환"""
        if not entities:
            return ""
        
        all_entities = []
        for entity_type, values in entities.items():
            if isinstance(values, list):
                all_entities.extend(values[:2])  # 타입당 최대 2개
            else:
                all_entities.append(str(values))
        
        return ",".join(all_entities)[:150]  # 길이 제한
    
    @staticmethod
    def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
        """메타데이터 정규화 (Chroma DB 호환)"""
        sanitized = {}
        
        for key, value in metadata.items():
            if value is None:
                continue
            
            # 문자열로 변환 (Chroma는 대부분 문자열 저장)
            if isinstance(value, (list, dict)):
                value = str(value)
            
            # 길이 제한 (Chroma 제약)
            if isinstance(value, str):
                value = value[:2000]
            
            sanitized[key] = value
        
        return sanitized


class EnrichedChunkingPipeline:
    """지능형 청킹을 포함한 완전한 파이프라인"""
    
    def __init__(
        self,
        use_intelligent_chunking: bool = True,
        chunk_size: int = 900,
        chunk_overlap: int = 200
    ):
        """
        Args:
            use_intelligent_chunking: Intelligent Chunking 활성화
            chunk_size: 청크 크기
            chunk_overlap: 오버랩
        """
        self.use_intelligent_chunking = use_intelligent_chunking
        
        if use_intelligent_chunking:
            self.adapter = IntelligentChunkingAdapter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            logger.info('✅ Intelligent Chunking Pipeline 활성화')
        else:
            self.adapter = None
            logger.info('⚠️ 기본 Chunking 사용 (Intelligent Chunking 비활성화)')
    
    def chunk_document(
        self,
        document: Dict[str, Any],
        breadcrumb_path: str = ""
    ) -> List[Dict[str, Any]]:
        """
        문서를 청킹 (지능형 또는 기본)
        
        Args:
            document: 문서 데이터
                - id: 문서 ID
                - title: 제목
                - content: 콘텐츠
                - updated_at: 수정 시간
                - url: URL (선택)
                - parent_id: 부모 ID (선택)
            breadcrumb_path: 계층 경로
        
        Returns:
            청크 리스트
        """
        if self.use_intelligent_chunking and self.adapter:
            return self.adapter.process_notion_page_with_enrichment(
                page_id=document.get('id'),
                title=document.get('title', 'Untitled'),
                content=document.get('content', ''),
                last_edited_time=document.get('updated_at', ''),
                breadcrumb_path=breadcrumb_path,
                parent_id=document.get('parent_id'),
                url=document.get('url', '')
            )
        else:
            # Fallback: 기본 청킹 (기존 로직)
            logger.warning('기본 청킹으로 폴백')
            return []
    
    def get_enrichment_stats(self) -> Dict[str, Any]:
        """Intelligent Chunking 통계 반환"""
        if not self.use_intelligent_chunking:
            return {'enabled': False}
        
        return {
            'enabled': True,
            'features': {
                'metadata_extraction': 'Temporal, Entities, Document Type',
                'keyword_extraction': 'TF-IDF based',
                'contextual_overlap': '200 characters',
                'metadata_header': 'Automatic injection'
            }
        }
