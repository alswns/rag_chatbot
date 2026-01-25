"""
NotionConnector - Notion API 연결 및 Delta Sync 구현

Notion API를 통해 데이터베이스에서 변경된 페이지만 조회하고,
ChromaDB에서 중복을 방지하며 저장합니다.
"""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
import time

from notion_client import Client
from notion_client.errors import APIResponseError

logger = logging.getLogger(__name__)


class NotionConnector:
    """Notion API를 통한 데이터 조회 및 동기화 클래스"""
    
    def __init__(self, token: str, database_id: Optional[str] = None):
        """
        Args:
            token: Notion Integration Token
            database_id: 동기화할 Notion Database ID (None이면 모든 데이터베이스)
        """
        self.token = token
        self.database_id = database_id
        self.client = Client(auth=token)
        
        if database_id:
            logger.info(f'Notion 클라이언트 초기화: Database ID = {database_id}')
        else:
            logger.info('Notion 클라이언트 초기화: 모든 데이터베이스에서 데이터 수집')
    
    def get_all_databases(self) -> List[Dict[str, Any]]:
        """
        사용자가 접근 가능한 모든 데이터베이스 조회
        
        Returns:
            데이터베이스 정보 리스트
        """
        try:
            databases = []
            has_more = True
            start_cursor = None
            
            while has_more:
                response = self.client.search(
                    filter={"value": "database", "property": "object"},
                    start_cursor=start_cursor,
                    page_size=100
                )
                
                databases.extend(response.get('results', []))
                has_more = response.get('has_more', False)
                start_cursor = response.get('next_cursor')
            
            logger.info(f'총 {len(databases)}개의 데이터베이스를 찾았습니다')
            for db in databases:
                # 데이터베이스 제목 추출
                db_title = 'Untitled'
                if 'title' in db:
                    if isinstance(db['title'], list) and len(db['title']) > 0:
                        db_title = db['title'][0].get('text', {}).get('content', 'Untitled')
                    elif isinstance(db['title'], str):
                        db_title = db['title']
                
                logger.info(f"  - {db_title} (ID: {db['id']})")
            
            return databases
            
        except Exception as e:
            logger.error(f'데이터베이스 조회 실패: {str(e)}')
            logger.warning('API 키가 정상적으로 설정되었는지 확인하세요')
            return []
    
    def test_connection(self) -> bool:
        """Notion API 연결 테스트"""
        try:
            if self.database_id:
                # 특정 데이터베이스가 지정된 경우
                self.client.databases.retrieve(database_id=self.database_id)
                logger.info('Notion API 연결 성공 (특정 데이터베이스)')
            else:
                # DATABASE_ID가 없으면 사용자 정보로 테스트
                logger.info('DATABASE_ID가 비어있음. 모든 데이터베이스에서 동기화합니다.')
                # 간단한 API 호출로 연결 확인
                self.client.users.me()
                logger.info(f'Notion API 연결 성공 (모든 데이터베이스 접근 가능)')
            return True
        except Exception as e:
            logger.error(f'Notion API 연결 실패: {str(e)}')
            logger.warning('NOTION_TOKEN이 올바른지 확인하세요')
            return False
    
    def get_updated_pages(self, last_sync_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        마지막 동기화 이후 변경된 페이지 조회 (Delta Sync)
        
        DATABASE_ID가 지정되면 해당 DB만, 없으면 Search API로 모든 페이지 조회
        
        Args:
            last_sync_time: ISO 8601 형식의 마지막 동기화 시간
                           None이면 모든 페이지 조회
        
        Returns:
            변경된 페이지 정보 리스트
        """
        logger.info(f'변경된 페이지 조회 시작 (last_sync_time={last_sync_time})')
        
        updated_pages = []
        
        try:
            if self.database_id:
                # 특정 데이터베이스에서만 조회
                logger.info(f'데이터베이스 ID [{self.database_id}]에서 페이지 조회...')
                updated_pages.extend(self._get_pages_from_database(
                    self.database_id, 
                    last_sync_time
                ))
            else:
                # Search API를 사용하여 모든 페이지 조회
                logger.info('Search API를 사용하여 워크스페이스의 모든 페이지 조회 중...')
                updated_pages.extend(self._get_all_pages_via_search(last_sync_time))
            
            logger.info(f'총 {len(updated_pages)}개의 변경된 페이지를 조회했습니다')
            return updated_pages
            
        except Exception as e:
            logger.error(f'페이지 조회 중 오류 발생: {str(e)}', exc_info=True)
            return []
    
    def _get_all_pages_via_search(self, last_sync_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search API를 사용하여 워크스페이스의 모든 페이지 조회
        
        Args:
            last_sync_time: 마지막 동기화 시간 (이후 수정된 페이지만)
        
        Returns:
            페이지 정보 리스트
        """
        pages_list = []
        has_more = True
        start_cursor = None
        
        logger.info('📋 Search API로 모든 페이지 검색 중...')
        
        while has_more:
            try:
                # Search API - 모든 'page' 객체 검색
                response = self.client.search(
                    filter={
                        "value": "page",
                        "property": "object"
                    },
                    start_cursor=start_cursor,
                    page_size=100,
                    sort={
                        "direction": "descending",
                        "timestamp": "last_edited_time"
                    }
                )
                
                for page in response.get('results', []):
                    page_id = page['id']
                    last_edited = page.get('last_edited_time')
                    
                    # last_sync_time 이후 수정된 페이지만 필터링
                    if last_sync_time:
                        if last_edited and last_edited > last_sync_time:
                            page_info = {
                                'page_id': page_id,
                                'id': page_id,
                                'title': page.get('title', [{}])[0].get('text', {}).get('content', 'Untitled') if page.get('title') else 'Untitled',
                                'last_edited_time': last_edited,
                                'created_time': page.get('created_time'),
                                'content': None
                            }
                            pages_list.append(page_info)
                            logger.debug(f"  ✓ {page_id} (수정: {last_edited})")
                        else:
                            logger.debug(f"  ✗ {page_id} (변경 없음 - {last_edited})")
                    else:
                        page_info = {
                            'page_id': page_id,
                            'id': page_id,
                            'title': page.get('title', [{}])[0].get('text', {}).get('content', 'Untitled') if page.get('title') else 'Untitled',
                            'last_edited_time': last_edited,
                            'created_time': page.get('created_time'),
                            'content': None
                        }
                        pages_list.append(page_info)
                        logger.debug(f"  ✓ {page_id}")
                
                has_more = response.get('has_more', False)
                start_cursor = response.get('next_cursor')
                
                logger.info(f"  현재까지 {len(pages_list)}개 페이지 수집됨")
                
            except Exception as e:
                logger.error(f'Search API 호출 실패: {str(e)}', exc_info=True)
                break
        
        logger.info(f'✓ 총 {len(pages_list)}개의 페이지를 Search API로 조회했습니다')
        return pages_list

    
    def _get_pages_from_database(self, database_id: str, last_sync_time: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        특정 데이터베이스에서 변경된 페이지 조회
        
        Args:
            database_id: Notion 데이터베이스 ID
            last_sync_time: 마지막 동기화 시간
        
        Returns:
            페이지 정보 리스트
        """
        pages_list = []
        has_more = True
        start_cursor = None
        
        while has_more:
            # Notion Query API로 페이지 조회
            query_dict = {
                'database_id': database_id,
                'page_size': 100,
                'sorts': [
                    {
                        'timestamp': 'last_edited_time',
                        'direction': 'descending'
                    }
                ]
            }
            
            if start_cursor:
                query_dict['start_cursor'] = start_cursor
            
            # 필터: last_edited_time > last_sync_time
            if last_sync_time:
                query_dict['filter'] = {
                    'timestamp': 'last_edited_time',
                    'last_edited_time': {
                        'after': last_sync_time
                    }
                }
            
            response = self.client.databases.query(**query_dict)
            
            # 페이지 정보 추출
            for page in response.get('results', []):
                page_info = {
                    'page_id': page['id'],
                    'title': self._extract_title(page),
                    'last_edited_time': page['last_edited_time'],
                    'created_time': page['created_time'],
                    'database_id': database_id,  # 어느 데이터베이스인지 추적
                    'content': None  # content는 추후 fetch_page_content에서 설정
                }
                pages_list.append(page_info)
            
            has_more = response.get('has_more', False)
            start_cursor = response.get('next_cursor')
            
            # API Rate Limiting 대응
            time.sleep(0.3)
        
        return pages_list
    
    def sync_pages(
        self,
        last_sync_time: Optional[str] = None,
        chunking_processor=None
    ) -> List[Dict[str, Any]]:
        """
        변경된 페이지를 조회하여 처리된 청크 반환 (complete pipeline)
        
        이 메서드는 다음을 수행합니다:
        1. Delta Sync로 변경된 페이지 조회
        2. 페이지 내용을 Markdown으로 변환
        3. ChunkingProcessor로 청킹
        4. 메타데이터 포함 (source, page_id)
        
        Args:
            last_sync_time: ISO 8601 형식의 마지막 동기화 시간
            chunking_processor: ChunkingProcessor 인스턴스 (청킹이 필요한 경우)
        
        Returns:
            처리된 문서 리스트 [{id, content, metadata}, ...]
        """
        logger.info(f'Notion 페이지 동기화 시작 (last_sync_time={last_sync_time})')
        
        # 변경된 페이지 조회
        updated_pages = self.get_updated_pages(last_sync_time)
        
        if not updated_pages:
            logger.info('변경된 페이지 없음')
            return []
        
        documents = []
        
        for page_info in updated_pages:
            try:
                page_id = page_info.get('page_id') or page_info.get('id')
                if not page_id:
                    logger.warning(f'page_id를 찾을 수 없습니다: {page_info.keys()}')
                    continue
                    
                title = page_info.get('title', 'Untitled')
                last_edited_time = page_info.get('last_edited_time')
                
                logger.debug(f'[{page_id}] 페이지 처리 중... (제목: {title})')
                # 페이지 내용 조회
                content = self.fetch_page_content(page_id)
                
                if not content.strip():
                    logger.warning(f'빈 페이지 내용: {page_id}')
                    continue
                
                # 청킹 처리 (ChunkingProcessor가 있는 경우)
                if chunking_processor:
                    chunks = chunking_processor.process_notion_page(
                        page_id=page_id,
                        title=title,
                        content=content,
                        last_edited_time=last_edited_time
                    )
                    
                    if not chunks:
                        logger.warning(f'청킹 실패: {page_id}')
                        continue
                    
                    # 페이지 URL 생성 (메타데이터용)
                    page_url = f'{page_id}'
                    
                    # 문서 리스트 생성
                    for i, chunk in enumerate(chunks):
                        doc = {
                            'id': f'{page_id}#{i}',
                            'content': chunk['content'],
                            'metadata': {
                                **chunk['metadata'],
                                'source': page_url,  # delete_by_source에서 사용
                            }
                        }
                        documents.append(doc)
                    
                    logger.info(f'페이지 처리 완료: {title} ({len(chunks)}개 청크)')
                
                else:
                    # ChunkingProcessor 없이 전체 페이지 반환
                    page_url = f'{page_id}'
                    doc = {
                        'id': page_id,
                        'content': content,
                        'metadata': {
                            'source': page_url,
                            'page_id': page_id,
                            'title': title,
                            'last_edited_time': last_edited_time,
                        }
                    }
                    documents.append(doc)
                    logger.info(f'페이지 조회 완료: {title}')
                
            except Exception as e:
                logger.error(f'페이지 처리 실패 ({page_id}): {str(e)}')
                continue
        
        logger.info(f'총 {len(documents)}개 문서 준비 완료')
        return documents
    
    def sync_pages_streaming(
        self,
        last_sync_time: Optional[str] = None,
        chunking_processor=None
    ):
        """
        변경된 페이지를 스트리밍 방식으로 처리 (페이지별로 즉시 반환)
        
        이 메서드는 다음을 수행합니다:
        1. Delta Sync로 변경된 페이지 조회
        2. 페이지별로 청킹
        3. 각 페이지 완료 시 즉시 청크들 반환 (제너레이터)
        
        Args:
            last_sync_time: ISO 8601 형식의 마지막 동기화 시간
            chunking_processor: ChunkingProcessor 인스턴스
        
        Yields:
            페이지별 청크 리스트 [{id, content, metadata}, ...]
        """
        logger.info(f'Notion 페이지 스트리밍 동기화 시작 (last_sync_time={last_sync_time})')
        
        # 변경된 페이지 조회
        updated_pages = self.get_updated_pages(last_sync_time)
        
        if not updated_pages:
            logger.info('변경된 페이지 없음')
            return
        
        for page_info in updated_pages:
            try:
                page_id = page_info.get('page_id') or page_info.get('id')
                if not page_id:
                    logger.warning(f'page_id를 찾을 수 없습니다: {page_info.keys()}')
                    continue
                
                title = page_info.get('title', 'Untitled')
                last_edited_time = page_info.get('last_edited_time')
                
                logger.debug(f'[{page_id}] 페이지 처리 중... (제목: {title})')
                
                # 페이지 내용 조회
                content = self.fetch_page_content(page_id)
                
                if not content.strip():
                    logger.warning(f'빈 페이지 내용: {page_id}')
                    continue
                
                # 청킹 처리
                if chunking_processor:
                    chunks = chunking_processor.process_notion_page(
                        page_id=page_id,
                        title=title,
                        content=content,
                        last_edited_time=last_edited_time
                    )
                    
                    if not chunks:
                        logger.warning(f'청킹 실패: {page_id}')
                        continue
                    
                    # 페이지 URL 생성 (메타데이터용)
                    page_url = f'{page_id}'
                    
                    # 페이지의 청크들을 문서 리스트로 변환
                    page_documents = []
                    for i, chunk in enumerate(chunks):
                        doc = {
                            'id': f'{page_id}#{i}',
                            'content': chunk['content'],
                            'metadata': {
                                **chunk['metadata'],
                                'source': page_url,
                                'title': title,  # 페이지 타이틀 추가
                            }
                        }
                        page_documents.append(doc)
                    
                    logger.debug(f'페이지 청킹 완료: {title} ({len(page_documents)}개 청크)')
                    
                    # 페이지 단위로 yield (즉시 반환)
                    yield page_documents
                
                else:
                    # ChunkingProcessor 없이 전체 페이지 반환
                    page_url = f'{page_id}'
                    doc = {
                        'id': page_id,
                        'content': content,
                        'metadata': {
                            'source': page_url,
                            'page_id': page_id,
                            'title': title,
                            'last_edited_time': last_edited_time,
                        }
                    }
                    yield [doc]
                
            except Exception as e:
                logger.error(f'페이지 처리 실패 ({page_id}): {str(e)}', exc_info=True)
                continue
    
    def fetch_page_content(self, page_id: str) -> str:
        """
        페이지의 전체 내용 조회
        
        Args:
            page_id: Notion Page ID
            
        Returns:
            페이지 내용 (Markdown 형식)
        """
        try:
            # 페이지 컨텐츠 블록 조회
            blocks = self._get_page_blocks(page_id)
            
            # 블록을 Markdown으로 변환
            markdown_content = self._blocks_to_markdown(blocks)
            
            logger.debug(f'페이지 컨텐츠 조회 완료: {page_id}')
            return markdown_content
            
        except Exception as e:
            logger.error(f'페이지 컨텐츠 조회 실패 ({page_id}): {str(e)}')
            return ''
    
    def _get_page_blocks(self, page_id: str, start_cursor: Optional[str] = None) -> List[Dict]:
        """페이지의 모든 블록 재귀적으로 조회 (부모-자식 순서 유지)"""
        all_blocks = []
        has_more = True
        cursor = start_cursor
        
        try:
            while has_more:
                response = self.client.blocks.children.list(
                    block_id=page_id,
                    page_size=100,
                    start_cursor=cursor
                )
                
                # 부모 블록과 자식 블록을 순차적으로 추가
                for block in response.get('results', []):
                    all_blocks.append(block)
                    # 해당 블록이 자식 블록을 가지면 재귀적으로 조회
                    if block.get('has_children'):
                        child_blocks = self._get_page_blocks(block['id'])
                        all_blocks.extend(child_blocks)
                
                has_more = response.get('has_more', False)
                cursor = response.get('next_cursor')
                time.sleep(0.3)
            
            return all_blocks
            
        except Exception as e:
            logger.error(f'블록 조회 실패 ({page_id}): {str(e)}')
            return []
    
    def _blocks_to_markdown(self, blocks: List[Dict]) -> str:
        """
        ✅ [긴급 데이터 품질 개선] 블록 리스트를 Markdown 형식으로 변환
        
        URL만 추출되는 버그 방지: 순수 텍스트만 추출하고 이미지/파일 URL 제외
        """
        markdown_lines = []
        
        for block in blocks:
            block_type = block.get('type')
            block_data = block.get(block_type, {})
            
            try:
                text = None
                
                if block_type == 'heading_1':
                    text = self._extract_rich_text(block_data.get('rich_text', []))
                    if text.strip():  # ✅ 공백만 있으면 제외
                        markdown_lines.append(f'# {text}\n')
                        logger.debug(f'[Notion] H1: {text[:50]}')
                
                elif block_type == 'heading_2':
                    text = self._extract_rich_text(block_data.get('rich_text', []))
                    if text.strip():
                        markdown_lines.append(f'## {text}\n')
                        logger.debug(f'[Notion] H2: {text[:50]}')
                
                elif block_type == 'heading_3':
                    text = self._extract_rich_text(block_data.get('rich_text', []))
                    if text.strip():
                        markdown_lines.append(f'### {text}\n')
                        logger.debug(f'[Notion] H3: {text[:50]}')
                
                elif block_type == 'paragraph':
                    text = self._extract_rich_text(block_data.get('rich_text', []))
                    if text.strip():
                        markdown_lines.append(f'{text}\n')
                        logger.debug(f'[Notion] 단락: {text[:50]}')
                
                elif block_type == 'bulleted_list_item':
                    text = self._extract_rich_text(block_data.get('rich_text', []))
                    if text.strip():
                        markdown_lines.append(f'- {text}\n')
                        logger.debug(f'[Notion] 불릿: {text[:50]}')
                
                elif block_type == 'numbered_list_item':
                    text = self._extract_rich_text(block_data.get('rich_text', []))
                    if text.strip():
                        markdown_lines.append(f'1. {text}\n')
                        logger.debug(f'[Notion] 번호: {text[:50]}')
                
                elif block_type == 'code':
                    text = self._extract_rich_text(block_data.get('rich_text', []))
                    language = block_data.get('language', 'text')
                    if text.strip():
                        markdown_lines.append(f'```{language}\n{text}\n```\n')
                        logger.debug(f'[Notion] 코드 ({language}): {text[:50]}')
                
                elif block_type == 'quote':
                    text = self._extract_rich_text(block_data.get('rich_text', []))
                    if text.strip():
                        markdown_lines.append(f'> {text}\n')
                        logger.debug(f'[Notion] 인용: {text[:50]}')
                
                elif block_type == 'divider':
                    markdown_lines.append('---\n')
                    logger.debug('[Notion] 구분선')
                
                # ✅ [중요] 이미지/파일 블록은 URL 대신 텍스트만 추출
                elif block_type == 'image':
                    # URL 대신, caption 텍스트만 가져오기
                    caption_text = self._extract_rich_text(block_data.get('caption', []))
                    if caption_text.strip():
                        markdown_lines.append(f'[이미지] {caption_text}\n')
                        logger.debug(f'[Notion] 이미지 캡션: {caption_text[:50]}')
                    else:
                        logger.debug('[Notion] 이미지 캡션 없음 (URL 제외)')
                
                elif block_type == 'file':
                    # 파일도 URL 대신 caption만
                    caption_text = self._extract_rich_text(block_data.get('caption', []))
                    if caption_text.strip():
                        markdown_lines.append(f'[파일] {caption_text}\n')
                        logger.debug(f'[Notion] 파일 캡션: {caption_text[:50]}')
                    else:
                        logger.debug('[Notion] 파일 캡션 없음 (URL 제외)')
                
                else:
                    logger.debug(f'[Notion] 미지원 블록 타입: {block_type}')
                
            except Exception as e:
                logger.warning(f'블록 변환 중 오류 ({block_type}): {str(e)}', exc_info=True)
                continue
        
        final_markdown = ''.join(markdown_lines)
        logger.info(f'✅ Markdown 변환 완료: {len(final_markdown)}자 (원본: {len(blocks)}개 블록)')
        
        return final_markdown
    
    def _extract_rich_text(self, rich_text_list: List[Dict]) -> str:
        """Rich Text 리스트를 일반 텍스트로 변환"""
        return ''.join([item.get('plain_text', '') for item in rich_text_list])
    
    def _extract_image_url(self, image_block: Dict) -> Optional[str]:
        """이미지 블록에서 URL 추출"""
        if 'external' in image_block:
            return image_block['external'].get('url')
        elif 'file' in image_block:
            return image_block['file'].get('url')
        return None
    
    def _extract_title(self, page: Dict) -> str:
        """페이지 객체에서 제목 추출"""
        try:
            properties = page.get('properties', {})
            
            # 일반적으로 'Name' 또는 'Title' 속성에서 제목 추출
            for prop_name, prop_value in properties.items():
                if prop_value.get('type') == 'title':
                    rich_text = prop_value.get('title', [])
                    return self._extract_rich_text(rich_text)
            
            return page.get('id', 'Untitled')
            
        except Exception as e:
            logger.warning(f'제목 추출 실패: {str(e)}')
            return 'Untitled'
