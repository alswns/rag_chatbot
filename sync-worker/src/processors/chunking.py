"""
ChunkingProcessor - 텍스트 분할 및 메타데이터 처리

LangChain의 MarkdownHeaderTextSplitter와 RecursiveCharacterTextSplitter를 사용하여
2단계 텍스트 분할을 수행합니다.
"""

import logging
from typing import List, Dict, Any

from langchain.text_splitter import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema import Document

logger = logging.getLogger(__name__)


class ChunkingProcessor:
    """텍스트 분할 및 청크 처리 클래스"""
    
    def __init__(
        self,
        markdown_chunk_size: int = 1000,
        markdown_chunk_overlap: int = 200,
        recursive_chunk_size: int = 500,
        recursive_chunk_overlap: int = 100
    ):
        """
        Args:
            markdown_chunk_size: Markdown 헤더 분할 시 청크 크기
            markdown_chunk_overlap: Markdown 헤더 분할 시 오버랩
            recursive_chunk_size: 재귀적 분할 시 청크 크기
            recursive_chunk_overlap: 재귀적 분할 시 오버랩
        """
        # 1단계: Markdown 헤더 기준 분할
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                ("##", "Header 2"),
                ("###", "Header 3"),
            ],
            return_each_line=False,
        )
        
        # 2단계: 재귀적 문자 기준 분할
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=recursive_chunk_size,
            chunk_overlap=recursive_chunk_overlap,
            separators=["\n\n", "\n", "。", "，", " ", ""]
        )
        
        logger.info('ChunkingProcessor 초기화 완료')
    
    def process_notion_page(
        self,
        page_id: str,
        title: str,
        content: str,
        last_edited_time: str
    ) -> List[Dict[str, Any]]:
        """
        Notion 페이지를 처리하여 청크 생성
        
        Args:
            page_id: Notion Page ID
            title: 페이지 제목
            content: 페이지 내용 (Markdown 형식)
            last_edited_time: 마지막 편집 시간
        
        Returns:
            메타데이터를 포함한 청크 리스트
        """
        logger.info(f'Notion 페이지 처리 시작: {page_id} - {title}')
        
        if not content:
            logger.warning(f'빈 컨텐츠: {page_id}')
            return []
        
        chunks = []
        
        try:
            # 1단계: Markdown 헤더 기준 분할
            header_chunks = self.markdown_splitter.split_text(content)
            
            logger.debug(f'Markdown 분할 완료: {len(header_chunks)}개 청크')
            
            # 2단계: 재귀적 문자 기준 분할
            for i, header_chunk in enumerate(header_chunks):
                # Document 객체로 변환
                if isinstance(header_chunk, Document):
                    chunk_content = header_chunk.page_content
                    chunk_metadata = header_chunk.metadata
                else:
                    chunk_content = header_chunk
                    chunk_metadata = {}
                
                # 재귀적 분할
                sub_chunks = self.recursive_splitter.split_text(chunk_content)
                
                # 각 청크에 메타데이터 추가
                for j, sub_chunk in enumerate(sub_chunks):
                    chunk_dict = {
                        'content': sub_chunk,
                        'metadata': {
                            'page_id': page_id,
                            'title': title,
                            'last_edited_time': last_edited_time,
                            'source': 'notion',
                            'chunk_index': f'{i}-{j}',
                            'header_1': chunk_metadata.get('Header 1', ''),
                            'header_2': chunk_metadata.get('Header 2', ''),
                            'header_3': chunk_metadata.get('Header 3', ''),
                        }
                    }
                    chunks.append(chunk_dict)
                    chunk_dict = {
                        'content': sub_chunk,
                        'metadata': {
                            'page_id': page_id,
                            'title': title,
                            'last_edited_time': last_edited_time,
                            'source': 'notion',
                            'chunk_index': f'{i}-{j}',
                            # Markdown 헤더 정보 포함
                            'header_1': chunk_metadata.get('Header 1', ''),
                            'header_2': chunk_metadata.get('Header 2', ''),
                            'header_3': chunk_metadata.get('Header 3', ''),
                        }
                    }
                    chunks.append(chunk_dict)
            
            logger.info(f'총 {len(chunks)}개 청크 생성: {page_id}')
            return chunks
            
        except Exception as e:
            logger.error(f'페이지 처리 실패 ({page_id}): {str(e)}')
            return []
    
    def process_gitea_file(
        self,
        repo_id: str,
        repo_name: str,
        file_path: str,
        content: str,
        last_commit_time: str,
        language: str = 'unknown'
    ) -> List[Dict[str, Any]]:
        """
        Gitea 저장소의 파일을 처리하여 청크 생성
        
        Args:
            repo_id: 저장소 ID
            repo_name: 저장소 이름
            file_path: 파일 경로
            content: 파일 내용
            last_commit_time: 마지막 커밋 시간
            language: 프로그래밍 언어
        
        Returns:
            메타데이터를 포함한 청크 리스트
        """
        logger.info(f'Gitea 파일 처리 시작: {repo_name}/{file_path}')
        
        if not content:
            logger.warning(f'빈 파일 내용: {file_path}')
            return []
        
        chunks = []
        
        try:
            # 코드 파일은 주로 함수/클래스 단위로 분할
            # 재귀적 분할만 적용 (코드는 헤더 기반 분할이 덜 효과적)
            sub_chunks = self.recursive_splitter.split_text(content)
            
            # 각 청크에 메타데이터 추가
            for i, sub_chunk in enumerate(sub_chunks):
                # 코드 라인 수 계산
                line_count = len(sub_chunk.split('\n'))
                
                chunk_dict = {
                    'content': sub_chunk,
                    'metadata': {
                        'repo_id': repo_id,
                        'repo_name': repo_name,
                        'file_path': file_path,
                        'last_commit_time': last_commit_time,
                        'source': 'gitea',
                        'language': language,
                        'chunk_index': str(i),
                        'line_count': line_count,
                    }
                }
                chunks.append(chunk_dict)
            
            logger.info(f'총 {len(chunks)}개 청크 생성: {file_path}')
            return chunks
            
        except Exception as e:
            logger.error(f'파일 처리 실패 ({file_path}): {str(e)}')
            return []
    
    @staticmethod
    def validate_chunk(chunk: Dict[str, Any]) -> bool:
        """
        청크 유효성 검사
        
        Args:
            chunk: 검증할 청크
        
        Returns:
            유효하면 True, 무효하면 False
        """
        if not isinstance(chunk, dict):
            return False
        
        if 'content' not in chunk or not chunk['content'].strip():
            return False
        
        if 'metadata' not in chunk:
            return False
        
        metadata = chunk['metadata']
        required_fields = ['source']
        
        # source별 필수 필드 검사
        source = metadata.get('source')
        if source == 'notion':
            required_fields.extend(['page_id', 'title', 'last_edited_time'])
        elif source == 'gitea':
            required_fields.extend(['repo_id', 'file_path', 'last_commit_time'])
        
        return all(field in metadata for field in required_fields)
