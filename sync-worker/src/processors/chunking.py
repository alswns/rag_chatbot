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
    """
    ✅ [최적화됨] 구조 기반 하이브리드 청킹 프로세서
    
    핵심 전략:
    1. H3 제거 → H1, H2만 사용 → 문맥 파편화 방지
    2. Context Injection → 본문 앞에 "Context: {Title} > {Header Path}" 주입
    3. Separator 최적화 → Notion 구분선, 코드블록 보호
    4. chunk_size=2000, overlap=200 (BGE-M3 최적)
    """
    
    def __init__(
        self,
        markdown_chunk_size: int = 2000,  # 사용 안 함 (호환성 유지)
        recursive_chunk_size: int = 2000,
        recursive_chunk_overlap: int = 200
    ):
        """
        Args:
            markdown_chunk_size: 호환성 유지용 (실제 미사용)
            recursive_chunk_size: 청크 최대 크기 (기본 2000)
            recursive_chunk_overlap: 청크 간 오버랩 (기본 200)
        """
        # 1단계: Markdown 헤더 기준 (H1, H2만 - H3 제거로 문맥 유지)
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "Header 1"),
                # ("##", "Header 2"),
                # ❌ H3 제거 - 문맥 파편화 방지
            ],
            strip_headers=False,  # 헤더 텍스트를 본문에 남겨둠
        )
        
        # 2단계: 재귀적 문자 기준 분할
        # ✅ Separator 최적화: 노션 구분선, 코드블록 보호
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=recursive_chunk_size,
            chunk_overlap=recursive_chunk_overlap,
            separators=[
                "\n\n",      # 빈 줄 (단락 구분)
                "\n---",     # Notion 구분선
                "\n## ",     # H2 헤더
                "```",       # 코드블록 (보호)
                "\n",        # 일반 줄바꿈
                ". ",        # 문장 끝
                "? ",
                "! ",
                " ",
                ""
            ]
        )
        
        logger.info(f'ChunkingProcessor 초기화 완료 (chunk_size={recursive_chunk_size}, overlap={recursive_chunk_overlap})')
    
    def process_notion_page(
        self,
        page_id: str,
        title: str,
        content: str,
        last_edited_time: str,
        breadcrumb_path: str = ''  # ✅ [추가] 부모 경로 (루트부터)
    ) -> List[Dict[str, Any]]:
        """
        Notion 페이지를 청크로 분할
        
        Args:
            page_id: 페이지 ID
            title: 페이지 제목
            content: 페이지 내용 (Markdown)
            last_edited_time: 마지막 수정 시간
            breadcrumb_path: 부모 경로 (예: "학사팀 매뉴얼 > 2024년 > 근로장학생 업무")
            
        Returns:
            청크 리스트
        """
        logger.info(f'Notion 페이지 처리 시작: {title}')
        
        if not content:
            return []
        
        chunks = []
        
        try:
            # 1단계: Markdown 헤더로 크게 덩어리 잡기
            header_chunks = self.markdown_splitter.split_text(content)
            
            for i, header_chunk in enumerate(header_chunks):
                chunk_content = header_chunk.page_content
                chunk_metadata = header_chunk.metadata
                
                # 2단계: 내용이 너무 길면 자르기 (Recursive)
                sub_chunks = self.recursive_splitter.split_text(chunk_content)
                
                # ✅ [개선] 문서 내 헤더 경로 (H1, H2)
                header_path_parts = []
                if chunk_metadata.get('Header 1'): header_path_parts.append(chunk_metadata['Header 1'])
                if chunk_metadata.get('Header 2'): header_path_parts.append(chunk_metadata['Header 2'])
                header_path = " > ".join(header_path_parts)
                
                for j, sub_chunk in enumerate(sub_chunks):
                    # ✅ [핵심] Context Injection - 부모 경로(루트) + 헤더 경로
                    # 예: "Context: 학사팀 매뉴얼 > 2024년 > 근로장학생 업무 > 신규채용 절차"
                    context_parts = []
                    
                    # 1. 부모 페이지 경로 (루트부터, pipeline에서 전달)
                    if breadcrumb_path:
                        context_parts.append(breadcrumb_path)
                    else:
                        context_parts.append(title)
                    
                    # 2. 문서 내 헤더 경로 (H1, H2)
                    if header_path:
                        context_parts.append(header_path)
                    
                    full_context = " > ".join(context_parts)
                    enriched_content = f"Context: {full_context}\n\n{sub_chunk}"

                    chunk_dict = {
                        'content': enriched_content,  # 임베딩 품질 극대화
                        'metadata': {
                            'page_id': page_id,
                            'title': title,
                            'breadcrumb_path': breadcrumb_path,  # ✅ 메타데이터에도 저장
                            'last_edited_time': last_edited_time,
                            'source': 'notion',
                            'chunk_index': f'{i}-{j}',
                            # 메타데이터는 필터링용으로 유지
                            'header_1': chunk_metadata.get('Header 1', ''),
                            'header_2': chunk_metadata.get('Header 2', ''),
                        }
                    }
                    chunks.append(chunk_dict)
            
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
