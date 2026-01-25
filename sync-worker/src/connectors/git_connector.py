"""
Unified Git Connector - GitHub & Gitea 통합 코드 수집기

필수 설치:
  pip install GitPython requests tree-sitter gitignore-parser langchain-text-splitters
  python -m tree_sitter build  (선택사항 - AST 파싱 사용 시)

특징:
  - BaseGitConnector 추상 클래스로 공통 로직 통합
  - AST 기반 Semantic Chunking (함수/클래스 단위)
  - .gitignore 자동 필터링
  - 메타데이터에 함수/클래스 정보 포함
"""

import logging
import os
import tempfile
import shutil
import time
import re
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from urllib.parse import urlparse
from collections import defaultdict

import requests
from git import Repo, GitCommandError
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

try:
    from gitignore_parser import parse_gitignore
    HAS_GITIGNORE_PARSER = True
except ImportError:
    HAS_GITIGNORE_PARSER = False

logger = logging.getLogger(__name__)

# 지원하는 파일 확장자와 언어 매핑
SUPPORTED_EXTENSIONS: Dict[str, Language] = {
    '.py': Language.PYTHON,
    '.js': Language.JS,
    '.java': Language.JAVA,
    '.cpp': Language.CPP,
    '.cc': Language.CPP,
    '.cs': Language.CSHARP,
    '.ts': Language.TS,
    '.go': Language.GO,
    '.rb': Language.RUBY,
}


class BaseGitConnector(ABC):
    """Git 저장소 분석의 공통 로직을 담당하는 추상 클래스"""
    
    def __init__(self, chunk_size: int = 1500, chunk_overlap: int = 200):
        """
        Args:
            chunk_size: 청크 최대 크기
            chunk_overlap: 청크 간 오버랩
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.temp_dir = tempfile.gettempdir()
        
        # 언어별 텍스트 분할기 초기화
        self.splitters: Dict[Language, RecursiveCharacterTextSplitter] = {}
        for ext, language in SUPPORTED_EXTENSIONS.items():
            try:
                self.splitters[language] = RecursiveCharacterTextSplitter.from_language(
                    language=language,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
            except Exception as e:
                logger.warning(f'분할기 로드 실패 ({ext}): {str(e)}')
        
        logger.info(f'BaseGitConnector 초기화: {len(self.splitters)}개 언어 지원')
    
    @abstractmethod
    def get_repositories(self) -> List[Dict[str, Any]]:
        """저장소 목록 조회 (구현체에서 정의)"""
        pass
    
    @abstractmethod
    def _add_auth_to_url(self, repo_url: str) -> str:
        """URL에 인증 정보 추가 (구현체에서 정의)"""
        pass
    
    def process_repository(
        self,
        repo_url: str,
        repo_name: str,
        platform: str
    ) -> List[Dict[str, Any]]:
        """
        Git 저장소를 clone하고 코드를 분석하여 청크 생성
        
        Args:
            repo_url: 저장소 URL
            repo_name: 저장소 이름
            platform: 플랫폼 ('github' 또는 'gitea')
        
        Returns:
            처리된 청크 리스트
        """
        logger.info(f'[{platform.upper()}] 저장소 분석 시작: {repo_url}')
        
        local_repo_path: Optional[str] = None
        documents: List[Dict[str, Any]] = []
        
        try:
            local_repo_path = self._clone_repository(repo_url)
            
            if not local_repo_path:
                logger.error(f'저장소 clone 실패: {repo_url}')
                return []
            
            # .gitignore 파일 파싱
            gitignore_matcher = self._load_gitignore(local_repo_path)
            
            # 저장소 분석
            documents = self._analyze_repository(
                local_repo_path=local_repo_path,
                repo_url=repo_url,
                repo_name=repo_name,
                platform=platform,
                gitignore_matcher=gitignore_matcher
            )
            
            logger.info(f'✓ 저장소 분석 완료: {len(documents)}개 청크 생성')
            return documents
            
        except Exception as e:
            logger.error(f'저장소 분석 오류 ({repo_url}): {str(e)}', exc_info=True)
            return []
        
        finally:
            if local_repo_path and os.path.exists(local_repo_path):
                try:
                    shutil.rmtree(local_repo_path)
                    logger.debug(f'임시 디렉토리 삭제: {local_repo_path}')
                except Exception as e:
                    logger.warning(f'임시 디렉토리 삭제 실패: {str(e)}')
    
    def _clone_repository(self, repo_url: str) -> Optional[str]:
        """
        Git 저장소를 임시 디렉토리에 clone
        
        Args:
            repo_url: 저장소 URL
        
        Returns:
            Clone된 디렉토리 경로 또는 None
        """
        clone_dir = os.path.join(
            self.temp_dir,
            f'git_{hash(repo_url) % 1000000}'
        )
        
        try:
            if os.path.exists(clone_dir):
                shutil.rmtree(clone_dir)
            
            auth_url = self._add_auth_to_url(repo_url)
            
            logger.debug(f'Clone 시작: {repo_url}')
            Repo.clone_from(auth_url, clone_dir, depth=1)
            
            logger.debug(f'Clone 완료: {clone_dir}')
            return clone_dir
            
        except GitCommandError as e:
            logger.error(f'Git clone 실패: {str(e)}')
            return None
        except Exception as e:
            logger.error(f'Clone 예외: {str(e)}')
            return None
    
    def _load_gitignore(self, repo_path: str) -> Optional[Any]:
        """
        .gitignore 파일을 파싱하여 매처 반환
        
        Args:
            repo_path: 저장소 경로
        
        Returns:
            gitignore 매처 또는 None
        """
        if not HAS_GITIGNORE_PARSER:
            logger.debug('gitignore_parser 미설치 - .gitignore 필터링 비활성화')
            return None
        
        gitignore_path = os.path.join(repo_path, '.gitignore')
        
        if not os.path.exists(gitignore_path):
            logger.debug('.gitignore 파일 없음')
            return None
        
        try:
            with open(gitignore_path, 'r', encoding='utf-8') as f:
                gitignore_matcher = parse_gitignore(f.read(), repo_path)
            logger.debug(f'.gitignore 파싱 완료')
            return gitignore_matcher
        except Exception as e:
            logger.warning(f'.gitignore 파싱 실패: {str(e)}')
            return None
    
    def _should_ignore(self, file_path: str, gitignore_matcher: Optional[Any]) -> bool:
        """
        파일이 .gitignore에 의해 무시되어야 하는지 확인
        
        Args:
            file_path: 파일 경로
            gitignore_matcher: gitignore 매처
        
        Returns:
            무시해야 하면 True
        """
        if not gitignore_matcher:
            return False
        
        try:
            return gitignore_matcher(file_path)
        except Exception:
            return False
    
    def _analyze_repository(
        self,
        local_repo_path: str,
        repo_url: str,
        repo_name: str,
        platform: str,
        gitignore_matcher: Optional[Any]
    ) -> List[Dict[str, Any]]:
        """
        저장소 디렉토리를 순회하여 코드 청크 생성
        
        Args:
            local_repo_path: 로컬 저장소 경로
            repo_url: 저장소 URL
            repo_name: 저장소 이름
            platform: 플랫폼
            gitignore_matcher: .gitignore 매처
        
        Returns:
            생성된 청크 리스트
        """
        documents: List[Dict[str, Any]] = []
        file_count = 0
        
        logger.debug(f'디렉토리 순회 시작: {local_repo_path}')
        
        for root, dirs, files in os.walk(local_repo_path):
            # .git, node_modules 등 무시할 디렉토리 제외
            dirs[:] = [
                d for d in dirs
                if not d.startswith('.') and d not in ['node_modules', '__pycache__', 'venv', 'env']
            ]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                
                # 지원되지 않는 확장자 스킵
                if file_ext not in SUPPORTED_EXTENSIONS:
                    continue
                
                # .gitignore 필터링
                relative_path = os.path.relpath(file_path, local_repo_path)
                if self._should_ignore(relative_path, gitignore_matcher):
                    logger.debug(f'gitignore로 제외: {relative_path}')
                    continue
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    if not content.strip():
                        continue
                    
                    file_chunks = self._create_semantic_chunks(
                        content=content,
                        file_path=relative_path,
                        file_ext=file_ext,
                        repo_url=repo_url,
                        repo_name=repo_name,
                        platform=platform
                    )
                    
                    documents.extend(file_chunks)
                    file_count += 1
                    
                    logger.debug(f'파일 처리: {relative_path} → {len(file_chunks)}개 청크')
                    
                except Exception as e:
                    logger.warning(f'파일 처리 오류 ({relative_path}): {str(e)}')
                    continue
        
        logger.info(f'디렉토리 순회 완료: {file_count}개 파일, {len(documents)}개 청크')
        return documents
    
    def _create_semantic_chunks(
        self,
        content: str,
        file_path: str,
        file_ext: str,
        repo_url: str,
        repo_name: str,
        platform: str
    ) -> List[Dict[str, Any]]:
        """
        코드를 의미론적 단위(함수/클래스)로 청크 생성
        
        Args:
            content: 파일 내용
            file_path: 상대 경로
            file_ext: 파일 확장자
            repo_url: 저장소 URL
            repo_name: 저장소 이름
            platform: 플랫폼
        
        Returns:
            청크 리스트
        """
        language = SUPPORTED_EXTENSIONS.get(file_ext)
        if not language:
            return []
        
        chunks: List[Dict[str, Any]] = []
        
        try:
            splitter = self.splitters.get(language)
            if not splitter:
                logger.debug(f'언어별 분할기 없음: {file_ext}')
                return []
            
            # 1단계: 의미론적 청크 추출 (함수/클래스 단위)
            semantic_chunks = self._extract_semantic_units(content, file_ext)
            
            # 2단계: 청크를 메타데이터와 함께 생성
            for chunk_idx, (unit_type, unit_name, unit_content) in enumerate(semantic_chunks):
                # 청크 분할
                text_chunks = splitter.split_text(unit_content)
                
                for sub_idx, chunk_text in enumerate(text_chunks):
                    # 헤더 정보 추가
                    header = f"# {file_path}\n"
                    if unit_name:
                        header += f"## {unit_type}: {unit_name}\n\n"
                    
                    full_content = header + chunk_text
                    
                    chunk_id = f'{repo_url}#{file_path}#{chunk_idx}#{sub_idx}'
                    
                    chunk_dict = {
                        'id': chunk_id,
                        'content': full_content,
                        'metadata': {
                            'source': repo_url,
                            'repo_name': repo_name,
                            'file_path': file_path,
                            'language': file_ext[1:],
                            'unit_type': unit_type,
                            'unit_name': unit_name if unit_name else '',
                            'chunk_index': f'{chunk_idx}-{sub_idx}',
                            'platform': platform
                        }
                    }
                    chunks.append(chunk_dict)
            
            logger.debug(f'{file_path}: {len(chunks)}개 청크 생성')
            return chunks
            
        except Exception as e:
            logger.error(f'청크 생성 오류 ({file_path}): {str(e)}')
            return []
    
    def _extract_semantic_units(
        self,
        content: str,
        file_ext: str
    ) -> List[Tuple[str, str, str]]:
        """
        코드에서 함수/클래스 단위로 의미론적 단위 추출
        
        Args:
            content: 파일 내용
            file_ext: 파일 확장자
        
        Returns:
            [(unit_type, unit_name, unit_content), ...] 리스트
        """
        units: List[Tuple[str, str, str]] = []
        
        try:
            if file_ext == '.py':
                units = self._extract_python_units(content)
            elif file_ext in ['.js', '.ts']:
                units = self._extract_js_units(content)
            elif file_ext == '.java':
                units = self._extract_java_units(content)
            else:
                # 기본: 전체 파일을 하나의 단위로 취급
                units = [('file', '', content)]
            
            return units if units else [('file', '', content)]
            
        except Exception as e:
            logger.debug(f'의미론적 추출 실패: {str(e)}')
            return [('file', '', content)]
    
    def _extract_python_units(self, content: str) -> List[Tuple[str, str, str]]:
        """Python 코드에서 함수/클래스 추출"""
        units: List[Tuple[str, str, str]] = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 클래스 정의
            if re.match(r'^\s*class\s+(\w+)', line):
                match = re.match(r'^\s*class\s+(\w+)', line)
                class_name = match.group(1) if match else 'Unknown'
                class_lines = [line]
                indent_level = len(line) - len(line.lstrip())
                i += 1
                
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.strip() and not next_line.startswith(' ' * (indent_level + 1)) and next_line[0] != '\n':
                        break
                    class_lines.append(next_line)
                    i += 1
                
                units.append(('class', class_name, '\n'.join(class_lines)))
                continue
            
            # 함수 정의
            if re.match(r'^\s*def\s+(\w+)', line):
                match = re.match(r'^\s*def\s+(\w+)', line)
                func_name = match.group(1) if match else 'Unknown'
                func_lines = [line]
                indent_level = len(line) - len(line.lstrip())
                i += 1
                
                while i < len(lines):
                    next_line = lines[i]
                    if next_line.strip() and not next_line.startswith(' ' * (indent_level + 1)) and next_line[0] != '\n':
                        break
                    func_lines.append(next_line)
                    i += 1
                
                units.append(('function', func_name, '\n'.join(func_lines)))
                continue
            
            i += 1
        
        return units
    
    def _extract_js_units(self, content: str) -> List[Tuple[str, str, str]]:
        """JavaScript/TypeScript 코드에서 함수/클래스 추출"""
        units: List[Tuple[str, str, str]] = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 클래스 정의
            if re.search(r'\bclass\s+(\w+)', line):
                match = re.search(r'\bclass\s+(\w+)', line)
                class_name = match.group(1) if match else 'Unknown'
                class_lines = [line]
                brace_count = line.count('{') - line.count('}')
                i += 1
                
                while i < len(lines) and brace_count > 0:
                    next_line = lines[i]
                    class_lines.append(next_line)
                    brace_count += next_line.count('{') - next_line.count('}')
                    i += 1
                
                units.append(('class', class_name, '\n'.join(class_lines)))
                continue
            
            # 함수 정의
            if re.search(r'function\s+(\w+)|(\w+)\s*\(.*\)\s*{|const\s+(\w+)\s*=\s*\(', line):
                func_name = 'Unknown'
                match = re.search(r'function\s+(\w+)', line)
                if match:
                    func_name = match.group(1)
                else:
                    match = re.search(r'(\w+)\s*\(', line)
                    if match:
                        func_name = match.group(1)
                
                func_lines = [line]
                brace_count = line.count('{') - line.count('}')
                i += 1
                
                while i < len(lines) and brace_count > 0:
                    next_line = lines[i]
                    func_lines.append(next_line)
                    brace_count += next_line.count('{') - next_line.count('}')
                    i += 1
                
                units.append(('function', func_name, '\n'.join(func_lines)))
                continue
            
            i += 1
        
        return units
    
    def _extract_java_units(self, content: str) -> List[Tuple[str, str, str]]:
        """Java 코드에서 클래스/메서드 추출"""
        units: List[Tuple[str, str, str]] = []
        lines = content.split('\n')
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 클래스 정의
            if re.search(r'\bclass\s+(\w+)', line):
                match = re.search(r'\bclass\s+(\w+)', line)
                class_name = match.group(1) if match else 'Unknown'
                class_lines = [line]
                brace_count = line.count('{') - line.count('}')
                i += 1
                
                while i < len(lines) and brace_count > 0:
                    next_line = lines[i]
                    class_lines.append(next_line)
                    brace_count += next_line.count('{') - next_line.count('}')
                    i += 1
                
                units.append(('class', class_name, '\n'.join(class_lines)))
                continue
            
            # 메서드 정의
            if re.search(r'(public|private|protected)?\s+\w+\s+(\w+)\s*\(', line):
                match = re.search(r'(public|private|protected)?\s+\w+\s+(\w+)\s*\(', line)
                method_name = match.group(2) if match else 'Unknown'
                method_lines = [line]
                brace_count = line.count('{') - line.count('}')
                i += 1
                
                while i < len(lines) and brace_count > 0:
                    next_line = lines[i]
                    method_lines.append(next_line)
                    brace_count += next_line.count('{') - next_line.count('}')
                    i += 1
                
                units.append(('method', method_name, '\n'.join(method_lines)))
                continue
            
            i += 1
        
        return units


class GitHubConnector(BaseGitConnector):
    """GitHub 저장소 분석 클래스"""
    
    def __init__(self, token: str, chunk_size: int = 1500, chunk_overlap: int = 200):
        """
        Args:
            token: GitHub Personal Access Token
            chunk_size: 청크 최대 크기
            chunk_overlap: 청크 간 오버랩
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.token = token
        self.api_url = 'https://api.github.com'
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        logger.info('GitHubConnector 초기화 완료')
    
    def test_connection(self) -> bool:
        """GitHub API 연결 테스트"""
        try:
            response = requests.get(
                f'{self.api_url}/user',
                headers=self.headers,
                timeout=5
            )
            response.raise_for_status()
            user_info = response.json()
            logger.info(f'✓ GitHub API 연결: {user_info.get("login")}')
            return True
        except Exception as e:
            logger.error(f'GitHub API 연결 실패: {str(e)}')
            return False
    
    def get_repositories(self) -> List[Dict[str, Any]]:
        """사용자의 모든 저장소 조회"""
        all_repos: List[Dict[str, Any]] = []
        page = 1
        
        try:
            while True:
                url = f'{self.api_url}/user/repos'
                params = {
                    'page': page,
                    'per_page': 100,
                    'sort': 'updated',
                    'direction': 'desc'
                }
                
                response = requests.get(url, headers=self.headers, params=params, timeout=10)
                response.raise_for_status()
                repos = response.json()
                
                if not repos:
                    break
                
                all_repos.extend(repos)
                
                if len(repos) < 100:
                    break
                
                page += 1
                time.sleep(0.3)
            
            logger.info(f'✓ GitHub 저장소 {len(all_repos)}개 조회')
            return all_repos
            
        except Exception as e:
            logger.error(f'저장소 조회 실패: {str(e)}')
            return []
    
    def _add_auth_to_url(self, repo_url: str) -> str:
        """GitHub URL에 토큰 인증 추가"""
        if 'https://' in repo_url:
            return repo_url.replace('https://', f'https://{self.token}@')
        return repo_url


class GiteaConnector(BaseGitConnector):
    """Gitea 저장소 분석 클래스"""
    
    def __init__(self, gitea_url: str, token: str, chunk_size: int = 1500, chunk_overlap: int = 200):
        """
        Args:
            gitea_url: Gitea 서버 URL (예: http://gitea:3000)
            token: Gitea Personal Access Token
            chunk_size: 청크 최대 크기
            chunk_overlap: 청크 간 오버랩
        """
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        # ✅ gitea_url이 None이거나 빈 문자열인 경우 방어 처리
        if gitea_url:
            self.gitea_url = gitea_url.rstrip('/')
        else:
            self.gitea_url = None
        self.token = token
        logger.info(f'GiteaConnector 초기화: {gitea_url if gitea_url else "(비활성화)"}')
    
    def test_connection(self) -> bool:
        """Gitea API 연결 테스트"""
        try:
            url = f'{self.gitea_url}/api/v1/user'
            headers = {'Authorization': f'token {self.token}'}
            response = requests.get(url, headers=headers, timeout=5)
            response.raise_for_status()
            user_info = response.json()
            logger.info(f'✓ Gitea API 연결: {user_info.get("username")}')
            return True
        except Exception as e:
            logger.error(f'Gitea API 연결 실패: {str(e)}')
            return False
    
    def get_repositories(self) -> List[Dict[str, Any]]:
        """사용자의 모든 저장소 조회"""
        all_repos: List[Dict[str, Any]] = []
        page = 1
        
        try:
            while True:
                url = f'{self.gitea_url}/api/v1/user/repos'
                headers = {'Authorization': f'token {self.token}'}
                params = {'page': page, 'limit': 50}
                
                response = requests.get(url, headers=headers, params=params, timeout=10)
                response.raise_for_status()
                repos = response.json()
                
                if not repos:
                    break
                
                all_repos.extend(repos)
                
                if len(repos) < 50:
                    break
                
                page += 1
                time.sleep(0.3)
            
            logger.info(f'✓ Gitea 저장소 {len(all_repos)}개 조회')
            return all_repos
            
        except Exception as e:
            logger.error(f'저장소 조회 실패: {str(e)}')
            return []
    
    def _add_auth_to_url(self, repo_url: str) -> str:
        """Gitea URL에 토큰 인증 추가"""
        parsed = urlparse(repo_url)
        
        if parsed.scheme and parsed.netloc:
            netloc = f'oauth2:{self.token}@{parsed.netloc}'
            return f'{parsed.scheme}://{netloc}{parsed.path}'
        
        return repo_url
