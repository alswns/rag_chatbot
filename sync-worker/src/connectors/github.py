"""
GitHubConnector - GitHub 저장소 분석 및 코드 학습

PyGithub 또는 requests를 사용하여 GitHub API로 저장소를 검색하고
git clone으로 전체 코드를 다운로드하여 분석합니다.
"""

import logging
import os
import tempfile
import shutil
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from urllib.parse import urlparse

import requests
from git import Repo, GitCommandError
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language

logger = logging.getLogger(__name__)

# 지원하는 파일 확장자와 언어 매핑
SUPPORTED_EXTENSIONS = {
    '.py': Language.PYTHON,
    '.js': Language.JS,
    '.java': Language.JAVA,
    '.cpp': Language.CPP,
    '.cc': Language.CPP,
    '.cs': Language.CSHARP,
    '.ts': Language.TS,
}


class GitHubConnector:
    """GitHub 저장소 분석 및 코드 학습 클래스"""
    
    def __init__(self, token: str):
        """
        Args:
            token: GitHub Personal Access Token
        """
        self.token = token
        self.api_url = 'https://api.github.com'
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.temp_dir = tempfile.gettempdir()
        
        # 언어별 텍스트 분할기 초기화
        self.splitters = {
            language: RecursiveCharacterTextSplitter.from_language(
                language=language,
                chunk_size=1500,
                chunk_overlap=200
            )
            for language in SUPPORTED_EXTENSIONS.values()
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
            logger.info(f'GitHub API 연결 성공: {user_info.get("login")}')
            return True
        except Exception as e:
            logger.error(f'GitHub API 연결 실패: {str(e)}')
            return False
    
    def get_user_repositories(self, exclude_private: bool = False, exclude_forks: bool = False) -> List[Dict]:
        """
        현재 사용자의 모든 저장소 조회
        
        Args:
            exclude_private: Private 저장소 제외 여부
            exclude_forks: Fork 저장소 제외 여부
        
        Returns:
            저장소 정보 리스트
        """
        logger.info('GitHub 사용자 저장소 조회 시작...')
        
        all_repos = []
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
                
                for repo in repos:
                    if exclude_private and repo['private']:
                        logger.debug(f'Private 저장소 제외: {repo["full_name"]}')
                        continue
                    if exclude_forks and repo['fork']:
                        logger.debug(f'Fork 저장소 제외: {repo["full_name"]}')
                        continue
                    
                    all_repos.append(repo)
                    logger.debug(f'저장소 추가: {repo["full_name"]}')
                
                if len(repos) < 100:
                    break
                
                page += 1
                time.sleep(0.3)  # Rate limiting
            
            logger.info(f'총 {len(all_repos)}개 사용자 저장소 조회 완료')
            return all_repos
            
        except Exception as e:
            logger.error(f'사용자 저장소 조회 실패: {str(e)}')
            return []
    
    def get_organization_repositories(self, org_list: List[str], exclude_private: bool = False, exclude_forks: bool = False) -> List[Dict]:
        """
        Organization의 모든 저장소 조회
        
        Args:
            org_list: Organization 이름 리스트
            exclude_private: Private 저장소 제외 여부
            exclude_forks: Fork 저장소 제외 여부
        
        Returns:
            저장소 정보 리스트
        """
        all_repos = []
        
        for org in org_list:
            try:
                logger.info(f'Organization "{org}" 저장소 조회 중...')
                repos = self._fetch_org_repos(org, exclude_private, exclude_forks)
                all_repos.extend(repos)
                logger.info(f'Organization "{org}": {len(repos)}개 저장소 조회 완료')
            except Exception as e:
                logger.error(f'Organization "{org}" 저장소 조회 실패: {str(e)}')
                continue
        
        logger.info(f'총 {len(all_repos)}개 Organization 저장소 조회 완료')
        return all_repos
    
    def _fetch_org_repos(self, org: str, exclude_private: bool = False, exclude_forks: bool = False) -> List[Dict]:
        """특정 Organization의 저장소 조회"""
        all_repos = []
        page = 1
        
        try:
            while True:
                url = f'{self.api_url}/orgs/{org}/repos'
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
                
                for repo in repos:
                    if exclude_private and repo['private']:
                        continue
                    if exclude_forks and repo['fork']:
                        continue
                    all_repos.append(repo)
                
                if len(repos) < 100:
                    break
                
                page += 1
                time.sleep(0.3)
            
            return all_repos
            
        except Exception as e:
            logger.error(f'Organization "{org}" 저장소 조회 실패: {str(e)}')
            return []
    
    def process_repository(self, repo_url: str, repo_name: str = None) -> List[Dict[str, Any]]:
        """
        GitHub 저장소를 clone하고 전체 구조를 분석하여 학습 가능한 청크 생성
        
        Args:
            repo_url: 저장소 URL
            repo_name: 저장소 이름 (메타데이터용)
        
        Returns:
            처리된 문서 청크 리스트
        """
        logger.info(f'GitHub 저장소 분석 시작: {repo_url}')
        
        local_repo_path = None
        documents = []
        
        try:
            # 저장소 clone
            local_repo_path = self._clone_repository(repo_url, repo_name)
            
            if not local_repo_path:
                logger.error(f'저장소 clone 실패: {repo_url}')
                return []
            
            # 저장소 구조 분석
            documents = self._analyze_repository(
                local_repo_path,
                repo_url,
                repo_name or urlparse(repo_url).path.split('/')[-1].replace('.git', '')
            )
            
            logger.info(f'{len(documents)}개 청크 생성 완료: {repo_url}')
            return documents
            
        except Exception as e:
            logger.error(f'저장소 분석 중 오류 ({repo_url}): {str(e)}')
            return []
        
        finally:
            # 임시 디렉토리 정리
            if local_repo_path and os.path.exists(local_repo_path):
                try:
                    shutil.rmtree(local_repo_path)
                    logger.debug(f'임시 디렉토리 삭제: {local_repo_path}')
                except Exception as e:
                    logger.warning(f'임시 디렉토리 삭제 실패: {str(e)}')
    
    def _clone_repository(self, repo_url: str, repo_name: str = None) -> Optional[str]:
        """
        GitHub 저장소를 임시 디렉토리에 clone
        
        Args:
            repo_url: 저장소 URL
            repo_name: 저장소 이름
        
        Returns:
            Clone된 디렉토리 경로
        """
        clone_dir = os.path.join(
            self.temp_dir,
            f'github_{hash(repo_url) % 1000000}'
        )
        
        try:
            if os.path.exists(clone_dir):
                shutil.rmtree(clone_dir)
            
            # GitHub URL에 토큰 추가
            auth_url = self._add_auth_to_url(repo_url)
            
            logger.info(f'저장소 clone 시작: {repo_url}')
            Repo.clone_from(auth_url, clone_dir, depth=1)
            
            logger.info(f'저장소 clone 완료: {clone_dir}')
            return clone_dir
            
        except GitCommandError as e:
            logger.error(f'Git clone 실패: {str(e)}')
            return None
        except Exception as e:
            logger.error(f'Clone 중 예상치 못한 오류: {str(e)}')
            return None
    
    def _add_auth_to_url(self, repo_url: str) -> str:
        """URL에 인증 정보 추가"""
        if 'https://' in repo_url:
            # https://github.com/user/repo.git → https://token@github.com/user/repo.git
            return repo_url.replace(
                'https://',
                f'https://{self.token}@'
            )
        return repo_url
    
    def _analyze_repository(
        self,
        local_repo_path: str,
        repo_url: str,
        repo_name: str
    ) -> List[Dict[str, Any]]:
        """
        저장소 디렉토리를 순회하며 지원되는 파일을 찾아 청크 생성
        
        Args:
            local_repo_path: 로컬 저장소 경로
            repo_url: 저장소 URL
            repo_name: 저장소 이름
        
        Returns:
            생성된 청크 리스트
        """
        documents = []
        file_count = 0
        
        logger.info(f'디렉토리 순회 시작: {local_repo_path}')
        
        for root, dirs, files in os.walk(local_repo_path):
            # .git 등 무시할 디렉토리 제외
            dirs[:] = [d for d in dirs if not d.startswith('.')]
            
            for file in files:
                file_path = os.path.join(root, file)
                file_ext = Path(file).suffix.lower()
                
                # 지원되는 확장자인지 확인
                if file_ext not in SUPPORTED_EXTENSIONS:
                    continue
                
                try:
                    # 상대 경로 계산
                    relative_path = os.path.relpath(file_path, local_repo_path)
                    
                    # 파일 내용 읽기
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    if not content.strip():
                        logger.debug(f'빈 파일 무시: {relative_path}')
                        continue
                    
                    # 파일별 청크 생성
                    file_chunks = self._create_file_chunks(
                        content,
                        relative_path,
                        file_ext,
                        repo_url,
                        repo_name
                    )
                    
                    documents.extend(file_chunks)
                    file_count += 1
                    
                    logger.debug(f'파일 처리 완료: {relative_path} ({len(file_chunks)}개 청크)')
                    
                except Exception as e:
                    logger.warning(f'파일 처리 실패 ({file_path}): {str(e)}')
                    continue
        
        logger.info(f'디렉토리 순회 완료: {file_count}개 파일 처리, {len(documents)}개 청크 생성')
        return documents
    
    def _create_file_chunks(
        self,
        content: str,
        file_path: str,
        file_ext: str,
        repo_url: str,
        repo_name: str
    ) -> List[Dict[str, Any]]:
        """
        파일 내용을 언어별 문법에 맞춰 청크로 분할
        
        Args:
            content: 파일 내용
            file_path: 상대 경로
            file_ext: 파일 확장자
            repo_url: 저장소 URL
            repo_name: 저장소 이름
        
        Returns:
            청크 리스트
        """
        language = SUPPORTED_EXTENSIONS.get(file_ext)
        if not language:
            return []
        
        chunks = []
        
        try:
            # 언어별 분할기 선택
            splitter = self.splitters.get(language)
            if not splitter:
                logger.warning(f'지원되지 않는 언어: {language}')
                return []
            
            # 파일 내용 분할
            text_chunks = splitter.split_text(content)
            
            # 각 청크에 파일 정보 접두사 추가
            for i, chunk_text in enumerate(text_chunks):
                # 청크 맨 앞에 파일 정보 삽입
                file_prefix = f'File: {file_path}\n\n'
                full_content = file_prefix + chunk_text
                
                # 청크 ID 생성
                chunk_id = f'{repo_url}#{file_path}#{i}'
                
                chunk_dict = {
                    'id': chunk_id,
                    'content': full_content,
                    'metadata': {
                        'source': repo_url,
                        'repo_name': repo_name,
                        'file_path': file_path,
                        'language': file_ext[1:],
                        'chunk_index': str(i),
                        'chunk_count': len(text_chunks),
                        'platform': 'github'
                    }
                }
                chunks.append(chunk_dict)
            
            logger.debug(f'{file_path}: {len(text_chunks)}개 청크 생성')
            return chunks
            
        except Exception as e:
            logger.error(f'청크 생성 실패 ({file_path}): {str(e)}')
            return []
