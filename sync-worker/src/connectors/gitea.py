"""
GiteaConnector - Gitea 저장소 구조 분석 및 코드 학습 (Unified Implementation)

git_connector.py의 GiteaConnector를 기반으로 하며,
backward compatibility를 위해 기존 인터페이스도 지원합니다.

주요 기능:
- Semantic Chunking (함수/클래스 단위)
- .gitignore 자동 필터링
- Gitea API 통합
"""

import logging
from typing import List, Dict, Any, Optional

from .git_connector import GiteaConnector as _BaseGiteaConnector

logger = logging.getLogger(__name__)


class GiteaConnector(_BaseGiteaConnector):
    """
    Gitea 저장소 분석 클래스 (Unified Implementation)
    
    git_connector.GiteaConnector를 상속받아,
    동일한 인터페이스로 작동하면서 새로운 기능을 활용합니다.
    
    기능:
    - Semantic Chunking: 함수/클래스 단위로 코드 분할
    - .gitignore 지원: 자동 노이즈 필터링
    - 메타데이터 확장: unit_type, unit_name 포함
    - Gitea API: 저장소 목록 조회
    """
    
    def __init__(self, gitea_url: str, token: str, chunk_size: int = 1500, chunk_overlap: int = 200):
        """
        Args:
            gitea_url: Gitea 서버 URL (예: http://gitea:3000)
            token: Gitea Personal Access Token
            chunk_size: 청크 최대 크기 (기본값: 1500)
            chunk_overlap: 청크 간 오버랩 (기본값: 200)
        """
        super().__init__(
            gitea_url=gitea_url,
            token=token,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.info(f'✓ GiteaConnector 초기화: {gitea_url} (Unified + Semantic Chunking)')
    
    def process_repository_by_path(self, repo_path: str) -> List[Dict[str, Any]]:
        """
        Gitea 저장소를 경로로 지정하여 분석 (하위 호환성)
        
        Args:
            repo_path: 저장소 경로 (예: 'organization/project' 또는 URL)
        
        Returns:
            처리된 청크 리스트
        """
        # repo_path가 URL이 아니면 Gitea URL 앞에 붙임
        if repo_path.startswith('http'):
            repo_url = repo_path
            repo_name = repo_path.split('/')[-1].replace('.git', '')
        else:
            repo_url = f'{self.gitea_url}/{repo_path}.git'
            repo_name = repo_path.split('/')[-1]
        
        return self.process_repository(
            repo_url=repo_url,
            repo_name=repo_name,
            platform='gitea'
        )


__all__ = ['GiteaConnector']
