"""
GitHubConnector - GitHub 저장소 분석 및 코드 학습 (Unified Implementation)

git_connector.py의 GitHubConnector를 기반으로 하며,
backward compatibility를 위해 기존 인터페이스도 지원합니다.

주요 기능:
- Semantic Chunking (함수/클래스 단위)
- .gitignore 자동 필터링
- GitHub API 통합
"""

import logging
from typing import List, Dict, Any, Optional

from .git_connector import GitHubConnector as _BaseGitHubConnector

logger = logging.getLogger(__name__)


class GitHubConnector(_BaseGitHubConnector):
    """
    GitHub 저장소 분석 클래스 (Unified Implementation)
    
    git_connector.GitHubConnector를 상속받아,
    동일한 인터페이스로 작동하면서 새로운 기능을 활용합니다.
    
    기능:
    - Semantic Chunking: 함수/클래스 단위로 코드 분할
    - .gitignore 지원: 자동 노이즈 필터링
    - 메타데이터 확장: unit_type, unit_name 포함
    """
    
    def __init__(self, token: str, chunk_size: int = 1500, chunk_overlap: int = 200):
        """
        Args:
            token: GitHub Personal Access Token
            chunk_size: 청크 최대 크기 (기본값: 1500)
            chunk_overlap: 청크 간 오버랩 (기본값: 200)
        """
        super().__init__(
            token=token,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        logger.info('✓ GitHubConnector 초기화 완료 (Unified + Semantic Chunking)')
    
    # 하위 호환성을 위한 추가 메서드들
    
    def get_user_repositories(
        self,
        exclude_private: bool = False,
        exclude_forks: bool = False
    ) -> List[Dict[str, Any]]:
        """
        현재 사용자의 모든 저장소 조회 (하위 호환성)
        
        Args:
            exclude_private: Private 저장소 제외 여부
            exclude_forks: Fork 저장소 제외 여부
        
        Returns:
            저장소 정보 리스트
        """
        all_repos = self.get_repositories()
        
        # 필터링
        filtered_repos = []
        for repo in all_repos:
            if exclude_private and repo.get('private'):
                logger.debug(f'Private 제외: {repo.get("full_name", repo.get("name"))}')
                continue
            if exclude_forks and repo.get('fork'):
                logger.debug(f'Fork 제외: {repo.get("full_name", repo.get("name"))}')
                continue
            filtered_repos.append(repo)
        
        logger.info(f'✓ {len(filtered_repos)}개 사용자 저장소 조회')
        return filtered_repos
    
    def get_organization_repositories(
        self,
        org_list: List[str],
        exclude_private: bool = False,
        exclude_forks: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Organization의 모든 저장소 조회 (하위 호환성)
        
        Args:
            org_list: Organization 이름 리스트
            exclude_private: Private 저장소 제외 여부
            exclude_forks: Fork 저장소 제외 여부
        
        Returns:
            저장소 정보 리스트
        """
        # 현재 버전: 사용자 저장소로 통합
        logger.warning('Organization 별도 조회는 구현되지 않았습니다. 사용자 저장소로 반환합니다.')
        return self.get_user_repositories(exclude_private, exclude_forks)


__all__ = ['GitHubConnector']
