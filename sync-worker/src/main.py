"""
Sync Worker - Notion & Gitea 데이터 동기화 메인 모듈

주기적으로 Notion 데이터베이스와 Gitea 저장소를 동기화하여
ChromaDB 벡터 데이터베이스에 저장합니다.
"""

import os
import logging
from datetime import datetime
import time
from dotenv import load_dotenv

# 로컬 모듈 임포트
from connectors import NotionConnector, GiteaConnector, GitHubConnector
from processors import ChunkingProcessor
from db import VectorStoreManager
from utils import SyncStateManager

# 환경변수 로드
load_dotenv()

# 로깅 설정
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SyncWorker:
    """Notion & Gitea 데이터 동기화를 담당하는 메인 클래스"""
    
    def __init__(self):
        """워커 초기화"""
        logger.info('=' * 60)
        logger.info('Sync Worker 초기화 시작')
        logger.info('=' * 60)
        
        # 환경변수 로드
        self.notion_token = os.getenv('NOTION_TOKEN')
        self.notion_database_id = os.getenv('NOTION_DATABASE_ID') or None  # 빈 문자열을 None으로 변환
        # ✅ GITEA_URL이 None이거나 빈 문자열이면 None으로 통일
        self.gitea_url = os.getenv('GITEA_URL') or None
        self.gitea_token = os.getenv('GITEA_TOKEN') or None
        self.github_token = os.getenv('GITHUB_TOKEN', '').strip()
        self.github_orgs = [org.strip() for org in os.getenv('GITHUB_ORGS', '').split(',') if org.strip()]
        self.github_include_user = os.getenv('GITHUB_INCLUDE_USER_REPOS', 'true').lower() == 'true'
        
        self.target_repos = os.getenv('TARGET_REPOS', '').split(',')
        self.target_repos = [repo.strip() for repo in self.target_repos if repo.strip()]
        
        self.chroma_host = os.getenv('CHROMA_HOST', 'localhost')
        self.chroma_port = int(os.getenv('CHROMA_PORT', '8000'))
        self.sync_interval = int(os.getenv('SYNC_INTERVAL', '300'))
        
        # 컴포넌트 초기화
        try:
            # 벡터 저장소
            self.vector_store = VectorStoreManager(
                chroma_host=self.chroma_host,
                chroma_port=self.chroma_port
            )
            logger.info('✓ VectorStoreManager 초기화 완료')
            
            # 동기화 상태 관리자
            self.sync_state = SyncStateManager()
            logger.info('✓ SyncStateManager 초기화 완료')
            
            # Notion 커넥터
            self.notion_connector = NotionConnector(
                token=self.notion_token,
                database_id=self.notion_database_id
            )
            if self.notion_connector.test_connection():
                logger.info('✓ NotionConnector 초기화 완료')
            else:
                logger.warning('⚠ Notion API 연결 실패 - 동기화 시 오류 가능')
            
            # Gitea 커넥터
            # ✅ GITEA_URL이 설정되지 않았거나 None이면 스킵
            if self.gitea_url:
                self.gitea_connector = GiteaConnector(
                    gitea_url=self.gitea_url,
                    token=self.gitea_token
                )
                logger.info(f'✓ GiteaConnector 초기화 완료 (대상 저장소: {len(self.target_repos)}개)')
            else:
                self.gitea_connector = None
                logger.info('⊘ Gitea URL 미설정 - Gitea 동기화 비활성화')
            
            # GitHub 커넥터
            if self.github_token:
                self.github_connector = GitHubConnector(token=self.github_token)
                if self.github_connector.test_connection():
                    logger.info('✓ GitHubConnector 초기화 완료')
                else:
                    logger.warning('⚠ GitHub API 연결 실패')
            else:
                self.github_connector = None
                logger.info('⊘ GitHub 토큰 미설정 - GitHub 동기화 비활성화')
            
            # 텍스트 분할 프로세서
            self.chunking_processor = ChunkingProcessor()
            logger.info('✓ ChunkingProcessor 초기화 완료')
            
        except Exception as e:
            logger.error(f'초기화 실패: {str(e)}')
            raise
        
        logger.info('=' * 60)
        logger.info('Sync Worker 초기화 완료')
        logger.info('=' * 60)
    
    def sync_notion(self) -> int:
        """
        Notion 데이터베이스 동기화
        
        - 마지막 동기화 시간 이후 변경된 페이지 조회
        - 페이지마다 청킹 후 즉시 저장 (스트리밍 방식)
        - ChromaDB에 저장 (기존 데이터는 먼저 삭제)
        
        Returns:
            동기화된 청크 개수
        """
        logger.info('[Notion] 동기화 시작...')
        
        try:
            # 마지막 동기화 시간 조회
            last_sync_time = self.sync_state.get_last_sync_time('notion')
            logger.info(f'[Notion] 마지막 동기화: {last_sync_time or "없음 (처음 동기화)"}')
            
            # 변경된 페이지 조회 및 처리 (Delta Sync)
            # 즉시 저장하기 위해 제너레이터 사용
            total_chunks = 0
            processed_pages = set()
            
            # 페이지별 스트리밍 처리
            for page_chunks in self.notion_connector.sync_pages_streaming(
                last_sync_time=last_sync_time,
                chunking_processor=self.chunking_processor
            ):
                if not page_chunks:
                    continue
                
                # 페이지별 청크가 완성되었을 때 즉시 저장
                source = page_chunks[0]['metadata'].get('source')
                
                # 기존 데이터 삭제 (중복 방지)
                if source and source not in processed_pages:
                    deleted_count = self.vector_store.delete_by_source(source)
                    if deleted_count > 0:
                        logger.debug(f'[Notion] 기존 데이터 삭제: {source} ({deleted_count}개)')
                    processed_pages.add(source)
                
                # 페이지의 청크들을 즉시 저장
                added_count = self.vector_store.add_documents(page_chunks)
                total_chunks += added_count
                
                # 페이지 타이틀 로깅
                page_title = page_chunks[0]['metadata'].get('title', 'Untitled')
                logger.info(f'✓ 페이지 저장 완료: {page_title} ({len(page_chunks)}개 청크)')
            
            # 마지막 동기화 시간 업데이트
            self.sync_state.set_last_sync_time('notion')
            
            if total_chunks == 0:
                logger.info('[Notion] 변경된 페이지 또는 청크 없음')
            else:
                logger.info(f'[Notion] 동기화 완료 (총 {total_chunks}개 청크)')
            
            return total_chunks
            
        except Exception as e:
            logger.error(f'[Notion] 동기화 중 오류: {str(e)}', exc_info=True)
            return 0
    
    def sync_gitea(self) -> int:
        """
        Gitea 저장소 동기화
        
        - 지정된 저장소 목록 반복 처리
        - Git clone으로 전체 코드 다운로드
        - 언어별 파일 필터링 및 청킹
        - ChromaDB에 저장 (기존 데이터는 먼저 삭제)
        
        Returns:
            동기화된 청크 개수
        """
        # ✅ Gitea가 설정되지 않았으면 스킵
        if not self.gitea_connector:
            logger.warning('[Gitea] 커넥터 미설정 - Gitea 동기화 건너뜀')
            return 0
        
        logger.info(f'[Gitea] 동기화 시작... (대상: {len(self.target_repos)}개 저장소)')
        
        total_chunks = 0
        
        for repo_path in self.target_repos:
            try:
                logger.info(f'[Gitea] 저장소 처리: {repo_path}')
                
                # 저장소 URL 구성
                repo_url = f'{self.gitea_url}/{repo_path}.git'
                
                # 저장소 분석 및 청크 생성
                documents = self.gitea_connector.process_repository(repo_path)
                
                if not documents:
                    logger.warning(f'[Gitea] 생성된 청크 없음: {repo_path}')
                    continue
                
                # 기존 데이터 삭제 (중복 방지)
                deleted_count = self.vector_store.delete_by_source(repo_url)
                if deleted_count > 0:
                    logger.debug(f'[Gitea] 기존 데이터 삭제: {repo_path} ({deleted_count}개)')
                
                # ChromaDB에 저장
                added_count = self.vector_store.add_documents(documents)
                total_chunks += added_count
                
                logger.info(f'[Gitea] 저장소 동기화 완료: {repo_path} ({added_count}개 청크)')
                
            except Exception as e:
                logger.error(f'[Gitea] 저장소 처리 실패 ({repo_path}): {str(e)}')
                continue
        
        # 마지막 동기화 시간 업데이트
        if self.target_repos:
            self.sync_state.set_last_sync_time('gitea')
        
        logger.info(f'[Gitea] 동기화 완료 (총 {total_chunks}개 청크)')
        return total_chunks
    
    def sync_github(self) -> int:
        """
        GitHub 저장소 동기화
        
        - GitHub API를 통해 저장소 목록 조회 (자동 검색 또는 지정)
        - Git clone으로 코드 다운로드
        - 언어별 파일 필터링 및 청킹
        - ChromaDB에 저장 (기존 데이터는 먼저 삭제)
        
        Returns:
            동기화된 청크 개수
        """
        if not self.github_connector:
            logger.info('[GitHub] GitHub 커넥터 미설정 - 동기화 스킵')
            return 0
        
        logger.info(f'[GitHub] 동기화 시작... (조직: {len(self.github_orgs)}개, 사용자 저장소: {self.github_include_user})')
        
        total_chunks = 0
        
        try:
            # 저장소 조회
            repos = []
            
            # 1. Organization 저장소 수집
            if self.github_orgs:
                try:
                    org_repos = self.github_connector.get_organization_repositories(
                        self.github_orgs,
                        exclude_private=os.getenv('GITHUB_EXCLUDE_PRIVATE', 'false').lower() == 'true',
                        exclude_forks=os.getenv('GITHUB_EXCLUDE_FORKS', 'false').lower() == 'true'
                    )
                    repos.extend(org_repos)
                    logger.info(f'[GitHub] Organization 저장소: {len(org_repos)}개')
                except Exception as e:
                    logger.error(f'[GitHub] Organization 저장소 조회 실패: {str(e)}')
            
            # 2. 사용자 저장소 수집 (설정된 경우)
            if self.github_include_user:
                try:
                    user_repos = self.github_connector.get_user_repositories(
                        exclude_private=os.getenv('GITHUB_EXCLUDE_PRIVATE', 'false').lower() == 'true',
                        exclude_forks=os.getenv('GITHUB_EXCLUDE_FORKS', 'false').lower() == 'true'
                    )
                    repos.extend(user_repos)
                    logger.info(f'[GitHub] 사용자 저장소: {len(user_repos)}개')
                except Exception as e:
                    logger.error(f'[GitHub] 사용자 저장소 조회 실패: {str(e)}')
            
            if not repos:
                logger.info('[GitHub] 조회된 저장소 없음')
                return 0
            
            # 기존 GitHub 데이터 삭제
            self.vector_store.delete_by_source('github')
            
            # 저장소별 처리
            for repo in repos:
                try:
                    repo_name = repo['full_name']
                    repo_url = repo['clone_url']
                    
                    logger.info(f'[GitHub] 저장소 처리: {repo_name}')
                    
                    # 저장소 처리 및 청크 생성
                    documents = self.github_connector.process_repository(repo_url, repo_name)
                    
                    if documents:
                        added_count = self.vector_store.add_documents(documents)
                        total_chunks += added_count
                        logger.info(f'[GitHub] 저장소 완료: {repo_name} ({added_count}개 청크)')
                    else:
                        logger.warning(f'[GitHub] 생성된 청크 없음: {repo_name}')
                    
                except Exception as e:
                    logger.error(f'[GitHub] 저장소 처리 실패 ({repo["full_name"]}): {str(e)}', exc_info=True)
                    continue
            
            # 마지막 동기화 시간 업데이트
            if repos:
                self.sync_state.set_last_sync_time('github')
            
            logger.info(f'[GitHub] 동기화 완료 (총 {total_chunks}개 청크)')
            return total_chunks
            
        except Exception as e:
            logger.error(f'[GitHub] 동기화 중 오류: {str(e)}', exc_info=True)
            return 0
    
    def print_stats(self) -> None:
        """벡터 저장소 통계 출력"""
        try:
            stats = self.vector_store.get_collection_stats()
            logger.info('=' * 60)
            logger.info('벡터 저장소 통계')
            logger.info(f'  - 컬렉션: {stats.get("collection_name")}')
            logger.info(f'  - 저장된 문서: {stats.get("document_count")}개')
            logger.info(f'  - 호스트: {stats.get("host")}:{stats.get("port")}')
            logger.info('=' * 60)
        except Exception as e:
            logger.error(f'통계 출력 실패: {str(e)}')
    
    def run(self) -> None:
        """메인 루프 - 주기적으로 동기화"""
        logger.info(f'Sync Worker 시작 (동기화 주기: {self.sync_interval}초)')
        
        cycle_count = 0
        
        while True:
            cycle_count += 1
            
            try:
                start_time = datetime.now()
                logger.info('=' * 70)
                logger.info(f'[사이클 #{cycle_count}] 동기화 시작: {start_time.isoformat()}')
                logger.info('=' * 70)
                
                notion_chunks = 0
                try:
                    logger.info('-' * 70)
                    logger.info('Step 1: Notion 페이지 동기화')
                    logger.info('-' * 70)
                    notion_chunks = self.sync_notion()
                except Exception as e:
                    logger.error(f'Notion 동기화 실패: {str(e)}', exc_info=True)
                

                # 1. Gitea 저장소 동기화
                gitea_chunks = 0
                try:
                    logger.info('-' * 70)
                    logger.info('Step 2: Gitea 저장소 동기화')
                    logger.info('-' * 70)
                    gitea_chunks = self.sync_gitea()
                except Exception as e:
                    logger.error(f'Gitea 동기화 실패: {str(e)}', exc_info=True)
                
                # 2. GitHub 저장소 동기화
                github_chunks = 0
                try:
                    logger.info('-' * 70)
                    logger.info('Step 3: GitHub 저장소 동기화')
                    logger.info('-' * 70)
                    github_chunks = self.sync_github()
                except Exception as e:
                    logger.error(f'GitHub 동기화 실패: {str(e)}', exc_info=True)
                
                # 3. Notion 페이지 동기화
                
                # 통계 출력
                try:
                    logger.info('-' * 70)
                    logger.info('벡터 저장소 통계')
                    logger.info('-' * 70)
                    self.print_stats()
                except Exception as e:
                    logger.error(f'통계 출력 실패: {str(e)}')
                
                # 소요 시간 계산
                end_time = datetime.now()
                elapsed = (end_time - start_time).total_seconds()
                
                logger.info('=' * 70)
                logger.info(f'[사이클 #{cycle_count}] 동기화 완료')
                logger.info(f'  • Gitea: {gitea_chunks}개 청크')
                logger.info(f'  • GitHub: {github_chunks}개 청크')
                logger.info(f'  • Notion: {notion_chunks}개 청크')
                logger.info(f'  • 총 처리: {gitea_chunks + github_chunks + notion_chunks}개 청크')
                logger.info(f'  • 소요시간: {elapsed:.2f}초')
                logger.info('=' * 70)
                
                # 3. 다음 동기화까지 대기
                remaining_time = self.sync_interval - elapsed
                if remaining_time > 0:
                    logger.info(f'{remaining_time:.0f}초 대기 중... (다음 동기화: {(end_time.timestamp() + remaining_time)})')
                    time.sleep(remaining_time)
                else:
                    logger.warning(f'동기화 시간이 설정값({self.sync_interval}초)을 초과했습니다. 즉시 다음 동기화 실행')
                
            except KeyboardInterrupt:
                logger.info('')
                logger.info('=' * 70)
                logger.info('Sync Worker 종료 (사용자 중단)')
                logger.info('=' * 70)
                break
                
            except Exception as e:
                logger.error(f'[사이클 #{cycle_count}] 예상치 못한 오류: {str(e)}', exc_info=True)
                logger.info('10초 후 재시도...')
                time.sleep(10)


if __name__ == '__main__':
    try:
        # 최대 3번 시도 (ChromaDB 준비 대기)
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                worker = SyncWorker()
                worker.run()
                break
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count  # 지수 백오프: 2, 4, 8초
                    logger.error(f'[시도 {retry_count}/{max_retries}] Sync Worker 초기화 실패: {str(e)}', exc_info=True)
                    logger.info(f'{wait_time}초 후 재시도...')
                    time.sleep(wait_time)
                else:
                    logger.error(f'Sync Worker 최종 실패: {str(e)}', exc_info=True)
                    exit(1)
                    
    except Exception as e:
        logger.error(f'예상치 못한 오류: {str(e)}', exc_info=True)
        exit(1)
