"""
Sync Worker - Notion & Gitea 데이터 동기화 메인 모듈

주기적으로 Notion 데이터베이스와 Gitea 저장소를 동기화하여
ChromaDB 벡터 데이터베이스에 저장합니다.
"""

import os
import sys
import argparse
import logging
from datetime import datetime
import time
from dotenv import load_dotenv

# 로컬 모듈 임포트
from connectors import NotionConnector, GiteaConnector, GitHubConnector
from processors import ChunkingProcessor
from processors.pipeline import GraphRAGPipeline
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

# 글로벌 설정
RELATION_COLUMN_NAME = os.getenv('NOTION_RELATION_COLUMN', '작업')  # 관계형 컬럼명
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')  # 임베딩 모델


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
            
            # Graph RAG 파이프라인 (Notion 기반)
            if self.notion_token:
                self.graph_rag_pipeline = GraphRAGPipeline(
                    notion_token=self.notion_token,
                    chroma_host=self.chroma_host,
                    chroma_port=self.chroma_port,
                    max_chunk_tokens=512,
                    traversal_depth=2
                )
                logger.info('✓ GraphRAGPipeline 초기화 완료')
            else:
                self.graph_rag_pipeline = None
                logger.info('⊘ Notion 토큰 미설정 - Graph RAG 파이프라인 비활성화')
            
        except Exception as e:
            logger.error(f'초기화 실패: {str(e)}')
            raise
        
        logger.info('=' * 60)
        logger.info('Sync Worker 초기화 완료')
        logger.info('=' * 60)
    
    def sync_notion(self) -> int:
        """
        Notion 데이터베이스 동기화 (Graph RAG 파이프라인 사용)
        
        - Graph RAG 파이프라인으로 Notion 문서 추출
        - NetworkX 그래프 구축 ("작업" 관계 + parent 계층)
        - Lazy Loading을 통한 메모리 최적화
        - ChromaDB에 저장
        
        Returns:
            동기화된 문서 개수
        """
        logger.info('[Notion Graph RAG] 동기화 시작...')
        
        try:
            # Graph RAG 파이프라인이 없으면 기존 방식으로 폴백
            if not self.graph_rag_pipeline:
                logger.warning('[Notion] Graph RAG 파이프라인 미활성화 - 기존 방식으로 동기화')
                return self._sync_notion_legacy()
            
            # ✅ Delta Sync: 마지막 동기화 시간 조회
            last_sync_time = self.sync_state.get_last_sync_time('notion')
            if last_sync_time:
                logger.info(f'[Notion] Delta Sync 모드 - 마지막 동기화: {last_sync_time}')
            else:
                logger.info('[Notion] Full Sync 모드 - 첫 동기화')
            
            # Graph RAG 파이프라인 실행 (Delta Sync 적용)
            logger.info('[Notion Graph RAG] 파이프라인 실행 중...')
            result = self.graph_rag_pipeline.run_full_pipeline(
                last_sync_time=last_sync_time  # ✅ Delta Sync 시간 전달
            )
            
            if result.get('status') == 'success':
                stored_count = result.get('stored_count', 0)
                nodes_count = result.get('nodes_count', 0)
                edges_count = result.get('edges_count', 0)
                
                logger.info('[Notion Graph RAG] 동기화 완료')
                logger.info(f'  • 문서: {result.get("documents_count", 0)}개')
                logger.info(f'  • 노드: {nodes_count}개')
                logger.info(f'  • 엣지: {edges_count}개')
                logger.info(f'  • 저장됨: {stored_count}개')
                
                # 마지막 동기화 시간 업데이트
                self.sync_state.set_last_sync_time('notion')
                
                return stored_count
            else:
                logger.error(f'[Notion Graph RAG] 파이프라인 실패: {result.get("message")}')
                return 0
            
        except Exception as e:
            logger.error(f'[Notion Graph RAG] 동기화 중 오류: {str(e)}', exc_info=True)
            logger.info('[Notion] 기존 방식으로 폴백...')
            try:
                return self._sync_notion_legacy()
            except Exception as fallback_error:
                logger.error(f'[Notion] 기존 방식도 실패: {str(fallback_error)}')
                return 0
    
    def _sync_notion_legacy(self) -> int:
        """
        Notion 데이터베이스 동기화 (기존 청킹 방식 - 폴백용)
        
        Returns:
            동기화된 청크 개수
        """
        logger.info('[Notion Legacy] 기존 청킹 방식으로 동기화 중...')
        
        try:
            # 마지막 동기화 시간 조회
            last_sync_time = self.sync_state.get_last_sync_time('notion')
            logger.info(f'[Notion] 마지막 동기화: {last_sync_time or "없음 (처음 동기화)"}')
            
            # 변경된 페이지 조회 및 처리
            total_chunks = 0
            processed_pages = set()
            
            # 페이지별 스트리밍 처리
            for page_chunks in self.notion_connector.sync_pages_streaming(
                last_sync_time=last_sync_time,
                chunking_processor=self.chunking_processor
            ):
                if not page_chunks:
                    continue
                
                source = page_chunks[0]['metadata'].get('source')
                
                # 중복 처리 방지 (삭제 없이 skip)
                if source and source in processed_pages:
                    continue
                processed_pages.add(source)
                
                # 페이지의 청크들을 즉시 저장
                added_count = self.vector_store.add_documents(page_chunks)
                total_chunks += added_count
                
                page_title = page_chunks[0]['metadata'].get('title', 'Untitled')
                logger.info(f'✓ 페이지 저장 완료: {page_title} ({len(page_chunks)}개 청크)')
            
            # 마지막 동기화 시간 업데이트
            self.sync_state.set_last_sync_time('notion')
            
            if total_chunks == 0:
                logger.info('[Notion Legacy] 변경된 페이지 또는 청크 없음')
            else:
                logger.info(f'[Notion Legacy] 동기화 완료 (총 {total_chunks}개 청크)')
            
            return total_chunks
            
        except Exception as e:
            logger.error(f'[Notion Legacy] 동기화 중 오류: {str(e)}', exc_info=True)
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
                
                # ChromaDB에 저장 (upsert 방식으로 중복 처리)
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
            
            # 저장소별 처리 (삭제 없이 upsert 방식)
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
        """일회성 동기화 - 한 번만 실행 후 종료"""
        try:
            start_time = datetime.now()
            logger.info('=' * 70)
            logger.info(f'Sync Worker 실행 시작: {start_time.isoformat()}')
            logger.info('=' * 70)
            
            notion_chunks = 0
            try:
                logger.info('-' * 70)
                logger.info('Step 1: Notion 페이지 동기화')
                logger.info('-' * 70)
                notion_chunks = self.sync_notion()
            except Exception as e:
                logger.error(f'Notion 동기화 실패: {str(e)}', exc_info=True)
            
            # Gitea 저장소 동기화
            gitea_chunks = 0
            try:
                logger.info('-' * 70)
                logger.info('Step 2: Gitea 저장소 동기화')
                logger.info('-' * 70)
                gitea_chunks = self.sync_gitea()
            except Exception as e:
                logger.error(f'Gitea 동기화 실패: {str(e)}', exc_info=True)
            
            # GitHub 저장소 동기화
            github_chunks = 0
            try:
                logger.info('-' * 70)
                logger.info('Step 3: GitHub 저장소 동기화')
                logger.info('-' * 70)
                github_chunks = self.sync_github()
            except Exception as e:
                logger.error(f'GitHub 동기화 실패: {str(e)}', exc_info=True)
            
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
            logger.info('✅ 동기화 완료')
            logger.info(f'  • Gitea: {gitea_chunks}개 청크')
            logger.info(f'  • GitHub: {github_chunks}개 청크')
            logger.info(f'  • Notion: {notion_chunks}개 청크')
            logger.info(f'  • 총 처리: {gitea_chunks + github_chunks + notion_chunks}개 청크')
            logger.info(f'  • 소요시간: {elapsed:.2f}초')
            logger.info('=' * 70)
            
        except Exception as e:
            logger.error(f'동기화 실패: {str(e)}', exc_info=True)
    
    def run_once(self) -> None:
        """단일 동기화 실행 (배치 작업용)"""
        logger.info('Sync Worker 단일 실행 모드')
        
        try:
            start_time = datetime.now()
            logger.info('=' * 70)
            logger.info(f'[단일 실행] 동기화 시작: {start_time.isoformat()}')
            logger.info('=' * 70)
            
            # Notion 동기화
            notion_chunks = 0
            try:
                logger.info('Notion 동기화 중...')
                notion_chunks = self.sync_notion()
            except Exception as e:
                logger.error(f'Notion 동기화 실패: {str(e)}', exc_info=True)
            
            # 통계 출력
            self.print_stats()
            
            # 소요 시간
            end_time = datetime.now()
            elapsed = (end_time - start_time).total_seconds()
            
            logger.info('=' * 70)
            logger.info(f'[단일 실행] 동기화 완료')
            logger.info(f'  • Notion: {notion_chunks}개')
            logger.info(f'  • 소요시간: {elapsed:.2f}초')
            logger.info('=' * 70)
            
        except Exception as e:
            logger.error(f'단일 실행 실패: {str(e)}', exc_info=True)
            raise


def parse_args():
    """명령줄 인자 파싱"""
    parser = argparse.ArgumentParser(description='RAG Sync Worker')
    parser.add_argument(
        '--once',
        action='store_true',
        help='1회만 실행 후 종료 (배치 작업용)'
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    sync_once = args.once or os.getenv('SYNC_ONCE', 'false').lower() == 'true'
    
    try:
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                worker = SyncWorker()
                
                if sync_once:
                    worker.run_once()
                else:
                    worker.run()
                break
                
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2 ** retry_count
                    logger.error(f'[시도 {retry_count}/{max_retries}] 실패: {str(e)}', exc_info=True)
                    logger.info(f'{wait_time}초 후 재시도...')
                    time.sleep(wait_time)
                else:
                    logger.error(f'최종 실패: {str(e)}', exc_info=True)
                    sys.exit(1)
                    
    except Exception as e:
        logger.error(f'예상치 못한 오류: {str(e)}', exc_info=True)
        sys.exit(1)
