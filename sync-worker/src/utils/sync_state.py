"""
SyncStateManager - 동기화 상태 관리 모듈

마지막 동기화 시간을 로컬 파일에 저장/조회하여 Delta Sync를 구현합니다.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class SyncStateManager:
    """동기화 상태를 관리하는 클래스"""
    
    def __init__(self, state_file: str = '/tmp/rag_sync_state.json'):
        """
        Args:
            state_file: 동기화 상태를 저장할 파일 경로
        """
        self.state_file = Path(state_file)
        self._load_state()
    
    def _load_state(self) -> None:
        """파일에서 동기화 상태 로드"""
        try:
            if self.state_file.exists():
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    self.state = json.load(f)
                logger.info(f'동기화 상태 로드 완료: {self.state}')
            else:
                self.state = {}
                logger.info('새로운 동기화 상태 파일 생성')
        except Exception as e:
            logger.error(f'동기화 상태 로드 실패: {str(e)}')
            self.state = {}
    
    def _save_state(self) -> None:
        """동기화 상태를 파일에 저장"""
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=2, ensure_ascii=False)
            logger.debug(f'동기화 상태 저장 완료: {self.state}')
        except Exception as e:
            logger.error(f'동기화 상태 저장 실패: {str(e)}')
    
    def get_last_sync_time(self, source: str) -> Optional[str]:
        """
        지정된 소스의 마지막 동기화 시간 조회
        
        Args:
            source: 동기화 소스 (예: 'notion', 'gitea')
            
        Returns:
            ISO 8601 형식의 마지막 동기화 시간, 없으면 None
        """
        return self.state.get(f'{source}_last_sync_time')
    
    def set_last_sync_time(self, source: str, timestamp: Optional[str] = None) -> None:
        """
        지정된 소스의 마지막 동기화 시간 저장
        
        Args:
            source: 동기화 소스 (예: 'notion', 'gitea')
            timestamp: ISO 8601 형식의 타임스탬프, None이면 현재 시간
        """
        if timestamp is None:
            timestamp = datetime.utcnow().isoformat() + 'Z'
        
        self.state[f'{source}_last_sync_time'] = timestamp
        logger.info(f'{source} 마지막 동기화 시간 업데이트: {timestamp}')
        self._save_state()
    
    def get_synced_pages(self, source: str) -> dict:
        """
        동기화된 페이지 목록 조회
        
        Args:
            source: 동기화 소스 (예: 'notion', 'gitea')
            
        Returns:
            페이지 ID와 정보의 딕셔너리
        """
        return self.state.get(f'{source}_synced_pages', {})
    
    def set_synced_pages(self, source: str, pages: dict) -> None:
        """
        동기화된 페이지 목록 저장
        
        Args:
            source: 동기화 소스 (예: 'notion', 'gitea')
            pages: 페이지 ID와 정보의 딕셔너리
        """
        self.state[f'{source}_synced_pages'] = pages
        self._save_state()
    
    def add_synced_page(self, source: str, page_id: str, metadata: dict) -> None:
        """
        동기화된 페이지 추가
        
        Args:
            source: 동기화 소스
            page_id: 페이지 ID
            metadata: 페이지 메타데이터
        """
        pages = self.get_synced_pages(source)
        pages[page_id] = metadata
        self.set_synced_pages(source, pages)
    
    def remove_synced_page(self, source: str, page_id: str) -> None:
        """
        동기화된 페이지 제거
        
        Args:
            source: 동기화 소스
            page_id: 페이지 ID
        """
        pages = self.get_synced_pages(source)
        if page_id in pages:
            del pages[page_id]
            self.set_synced_pages(source, pages)
