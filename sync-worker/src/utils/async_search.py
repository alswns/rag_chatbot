"""
Async Search Manager - 비동기 검색 최적화 및 병렬 처리

멀티패스 검색을 효율적으로 병렬화하여 성능을 높입니다.
"""

import asyncio
import logging
from typing import List, Dict, Any, Callable, TypeVar, Coroutine

logger = logging.getLogger(__name__)

T = TypeVar('T')


class AsyncSearchManager:
    """비동기 검색 관리 및 병렬 처리 최적화"""
    
    @staticmethod
    async def parallel_search(
        search_tasks: List[Coroutine],
        timeout: int = 30,
        return_exceptions: bool = True
    ) -> List[Any]:
        """
        여러 검색을 병렬로 실행
        
        Args:
            search_tasks: 실행할 코루틴 리스트
            timeout: 각 작업의 타임아웃 (초)
            return_exceptions: 예외 발생 시 예외를 반환할지 여부
        
        Returns:
            각 작업의 결과 리스트
        """
        try:
            # 타임아웃 설정
            tasks_with_timeout = [
                asyncio.wait_for(task, timeout=timeout)
                for task in search_tasks
            ]
            
            # 병렬 실행
            results = await asyncio.gather(
                *tasks_with_timeout,
                return_exceptions=return_exceptions
            )
            
            logger.info(f'✅ {len(results)}개 병렬 검색 완료')
            
            # 실패한 작업 로깅
            failed = sum(1 for r in results if isinstance(r, Exception))
            if failed > 0:
                logger.warning(f'⚠️ {failed}개 검색 실패')
            
            return results
            
        except Exception as e:
            logger.error(f'❌ 병렬 검색 실패: {str(e)}')
            return []
    
    @staticmethod
    async def batch_search_with_rate_limit(
        queries: List[str],
        search_fn: Callable[[str], Coroutine],
        batch_size: int = 3,
        delay_between_batches: float = 0.1
    ) -> List[Any]:
        """
        배치 단위로 검색을 실행하여 rate limit 조절
        
        Args:
            queries: 검색할 쿼리 리스트
            search_fn: 쿼리를 받아 결과를 반환하는 비동기 함수
            batch_size: 한 번에 병렬 실행할 검색 개수
            delay_between_batches: 배치 간 지연 (초)
        
        Returns:
            모든 검색 결과
        """
        all_results = []
        
        for i in range(0, len(queries), batch_size):
            batch = queries[i:i+batch_size]
            logger.info(f'🔄 배치 {i//batch_size + 1}: {len(batch)}개 쿼리 처리')
            
            # 배치 내 검색 병렬 실행
            tasks = [search_fn(query) for query in batch]
            batch_results = await AsyncSearchManager.parallel_search(tasks)
            all_results.extend(batch_results)
            
            # 다음 배치 전 대기
            if i + batch_size < len(queries):
                await asyncio.sleep(delay_between_batches)
        
        logger.info(f'✅ 배치 검색 완료: {len(all_results)}개 결과')
        
        return all_results
    
    @staticmethod
    async def search_with_timeout_fallback(
        primary_search: Coroutine,
        fallback_search: Coroutine,
        timeout: int = 10
    ) -> Any:
        """
        Primary 검색이 타임아웃되면 Fallback 검색 실행
        
        Args:
            primary_search: 먼저 실행할 검색
            fallback_search: Primary 실패 시 실행할 검색
            timeout: Primary 검색 타임아웃 (초)
        
        Returns:
            검색 결과 (Primary 또는 Fallback)
        """
        try:
            logger.info('🔍 Primary 검색 시작...')
            result = await asyncio.wait_for(primary_search, timeout=timeout)
            logger.info('✅ Primary 검색 성공')
            return result
            
        except asyncio.TimeoutError:
            logger.warning(f'⏱️ Primary 검색 타임아웃 ({timeout}초) → Fallback 검색 시작')
            try:
                result = await fallback_search
                logger.info('✅ Fallback 검색 성공')
                return result
            except Exception as e:
                logger.error(f'❌ Fallback 검색도 실패: {str(e)}')
                return None
        
        except Exception as e:
            logger.error(f'❌ Primary 검색 실패: {str(e)} → Fallback 검색 시작')
            try:
                result = await fallback_search
                logger.info('✅ Fallback 검색 성공')
                return result
            except Exception as fallback_e:
                logger.error(f'❌ Fallback 검색도 실패: {str(fallback_e)}')
                return None
    
    @staticmethod
    async def search_with_circuit_breaker(
        search_fn: Callable[[str], Coroutine],
        queries: List[str],
        failure_threshold: int = 3,
        recovery_timeout: int = 60
    ) -> List[Any]:
        """
        Circuit Breaker 패턴으로 연쇄 실패 방지
        
        연속 실패가 threshold를 초과하면 일시적으로 검색 중단 후 복구 시도
        
        Args:
            search_fn: 검색 함수
            queries: 검색할 쿼리 리스트
            failure_threshold: 실패 임계값
            recovery_timeout: 복구 대기 시간 (초)
        
        Returns:
            검색 결과 리스트
        """
        results = []
        consecutive_failures = 0
        circuit_open = False
        
        for i, query in enumerate(queries):
            # Circuit이 열려 있으면 대기
            if circuit_open:
                logger.warning(f'⚠️ Circuit Open - {recovery_timeout}초 대기 후 복구')
                await asyncio.sleep(recovery_timeout)
                circuit_open = False
                consecutive_failures = 0
            
            try:
                logger.debug(f'[{i+1}/{len(queries)}] 검색: {query[:50]}...')
                result = await asyncio.wait_for(
                    search_fn(query),
                    timeout=30
                )
                results.append(result)
                consecutive_failures = 0  # 성공 시 카운터 리셋
                
            except Exception as e:
                logger.warning(f'❌ 검색 실패: {str(e)}')
                results.append(None)
                consecutive_failures += 1
                
                if consecutive_failures >= failure_threshold:
                    logger.error(f'❌ 연쇄 실패 {consecutive_failures}회 → Circuit Open')
                    circuit_open = True
        
        logger.info(f'✅ Circuit Breaker 검색 완료: {len([r for r in results if r])}개 성공')
        
        return results
