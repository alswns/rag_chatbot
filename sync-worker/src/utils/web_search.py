"""
웹 검색 모듈 (Refactored - Intelligent Integration)

판단 로직과 생성 로직을 분리하여 LLM 성능 한계 극복:
1. SearchDecisionMaker: Binary Decision (YES/NO)
2. QueryGenerator: Clean Query Generation + Query Expansion
3. DuckDuckGoSearcher: Web Search Execution
4. WebSearchResultFusion: 웹 검색 결과를 RAG 결과와 통합 (RRF)

특징:
- Query Expansion: 웹 검색 쿼리도 3가지 버전 생성
- Result Fusion: 웹 검색 + RAG 결과를 RRF로 통합
- Async-First: 모든 작업 비동기 처리
"""

import os
import re
import logging
import asyncio
from typing import List, Dict, Any, Optional, Tuple
import openai

logger = logging.getLogger(__name__)

# DuckDuckGo 라이브러리 임포트
try:
    from duckduckgo_search import AsyncDDGS
except ImportError:
    AsyncDDGS = None
    logger.warning('⚠️ duckduckgo-search 미설치 - 웹 검색 비활성화')


# ==================== 1. Binary Decision Maker ====================

class SearchDecisionMaker:
    """웹 검색 필요성 판단 (Binary: YES/NO)"""
    
    DECISION_PROMPT = """당신은 **정답 확인기(Answer Checker)**입니다.

## 작업
사용자의 질문에 대한 답이 [내부 문서]에 충분히 포함되어 있는지 확인하세요.

## 판단 기준

### YES (웹 검색 필요) 👉 다음 경우:
- 내부 문서가 **완전히 비어있음**
- 내부 문서에 **관련 정보가 전혀 없음**
- 질문이 **최신 정보**(뉴스, 버전, 트렌드)를 요구함
- 질문이 **일반 상식/개념**을 요구함 (위키피디아 수준)
- 내부 문서에 **힌트만 있고 구체적 답이 없음** (예: URL만 있고 설명 없음)

### NO (내부 문서로 충분) 👉 다음 경우:
- 내부 문서에 **직접적인 답**이 있음
- 설정값, 코드, 경로, IP 등 **구체적 정보**가 있음
- 문서가 질문의 맥락과 **완전히 일치**함

## ⚠️ 중요
- URL, 코드 스니펫, 설정값 등이 **힌트로만** 있으면 → **NO** (충분함)
- 외부 정보가 **반드시** 필요할 때만 → **YES**

## 출력 형식
**오직 'YES' 또는 'NO' 단어 하나만 출력하세요.**
(설명, 이유, 추가 문장 금지)

---

## 사용자 질문
{query}

## 내부 문서
{context}

## 당신의 판단:"""

    def __init__(self, vllm_api_url: str = None):
        self.vllm_api_url = vllm_api_url or os.getenv('VLLM_API_URL', 'http://localhost:8000/v1')
        self.client = None
    
    def _get_client(self) -> openai.OpenAI:
        if self.client is None:
            self.client = openai.OpenAI(
                api_key='sk-not-needed',
                base_url=self.vllm_api_url,
                timeout=30.0
            )
        return self.client
    
    async def needs_search(self, query: str, context: str) -> bool:
        """
        웹 검색 필요 여부 판단
        
        Returns:
            True: 웹 검색 필요
            False: 내부 문서로 충분
        """
        try:
            # 컨텍스트 요약 (너무 길면 잘라냄)
            context_summary = context[:2000] if context else "검색 결과 없음"
            
            prompt = self.DECISION_PROMPT.format(
                query=query,
                context=context_summary
            )
            
            client = self._get_client()
            response = client.chat.completions.create(
                model=os.getenv('LLM_MODEL_ID', 'DeepSeek-R1'),
                messages=[
                    {'role': 'system', 'content': '정답 확인기입니다.'},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.0,
                max_tokens=10,  # YES/NO만 필요
                stream=False
            )
            
            decision = response.choices[0].message.content.strip().upper()
            
            # Parsing: YES 포함 여부
            needs_search = 'YES' in decision
            
            logger.info(f'🔍 검색 판단: {"YES (웹 검색 필요)" if needs_search else "NO (내부 문서 충분)"}')
            logger.debug(f'LLM 원본 응답: "{decision}"')
            
            return needs_search
            
        except Exception as e:
            logger.error(f'❌ 검색 판단 실패: {str(e)}')
            # 에러 시 안전하게 검색하지 않음 (보수적)
            return False


# ==================== 2. Query Generator ====================

# ==================== 2. Query Generator with Expansion ====================

class QueryGenerator:
    """검색 엔진용 쿼리 생성 + Query Expansion"""
    
    GENERATION_PROMPT = """DuckDuckGo 검색을 위한 **영어 키워드**를 생성하세요.

## 규칙
1. **3-5 단어**로 구성
2. **일반적인 기술 용어**만 사용
3. **프로젝트명, 파일명, 변수명, IP 주소 제거**
4. 설명이나 접두어 없이 **오직 키워드만** 출력

## 예시
- 입력: "우리 프로젝트의 FastAPI 비동기 처리 방법"
- 출력: fastapi async processing tutorial

- 입력: "최신 리액트 버전"
- 출력: latest react version 2026

## 사용자 질문
{query}

## 검색 키워드:"""

    EXPANSION_PROMPT = """사용자의 웹 검색 질문을 3가지 관점에서 다양하게 표현하세요.

## 규칙
1. **원본(Original)**: 원래 질문을 더 명확하고 자세하게
2. **개념(Conceptual)**: 기술적 배경이나 이론 중심
3. **실용(Practical)**: 실제 사용 사례나 구현 중심

각각 3-5 단어의 영어 검색어로 표현.

## 예시
사용자 질문: "FastAPI 성능 최적화"
- Original: fastapi performance optimization
- Conceptual: fastapi async frameworks architecture
- Practical: fastapi best practices tutorial

## 사용자 질문
{query}

## 응답 형식 (각 한 줄씩):
Original:
Conceptual:
Practical:"""

    def __init__(self, vllm_api_url: str = None):
        self.vllm_api_url = vllm_api_url or os.getenv('VLLM_API_URL', 'http://localhost:8000/v1')
        self.client = None
    
    def _get_client(self) -> openai.OpenAI:
        if self.client is None:
            self.client = openai.OpenAI(
                api_key='sk-not-needed',
                base_url=self.vllm_api_url,
                timeout=30.0
            )
        return self.client
    
    def _clean_query(self, raw_query: str) -> str:
        """LLM 출력 정제"""
        # 1. 마크다운 코드 블록 제거
        cleaned = re.sub(r'```[a-z]*\n?', '', raw_query)
        
        # 2. 인용부호 제거
        cleaned = cleaned.replace('"', '').replace("'", '').strip()
        
        # 3. 접두어 제거 (다국어)
        prefixes = [
            '검색 키워드:', '검색어:', 'search query:', 'query:', 
            'keywords:', '영어 검색어:', '세탁된 검색어:',
            'original:', 'conceptual:', 'practical:'
        ]
        for prefix in prefixes:
            if cleaned.lower().startswith(prefix.lower()):
                cleaned = cleaned[len(prefix):].strip()
        
        # 4. 첫 줄만 추출
        cleaned = cleaned.split('\n')[0].strip()
        
        # 5. 최대 길이 제한 (10 단어)
        words = cleaned.split()
        if len(words) > 10:
            cleaned = ' '.join(words[:10])
        
        return cleaned
    
    async def generate_query(self, query: str) -> str:
        """
        검색 엔진용 쿼리 생성
        
        Args:
            query: 사용자 질문
        
        Returns:
            세탁된 영어 검색어
        """
        try:
            prompt = self.GENERATION_PROMPT.format(query=query)
            
            client = self._get_client()
            response = client.chat.completions.create(
                model=os.getenv('LLM_MODEL_ID', 'DeepSeek-R1'),
                messages=[
                    {'role': 'system', 'content': '검색 키워드 생성기입니다.'},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.1,
                max_tokens=30,
                stream=False
            )
            
            raw_output = response.choices[0].message.content.strip()
            cleaned_query = self._clean_query(raw_output)
            
            if cleaned_query:
                logger.info(f'🧼 쿼리 생성: "{query}" → "{cleaned_query}"')
                return cleaned_query
            else:
                # Fallback: 간단한 키워드 추출
                fallback = ' '.join(query.split()[:5])
                logger.warning(f'⚠️ 생성 결과 비어있음, Fallback: "{fallback}"')
                return fallback
                
        except Exception as e:
            logger.error(f'❌ 쿼리 생성 실패: {str(e)}')
            # Fallback
            fallback = ' '.join(query.split()[:5])
            logger.info(f'🔄 Fallback 쿼리: "{fallback}"')
            return fallback
    
    async def expand_query(self, query: str) -> List[str]:
        """
        웹 검색용 Query Expansion (3가지 버전)
        
        Args:
            query: 사용자 질문
        
        Returns:
            [original_query, conceptual_query, practical_query]
        """
        try:
            prompt = self.EXPANSION_PROMPT.format(query=query)
            
            client = self._get_client()
            response = client.chat.completions.create(
                model=os.getenv('LLM_MODEL_ID', 'DeepSeek-R1'),
                messages=[
                    {'role': 'system', 'content': '웹 검색 쿼리 확장 전문가입니다.'},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.5,
                max_tokens=100,
                stream=False
            )
            
            response_text = response.choices[0].message.content.strip()
            
            # 응답 파싱
            expanded_queries = []
            for line in response_text.split('\n'):
                cleaned = self._clean_query(line)
                if cleaned and len(cleaned) > 3:
                    expanded_queries.append(cleaned)
            
            # 최대 3개까지만
            expanded_queries = expanded_queries[:3]
            
            if len(expanded_queries) < 3:
                # Fallback: 기본 쿼리만이라도 반환
                base_query = await self.generate_query(query)
                expanded_queries = [base_query]
            
            logger.info(f'📝 Query Expansion: {len(expanded_queries)}개 쿼리')
            for i, q in enumerate(expanded_queries, 1):
                logger.debug(f'   [{i}] {q}')
            
            return expanded_queries
            
        except Exception as e:
            logger.error(f'❌ Query Expansion 실패: {str(e)}')
            # Fallback: 기본 쿼리 하나만 반환
            base_query = await self.generate_query(query)
            return [base_query]


# ==================== 3. DuckDuckGo Searcher ====================

class DuckDuckGoSearcher:
    """DuckDuckGo 웹 검색 실행"""
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        DuckDuckGo 검색 수행
        
        Args:
            query: 검색어
            max_results: 최대 결과 수
        
        Returns:
            [{'title': ..., 'body': ..., 'href': ...}, ...]
        """
        if AsyncDDGS is None:
            logger.error('❌ duckduckgo-search 미설치')
            return []
        
        try:
            logger.info(f'🌐 DuckDuckGo 검색: "{query}"')
            
            # ✅ Bug Fix: async with 패턴 강제
            async with AsyncDDGS() as ddgs:
                search_results = await ddgs.text(query, max_results=max_results)
                
                results = []
                for result in search_results:
                    results.append({
                        'title': result.get('title', ''),
                        'body': result.get('body', ''),
                        'href': result.get('href', '')
                    })
                
                logger.info(f'✅ 검색 완료: {len(results)}개 결과')
                return results
                
        except Exception as e:
            logger.error(f'❌ DuckDuckGo 검색 실패: {str(e)}')
            return []
    
    def format_results(self, results: List[Dict[str, Any]]) -> str:
        """검색 결과를 LLM용 컨텍스트로 포맷팅"""
        if not results:
            return ""
        
        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get('title', 'Untitled')
            body = result.get('body', '')
            href = result.get('href', '')
            
            formatted.append(f"""[웹 검색 결과 {i}]
제목: {title}
내용: {body}
출처: {href}
""")
        
        return '\n---\n'.join(formatted)


# ==================== 4. Web Search Result Fusion ====================

class WebSearchResultFusion:
    """웹 검색 결과를 RAG 결과와 통합 (RRF)"""
    
    @staticmethod
    def fuse_with_rrf(
        rag_results: List[Dict[str, Any]],
        web_results: List[Dict[str, Any]],
        k: int = 10,
        rag_weight: float = 0.6,
        web_weight: float = 0.4
    ) -> List[Dict[str, Any]]:
        """
        RAG 결과와 웹 검색 결과를 RRF로 통합
        
        Args:
            rag_results: RAG 검색 결과 (이미 정렬됨)
            web_results: 웹 검색 결과 (이미 정렬됨)
            k: 최종 반환 개수
            rag_weight: RAG 결과 가중치 (0.6 = 60%)
            web_weight: 웹 검색 결과 가중치 (0.4 = 40%)
        
        Returns:
            통합된 결과 리스트
        """
        K = 60  # RRF 하이퍼파라미터
        
        # 각 소스별 점수 계산
        source_scores: Dict[str, Dict[str, Any]] = {}
        
        # RAG 결과 처리
        for rank, result in enumerate(rag_results, 1):
            doc_id = result.get('id', result.get('document_id', str(rank)))
            rrf_score = rag_weight * (1 / (K + rank))
            
            if doc_id not in source_scores:
                source_scores[doc_id] = {
                    'result': result,
                    'total_score': 0,
                    'sources': 0,
                    'rag_rank': rank,
                    'web_rank': None
                }
            
            source_scores[doc_id]['total_score'] += rrf_score
            source_scores[doc_id]['sources'] += 1
        
        # 웹 검색 결과 처리
        for rank, result in enumerate(web_results, 1):
            # 웹 검색 결과는 ID가 없으므로 제목으로 식별
            doc_id = f"web_{hash(result.get('title', str(rank))) % 10000}"
            rrf_score = web_weight * (1 / (K + rank))
            
            if doc_id not in source_scores:
                source_scores[doc_id] = {
                    'result': result,
                    'total_score': 0,
                    'sources': 0,
                    'rag_rank': None,
                    'web_rank': rank
                }
            
            source_scores[doc_id]['total_score'] += rrf_score
            source_scores[doc_id]['sources'] += 1
            if source_scores[doc_id].get('rag_rank') is None:
                source_scores[doc_id]['web_rank'] = rank
        
        # 점수 기준 정렬
        sorted_results = sorted(
            source_scores.items(),
            key=lambda x: (
                x[1]['sources'],  # 여러 소스에서 등장한 결과 우선
                x[1]['total_score']
            ),
            reverse=True
        )
        
        # 최종 결과 구성
        fused = []
        for doc_id, metadata in sorted_results[:k]:
            result = metadata['result'].copy()
            result['fusion_score'] = metadata['total_score']
            result['source_count'] = metadata['sources']
            result['rag_rank'] = metadata.get('rag_rank')
            result['web_rank'] = metadata.get('web_rank')
            result['source'] = 'hybrid'  # RAG + Web
            fused.append(result)
        
        logger.info(f'🔀 Result Fusion 완료: RAG {len(rag_results)} + Web {len(web_results)} → {len(fused)}개')
        
        return fused
    
    @staticmethod
    def format_fused_results(results: List[Dict[str, Any]]) -> str:
        """통합된 결과를 LLM용 컨텍스트로 포맷팅"""
        if not results:
            return ""
        
        formatted_lines = ["## 통합 검색 결과 (RAG + 웹 검색)\n"]
        
        for i, result in enumerate(results, 1):
            # 소스 표시
            source_info = []
            if result.get('rag_rank'):
                source_info.append(f"RAG #{result['rag_rank']}")
            if result.get('web_rank'):
                source_info.append(f"Web #{result['web_rank']}")
            source_str = " | ".join(source_info) if source_info else "Hybrid"
            
            # RAG 결과 vs 웹 결과에 따라 포맷 변경
            if 'web_rank' in result and result['web_rank']:
                # 웹 검색 결과
                title = result.get('title', 'Untitled')
                body = result.get('body', '')
                href = result.get('href', '')
                
                formatted_lines.append(f"""### [{i}] {title} ({source_str})
{body}
[출처: {href}]
""")
            else:
                # RAG 결과
                title = result.get('title', 'Untitled')
                content = result.get('content', '')[:300]
                
                formatted_lines.append(f"""### [{i}] {title} ({source_str})
{content}
""")
        
        return "\n".join(formatted_lines)


# ==================== 5. Unified Interface ====================

class WebSearchService:
    """웹 검색 통합 인터페이스"""
    
    def __init__(self, vllm_api_url: str = None):
        self.decision_maker = SearchDecisionMaker(vllm_api_url)
        self.query_generator = QueryGenerator(vllm_api_url)
        self.searcher = DuckDuckGoSearcher()
        self.fusion = WebSearchResultFusion()
    
    async def search_if_needed(
        self,
        user_query: str,
        internal_context: str,
        force_search: bool = False
    ) -> str:
        """
        조건부 웹 검색 수행
        
        Args:
            user_query: 사용자 질문
            internal_context: 내부 RAG 검색 결과
            force_search: 강제 검색 여부
        
        Returns:
            포맷된 웹 검색 결과 (또는 빈 문자열)
        """
        try:
            # Step 1: 검색 필요성 판단 (강제 검색이 아닐 때만)
            if not force_search:
                needs_search = await self.decision_maker.needs_search(user_query, internal_context)
                if not needs_search:
                    logger.info('ℹ️  내부 문서로 충분 → 웹 검색 스킵')
                    return ""
            else:
                logger.info('🔍 강제 검색 모드')
            
            # Step 2: 검색 쿼리 생성 + 확장
            search_query = await self.query_generator.generate_query(user_query)
            
            # Step 3: 웹 검색 실행
            results = await self.searcher.search(search_query, max_results=5)
            
            # Step 4: 결과 포맷팅
            if results:
                formatted = self.searcher.format_results(results)
                logger.info(f'✅ 웹 검색 완료: {len(results)}개 결과 반환')
                return formatted
            else:
                logger.warning('⚠️ 웹 검색 결과 없음')
                return ""
                
        except Exception as e:
            logger.error(f'❌ 웹 검색 처리 실패: {str(e)}')
            return ""
    
    async def search_with_expansion(
        self,
        user_query: str
    ) -> List[Dict[str, Any]]:
        """
        Query Expansion을 사용한 웹 검색 (3가지 버전)
        
        Args:
            user_query: 사용자 질문
        
        Returns:
            웹 검색 결과 리스트
        """
        try:
            # Step 1: Query Expansion
            expanded_queries = await self.query_generator.expand_query(user_query)
            logger.info(f'📝 웹 검색 쿼리 확장: {len(expanded_queries)}개')
            
            # Step 2: 각 쿼리마다 웹 검색 수행
            all_results = []
            for i, query in enumerate(expanded_queries, 1):
                logger.info(f'   [{i}] "{query}" 검색 중...')
                results = await self.searcher.search(query, max_results=3)
                all_results.extend(results)
            
            # Step 3: 중복 제거 (제목 기반)
            seen_titles = set()
            unique_results = []
            for result in all_results:
                title = result.get('title', '')
                if title and title not in seen_titles:
                    seen_titles.add(title)
                    unique_results.append(result)
            
            logger.info(f'✅ 웹 검색 확장 완료: {len(unique_results)}개 고유 결과')
            
            return unique_results
            
        except Exception as e:
            logger.error(f'❌ 웹 검색 확장 실패: {str(e)}')
            return []
    
    async def fuse_with_rag_results(
        self,
        rag_results: List[Dict[str, Any]],
        user_query: str,
        enable_expansion: bool = False
    ) -> List[Dict[str, Any]]:
        """
        RAG 결과와 웹 검색 결과를 통합 (RRF)
        
        Args:
            rag_results: RAG 검색 결과
            user_query: 사용자 질문
            enable_expansion: Query Expansion 활성화 여부
        
        Returns:
            통합된 결과 리스트
        """
        try:
            # 웹 검색 수행
            if enable_expansion:
                web_results = await self.search_with_expansion(user_query)
            else:
                web_query = await self.query_generator.generate_query(user_query)
                web_results = await self.searcher.search(web_query, max_results=5)
            
            # RAG와 웹 결과 통합
            if web_results:
                fused = WebSearchResultFusion.fuse_with_rrf(
                    rag_results=rag_results,
                    web_results=web_results,
                    k=10,
                    rag_weight=0.6,
                    web_weight=0.4
                )
                logger.info(f'🔀 RAG + Web 통합 완료: {len(fused)}개 결과')
                return fused
            else:
                logger.info('⚠️ 웹 검색 결과 없음 → RAG 결과만 반환')
                return rag_results
                
        except Exception as e:
            logger.error(f'❌ Result Fusion 실패: {str(e)}')
            return rag_results


# ==================== 싱글톤 인스턴스 ====================

_web_search_service: Optional[WebSearchService] = None

def get_web_search_service() -> WebSearchService:
    """WebSearchService 싱글톤 반환"""
    global _web_search_service
    
    if _web_search_service is None:
        _web_search_service = WebSearchService()
        logger.info('✅ WebSearchService 초기화 완료')
    
    return _web_search_service
