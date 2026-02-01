"""
웹 검색 모듈 (Refactored - Separation of Concerns)

판단 로직과 생성 로직을 분리하여 LLM 성능 한계 극복:
1. SearchDecisionMaker: Binary Decision (YES/NO)
2. QueryGenerator: Clean Query Generation
3. DuckDuckGoSearcher: Web Search Execution
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
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

class QueryGenerator:
    """검색 엔진용 쿼리 생성 (Clean Output)"""
    
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
            'keywords:', '영어 검색어:', '세탁된 검색어:'
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


# ==================== 4. Unified Interface ====================

class WebSearchService:
    """웹 검색 통합 인터페이스"""
    
    def __init__(self, vllm_api_url: str = None):
        self.decision_maker = SearchDecisionMaker(vllm_api_url)
        self.query_generator = QueryGenerator(vllm_api_url)
        self.searcher = DuckDuckGoSearcher()
    
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
            
            # Step 2: 검색 쿼리 생성
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


# ==================== 싱글톤 인스턴스 ====================

_web_search_service: Optional[WebSearchService] = None

def get_web_search_service() -> WebSearchService:
    """WebSearchService 싱글톤 반환"""
    global _web_search_service
    
    if _web_search_service is None:
        _web_search_service = WebSearchService()
        logger.info('✅ WebSearchService 초기화 완료')
    
    return _web_search_service
