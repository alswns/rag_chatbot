"""
조건부 웹 검색 모듈 (Conditional Web Search with Sanitization)

보안 최우선:
- 내부 정보(프로젝트명, 변수명, IP 등) 유출 방지
- LLM 기반 검색 필요성 판단
- 세탁된 검색어만 외부로 전송

효율성:
- 내부 검색 결과가 충분하면 웹 검색 스킵
- max_tokens=50으로 빠른 판단
"""

import logging
import os
from typing import Optional, Tuple, List, Dict, Any
import openai
import asyncio

logger = logging.getLogger(__name__)

# DuckDuckGo 검색 라이브러리
try:
    from duckduckgo_search import AsyncDDGS
except ImportError:
    AsyncDDGS = None
    logger.warning('⚠️ duckduckgo-search 미설치 - 웹 검색 비활성화')


class SearchDecisionMaker:
    """
    ✅ 내부 RAG 검색 결과 분석 및 웹 검색 필요성 판단
    
    보안 철칙:
    - 내부 문서 내용을 외부로 전송하지 않음
    - 검색어에서 민감 정보 제거 (프로젝트명, 변수명, IP, 경로 등)
    """
    
    # 🔒 검색 판단 프롬프트 (보안 강화)
    DECISION_PROMPT = """당신은 **보안 검색 어시스턴트**입니다.

## 작업
사용자 질문과 내부 검색 결과를 분석하여 **웹 검색이 필요한지** 판단하세요.

## ⚠️ 중요: 비판적 평가 원칙
1. **구체적 정답이 없으면 검색하라**
   - 단순히 키워드가 겹친다고 답변 가능하다고 판단하지 마세요.
   - 예: "파이썬 버전"이라는 단어만 있고 **구체적 숫자(3.11, 3.12)**가 없으면 불충분!

2. **내부 지식을 사용하지 마라**
   - 오직 제공된 [Context] 안에 정보가 있는지**만** 확인하세요.
   - LLM의 사전 학습 지식으로 추측하지 마세요.

3. **최신 정보는 웹 검색 필수**
   - "최신", "현재", "2025년", "최근" 등의 시간성 표현이 있으면 즉시 검색.

## 판단 기준
1. **내부 문서가 충분함** (웹 검색 불필요)
   - 질문에 **직접적으로** 답할 수 있는 구체적 정보가 있음
   - 문서에 상세한 설명/코드/가이드 포함
   - 유사도가 **0.7 이상**이고 관련성이 명확함
   → 반환: "NO_SEARCH"

2. **내부 문서가 불충분함** (웹 검색 필요)
   - 문서가 아예 없거나 관련성이 낮음
   - 일반적인 기술 질문/개념 설명이 필요함
   - 최신 정보가 필요함 (버전, 업데이트 등)
   - **부분적 정보**만 있음 (예: 개념 언급만 있고 구체적 방법 없음)
   → 반환: 세탁된 영어 검색어

## 🔒 보안 규칙 (절대 준수)
웹 검색어에 다음을 **절대 포함하지 마세요**:
- 프로젝트명 (예: "rag_chatbot", "my-project")
- 파일/경로 (예: "/src/main.py", "config.json")
- 변수/함수명 (예: "get_user_data", "API_KEY")
- IP/도메인 (예: "192.168.1.1", "internal.company.com")
- 조직명 (예: "우리팀", "회사")

**일반적인 기술 용어만 사용하세요!**

## 출력 형식
- 웹 검색 불필요: "NO_SEARCH"
- 웹 검색 필요: "영어 검색어" (3-5 단어, 일반 기술 용어만)

## 사용자 질문
{user_query}

## 내부 검색 결과 요약
{internal_summary}

## 당신의 판단 (한 줄로):"""

    def __init__(self, vllm_api_url: str = None):
        """
        Args:
            vllm_api_url: vLLM API 엔드포인트 (None이면 환경변수 사용)
        """
        self.vllm_api_url = vllm_api_url or os.getenv('VLLM_API_URL', 'http://localhost:8000/v1')
        self.client = None
        
    def _get_client(self) -> openai.OpenAI:
        """vLLM 클라이언트 반환 (lazy initialization)"""
        if self.client is None:
            self.client = openai.OpenAI(
                api_key='sk-not-needed',
                base_url=self.vllm_api_url,
                timeout=30.0  # 빠른 판단용
            )
        return self.client
    
    def _summarize_internal_results(self, documents: List[Dict[str, Any]], max_docs: int = 3) -> str:
        """
        내부 검색 결과 요약 (보안 고려)
        
        Args:
            documents: 검색된 문서 리스트
            max_docs: 요약할 최대 문서 수
        
        Returns:
            요약 텍스트 (또는 "검색 결과 없음")
        """
        if not documents:
            return "검색 결과 없음 - 내부 문서에서 관련 정보를 찾지 못했습니다."
        
        summaries = []
        for i, doc in enumerate(documents[:max_docs], 1):
            title = doc.get('metadata', {}).get('title', 'Untitled')
            content_preview = doc.get('content', '')[:200]  # 처음 200자만
            score = doc.get('score', 0.0)
            
            summaries.append(f"[문서 {i}] 제목: {title}, 유사도: {score:.2f}, 내용 일부: {content_preview}...")
        
        return '\n'.join(summaries)
    
    async def sanitize_query(self, user_query: str) -> str:
        """
        검색어 세탁 (보안 정보 제거)
        
        Args:
            user_query: 사용자 질문
        
        Returns:
            세탁된 검색어
        """
        try:
            client = self._get_client()
            sanitize_prompt = f"""다음 질문에서 프로젝트명, 변수명, 파일명 등의 민감한 내부 정보를 제거하고,
일반적인 기술 용어로만 구성된 **영어 검색어**(3-5 단어)를 생성하세요.

사용자 질문: {user_query}

세탁된 검색어 (영어, 3-5 단어):"""
            
            response = client.chat.completions.create(
                model=os.getenv('LLM_MODEL_ID', 'DeepSeek-R1'),
                messages=[
                    {'role': 'system', 'content': '보안 검색 어시스턴트입니다.'},
                    {'role': 'user', 'content': sanitize_prompt}
                ],
                temperature=0.1,
                max_tokens=30,
                stream=False
            )
            
            sanitized = response.choices[0].message.content.strip()
            sanitized = sanitized.replace('"', '').replace("'", "").strip()
            logger.info(f'🧼 검색어 세탁: "{user_query}" → "{sanitized}"')
            return sanitized
            
        except Exception as e:
            logger.error(f'❌ 검색어 세탁 실패: {str(e)}')
            # Fallback: 간단한 키워드 추출
            return ' '.join(user_query.split()[:5])
    
    async def decide_and_sanitize(
        self,
        user_query: str,
        internal_documents: List[Dict[str, Any]],
        force_search: bool = False
    ) -> str:
        """
        웹 검색 필요성 판단 및 검색어 세탁
        
        Args:
            user_query: 사용자 질문
            internal_documents: 내부 RAG 검색 결과
            force_search: True일 경우 판단 스킵하고 무조건 검색어 생성
        
        Returns:
            'NO_SEARCH' 또는 세탁된 검색어
        """
        try:
            # Force Search: 판단 스킵
            if force_search:
                logger.info('🔍 강제 검색 모드: 판단 스킵 → 즉시 검색어 생성')
                return await self.sanitize_query(user_query)
            
            # 내부 검색 결과 요약
            internal_summary = self._summarize_internal_results(internal_documents)
            
            # 프롬프트 구성
            prompt = self.DECISION_PROMPT.format(
                user_query=user_query,
                internal_summary=internal_summary
            )
            
            # LLM 호출 (빠른 판단)
            client = self._get_client()
            response = client.chat.completions.create(
                model=os.getenv('LLM_MODEL_ID', 'DeepSeek-R1'),
                messages=[
                    {'role': 'system', 'content': '보안 검색 어시스턴트입니다.'},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.1,  # 일관성 있는 판단
                max_tokens=50,  # 빠른 응답
                stream=False
            )
            
            decision = response.choices[0].message.content.strip()
            
            # 검증 및 정제
            if 'NO_SEARCH' in decision.upper():
                logger.info('🔍 웹 검색 판단: 내부 문서로 충분 → 웹 검색 스킵')
                return 'NO_SEARCH'
            else:
                # 검색어 추출 (따옴표, 괄호 제거)
                sanitized = decision.replace('"', '').replace("'", "").strip()
                logger.info(f'🔍 웹 검색 판단: 필요 → 세탁된 검색어: "{sanitized}"')
                return sanitized
                
        except Exception as e:
            logger.error(f'❌ 검색 판단 실패: {str(e)} → 안전하게 웹 검색 스킵')
            return 'NO_SEARCH'
    
    async def search(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        DuckDuckGo 웹 검색 수행
        
        Args:
            query: 검색어 (이미 세탁됨)
            max_results: 최대 결과 수
        
        Returns:
            검색 결과 리스트 [{'title': ..., 'body': ..., 'href': ...}, ...]
        """
        if AsyncDDGS is None:
            logger.error('❌ duckduckgo-search 미설치 - 웹 검색 불가')
            return []
        
        try:
            logger.info(f'🌐 DuckDuckGo 검색 시작: "{query}"')
            
            async with AsyncDDGS() as ddgs:
                results = []
                async for result in ddgs.text(query, max_results=max_results):
                    results.append({
                        'title': result.get('title', ''),
                        'body': result.get('body', ''),
                        'href': result.get('href', '')
                    })
                
                logger.info(f'✅ 웹 검색 완료: {len(results)}개 결과')
                return results
                
        except Exception as e:
            logger.error(f'❌ DuckDuckGo 검색 실패: {str(e)}')
            return []
    
    def format_web_results(self, results: List[Dict[str, Any]]) -> str:
        """
        웹 검색 결과를 LLM 프롬프트용으로 포맷
        
        Args:
            results: 검색 결과 리스트
        
        Returns:
            포맷된 텍스트
        """
        if not results:
            return ""
        
        lines = ["### 웹 검색 결과 (외부 정보)\n"]
        for i, result in enumerate(results[:3], 1):  # 상위 3개만
            title = result.get('title', 'No title')
            body = result.get('body', '')[:300]  # 300자 제한
            url = result.get('href', '')
            
            lines.append(f"**[웹 검색 {i}] {title}**")
            lines.append(f"내용: {body}...")
            lines.append(f"출처: {url}\n")
        
        return '\n'.join(lines)


# 싱글톤 인스턴스
_search_decision_maker: Optional[SearchDecisionMaker] = None

def get_search_decision_maker() -> SearchDecisionMaker:
    """SearchDecisionMaker 싱글톤 반환"""
    global _search_decision_maker
    if _search_decision_maker is None:
        _search_decision_maker = SearchDecisionMaker()
    return _search_decision_maker
