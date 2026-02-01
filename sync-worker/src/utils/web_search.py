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
    
    # 🎯 Answerability Check Prompt (정답 충족성 중심)
    DECISION_PROMPT = """당신은 **답변 가능성 판단기(Answerability Judge)**입니다.

## 작업
사용자 질문과 [내부 문서 요약]을 대조하여, **질문에 대한 구체적인 정답**이 있는지 판단하세요.

## 판단 기준 (Exact Answer Rule)

### 1️⃣ **직접적 정답 유무 확인**
- ❌ **나쁨 예:**
  - 질문: "최신 리액트 버전이 뭐야?"
  - 문서: "리액트 16 사용 중" (주제는 같지만 '최신' 정보 없음)
  - 판단: **SEARCH** → `latest react version 2026`

- ❌ **나쁨 예:**
  - 질문: "S3 버킷 생성 방법 알려줘"
  - 문서: "S3 버킷 이름은 'my-bucket'이다" (설정값만 있고 '생성 방법' 없음)
  - 판단: **SEARCH** → `aws s3 create bucket tutorial`

- ✅ **좋은 예:**
  - 질문: "우리 프로젝트 서버 IP 뭐야?"
  - 문서: "Server IP: 10.0.0.1" (직접적 정답 있음)
  - 판단: **NO_SEARCH**

### 2️⃣ **외부 지식 규칙 (External Knowledge Rule)**
다음 유형의 질문은 무조건 웹 검색:
- "최신 뉴스", "현재 트렌드", "2026년 기준"
- 일반 상식/개념 설명 (예: "머신러닝이 뭐야?")
- 외부 라이브러리 사용법 (예: "FastAPI 비동기 처리 방법")
- 비교 분석 (예: "React vs Vue 장단점")

### 3️⃣ **키워드 겹침 함정 경계**
- 문서에 키워드만 언급되고 **구체적 수치/방법/코드**가 없으면 → **SEARCH**
- 예: "파이썬 버전" 단어만 있고 "3.11", "3.12" 같은 숫자 없음 → 불충분

## 🔒 보안 규칙 (검색어 세탁)
웹 검색어에 **절대 포함 금지**:
- 프로젝트명, 파일명, 변수명
- IP 주소, 내부 도메인
- 조직명, 팀명

**일반적인 기술 용어만 사용!**

## 출력 형식
- 내부 문서로 충분: `NO_SEARCH`
- 외부 검색 필요: `영어 검색어` (3-5 단어)

---

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
    
    def _summarize_internal_results(self, documents: List[Dict[str, Any]], max_docs: int = 3) -> tuple[str, float]:
        """
        내부 검색 결과 요약 (보안 고려)
        
        Args:
            documents: 검색된 문서 리스트
            max_docs: 요약할 최대 문서 수
        
        Returns:
            (요약 텍스트, 평균 유사도)
        """
        if not documents:
            return "검색 결과 없음 - 내부 문서에서 관련 정보를 찾지 못했습니다.", 0.0
        
        summaries = []
        total_score = 0.0
        
        for i, doc in enumerate(documents[:max_docs], 1):
            title = doc.get('metadata', {}).get('title', 'Untitled')
            content_preview = doc.get('content', '')[:200]  # 처음 200자만
            score = doc.get('score', 0.0)
            total_score += score
            
            summaries.append(f"[문서 {i}] 제목: {title}, 유사도: {score:.2f}, 내용 일부: {content_preview}...")
        
        avg_score = total_score / min(len(documents), max_docs)
        return '\n'.join(summaries), avg_score
    
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
            
            # ✅ 출력 정제 (오염 제거)
            # 1. 마크다운 코드 블럭 제거
            sanitized = sanitized.replace('```', '').strip()
            
            # 2. 인용부호 제거
            sanitized = sanitized.replace('"', '').replace("'", '').strip()
            
            # 3. 설명 문구 제거 ("세탁된 검색어:", "영어 검색어:" 등)
            for prefix in ['세탁된 검색어:', '영어 검색어:', 'Search query:', 'Query:']:
                if sanitized.lower().startswith(prefix.lower()):
                    sanitized = sanitized[len(prefix):].strip()
            
            # 4. 첫 줄만 추출 (여러 줄 응답 방지)
            sanitized = sanitized.split('\n')[0].strip()
            
            # 5. 최대 길이 제한 (10 단어)
            words = sanitized.split()
            if len(words) > 10:
                sanitized = ' '.join(words[:10])
            
            if sanitized:
                logger.info(f'🧼 검색어 세탁: "{user_query}" → "{sanitized}"')
                return sanitized
            else:
                # Fallback: 간단한 키워드 추출
                fallback = ' '.join(user_query.split()[:5])
                logger.warning(f'⚠️ 세탁 결과 비어있음, Fallback 사용: "{fallback}"')
                return fallback
            
        except Exception as e:
            logger.error(f'❌ 검색어 세탁 실패: {str(e)}')
            # Fallback: 간단한 키워드 추출 (5 단어)
            fallback = ' '.join(user_query.split()[:5])
            logger.info(f'🔄 Fallback 검색어 사용: "{fallback}"')
            return fallback
    
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
            internal_summary, avg_similarity = self._summarize_internal_results(internal_documents)
            
            # 프롬프트 구성 (유사도 정보 포함)
            enhanced_summary = f"""평균 유사도: {avg_similarity:.2f}

{internal_summary}"""
            
            prompt = self.DECISION_PROMPT.format(
                user_query=user_query,
                internal_summary=enhanced_summary
            )
            
            # LLM 호출 (빠른 판단)
            client = self._get_client()
            response = client.chat.completions.create(
                model=os.getenv('LLM_MODEL_ID', 'DeepSeek-R1'),
                messages=[
                    {'role': 'system', 'content': '답변 가능성 판단기입니다.'},
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0.0,  # ✅ 일관성 확보 (0.1 → 0.0)
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
                # ✅ Fix: async for → await + for 패턴
                search_results = await ddgs.text(query, max_results=max_results)
                
                results = []
                for result in search_results:
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
