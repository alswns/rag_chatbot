"""
ScalableIntentRouter - 확장 가능한 의도 분류 라우터

Multi-Source RAG 시스템을 위한 Intent & Domain 분류기
- 현재: Notion 지식 검색 활성화
- 예정: GitHub 코드 검색 연동

분류 카테고리:
1. search_knowledge: 문서/지식 검색 (Notion, Wiki)
2. search_code: 코드/구현 검색 (GitHub) [Future]
3. summary: 요약 요청
4. chat: 일상 대화 (검색 불필요)
"""

import logging
import re
import json
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class Intent(str, Enum):
    """검색 의도 유형"""
    SEARCH_KNOWLEDGE = "search_knowledge"  # 문서/지식 검색
    SEARCH_CODE = "search_code"            # 코드/구현 검색 [Future]
    SUMMARY = "summary"                     # 요약 요청
    CHAT = "chat"                          # 일상 대화


class Domain(str, Enum):
    """검색 대상 도메인"""
    NOTION = "notion"      # Notion 문서
    GITHUB = "github"      # GitHub 코드 [Future]
    GENERAL = "general"    # 일반 (검색 불필요 또는 혼합)


@dataclass
class RouterResult:
    """라우팅 결과 데이터 클래스"""
    intent: str
    domain: str
    keywords: List[str]
    reasoning: str
    confidence: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent,
            "domain": self.domain,
            "keywords": self.keywords,
            "reasoning": self.reasoning,
            "confidence": self.confidence
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class ScalableIntentRouter:
    """
    확장 가능한 의도 분류 라우터
    
    2단계 분류 전략:
    1. 규칙 기반 빠른 분류 (키워드 매칭)
    2. LLM 기반 정밀 분류 (필요 시)
    
    확장 포인트:
    - 새로운 도메인 추가: Domain Enum + 패턴 추가
    - 새로운 의도 추가: Intent Enum + 패턴 추가
    """
    
    # =========================================================================
    # 패턴 정의 (확장 시 여기에 추가)
    # =========================================================================
    
    # 코드 검색 키워드 패턴 (GitHub 연동 시 활성화)
    CODE_PATTERNS = {
        'keywords': [
            # 함수/클래스 관련
            '함수', '클래스', 'class', 'function', 'def ', 'method',
            '메서드', '구현', '코드', 'code', '소스',
            # 파일 관련
            '파일', 'file', '.py', '.js', '.ts', '.java', '.go',
            'dockerfile', 'docker-compose', 'requirements',
            # Git 관련
            'commit', '커밋', 'branch', '브랜치', 'merge', 'pr', 'pull request',
            # 에러/디버깅
            'error', '에러', 'exception', 'traceback', 'bug', '버그',
            'debug', '디버그', 'log', '로그',
            # 기술 스택
            'api', 'endpoint', 'route', '라우트', 'import', 'package',
        ],
        'patterns': [
            r'[a-z_]+\(\)',           # function_name()
            r'[A-Z][a-zA-Z]+Class',   # ClassName
            r'def\s+\w+',             # def function
            r'class\s+\w+',           # class Name
            r'\w+\.py',               # file.py
            r'`[^`]+`',               # `code`
            r'[A-Z][a-z]+Error',      # SomeError
        ]
    }
    
    # 지식 검색 키워드 패턴 (Notion 등)
    KNOWLEDGE_PATTERNS = {
        'keywords': [
            # 규정/절차
            '규정', '규칙', '절차', '방법', '기준', '정책', '매뉴얼',
            '가이드', '안내', '지침', '프로세스',
            # 정보 요청
            '뭐야', '뭔가요', '알려줘', '설명', '정의', '개념',
            '이유', '왜', '어떻게', '언제', '어디',
            # 업무 관련
            '신청', '승인', '결재', '보고', '일정', '기한', '마감',
            '담당', '연락처', '문의',
            # 장학/인사 (도메인 특화)
            '장학', '급여', '시급', '근로', '채용', '인사', '휴가',
        ],
        'patterns': [
            r'.+이?란\??',            # ~란?, ~이란?
            r'.+[이가]\s*뭐',         # ~이 뭐, ~가 뭐
            r'어떻게\s*.+[하해]',     # 어떻게 ~하
            r'.+방법',                # ~방법
        ]
    }
    
    # 요약 요청 패턴
    SUMMARY_PATTERNS = {
        'keywords': [
            '요약', '정리', '줄여', '짧게', '간단히', '핵심',
            'summary', 'summarize', 'tldr', 'tl;dr',
            '무슨 내용', '어떤 내용', '대략', '개요',
        ],
        'patterns': [
            r'.+요약해\s*줘',
            r'.+정리해\s*줘',
            r'.+설명해\s*줘',
        ]
    }
    
    # 일상 대화 패턴
    CHAT_PATTERNS = {
        'keywords': [
            # 인사
            '안녕', '하이', 'hi', 'hello', '반가워', '수고',
            # 감사
            '고마워', '감사', 'thanks', 'thank you', 'thx',
            # 잡담
            '뭐해', '심심', '재밌', '웃겨', 'ㅋㅋ', 'ㅎㅎ',
            # 시스템
            '넌 뭐야', '누구야', '이름이 뭐', '뭘 할 수 있',
        ],
        'patterns': [
            r'^안녕',
            r'^ㅎ+$',
            r'^ㅋ+$',
        ]
    }
    
    # 도메인 힌트 패턴
    DOMAIN_HINTS = {
        Domain.NOTION: [
            'notion', '노션', '문서', '페이지', '워크스페이스',
            '매뉴얼', '가이드', '규정', '정책', '업무',
        ],
        Domain.GITHUB: [
            'github', '깃허브', 'git', '깃', 'repo', '저장소',
            'commit', 'branch', 'pull request', 'pr', 'issue',
            '코드', 'code', '소스', 'source',
        ]
    }
    
    def __init__(
        self,
        llm_client=None,
        use_llm_fallback: bool = True,
        active_domains: List[str] = None
    ):
        """
        Args:
            llm_client: LLM API 클라이언트 (OpenAI 호환)
            use_llm_fallback: 규칙 기반 분류 실패 시 LLM 사용 여부
            active_domains: 활성화된 도메인 목록 (기본: ['notion'])
        """
        self.llm_client = llm_client
        self.use_llm_fallback = use_llm_fallback
        self.active_domains = active_domains or ['notion']
        
        logger.info(f'✅ ScalableIntentRouter 초기화')
        logger.info(f'   - 활성 도메인: {self.active_domains}')
        logger.info(f'   - LLM Fallback: {use_llm_fallback}')
    
    def route(self, query: str) -> RouterResult:
        """
        쿼리를 분석하여 Intent와 Domain 결정
        
        Args:
            query: 사용자 질문
        
        Returns:
            RouterResult 객체
        """
        query_lower = query.lower().strip()
        
        logger.debug(f'🔍 라우팅 시작: "{query[:50]}..."')
        
        # =====================================================
        # Step 1: 규칙 기반 빠른 분류
        # =====================================================
        result = self._rule_based_classify(query, query_lower)
        
        if result.confidence >= 0.7:
            logger.info(f'✓ 규칙 기반 분류: intent={result.intent}, domain={result.domain}')
            return result
        
        # =====================================================
        # Step 2: LLM 기반 정밀 분류 (필요 시)
        # =====================================================
        if self.use_llm_fallback and self.llm_client:
            try:
                llm_result = self._llm_classify(query)
                if llm_result:
                    logger.info(f'✓ LLM 분류: intent={llm_result.intent}, domain={llm_result.domain}')
                    return llm_result
            except Exception as e:
                logger.warning(f'LLM 분류 실패: {str(e)}')
        
        # =====================================================
        # Step 3: 기본값 반환
        # =====================================================
        # 확신이 낮으면 기본적으로 지식 검색으로 처리
        if result.confidence < 0.3:
            result = RouterResult(
                intent=Intent.SEARCH_KNOWLEDGE.value,
                domain=Domain.NOTION.value if 'notion' in self.active_domains else Domain.GENERAL.value,
                keywords=self._extract_keywords(query),
                reasoning="기본 분류: 지식 검색으로 처리",
                confidence=0.5
            )
        
        logger.info(f'✓ 최종 분류: intent={result.intent}, domain={result.domain}')
        return result
    
    def _rule_based_classify(self, query: str, query_lower: str) -> RouterResult:
        """규칙 기반 분류"""
        scores = {
            Intent.CHAT: 0.0,
            Intent.SUMMARY: 0.0,
            Intent.SEARCH_CODE: 0.0,
            Intent.SEARCH_KNOWLEDGE: 0.0,
        }
        
        # 1. 일상 대화 체크 (최우선)
        chat_score = self._calculate_pattern_score(query_lower, self.CHAT_PATTERNS)
        scores[Intent.CHAT] = chat_score
        
        if chat_score >= 0.8:
            return RouterResult(
                intent=Intent.CHAT.value,
                domain=Domain.GENERAL.value,
                keywords=[],
                reasoning="인사/잡담 패턴 감지",
                confidence=chat_score
            )
        
        # 2. 요약 요청 체크
        summary_score = self._calculate_pattern_score(query_lower, self.SUMMARY_PATTERNS)
        scores[Intent.SUMMARY] = summary_score
        
        # 3. 코드 검색 체크
        code_score = self._calculate_pattern_score(query_lower, self.CODE_PATTERNS)
        scores[Intent.SEARCH_CODE] = code_score
        
        # 4. 지식 검색 체크
        knowledge_score = self._calculate_pattern_score(query_lower, self.KNOWLEDGE_PATTERNS)
        scores[Intent.SEARCH_KNOWLEDGE] = knowledge_score
        
        # 5. 도메인 결정
        domain = self._determine_domain(query_lower)
        
        # 6. 최고 점수 Intent 선택
        best_intent = max(scores, key=scores.get)
        best_score = scores[best_intent]
        
        # 7. 키워드 추출
        keywords = self._extract_keywords(query)
        
        # 8. 활성 도메인 체크
        if best_intent == Intent.SEARCH_CODE and 'github' not in self.active_domains:
            # GitHub 미활성화 시 지식 검색으로 전환
            best_intent = Intent.SEARCH_KNOWLEDGE
            domain = Domain.NOTION.value if 'notion' in self.active_domains else Domain.GENERAL.value
            reasoning = "코드 검색 요청이나 GitHub 미활성화 → 지식 검색으로 전환"
        elif best_intent == Intent.SEARCH_KNOWLEDGE:
            domain = Domain.NOTION.value if 'notion' in self.active_domains else domain
            reasoning = f"지식 검색 패턴 감지 (score={best_score:.2f})"
        elif best_intent == Intent.SUMMARY:
            reasoning = "요약 요청 패턴 감지"
        elif best_intent == Intent.CHAT:
            reasoning = "일상 대화 패턴 감지"
        else:
            reasoning = f"Intent={best_intent.value}, Score={best_score:.2f}"
        
        return RouterResult(
            intent=best_intent.value,
            domain=domain,
            keywords=keywords,
            reasoning=reasoning,
            confidence=best_score
        )
    
    def _calculate_pattern_score(
        self,
        query_lower: str,
        patterns: Dict[str, List]
    ) -> float:
        """패턴 매칭 점수 계산"""
        score = 0.0
        max_score = 1.0
        
        # 키워드 매칭
        keywords = patterns.get('keywords', [])
        keyword_matches = sum(1 for kw in keywords if kw in query_lower)
        if keyword_matches > 0:
            score += min(keyword_matches * 0.2, 0.6)
        
        # 정규식 패턴 매칭
        regex_patterns = patterns.get('patterns', [])
        for pattern in regex_patterns:
            if re.search(pattern, query_lower):
                score += 0.3
                break
        
        return min(score, max_score)
    
    def _determine_domain(self, query_lower: str) -> str:
        """도메인 결정"""
        domain_scores = {}
        
        for domain, hints in self.DOMAIN_HINTS.items():
            score = sum(1 for hint in hints if hint in query_lower)
            domain_scores[domain] = score
        
        if not domain_scores:
            return Domain.GENERAL.value
        
        best_domain = max(domain_scores, key=domain_scores.get)
        
        # 활성 도메인 체크
        if best_domain.value not in self.active_domains:
            # 활성화된 도메인 중 선택
            for d in self.active_domains:
                return d
            return Domain.GENERAL.value
        
        return best_domain.value
    
    def _extract_keywords(self, query: str) -> List[str]:
        """쿼리에서 핵심 키워드 추출"""
        # 불용어 정의
        stopwords = {
            '은', '는', '이', '가', '을', '를', '의', '에', '에서', '와', '과',
            '하고', '하는', '하면', '해서', '해줘', '해주세요', '알려줘', '알려주세요',
            '뭐야', '뭔가요', '있어', '없어', '있나요', '없나요',
            '어떻게', '무엇', '언제', '어디', '누가', '왜',
            '좀', '그', '저', '이', '것', '수', '때', '등',
        }
        
        # 특수문자 제거 및 토큰화
        tokens = re.findall(r'[가-힣a-zA-Z0-9_]+', query)
        
        # 불용어 제거 및 길이 필터
        keywords = [
            token for token in tokens
            if token not in stopwords and len(token) >= 2
        ]
        
        # 중복 제거 및 상위 5개
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw.lower() not in seen:
                seen.add(kw.lower())
                unique_keywords.append(kw)
        
        return unique_keywords[:5]
    
    def _llm_classify(self, query: str) -> Optional[RouterResult]:
        """LLM 기반 분류 (OpenAI 호환 API)"""
        if not self.llm_client:
            return None
        
        system_prompt = """당신은 사용자 질문을 분석하여 의도와 검색 대상을 분류하는 AI입니다.

분류 카테고리:
1. search_knowledge: 문서/지식/규정/절차에 대한 질문
2. search_code: 코드/함수/파일/구현에 대한 질문
3. summary: 요약 요청
4. chat: 인사/잡담

도메인:
- notion: 문서, 매뉴얼, 규정, 업무 관련
- github: 코드, 소스, 저장소 관련
- general: 일반 대화 또는 혼합

반드시 아래 JSON 형식으로만 응답하세요:
{"intent": "...", "domain": "...", "keywords": [...], "reasoning": "..."}"""

        user_prompt = f"질문: {query}"
        
        try:
            response = self.llm_client.chat.completions.create(
                model="DeepSeek-R1-Distill-8B",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            content = response.choices[0].message.content
            
            # JSON 파싱
            json_match = re.search(r'\{[^}]+\}', content)
            if json_match:
                data = json.loads(json_match.group())
                return RouterResult(
                    intent=data.get('intent', Intent.SEARCH_KNOWLEDGE.value),
                    domain=data.get('domain', Domain.NOTION.value),
                    keywords=data.get('keywords', []),
                    reasoning=data.get('reasoning', 'LLM 분류'),
                    confidence=0.85
                )
        except Exception as e:
            logger.warning(f'LLM 분류 파싱 실패: {str(e)}')
        
        return None
    
    def add_domain(self, domain: str) -> None:
        """새로운 도메인 활성화"""
        if domain not in self.active_domains:
            self.active_domains.append(domain)
            logger.info(f'✓ 도메인 추가: {domain}')
    
    def remove_domain(self, domain: str) -> None:
        """도메인 비활성화"""
        if domain in self.active_domains:
            self.active_domains.remove(domain)
            logger.info(f'✓ 도메인 제거: {domain}')


# =========================================================================
# 편의 함수 및 싱글톤
# =========================================================================

_router_instance: Optional[ScalableIntentRouter] = None


def get_intent_router(
    llm_client=None,
    active_domains: List[str] = None
) -> ScalableIntentRouter:
    """IntentRouter 싱글톤 반환"""
    global _router_instance
    
    if _router_instance is None:
        _router_instance = ScalableIntentRouter(
            llm_client=llm_client,
            active_domains=active_domains or ['notion']
        )
    
    return _router_instance


def route_query(query: str, llm_client=None) -> RouterResult:
    """쿼리 라우팅 헬퍼 함수"""
    router = get_intent_router(llm_client=llm_client)
    return router.route(query)


# =========================================================================
# 테스트용 예제
# =========================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    
    router = ScalableIntentRouter(active_domains=['notion'])
    
    test_queries = [
        # search_knowledge
        "근로장학생 시급 규정이 뭐야?",
        "RAG 아키텍처 설계 의도가 뭐야?",
        "장학금 신청 기간 알려줘.",
        "채용 절차가 어떻게 돼?",
        
        # search_code (GitHub 미활성화 → 지식 검색으로 전환)
        "로그인 함수 어디에 정의되어 있어?",
        "도커파일 설정 보여줘.",
        "Retriever 클래스 코드 보여줘.",
        
        # summary
        "이 문서 요약해줘.",
        "핵심만 정리해줘.",
        
        # chat
        "안녕하세요!",
        "고마워~",
        "ㅋㅋㅋ",
    ]
    
    print("\n" + "=" * 60)
    print(" ScalableIntentRouter 테스트")
    print("=" * 60)
    
    for query in test_queries:
        result = router.route(query)
        print(f"\n📝 Query: {query}")
        print(f"   → Intent: {result.intent}")
        print(f"   → Domain: {result.domain}")
        print(f"   → Keywords: {result.keywords}")
        print(f"   → Confidence: {result.confidence:.2f}")
        print(f"   → Reasoning: {result.reasoning}")
