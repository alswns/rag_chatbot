"""
🚀 RAG Inference API Server - FastAPI 기반 (Production-Ready)

3가지 핵심 역할:
1. 벡터 DB 검색 (ChromaDB → XML 포맷 Context)
2. 모델 관리 (vLLM 모델 정보 제공)
3. 질의응답 (DeepSeek-R1 추론)

Open WebUI와 완전 호환되는 OpenAI API 구현

✅ Production-Ready 기능:
- Async & Non-blocking I/O (ThreadPoolExecutor)
- Smart Token Management (동적 컨텍스트 윈도우)
- Structured Prompt Engineering (System Prompt에 Context 주입)
- Observability (time_logger 데코레이터)
"""

import os
import sys
import logging
import json
import re
import asyncio
import functools
from typing import Optional, List, Generator, Dict, Any, Callable
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time

# 모듈 경로
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
import uvicorn
import openai
import requests
from dotenv import load_dotenv

try:
    from db.vector_store import VectorStoreManager
except ImportError:
    from db import VectorStoreManager

try:
    from db.drill_down_retriever import GraphDrillDownRetriever, create_drill_down_retriever
except ImportError:
    GraphDrillDownRetriever = None
    create_drill_down_retriever = None

try:
    from utils.intent_router import ScalableIntentRouter, Intent, Domain, get_intent_router
except ImportError:
    ScalableIntentRouter = None
    Intent = None
    Domain = None
    get_intent_router = None

try:
    from processors.graph_rag import GraphRAGProcessor
except ImportError:
    GraphRAGProcessor = None

# 환경변수 로드
load_dotenv()

# 로깅 설정
log_level = os.getenv('LOG_LEVEL', 'INFO')
logging.basicConfig(
    level=getattr(logging, log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ==================== 4️⃣ Observability: time_logger 데코레이터 ====================

def time_logger(func: Callable) -> Callable:
    """
    ⏱️ 함수 실행 시간 측정 데코레이터
    
    동기/비동기 함수 모두 지원
    """
    @functools.wraps(func)
    async def async_wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.info(f'⏱️ [{func.__name__}] {elapsed:.4f}초')
    
    @functools.wraps(func)
    def sync_wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            elapsed = time.perf_counter() - start
            logger.info(f'⏱️ [{func.__name__}] {elapsed:.4f}초')
    
    if asyncio.iscoroutinefunction(func):
        return async_wrapper
    return sync_wrapper


# ==================== FastAPI 앱 ====================

app = FastAPI(
    title='RAG Inference API',
    description='Enterprise RAG - OpenAI Compatible (Production-Ready)',
    version='2.0.0'
)

# ==================== 설정 ====================

MODEL_NAME = os.getenv('LLM_MODEL_ID', 'DeepSeek-R1-Distill-8B')
LLM_BACKEND = os.getenv('LLM_BACKEND', 'vllm').lower()
VLLM_API_URL = os.getenv('VLLM_API_URL', 'http://localhost:8000/v1')
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
CHROMA_HOST = os.getenv('CHROMA_HOST', 'localhost')
CHROMA_PORT = int(os.getenv('CHROMA_PORT', '8000'))
GRAPH_PERSIST_PATH = os.getenv('GRAPH_PERSIST_PATH', './data/graph.pkl')
SEARCH_TOP_K = int(os.getenv('SEARCH_TOP_K', '5'))

# ✅ Reranking 설정
ENABLE_RERANKING = os.getenv('ENABLE_RERANKING', 'true').lower() == 'true'
RERANKER_MODEL = os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')
RERANKER_TOP_K = int(os.getenv('RERANKER_TOP_K', '50'))

# ✅ Token Management 설정
MAX_MODEL_LEN = int(os.getenv('MAX_MODEL_LEN', '8192'))
RESERVED_OUTPUT_TOKENS = int(os.getenv('RESERVED_OUTPUT_TOKENS', '1024'))
MAX_CONTEXT_TOKENS = MAX_MODEL_LEN - RESERVED_OUTPUT_TOKENS

# ✅ ThreadPool 설정 (Non-blocking I/O)
THREAD_POOL_SIZE = int(os.getenv('THREAD_POOL_SIZE', '4'))
executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)

logger.info(f'✅ 모델: {MODEL_NAME}')
logger.info(f'✅ LLM Backend: {LLM_BACKEND.upper()}')
logger.info(f'✅ Reranking: {"활성화" if ENABLE_RERANKING else "비활성화"} (모델: {RERANKER_MODEL if ENABLE_RERANKING else "N/A"})')
logger.info(f'✅ Token Limit: {MAX_MODEL_LEN} (Output 예약: {RESERVED_OUTPUT_TOKENS})')
logger.info(f'✅ ThreadPool: {THREAD_POOL_SIZE} workers')

# ==================== 글로벌 변수 ====================

vector_store: Optional[VectorStoreManager] = None
graph_processor: Optional['GraphRAGProcessor'] = None
drill_down_retriever: Optional['GraphDrillDownRetriever'] = None
intent_router: Optional['ScalableIntentRouter'] = None
semantic_router: Optional['SemanticIntentRouter'] = None  # Fix: Embedding-based Intent Router
vllm_client: Optional[openai.OpenAI] = None
available_models: List[Dict[str, Any]] = []


# ==================== 데이터 모델 ====================

class ChatMessage(BaseModel):
    role: str = Field(..., description="역할: system/user/assistant")
    content: str = Field(..., description="메시지 내용")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=MODEL_NAME)
    messages: List[ChatMessage]
    temperature: float = Field(default=0.1, ge=0.0, le=2.0) 
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=2048)
    stream: bool = Field(default=False)


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Dict]
    usage: Dict


# ==================== 2️⃣ Smart Token Management ====================

class TokenManager:
    """
    ✅ Smart Token Management
    
    - 토큰 수 추정 (한글 1.5, 영어 1.0 char/token)
    - 동적 컨텍스트 윈도우 관리
    - 과거 대화 히스토리 최적화
    """
    
    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        토큰 수 추정 (Qwen2.5 최적화)
        - 한글: ~2.0 chars/token (Qwen 토크나이저 특성 반영)
        - 영어/숫자: ~3.5 chars/token (vLLM 기준)
        - 특수문자/공백: ~1.5 chars/token
        """
        if not text:
            return 0
        
        # Fix: Qwen 토크나이저는 한글을 더 잘게 쪼개므로 가중치 상향
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z0-9]', text))
        other_chars = len(text) - korean_chars - english_chars
        
        # Qwen2.5 실측 기준 가중치
        korean_tokens = korean_chars / 2.0  # 1.2 → 2.0 상향
        english_tokens = english_chars / 3.5
        other_tokens = other_chars / 1.5
        
        total = int(korean_tokens + english_tokens + other_tokens)
        
        # 최소값 보장 (매우 짧은 텍스트)
        return max(total, len(text.split()) if text.strip() else 0)
    
    @staticmethod
    def estimate_messages_tokens(messages: List[Dict[str, str]]) -> int:
        """메시지 리스트의 총 토큰 수 추정"""
        total = 0
        for msg in messages:
            # role 토큰 (~4)
            total += 4
            # content 토큰
            total += TokenManager.estimate_tokens(msg.get('content', ''))
        return total
    
    @staticmethod
    def manage_context_window(
        system_prompt: str,
        context: str,
        current_query: str,
        history: List[ChatMessage],
        max_tokens: int = 8192 # vLLM 설정값
    ) -> List[Dict[str, str]]:
        """
        ✅ [완전판] 토큰 예산 내에서 출력 공간을 보장하며 히스토리 관리
        """
        # 1️⃣ 출력(답변) 공간 예약 (매우 중요!)
        # 모델이 답변을 생성할 공간을 최소 2,048 토큰은 남겨둬야 합니다.
        RESERVED_OUTPUT = 2048 
        effective_limit = max_tokens - RESERVED_OUTPUT 

        # 2️⃣ 시스템 프롬프트 조립
        if context:
            full_system = f"""{system_prompt}
            ---
            ### 제공된 참고 문서 (Reference Context)
            사용자의 질문에 답변하기 위해 아래의 문서들을 최우선으로 참고하세요.

            <context>
            {context}
            </context>
            ---
            """
        else:
            full_system = system_prompt

        # 3️⃣ 고정 비용 계산 (시스템 + 현재 질문)
        # 🌟 한글은 TokenManager.estimate_tokens에서 글자수 * 2.5~3배로 잡아야 안전합니다.
        system_tokens = TokenManager.estimate_tokens(full_system)
        query_tokens = TokenManager.estimate_tokens(current_query)
        
        fixed_tokens = system_tokens + query_tokens + 50 # 여유분 살짝 추가
        
        # 🚨 [Critical Fix] 컨텍스트+질문이 한계 초과 시 context 강제 트리밍
        if fixed_tokens > effective_limit:
            excess_tokens = fixed_tokens - effective_limit
            # Fix: Prevent OOM on L4 GPU - 초과 토큰 * 3글자 제거 (안전 여유)
            chars_to_remove = int(excess_tokens * 3)
            
            if context and len(context) > chars_to_remove:
                trimmed_context = context[:-chars_to_remove]
                logger.warning(f"⚠️ Context 강제 트리밍: {len(context)} → {len(trimmed_context)}자 (초과 {excess_tokens} 토큰)")
                
                # 재계산
                full_system = f"""{system_prompt}
            ---
            ### 제공된 참고 문서 (Reference Context)
            사용자의 질문에 답변하기 위해 아래의 문서들을 최우선으로 참고하세요.

            <context>
            {trimmed_context}
            </context>
            ---
            """
                system_tokens = TokenManager.estimate_tokens(full_system)
                fixed_tokens = system_tokens + query_tokens + 50
            else:
                logger.error(f"❌ Context 트리밍 실패: 여전히 초과 ({fixed_tokens} > {effective_limit})")
                fixed_tokens = effective_limit 

        remaining_tokens = effective_limit - fixed_tokens
        
        logger.debug(f'📊 토큰 예산: 가용={effective_limit}, 고정={fixed_tokens}, 히스토리용={remaining_tokens}')

        # 4️⃣ 히스토리 동적 포함 (최신 → 과거)
        selected_history = []
        history_tokens = 0
        
        # 🌟 히스토리가 너무 많으면(예: 52개) 모델이 혼란스러우니 최대 10개 정도로 제한하는 것을 권장합니다.
        max_history_count = 10 
        
        for msg in reversed(history[-max_history_count:]):
            msg_tokens = TokenManager.estimate_tokens(msg.content) + 10
            
            if history_tokens + msg_tokens <= remaining_tokens:
                selected_history.insert(0, {'role': msg.role, 'content': msg.content})
                history_tokens += msg_tokens
            else:
                break
        
        # 5️⃣ 최종 메시지 조립
        messages = [{'role': 'system', 'content': full_system}]
        messages.extend(selected_history)
        messages.append({'role': 'user', 'content': current_query})
        
        # 최종 확인 로그
        final_input_tokens = TokenManager.estimate_messages_tokens(messages)
        logger.info(f'✅ 토큰 관리 완료: 히스토리 {len(selected_history)}개, 예측 총합 ~{final_input_tokens} (한도 {max_tokens})')
        
        return messages


# ==================== Semantic Intent Router (Embedding-based, No LLM) ====================

class SemanticIntentRouter:
    """
    ✅ LLM 없이 Embedding 기반 의도 분류
    
    - 메모리 효율: 기존 embedding_service 재사용
    - 속도: cosine_similarity로 즉시 분류
    - GPU 절약: No LLM inference
    """
    
    # Intent별 Anchor 문장들
    INTENT_ANCHORS = {
        'coding': [
            "코드 작성해줘",
            "에러 수정해줘",
            "함수 구현",
            "디버깅 도와줘",
            "코드 리뷰",
            "버그 찾아줘"
        ],
        'explanation': [
            "설명해줘",
            "이게 무슨 뜻이야",
            "개념 알려줘",
            "어떻게 작동하는지",
            "원리가 뭐야",
            "차이점이 뭐야"
        ],
        'chat': [
            "안녕",
            "반가워",
            "너 누구니",
            "고마워",
            "괜찮아",
            "알겠어"
        ]
    }
    
    def __init__(self, embedding_service):
        """Fix: Prevent GPU memory waste - 기존 embedding_service 재사용"""
        self.embedding_service = embedding_service
        self.anchor_embeddings = {}
        self._build_anchors()
    
    def _build_anchors(self) -> None:
        """Anchor 문장들의 평균 임베딩 생성"""
        for intent, anchors in self.INTENT_ANCHORS.items():
            embeddings = self.embedding_service.encode(anchors)
            # 평균 임베딩 계산
            avg_embedding = embeddings.mean(axis=0)
            self.anchor_embeddings[intent] = avg_embedding
        
        logger.info(f"✅ SemanticIntentRouter 초기화: {len(self.anchor_embeddings)}개 Intent")
    
    def classify(self, query: str, threshold: float = 0.5) -> tuple[str, float]:
        """
        쿼리를 분류하여 Intent와 신뢰도 반환
        
        Returns:
            (intent, confidence) - 예: ('coding', 0.82)
        """
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        query_embedding = self.embedding_service.encode([query])[0]
        
        max_similarity = -1.0
        best_intent = 'explanation'  # default
        
        for intent, anchor_emb in self.anchor_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                anchor_emb.reshape(1, -1)
            )[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_intent = intent
        
        # threshold 이하면 'explanation' (안전한 기본값)
        if max_similarity < threshold:
            return 'explanation', max_similarity
        
        return best_intent, float(max_similarity)


# ==================== 1️⃣ 벡터 DB 검색 (RAG) - Async 전환 ====================

class VectorSearchManager:
    """
    ✅ [Production-Ready] Intent Router + Drill-Down Retriever 통합
    
    - Non-blocking I/O: ThreadPoolExecutor 사용
    - Observability: time_logger 데코레이터 적용
    - 웹 검색 판단용 결과 저장
    """
    
    # ✅ 마지막 검색 결과 저장 (웹 검색 판단용)
    _last_search_results: List[Dict[str, Any]] = []
    
    @staticmethod
    @time_logger
    def _search_sync(query: str, top_k: int = SEARCH_TOP_K) -> str:
        """
        ✅ [동기 버전] 실제 검색 로직
        
        ThreadPoolExecutor에서 실행됨
        """
        if vector_store is None:
            logger.warning('❌ Vector Store 미초기화')
            return ""
        
        try:
            # =====================================================
            # Step 1: Intent 분류 (의도 파악)
            # =====================================================
            intent_result = None
            if intent_router is not None:
                intent_result = intent_router.route(query)
                logger.info(f'🎯 Intent: {intent_result.intent} | Domain: {intent_result.domain} | Conf: {intent_result.confidence:.2f}')
                
                if intent_result.intent == 'chat':
                    logger.info('💬 일상 대화 감지 → 검색 스킵')
                    return ""
            
            # =====================================================
            # Step 2: Drill-Down 검색 (3단계 드릴다운)
            # =====================================================
            if drill_down_retriever is not None:
                logger.info(f'🔍 드릴다운 검색 시작: "{query[:50]}..." (top_k={top_k}, rerank={ENABLE_RERANKING})')
                
                search_k = RERANKER_TOP_K if ENABLE_RERANKING else top_k
                logger.info(f'🔍 drill_down_retriever.retrieve 호출: k={search_k}, use_reranking={ENABLE_RERANKING}')
                
                documents = drill_down_retriever.retrieve(
                    query=query,
                    k=search_k,
                    use_reranking=ENABLE_RERANKING
                )
                
                if documents:
                    documents = documents[:top_k]
                    logger.info(f'🔍 결과 자르기: {len(documents)}개 → {top_k}개')
                    
                    # ✅ 웹 검색 판단용 결과 저장
                    VectorSearchManager._last_search_results = [
                        {
                            'content': doc.content,
                            'metadata': doc.metadata,
                            'score': doc.score
                        } for doc in documents
                    ]
                    
                    context_xml = drill_down_retriever._format_as_xml(documents)
                    logger.info(f'✅ 드릴다운 검색 완료: {len(documents)}개 문서')
                    return context_xml
                else:
                    logger.info('⚠️ 드릴다운 검색 결과 없음 → Fallback')
                    VectorSearchManager._last_search_results = []
            
            # =====================================================
            # Step 3: Fallback - 기존 검색 방식
            # =====================================================
            logger.info(f'🔍 Fallback 검색: "{query[:50]}..."')
            
            context = vector_store.retrieve_context(
                query, 
                top_k=top_k, 
                use_hybrid=True,
                use_reranking=ENABLE_RERANKING
            )
            
            if not context:
                logger.info('⚠️ 검색 결과 없음')
                return ""
            
            logger.info(f'✅ 컨텍스트 생성 완료: {len(context)}자')
            return context
            
        except Exception as e:
            logger.error(f'❌ 검색 실패: {str(e)}', exc_info=True)
            return ""
    
    @staticmethod
    async def search(query: str, top_k: int = SEARCH_TOP_K) -> str:
        """
        ✅ [비동기 버전] Non-blocking 검색
        
        Event Loop를 차단하지 않고 ThreadPoolExecutor에서 실행
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            VectorSearchManager._search_sync,
            query,
            top_k
        )
    
    @staticmethod
    @time_logger
    def _search_with_intent_sync(query: str, top_k: int = SEARCH_TOP_K) -> Dict[str, Any]:
        """[동기 버전] Intent 포함 검색"""
        result = {
            'intent': None,
            'documents': [],
            'context': '',
            'search_type': 'fallback'
        }
        
        if vector_store is None:
            return result
        
        try:
            if intent_router is not None:
                intent_result = intent_router.route(query)
                result['intent'] = intent_result.to_dict()
                
                if intent_result.intent == 'chat':
                    result['search_type'] = 'skip'
                    return result
            
            if drill_down_retriever is not None:
                documents = drill_down_retriever.retrieve(
                    query, 
                    k=top_k,
                    use_reranking=ENABLE_RERANKING
                )
                if documents:
                    result['documents'] = [doc.to_dict() for doc in documents]
                    result['context'] = drill_down_retriever._format_as_xml(documents)
                    result['search_type'] = 'drill_down'
                    return result
            
            context = vector_store.retrieve_context(
                query, 
                top_k=top_k, 
                use_hybrid=True,
                use_reranking=ENABLE_RERANKING
            )
            result['context'] = context
            result['search_type'] = 'fallback'
            
            return result
            
        except Exception as e:
            logger.error(f'❌ 검색 실패: {str(e)}')
            return result
    
    @staticmethod
    async def search_with_intent(query: str, top_k: int = SEARCH_TOP_K) -> Dict[str, Any]:
        """[비동기 버전] Intent 포함 Non-blocking 검색"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            VectorSearchManager._search_with_intent_sync,
            query,
            top_k
        )


# ==================== 2️⃣ 모델 관리 ====================

class ModelManager:
    """vLLM 모델 정보 관리"""
    
    @staticmethod
    def get_models() -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록 반환"""
        global available_models
        
        if not available_models:
            try:
                logger.info('📡 vLLM 모델 정보 조회 중...')
                response = requests.get(
                    f'{VLLM_API_URL.replace("/v1", "")}/v1/models',
                    timeout=5
                )
                
                if response.status_code == 200:
                    data = response.json()
                    available_models = data.get('data', [])
                    logger.info(f'✅ vLLM에서 {len(available_models)}개 모델 로드')
                    return available_models
                    
            except Exception as e:
                logger.warning(f'⚠️  vLLM 모델 조회 실패: {str(e)}')
        
        return [{
            'id': MODEL_NAME,
            'object': 'model',
            'created': int(datetime.now().timestamp()),
            'owned_by': 'vllm',
            'permission': [{
                'id': 'modelperm-default',
                'object': 'model_permission',
                'created': int(datetime.now().timestamp()),
                'allow_create_engine': False,
                'allow_sampling': True,
                'allow_logprobs': False,
                'allow_search_indices': False,
                'allow_view': True,
                'allow_fine_tuning': False,
                'organization': '*',
                'group_id': None,
                'is_blocking': False
            }]
        }]


# ==================== 3️⃣ 질의응답 (LLM) ====================

class QuestionAnsweringManager:
    """DeepSeek-R1 기반 답변 생성 (Production-Ready)"""
    
    # ✅ Base Persona (공통)
    BASE_PERSONA = """당신은 소프트웨어 엔지니어링 분야의 **수석 엔지니어(Senior Technical Lead) 어시스턴트**입니다.
사용자의 질문에 대해 제공된 **Context(맥락)**를 바탕으로 가장 정확하고 기술적으로 깊이 있는 답변을 제공해야 합니다.

## 1. 답변 원칙 (Core Principles)
- **언어:** 설명은 **한국어(Korean)**로 하되, 기술 용어(Technical Terms), 라이브러리 명, 함수 이름 등은 **영어 원문**을 그대로 유지합니다. (예: "ROS Node를 실행합니다" (O) / "로스 노드를 실행합니다" (X))
- **근거 기반:** 반드시 `<context>` 태그 안에 제공된 정보에 기반하여 답변합니다. 문맥에 없는 내용은 사용자의 일반적인 질문이 아닌 이상 추측해서 답하지 말고, 정보가 부족하면 솔직하게 말합니다.
- **출처 표기:** 답변의 신뢰도를 높이기 위해, 가능한 경우 정보가 포함된 **파일 이름이나 문서 제목**을 인용합니다. (예: "`ik_solver.cpp` 파일의 로직에 따르면...")

## 2. 코드 작성 가이드 (Code Guidelines)
- **완전성:** 코드를 예시로 들 때 핵심 로직을 생략(`...`)하지 말고, 실행 가능한 형태로 작성합니다.
- **주석:** 코드의 주요 라인에는 **한국어 주석**을 달아 동작 원리를 설명합니다.
- **포맷:** Markdown Code Block(```language)을 반드시 사용합니다.

## 3. 답변 스타일 (Style)
- **두괄식:** 결론이나 핵심 해결책을 먼저 제시하고, 그 뒤에 상세 설명이나 근거를 덧붙입니다.
- **구조화:** 긴 설명이 필요할 경우 번호 매기기(1., 2.)나 불렛 포인트(-)를 사용하여 가독성을 높입니다.
- **전문성:** 초보자용 비유보다는 엔지니어 간의 대화처럼 명확하고 직관적인 기술 용어를 사용합니다."""
    
    # ✅ Intent별 동적 지침 (Dynamic Persona Injection)
    INTENT_INSTRUCTIONS = {
        'coding': """

## 🎯 추가 지침 (코드 작성 모드)
- **코드 우선:** 설명은 간결하게, 코드는 주석을 포함하여 완벽하게 작성하라.
- **실행 가능:** 코드 스니펫은 복사-붙여넣기 즉시 실행 가능해야 한다.
- **엣지 케이스:** 에러 처리와 경계 조건을 반드시 포함하라.""",
        
        'explanation': """

## 🎯 추가 지침 (개념 설명 모드)
- **비유 활용:** 초보자도 이해하기 쉽게 일상적 비유를 사용하라.
- **단계별:** 복잡한 개념은 작은 단위로 쪼개어 단계별로 설명하라.
- **시각화:** 가능하면 다이어그램이나 플로우차트 형태로 표현하라.""",
        
        'chat': """

## 🎯 추가 지침 (일반 대화 모드)
- **친근함:** 기술적 깊이보다는 친절하고 간결한 답변을 우선하라.
- **간결성:** 불필요한 상세 설명은 생략하라."""
    }
    
    # ✅ 통합 시스템 프롬프트 (호환성 유지)
    SYSTEM_PROMPT = BASE_PERSONA
    
    @staticmethod
    def get_dynamic_prompt(intent: str = 'explanation') -> str:
        """
        Intent에 따른 동적 시스템 프롬프트 생성
        
        Args:
            intent: 'coding', 'explanation', 'chat'
        
        Returns:
            확장된 시스템 프롬프트
        """
        base = QuestionAnsweringManager.BASE_PERSONA
        instruction = QuestionAnsweringManager.INTENT_INSTRUCTIONS.get(intent, '')
        return base + instruction
    
    @staticmethod
    def extract_user_message(messages: List[ChatMessage]) -> Optional[str]:
        """마지막 사용자 메시지 추출"""
        for msg in reversed(messages):
            if msg.role == 'user':
                return msg.content
        return None
    

    
    @staticmethod
    @time_logger
    def call_llm(
        messages: List[Dict],
        temperature: float = 0.6,
        max_tokens: Optional[int] = 2048,
        stream: bool = False,
        max_retries: int = 3,
        retry_delay: float = 2.0
    ) -> Any:
        """
        vLLM 호출 (재시도 로직 포함)
        """
        global vllm_client
        
        if vllm_client is None:
            # Fix: Extend timeout for 16k context processing on L4 GPU
            vllm_client = openai.OpenAI(
                api_key='sk-not-needed',
                base_url=VLLM_API_URL,
                timeout=180.0  # 60 → 180초 연장
            )
        
        last_error = None
        
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f'📡 vLLM 호출 중... (시도 {attempt}/{max_retries})')
                
                response = vllm_client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=temperature,
                    top_p=0.95,
                    max_tokens=max_tokens,
                    stream=stream,
                    stop=[
                        "<|file_sep|>",
                        "### <CONTEXT>",
                        "[관련 문서 없음]"
                    ],
                )
                
                logger.info('✅ vLLM 응답 수신')
                return response
                
            except openai.APIConnectionError as e:
                last_error = e
                logger.warning(f'⚠️ vLLM 연결 실패 (시도 {attempt}/{max_retries}): {str(e)[:50]}')
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
                    vllm_client = openai.OpenAI(
                        api_key='sk-not-needed',
                        base_url=VLLM_API_URL,
                        timeout=180.0  # Fix: 16k context support
                    )
            except Exception as e:
                last_error = e
                logger.error(f'❌ vLLM 호출 실패: {str(e)}')
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    break
        
        logger.error(f'❌ vLLM 호출 최종 실패: {str(last_error)}', exc_info=True)
        raise last_error
    
    @staticmethod
    async def call_llm_async(
        messages: List[Dict],
        temperature: float = 0.6,
        max_tokens: Optional[int] = 2048,
        stream: bool = False
    ) -> Any:
        """
        ✅ [비동기 버전] vLLM 호출
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            executor,
            lambda: QuestionAnsweringManager.call_llm(
                messages, temperature, max_tokens, stream
            )
        )
    
    @staticmethod
    def stream_response(response) -> Generator[str, None, None]:
        """스트리밍 응답 생성"""
        try:
            chunk_count = 0
            
            for chunk in response:
                chunk_count += 1
                
                if (hasattr(chunk, 'choices') and chunk.choices and 
                    len(chunk.choices) > 0):
                    
                    choice = chunk.choices[0]
                    if (hasattr(choice, 'delta') and choice.delta and
                        hasattr(choice.delta, 'content') and 
                        choice.delta.content):
                        
                        content = choice.delta.content
                        
                        data = {
                            "id": f"chatcmpl-{datetime.now().timestamp()}",
                            "object": "chat.completion.chunk",
                            "created": int(datetime.now().timestamp()),
                            "model": MODEL_NAME,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": content},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(data)}\n\n"
            
            logger.info(f'✅ 스트리밍 완료 ({chunk_count}개 청크)')
            
            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f'❌ 스트리밍 오류: {str(e)}', exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ==================== 초기화 ====================

def wait_for_vllm(max_retries: int = 30, retry_interval: int = 10) -> bool:
    """vLLM 서버가 준비될 때까지 대기"""
    global vllm_client
    
    logger.info(f'⏳ vLLM 서버 연결 대기 중... (최대 {max_retries * retry_interval}초)')
    
    for attempt in range(1, max_retries + 1):
        try:
            if vllm_client is None:
                vllm_client = openai.OpenAI(
                    api_key='sk-not-needed',
                    base_url=VLLM_API_URL,
                    timeout=30.0
                )
            
            models = vllm_client.models.list()
            logger.info(f'✅ vLLM 연결 성공 (시도 {attempt}/{max_retries})')
            return True
            
        except Exception as e:
            logger.warning(f'⏳ vLLM 연결 대기 중... ({attempt}/{max_retries}) - {str(e)[:50]}')
            if attempt < max_retries:
                time.sleep(retry_interval)
            else:
                logger.error(f'❌ vLLM 연결 실패: {max_retries}회 시도 후 포기')
                return False
    
    return False


def preload_embedding_service():
    """임베딩 서비스 사전 로드"""
    try:
        from utils.embedding_service import get_embedding_service
        logger.info('🔄 임베딩 서비스 사전 로드 중...')
        embedding_service = get_embedding_service()
        logger.info(f'✅ 임베딩 서비스 로드 완료: {embedding_service.model_name}')
    except Exception as e:
        logger.warning(f'⚠️ 임베딩 서비스 사전 로드 실패: {str(e)}')


def preload_reranker():
    """Reranker 사전 로드 (GPU 활용)"""
    if not ENABLE_RERANKING:
        logger.info('ℹ️  Reranking 비활성화 - Reranker 로드 스킵')
        return
    
    try:
        from db.vector_store import get_reranker
        logger.info(f'🔄 Reranker 사전 로드 중... (모델: {RERANKER_MODEL})')
        reranker = get_reranker()
        if reranker:
            logger.info(f'✅ Reranker 로드 완료: {RERANKER_MODEL}')
        else:
            logger.warning('⚠️ Reranker 로드 실패')
    except Exception as e:
        logger.warning(f'⚠️ Reranker 사전 로드 실패: {str(e)}')


@app.on_event('startup')
async def startup():
    """서버 시작"""
    global vector_store, graph_processor, drill_down_retriever, intent_router, vllm_client, semantic_router
    
    logger.info('=' * 70)
    logger.info('🚀 RAG API 시작 중... (Production-Ready v2.0)')
    logger.info('=' * 70)
    
    try:
        # 0️⃣ 임베딩 서비스 + Reranker 사전 로드
        logger.info('0️⃣  임베딩 서비스 사전 로드...')
        preload_embedding_service()
        
        if ENABLE_RERANKING:
            logger.info('0️⃣  Reranker 사전 로드...')
            preload_reranker()
        
        # 1️⃣ 벡터 DB 초기화
        logger.info('1️⃣  벡터 DB 초기화...')
        vector_store = VectorStoreManager(
            chroma_host=CHROMA_HOST,
            chroma_port=CHROMA_PORT
        )
        stats = vector_store.get_collection_stats()
        logger.info(f'✅ 벡터 DB 준비: {stats.get("document_count", 0)}개 문서')
        
        # 2️⃣ 그래프 로드
        logger.info('2️⃣  그래프 로드...')
        graph = None
        if GraphRAGProcessor and os.path.exists(GRAPH_PERSIST_PATH):
            graph_processor = GraphRAGProcessor.from_file(GRAPH_PERSIST_PATH)
            if graph_processor:
                graph = graph_processor.graph
                logger.info(f'✅ 그래프 로드 완료: {graph.number_of_nodes()}개 노드, {graph.number_of_edges()}개 엣지')
            else:
                logger.warning('⚠️  그래프 로드 실패 - 벡터 검색만 사용')
        else:
            logger.info('ℹ️  그래프 파일 없음 - 벡터 검색만 사용')
        
        # 3️⃣ Intent Router 초기화
        logger.info('3️⃣  Intent Router 초기화...')
        if ScalableIntentRouter is not None:
            intent_router = ScalableIntentRouter(
                active_domains=['notion'],
                use_llm_fallback=False
            )
            logger.info(f'✅ Intent Router 준비: 활성 도메인 = {intent_router.active_domains}')
        else:
            logger.warning('⚠️  Intent Router 미사용')
        
        # 4️⃣ Drill-Down Retriever 초기화
        logger.info('4️⃣  Drill-Down Retriever 초기화...')
        if GraphDrillDownRetriever is not None and graph is not None:
            drill_down_retriever = GraphDrillDownRetriever(
                vector_store=vector_store,
                graph=graph,
                hub_types=['page', 'root'],
                hub_score_threshold=0.3,
                include_mention_depth=1
            )
            logger.info(f'✅ Drill-Down Retriever 준비')
        else:
            logger.info('ℹ️  Drill-Down Retriever 미사용 (그래프 없음)')
        
        # 5️⃣ Semantic Intent Router 초기화
        logger.info('5️⃣  Semantic Intent Router 초기화...')
        try:
            from utils.embedding_service import get_embedding_service
            embedding_service = get_embedding_service()
            semantic_router = SemanticIntentRouter(embedding_service)
            logger.info(f'✅ Semantic Intent Router 준비 (No LLM)')
        except Exception as e:
            logger.warning(f'⚠️  Semantic Intent Router 초기화 실패: {str(e)}')
            semantic_router = None
        
        # 5️⃣ vLLM 연결 대기
        logger.info('5️⃣  vLLM 연결 대기...')
        vllm_ready = wait_for_vllm(max_retries=30, retry_interval=10)
        
        # 6️⃣ 모델 정보 로드
        logger.info('6️⃣  모델 정보 로드...')
        models = ModelManager.get_models()
        logger.info(f'✅ {len(models)}개 모델 감지')
        
        logger.info('=' * 70)
        logger.info('🚀 RAG API 준비 완료! (Production-Ready)')
        logger.info(f'   - Vector Store: ✅ ({stats.get("document_count", 0)}개 문서)')
        logger.info(f'   - Graph: {"✅" if graph else "❌"}')
        logger.info(f'   - Intent Router: {"✅" if intent_router else "❌"}')
        logger.info(f'   - Drill-Down Retriever: {"✅" if drill_down_retriever else "❌"}')
        logger.info(f'   - Reranking: {"✅" if ENABLE_RERANKING else "❌"}')
        logger.info(f'   - Token Management: ✅ (Max: {MAX_CONTEXT_TOKENS})')
        logger.info(f'   - ThreadPool: ✅ ({THREAD_POOL_SIZE} workers)')
        logger.info(f'   - vLLM: {"✅" if vllm_ready else "❌"}')
        logger.info('=' * 70)
        
    except Exception as e:
        logger.error(f'❌ 초기화 실패: {str(e)}', exc_info=True)
        raise


@app.on_event('shutdown')
async def shutdown():
    """서버 종료 시 리소스 정리"""
    global vllm_client, executor
    
    logger.info('🛑 서버 종료 중...')
    
    # Fix: Resource cleanup for L4 GPU
    if vllm_client:
        try:
            vllm_client.close()
            logger.info('✅ vLLM Client 정리 완료')
        except Exception as e:
            logger.warning(f'⚠️ vLLM Client 정리 실패: {str(e)}')
    
    executor.shutdown(wait=True)
    logger.info('✅ ThreadPool 정리 완료')


# ==================== API 엔드포인트 ====================

@app.get('/health')
async def health_check() -> Dict:
    """헬스 체크"""
    try:
        if vector_store is None:
            return {'status': 'unhealthy', 'error': 'Vector store not initialized'}
        
        stats = vector_store.get_collection_stats()
        graph_stats = None
        if graph_processor:
            graph_stats = {
                'nodes': graph_processor.graph.number_of_nodes(),
                'edges': graph_processor.graph.number_of_edges()
            }
        
        return {
            'status': 'ok',
            'version': '2.0.0',
            'timestamp': datetime.now().isoformat(),
            'vector_store': stats,
            'graph': graph_stats,
            'model': MODEL_NAME,
            'documents': stats.get('document_count', 0),
            'token_limit': MAX_CONTEXT_TOKENS,
            'thread_pool_size': THREAD_POOL_SIZE,
            'components': {
                'vector_store': vector_store is not None,
                'graph': graph_processor is not None,
                'intent_router': intent_router is not None,
                'drill_down_retriever': drill_down_retriever is not None,
                'semantic_router': semantic_router is not None,
                'reranking': ENABLE_RERANKING
            }
        }
    except Exception as e:
        logger.error(f'❌ 헬스 체크 실패: {str(e)}')
        return {'status': 'unhealthy', 'error': str(e)}


@app.get('/v1/models')
async def list_models() -> Dict:
    """모델 목록 반환"""
    logger.info('📊 /v1/models 요청')
    models = ModelManager.get_models()
    return {'object': 'list', 'data': models}


# ==================== 디버그 엔드포인트 ====================

@app.post('/v1/search')
async def search_documents(query: str, top_k: int = 5) -> Dict:
    """✅ [비동기] 검색 결과 확인 엔드포인트"""
    logger.info(f'🔍 /v1/search 요청: "{query[:50]}..."')
    
    result = await VectorSearchManager.search_with_intent(query, top_k=top_k)
    
    return {
        'query': query,
        'intent': result.get('intent'),
        'search_type': result.get('search_type'),
        'document_count': len(result.get('documents', [])),
        'documents': result.get('documents', [])[:3],
        'context_length': len(result.get('context', ''))
    }


@app.post('/v1/intent')
async def analyze_intent(query: str) -> Dict:
    """✅ [디버그용] Intent 분석 엔드포인트"""
    if intent_router is None:
        return {'error': 'Intent Router not initialized'}
    
    result = intent_router.route(query)
    return result.to_dict()


@app.post('/v1/chat/completions')
@time_logger
async def chat_completions(request: ChatCompletionRequest) -> Any:
    """
    ✅ [Production-Ready] Chat Completion 엔드포인트
    
    개선사항:
    1. Non-blocking I/O (검색, LLM 호출)
    2. Smart Token Management (동적 히스토리 관리)
    3. Structured Prompt (System에 Context 주입)
    4. Observability (time_logger)
    """
    logger.info('=' * 70)
    logger.info(f'💬 Chat: {len(request.messages)}개 메시지, stream={request.stream}')
    logger.info('=' * 70)
    
    try:
        # 1️⃣ 사용자 질문 추출
        user_message = QuestionAnsweringManager.extract_user_message(request.messages)
        if not user_message:
            raise HTTPException(status_code=400, detail="사용자 메시지 없음")
        
        logger.info(f'질문: {user_message[:100]}...')
        
        # ✅ Open WebUI 내부 Task 요청 감지
        is_internal_task = user_message.strip().startswith('### Task:')
        
        # ✅ follow-up questions 요청 차단
        if 'follow-up questions' in user_message.lower() or 'Suggest 3-5 relevant' in user_message:
            logger.info('🚫 Follow-up questions 요청 차단')
            return {
                'id': f'chatcmpl-{datetime.now().timestamp()}',
                'object': 'chat.completion',
                'created': int(datetime.now().timestamp()),
                'model': MODEL_NAME,
                'choices': [{
                    'index': 0,
                    'message': {'role': 'assistant', 'content': ''},
                    'finish_reason': 'stop'
                }],
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            }
        
        if is_internal_task:
            # 내부 Task는 RAG 검색 없이 바로 LLM에 전달
            logger.info('🔧 Open WebUI 내부 Task 감지 → RAG 스킵')
            vllm_messages = [
                {'role': 'system', 'content': 'You are a helpful assistant. Respond in the requested format.'},
                {'role': 'user', 'content': user_message}
            ]
        else:
            # 2️⃣ 벡터 DB 검색 (RAG) - Non-blocking
            logger.info('Step 1: RAG 검색 (Non-blocking)...')
            context = await VectorSearchManager.search(user_message, top_k=SEARCH_TOP_K)
            logger.info(f'검색 완료: {len(context)}자')
            
            # ✅ 조건부 웹 검색 (Conditional Web Search)
            web_context = ""
            internal_docs_for_decision = []  # 웹 검색 판단용
            
            # 내부 검색 결과 파싱 (간단한 휴리스틱)
            if drill_down_retriever and hasattr(VectorSearchManager, '_last_search_results'):
                internal_docs_for_decision = getattr(VectorSearchManager, '_last_search_results', [])
            
            # Step 1.5: 웹 검색 필요성 판단 (보안 우선)
            try:
                from utils.web_search import get_search_decision_maker
                
                decision_maker = get_search_decision_maker()
                decision = await decision_maker.decide_and_sanitize(
                    user_query=user_message,
                    internal_documents=internal_docs_for_decision
                )
                
                if decision != 'NO_SEARCH':
                    # 웹 검색 실행
                    logger.info(f'🌐 웹 검색 실행: "{decision}"')
                    web_results = await decision_maker.search(decision, max_results=5)
                    web_context = decision_maker.format_web_results(web_results)
                    logger.info(f'✅ 웹 검색 완료: {len(web_results)}개 결과')
                else:
                    logger.info('ℹ️  웹 검색 스킵: 내부 문서로 충분')
                    
            except ImportError:
                logger.warning('⚠️  web_search 모듈 없음 - 웹 검색 스킵')
            except Exception as e:
                logger.error(f'❌ 웹 검색 실패: {str(e)} - 내부 검색 결과만 사용')
            
            # ✅ Semantic Intent 분류 (No LLM)
            detected_intent = 'explanation'  # default
            if semantic_router:
                try:
                    detected_intent, confidence = semantic_router.classify(user_message)
                    logger.info(f'🎯 Intent: {detected_intent} (confidence={confidence:.2f})')
                except Exception as e:
                    logger.warning(f'⚠️ Intent 분류 실패: {str(e)}')
            
            # 3️⃣ Smart Token Management + ✅ Dynamic Persona Injection
            logger.info('Step 2: 토큰 관리 & 동적 프롬프트 구성...')
            
            # 현재 메시지 이전의 히스토리 추출
            history = [msg for msg in request.messages[:-1] if msg.role in ['user', 'assistant']]
            
            # ✅ Dynamic Persona: Intent에 따른 시스템 프롬프트 선택
            dynamic_prompt = QuestionAnsweringManager.get_dynamic_prompt(detected_intent)
            
            # ✅ 통합 컨텍스트 구성 (내부 + 웹)
            combined_context = context
            if web_context:
                combined_context = f"{context}\n\n{web_context}"
                logger.info(f'📚 통합 컨텍스트: 내부 {len(context)}자 + 웹 {len(web_context)}자')
            
            # ✅ TokenManager를 사용한 동적 컨텍스트 윈도우 관리
            vllm_messages = TokenManager.manage_context_window(
                system_prompt=dynamic_prompt,  # ✅ Intent-aware prompt
                context=combined_context,  # ✅ 내부 + 웹 검색 결과
                current_query=user_message,
                history=history,
                max_tokens=MAX_CONTEXT_TOKENS
            )
        
        # 4️⃣ LLM 호출 - Non-blocking
        logger.info('Step 3: LLM 호출 (Non-blocking)...')
        
        # ✅ 동적 max_tokens 계산 (토큰 오버플로우 방지)
        estimated_input_tokens = TokenManager.estimate_messages_tokens(vllm_messages)
        available_output_tokens = MAX_MODEL_LEN - estimated_input_tokens - 100  # 100토큰 여유
        
        # max_tokens를 사용 가능한 범위로 제한
        dynamic_max_tokens = min(
            request.max_tokens or 2048,
            available_output_tokens,
            1024  # 최대 1024토큰으로 제한 (안전장치)
        )
        
        if dynamic_max_tokens < 50:
            logger.warning(f'⚠️ 출력 토큰 부족: 입력 {estimated_input_tokens}, 사용가능 {available_output_tokens}')
            dynamic_max_tokens = 50  # 최소 50토큰 보장
        
        logger.info(f'📊 토큰 계산: 입력 ~{estimated_input_tokens}, 출력 {dynamic_max_tokens} (한도 {MAX_MODEL_LEN})')
        
        response = await QuestionAnsweringManager.call_llm_async(
            messages=vllm_messages,
            temperature=request.temperature,
            max_tokens=dynamic_max_tokens,
            stream=request.stream
        )
        
        # 5️⃣ 응답 반환
        if request.stream:
            logger.info('스트리밍 응답 반환')
            return StreamingResponse(
                QuestionAnsweringManager.stream_response(response),
                media_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no'
                }
            )
        else:
            logger.info('일반 응답 반환')
            
            if not response.choices:
                raise HTTPException(status_code=500, detail='vLLM 응답 오류')
            
            assistant_message = response.choices[0].message.content
            
            logger.info(f'✅ 응답: {assistant_message[:100]}...')
            
            return {
                'id': f'chatcmpl-{datetime.now().timestamp()}',
                'object': 'chat.completion',
                'created': int(datetime.now().timestamp()),
                'model': MODEL_NAME,
                'choices': [{
                    'index': 0,
                    'message': {'role': 'assistant', 'content': assistant_message},
                    'finish_reason': 'stop'
                }],
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'❌ 처리 오류: {str(e)}', exc_info=True)
        raise HTTPException(status_code=500, detail=f'처리 실패: {str(e)}')


# ==================== 서버 실행 ====================

if __name__ == '__main__':
    host = os.getenv('SERVER_HOST', '0.0.0.0')
    port = int(os.getenv('SERVER_PORT', '8010'))
    
    logger.info(f'🚀 서버 시작: {host}:{port}')
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level.lower()
    )
