"""
🚀 RAG Inference API Server - FastAPI 기반 (Production-Ready)

3가지 핵심 역할:
1. 벡터 DB 검색 (ChromaDB → XML 포맷 Context)
2. 모델 관리 (vLLM 모델 정보 제공)
3. 질의응답 (DeepSeek-R1 추론 → <think> 태그 필터링)

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
SEARCH_TOP_K = int(os.getenv('SEARCH_TOP_K', '2'))

# ✅ Reranking 설정
ENABLE_RERANKING = os.getenv('ENABLE_RERANKING', 'true').lower() == 'true'
RERANKER_MODEL = os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')
RERANKER_TOP_K = int(os.getenv('RERANKER_TOP_K', '10'))

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
        토큰 수 추정 (근사치)
        - 한글: ~1.5 chars/token
        - 영어/숫자: ~4 chars/token (OpenAI 기준)
        - 혼합 텍스트 평균: ~2 chars/token
        """
        if not text:
            return 0
        
        # 한글 문자 수
        korean_chars = len(re.findall(r'[가-힣]', text))
        # 영어/숫자/특수문자
        other_chars = len(text) - korean_chars
        
        # 한글은 1.5 char/token, 영어는 4 char/token
        korean_tokens = korean_chars / 1.5
        other_tokens = other_chars / 4
        
        return int(korean_tokens + other_tokens)
    
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
        max_tokens: int = MAX_CONTEXT_TOKENS
    ) -> List[Dict[str, str]]:
        """
        ✅ 동적 컨텍스트 윈도우 관리
        
        토큰 예산 내에서 최대한 많은 히스토리를 포함
        
        Args:
            system_prompt: 시스템 프롬프트
            context: 검색된 문서 컨텍스트
            current_query: 현재 사용자 질문
            history: 과거 대화 히스토리
            max_tokens: 최대 허용 토큰 수
        
        Returns:
            최적화된 메시지 리스트
        """
        messages = []
        
        # =====================================================
        # Step 1: 고정 비용 계산 (System + Context + Current)
        # =====================================================
        
        # 3️⃣ Structured Prompt: System Prompt에 Context 주입
        if context:
            full_system = f"""{system_prompt}

### Reference Context
<context>
{context}
</context>

위 문서를 참고하여 사용자 질문에 답변하세요."""
        else:
            full_system = system_prompt
        
        system_tokens = TokenManager.estimate_tokens(full_system)
        query_tokens = TokenManager.estimate_tokens(current_query)
        
        fixed_tokens = system_tokens + query_tokens + 20  # 여유분
        remaining_tokens = max_tokens - fixed_tokens
        
        logger.debug(f'토큰 예산: 전체={max_tokens}, 고정={fixed_tokens}, 히스토리용={remaining_tokens}')
        
        # =====================================================
        # Step 2: 히스토리 동적 포함 (최신 → 과거 순)
        # =====================================================
        
        selected_history = []
        history_tokens = 0
        
        # 역순으로 순회하며 토큰 예산 내에서 추가
        for msg in reversed(history):
            msg_tokens = TokenManager.estimate_tokens(msg.content) + 4  # role 포함
            
            if history_tokens + msg_tokens <= remaining_tokens:
                selected_history.insert(0, {'role': msg.role, 'content': msg.content})
                history_tokens += msg_tokens
            else:
                # 토큰 예산 초과 시 중단
                break
        
        # =====================================================
        # Step 3: 최종 메시지 조립
        # =====================================================
        
        messages.append({'role': 'system', 'content': full_system})
        messages.extend(selected_history)
        messages.append({'role': 'user', 'content': current_query})
        
        total_tokens = TokenManager.estimate_messages_tokens(messages)
        logger.info(f'📊 토큰 관리: 히스토리 {len(selected_history)}개 포함, 총 ~{total_tokens} 토큰')
        
        return messages


# ==================== 1️⃣ 벡터 DB 검색 (RAG) - Async 전환 ====================

class VectorSearchManager:
    """
    ✅ [Production-Ready] Intent Router + Drill-Down Retriever 통합
    
    - Non-blocking I/O: ThreadPoolExecutor 사용
    - Observability: time_logger 데코레이터 적용
    """
    
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
                
                documents = drill_down_retriever.retrieve(
                    query=query,
                    k=search_k,
                    use_reranking=ENABLE_RERANKING
                )
                
                if documents:
                    documents = documents[:top_k]
                    context_xml = drill_down_retriever._format_as_xml(documents)
                    logger.info(f'✅ 드릴다운 검색 완료: {len(documents)}개 문서')
                    return context_xml
                else:
                    logger.info('⚠️ 드릴다운 검색 결과 없음 → Fallback')
            
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
    
    # ✅ 3️⃣ Structured Prompt: Context는 별도로 주입됨
    SYSTEM_PROMPT = """당신은 한국어로 답변하는 RAG 어시스턴트입니다.

## 핵심 규칙
1. **한국어 전용**: 모든 응답은 반드시 한국어로 작성합니다. 중국어, 일본어는 절대 사용하지 마세요.
2. **문서 기반 답변**: <context> 안의 문서 내용을 기반으로 답변합니다.
3. **정보 없음 처리**: 문서에 관련 정보가 없으면 "제공된 문서에는 해당 정보가 없습니다."라고 답합니다.
4. **자연스러운 대화**: 딱딱한 형식 없이 자연스럽게 설명합니다.

## 답변 스타일
- 핵심 내용을 먼저 말하고 세부 사항을 설명
- 불필요한 인사말이나 미사여구 생략
- 기술 용어는 영어 그대로 사용 가능 (예: Docker, API)"""
    
    @staticmethod
    def extract_user_message(messages: List[ChatMessage]) -> Optional[str]:
        """마지막 사용자 메시지 추출"""
        for msg in reversed(messages):
            if msg.role == 'user':
                return msg.content
        return None
    
    @staticmethod
    def extract_think_content(text: str) -> tuple[str, str]:
        """<think> 태그 분리"""
        think_pattern = r'<think>(.*?)</think>'
        match = re.search(think_pattern, text, re.DOTALL)
        
        if match:
            think = match.group(1).strip()
            response = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
            logger.debug(f'[추론] {len(think)}자')
            return think, response
        
        return "", text
    
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
            vllm_client = openai.OpenAI(
                api_key='sk-not-needed',
                base_url=VLLM_API_URL,
                timeout=60.0
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
                        "[관련 문서 없음]",
                        "<｜end of sentence｜>",
                        "好，",
                        "首先",
                        "接下来",
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
                        timeout=60.0
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
        """스트리밍 응답 생성 (중국어 필터링 포함)"""
        chinese_pattern = re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
        
        try:
            in_think = False
            chunk_count = 0
            chinese_detected = False
            
            for chunk in response:
                chunk_count += 1
                
                if (hasattr(chunk, 'choices') and chunk.choices and 
                    len(chunk.choices) > 0):
                    
                    choice = chunk.choices[0]
                    if (hasattr(choice, 'delta') and choice.delta and
                        hasattr(choice.delta, 'content') and 
                        choice.delta.content):
                        
                        content = choice.delta.content
                        
                        if '<think>' in content:
                            in_think = True
                        
                        if chinese_pattern.search(content):
                            chinese_detected = True
                            continue
                        
                        if not in_think:
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
                        
                        if '</think>' in content:
                            in_think = False
            
            if chinese_detected:
                logger.warning('⚠️ 중국어 출력 감지 및 필터링됨')
            
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
    global vector_store, graph_processor, drill_down_retriever, intent_router, vllm_client
    
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
    logger.info('🛑 서버 종료 중...')
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
            
            # 3️⃣ Smart Token Management + Structured Prompt
            logger.info('Step 2: 토큰 관리 & 프롬프트 구성...')
            
            # 현재 메시지 이전의 히스토리 추출
            history = [msg for msg in request.messages[:-1] if msg.role in ['user', 'assistant']]
            
            # ✅ TokenManager를 사용한 동적 컨텍스트 윈도우 관리
            vllm_messages = TokenManager.manage_context_window(
                system_prompt=QuestionAnsweringManager.SYSTEM_PROMPT,
                context=context,
                current_query=user_message,
                history=history,
                max_tokens=MAX_CONTEXT_TOKENS
            )
        
        # 4️⃣ LLM 호출 - Non-blocking
        logger.info('Step 3: LLM 호출 (Non-blocking)...')
        response = await QuestionAnsweringManager.call_llm_async(
            messages=vllm_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
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
            think, clean_message = QuestionAnsweringManager.extract_think_content(assistant_message)
            
            logger.info(f'✅ 응답: {clean_message[:100]}...')
            
            return {
                'id': f'chatcmpl-{datetime.now().timestamp()}',
                'object': 'chat.completion',
                'created': int(datetime.now().timestamp()),
                'model': MODEL_NAME,
                'choices': [{
                    'index': 0,
                    'message': {'role': 'assistant', 'content': clean_message},
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
