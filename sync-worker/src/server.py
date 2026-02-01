"""
🚀 RAG Inference API Server - FastAPI 기반

3가지 핵심 역할:
1. 벡터 DB 검색 (ChromaDB → XML 포맷 Context)
2. 모델 관리 (vLLM 모델 정보 제공)
3. 질의응답 (DeepSeek-R1 추론 → <think> 태그 필터링)

Open WebUI와 완전 호환되는 OpenAI API 구현
"""

import os
import sys
import logging
import json
import re
from typing import Optional, List, Generator, Dict, Any
from datetime import datetime
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

# ==================== FastAPI 앱 ====================

app = FastAPI(
    title='RAG Inference API',
    description='Enterprise RAG - OpenAI Compatible (DeepSeek-R1)',
    version='1.0.0'
)

# ==================== 설정 ====================

MODEL_NAME = os.getenv('LLM_MODEL_ID', 'DeepSeek-R1-Distill-8B')
LLM_BACKEND = os.getenv('LLM_BACKEND', 'vllm').lower()
VLLM_API_URL = os.getenv('VLLM_API_URL', 'http://localhost:8000/v1')
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
CHROMA_HOST = os.getenv('CHROMA_HOST', 'localhost')
CHROMA_PORT = int(os.getenv('CHROMA_PORT', '8000'))
GRAPH_PERSIST_PATH = os.getenv('GRAPH_PERSIST_PATH', './data/graph.pkl')
# ✅ [업그레이드] Full Page Retrieval 적용: 페이지 전체를 가져오므로 개수를 5에서 2로 줄임
# 2개만 찾아도 페이지 2개 분량이 통째로 들어가므로 충분함
SEARCH_TOP_K = int(os.getenv('SEARCH_TOP_K', '2'))

logger.info(f'✅ 모델: {MODEL_NAME}')
logger.info(f'✅ LLM Backend: {LLM_BACKEND.upper()}')

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


# ==================== 1️⃣ 벡터 DB 검색 (RAG) ====================

class VectorSearchManager:
    """
    ✅ [업그레이드] Intent Router + Drill-Down Retriever 통합
    
    검색 플로우:
    1. Intent Router: 쿼리 의도 분류 (search_knowledge/chat/summary)
    2. Drill-Down Retriever: 3단계 드릴다운 검색
    3. 컨텍스트 포맷팅: XML 형식으로 반환
    """
    
    @staticmethod
    def search(query: str, top_k: int = SEARCH_TOP_K) -> str:
        """
        ✅ [통합 검색] Intent 기반 라우팅 + 드릴다운 검색
        
        Args:
            query: 검색 질문
            top_k: 반환할 문서 개수
        
        Returns:
            XML 형식의 컨텍스트
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
                
                # 일상 대화는 검색 불필요
                if intent_result.intent == 'chat':
                    logger.info('💬 일상 대화 감지 → 검색 스킵')
                    return ""
            
            # =====================================================
            # Step 2: Drill-Down 검색 (3단계 드릴다운)
            # =====================================================
            if drill_down_retriever is not None:
                logger.info(f'🔍 드릴다운 검색 시작: "{query[:50]}..." (top_k={top_k})')
                
                documents, context_xml = drill_down_retriever.retrieve_with_context(
                    query=query,
                    k=top_k,
                    context_format='xml'
                )
                
                if documents:
                    logger.info(f'✅ 드릴다운 검색 완료: {len(documents)}개 문서')
                    return context_xml
                else:
                    logger.info('⚠️ 드릴다운 검색 결과 없음 → Fallback')
            
            # =====================================================
            # Step 3: Fallback - 기존 검색 방식
            # =====================================================
            logger.info(f'🔍 Fallback 검색: "{query[:50]}..." (top_k={top_k})')
            
            # 기존 retrieve_context 사용
            context = vector_store.retrieve_context(query, top_k=top_k, use_hybrid=True)
            
            if not context:
                logger.info('⚠️ 검색 결과 없음')
                return ""
            
            logger.info(f'✅ 컨텍스트 생성 완료: {len(context)}자')
            return context
            
        except Exception as e:
            logger.error(f'❌ 검색 실패: {str(e)}', exc_info=True)
            return ""
    
    @staticmethod
    def search_with_intent(query: str, top_k: int = SEARCH_TOP_K) -> Dict[str, Any]:
        """
        ✅ [상세 버전] Intent 정보와 함께 검색 결과 반환
        
        Returns:
            {
                'intent': {...},
                'documents': [...],
                'context': '...',
                'search_type': 'drill_down' | 'fallback'
            }
        """
        result = {
            'intent': None,
            'documents': [],
            'context': '',
            'search_type': 'fallback'
        }
        
        if vector_store is None:
            return result
        
        try:
            # Intent 분류
            if intent_router is not None:
                intent_result = intent_router.route(query)
                result['intent'] = intent_result.to_dict()
                
                if intent_result.intent == 'chat':
                    result['search_type'] = 'skip'
                    return result
            
            # Drill-Down 검색
            if drill_down_retriever is not None:
                documents = drill_down_retriever.retrieve(query, k=top_k)
                if documents:
                    result['documents'] = [doc.to_dict() for doc in documents]
                    result['context'] = drill_down_retriever._format_as_xml(documents)
                    result['search_type'] = 'drill_down'
                    return result
            
            # Fallback
            context = vector_store.retrieve_context(query, top_k=top_k, use_hybrid=True)
            result['context'] = context
            result['search_type'] = 'fallback'
            
            return result
            
        except Exception as e:
            logger.error(f'❌ 검색 실패: {str(e)}')
            return result


# ==================== 2️⃣ 모델 관리 ====================

class ModelManager:
    """vLLM 모델 정보 관리"""
    
    @staticmethod
    def get_models() -> List[Dict[str, Any]]:
        """
        사용 가능한 모델 목록 반환
        
        Returns:
            OpenAI 호환 모델 정보
        """
        global available_models
        
        # vLLM에서 모델 정보 가져오기 시도
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
        
        # 폴백: 기본 모델 반환
        return [{
            'id': MODEL_NAME,
            'object': 'model',
            'created': int(datetime.now().timestamp()),
            'owned_by': 'vllm',
            'permission': [
                {
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
                }
            ]
        }]


# ==================== 3️⃣ 질의응답 (LLM) ====================

class QuestionAnsweringManager:
    """DeepSeek-R1 기반 답변 생성"""
    
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
        
        Args:
            messages: 대화 메시지
            temperature: 샘플링 온도
            max_tokens: 최대 토큰 수
            stream: 스트리밍 여부
            max_retries: 최대 재시도 횟수
            retry_delay: 재시도 대기 시간 (초)
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
                        "好，",  # 중국어 시작 패턴 차단
                        "首先",  # 중국어 시작 패턴 차단
                        "接下来",  # 중국어 시작 패턴 차단
                    ],
                )
                
                logger.info('✅ vLLM 응답 수신')
                return response
                
            except openai.APIConnectionError as e:
                last_error = e
                logger.warning(f'⚠️ vLLM 연결 실패 (시도 {attempt}/{max_retries}): {str(e)[:50]}')
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)  # 점진적 대기
                    # 클라이언트 재초기화
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
    def stream_response(response) -> Generator[str, None, None]:
        """스트리밍 응답 생성 (중국어 필터링 포함)"""
        import re
        
        # 중국어 문자 감지 패턴 (한자 범위)
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
                        
                        # <think> 태그 필터링
                        if '<think>' in content:
                            in_think = True
                        
                        # 중국어 감지 시 스킵
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
            
            # 종료
            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f'❌ 스트리밍 오류: {str(e)}', exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ==================== 초기화 ====================

def wait_for_vllm(max_retries: int = 30, retry_interval: int = 10) -> bool:
    """
    vLLM 서버가 준비될 때까지 대기
    
    Args:
        max_retries: 최대 재시도 횟수
        retry_interval: 재시도 간격 (초)
    
    Returns:
        성공 여부
    """
    global vllm_client
    
    logger.info(f'⏳ vLLM 서버 연결 대기 중... (최대 {max_retries * retry_interval}초)')
    
    for attempt in range(1, max_retries + 1):
        try:
            # vLLM 클라이언트 초기화
            if vllm_client is None:
                vllm_client = openai.OpenAI(
                    api_key='sk-not-needed',
                    base_url=VLLM_API_URL,
                    timeout=30.0
                )
            
            # 모델 목록 조회로 연결 테스트
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
    """
    임베딩 서비스 사전 로드 (HuggingFace 모델 싱글톤)
    """
    try:
        from utils.embedding_service import get_embedding_service
        logger.info('🔄 임베딩 서비스 사전 로드 중...')
        embedding_service = get_embedding_service()
        logger.info(f'✅ 임베딩 서비스 로드 완료: {embedding_service.model_name}')
    except Exception as e:
        logger.warning(f'⚠️ 임베딩 서비스 사전 로드 실패: {str(e)}')


@app.on_event('startup')
async def startup():
    """서버 시작"""
    global vector_store, graph_processor, drill_down_retriever, intent_router, vllm_client
    
    logger.info('=' * 70)
    logger.info('RAG API 시작 중...')
    logger.info('=' * 70)
    
    try:
        # 0️⃣ 임베딩 서비스 사전 로드 (HuggingFace 모델)
        logger.info('0️⃣  임베딩 서비스 사전 로드...')
        preload_embedding_service()
        
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
                active_domains=['notion'],  # 현재 Notion만 활성화
                use_llm_fallback=False       # 규칙 기반만 사용 (속도 최적화)
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
        logger.info('🚀 RAG API 준비 완료!')
        logger.info(f'   - Vector Store: ✅ ({stats.get("document_count", 0)}개 문서)')
        logger.info(f'   - Graph: {"✅" if graph else "❌"}')
        logger.info(f'   - Intent Router: {"✅" if intent_router else "❌"}')
        logger.info(f'   - Drill-Down Retriever: {"✅" if drill_down_retriever else "❌"}')
        logger.info(f'   - vLLM: {"✅" if vllm_ready else "❌ (백그라운드 연결 시도)"}')
        logger.info('=' * 70)
        
    except Exception as e:
        logger.error(f'❌ 초기화 실패: {str(e)}', exc_info=True)
        raise


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
            'timestamp': datetime.now().isoformat(),
            'vector_store': stats,
            'graph': graph_stats,
            'model': MODEL_NAME,
            'documents': stats.get('document_count', 0),
            'components': {
                'vector_store': vector_store is not None,
                'graph': graph_processor is not None,
                'intent_router': intent_router is not None,
                'drill_down_retriever': drill_down_retriever is not None
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
    
    return {
        'object': 'list',
        'data': models
        # "data": [{"id": "DeepSeek-R1-Distill-Qwen-14B", "object": "model"}]
    }


# ==================== 디버그 엔드포인트 ====================

@app.post('/v1/search')
async def search_documents(query: str, top_k: int = 5) -> Dict:
    """
    ✅ [디버그용] 검색 결과 확인 엔드포인트
    
    Intent Router + Drill-Down Retriever 동작 확인용
    """
    logger.info(f'🔍 /v1/search 요청: "{query[:50]}..."')
    
    result = VectorSearchManager.search_with_intent(query, top_k=top_k)
    
    return {
        'query': query,
        'intent': result.get('intent'),
        'search_type': result.get('search_type'),
        'document_count': len(result.get('documents', [])),
        'documents': result.get('documents', [])[:3],  # 상위 3개만 반환
        'context_length': len(result.get('context', ''))
    }


@app.post('/v1/intent')
async def analyze_intent(query: str) -> Dict:
    """
    ✅ [디버그용] Intent 분석 엔드포인트
    """
    if intent_router is None:
        return {'error': 'Intent Router not initialized'}
    
    result = intent_router.route(query)
    return result.to_dict()


@app.post('/v1/chat/completions')
async def chat_completions(request: ChatCompletionRequest) -> Any:
    """
    Chat Completion 엔드포인트
    
    플로우:
    1. 사용자 질문 추출
    2. Intent 분류 → Drill-Down 검색
    3. System Prompt + Context + Query 구성
    4. vLLM (DeepSeek-R1) 호출
    5. <think> 태그 필터링
    6. 응답 반환
    """
    logger.info('=' * 70)
    logger.info(f'💬 Chat: {len(request.messages)}개 메시지, stream={request.stream}')
    logger.info('=' * 70)
    
    try:
        # 1️⃣  사용자 질문 추출
        user_message = QuestionAnsweringManager.extract_user_message(request.messages)
        if not user_message:
            raise HTTPException(status_code=400, detail="사용자 메시지 없음")
        
        logger.info(f'질문: {user_message[:100]}...')
        
        # ✅ Open WebUI 내부 Task 요청 감지 (RAG 검색 불필요)
        is_internal_task = user_message.strip().startswith('### Task:')
        
        # ✅ follow-up questions 요청은 완전히 차단
        if 'follow-up questions' in user_message.lower() or 'Suggest 3-5 relevant' in user_message:
            logger.info('🚫 Follow-up questions 요청 차단 → 빈 응답 반환')
            return {
                'id': f'chatcmpl-{datetime.now().timestamp()}',
                'object': 'chat.completion',
                'created': int(datetime.now().timestamp()),
                'model': MODEL_NAME,
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': ''
                    },
                    'finish_reason': 'stop'
                }],
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            }
        
        if is_internal_task:
            # 내부 Task는 RAG 검색 없이 바로 LLM에 전달
            logger.info('🔧 Open WebUI 내부 Task 감지 → RAG 스킵')
            context = ""
            final_user_content = user_message
            
            # 내부 Task용 간단한 시스템 프롬프트
            vllm_messages = [
                {'role': 'system', 'content': 'You are a helpful assistant. Respond in the requested format.'}
            ]
        else:
            # 2️⃣  벡터 DB 검색 (RAG)
            logger.info('Step 1: RAG 검색...')
            context = VectorSearchManager.search(user_message, top_k=SEARCH_TOP_K)
            logger.info(f'검색 완료: {len(context)}자')
            
            # 3️⃣  프롬프트 구성
            logger.info('Step 2: 프롬프트 구성...')
            final_user_content = f"""다음은 검색된 참고 문서입니다:
{context}

---

사용자 질문:
{user_message}

위 문서를 참고하여 질문에 답하세요."""
        
            # 메시지 재구성 (일반 RAG 질문)
            vllm_messages = [
                {'role': 'system', 'content': QuestionAnsweringManager.SYSTEM_PROMPT}
            ]
        
        # ✅ [수정] 토큰 제한을 고려하여 최근 메시지만 포함
        # 너무 오래된 대화는 제외하여 토큰 절약
        # 하지만 최근 2-3개는 포함하여 대화의 연결성 유지
        max_history = 3  # 최근 3개 메시지까지만 포함
        
        # 현재 메시지 이전의 모든 메시지 중 최근 max_history개만 선택
        # (내부 Task는 히스토리 포함하지 않음)
        if not is_internal_task:
            history_messages = request.messages[:-1]  # 현재 메시지 제외
            if len(history_messages) > max_history:
                history_messages = history_messages[-max_history:]  # 최근 max_history개만
            
            # 선택된 과거 메시지 추가
            for msg in history_messages:
                if msg.role in ['user', 'assistant']:
                    vllm_messages.append({'role': msg.role, 'content': msg.content})
            
            logger.info(f'메시지: System + History({len(history_messages)}) + Current = {len(vllm_messages)+1}개')
        else:
            logger.info(f'메시지: Internal Task (히스토리 없음)')
        
        # 최종 사용자 메시지 추가 (검색된 컨텍스트 포함)
        vllm_messages.append({'role': 'user', 'content': final_user_content})
        
        # 4️⃣  LLM 호출
        logger.info('Step 3: LLM 호출...')
        response = QuestionAnsweringManager.call_llm(
            messages=vllm_messages,
            temperature=request.temperature,
            max_tokens=request.max_tokens,
            stream=request.stream
        )
        
        # 5️⃣  응답 반환
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
            
            # <think> 태그 제거
            think, clean_message = QuestionAnsweringManager.extract_think_content(
                assistant_message
            )
            
            logger.info(f'✅ 응답: {clean_message[:100]}...')
            
            return {
                'id': f'chatcmpl-{datetime.now().timestamp()}',
                'object': 'chat.completion',
                'created': int(datetime.now().timestamp()),
                'model': MODEL_NAME,
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': clean_message
                    },
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
