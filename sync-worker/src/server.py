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

MODEL_NAME = os.getenv('LLM_MODEL_ID', 'DeepSeek-R1-Distill-Qwen-14B')
LLM_BACKEND = os.getenv('LLM_BACKEND', 'vllm').lower()
VLLM_API_URL = os.getenv('VLLM_API_URL', 'http://localhost:8000/v1')
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
CHROMA_HOST = os.getenv('CHROMA_HOST', 'localhost')
CHROMA_PORT = int(os.getenv('CHROMA_PORT', '8000'))
SEARCH_TOP_K = int(os.getenv('SEARCH_TOP_K', '5'))

logger.info(f'✅ 모델: {MODEL_NAME}')
logger.info(f'✅ LLM Backend: {LLM_BACKEND.upper()}')

# ==================== 글로벌 변수 ====================

vector_store: Optional[VectorStoreManager] = None
vllm_client: Optional[openai.OpenAI] = None
available_models: List[Dict[str, Any]] = []

# ==================== 데이터 모델 ====================

class ChatMessage(BaseModel):
    role: str = Field(..., description="역할: system/user/assistant")
    content: str = Field(..., description="메시지 내용")


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=MODEL_NAME)
    messages: List[ChatMessage]
    temperature: float = Field(default=0.6, ge=0.0, le=2.0)
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
    """ChromaDB 벡터 검색 담당"""
    
    @staticmethod
    def search(query: str, top_k: int = SEARCH_TOP_K) -> str:
        """
        벡터 DB에서 관련 문서 검색
        
        Args:
            query: 검색 질문
            top_k: 상위 K개 결과
        
        Returns:
            XML 포맷의 Context 문서
        """
        if vector_store is None:
            logger.warning('❌ Vector Store 미초기화')
            return "<documents></documents>"
        
        try:
            logger.info(f'🔍 검색: "{query[:50]}..." (top_k={top_k})')
            
            # ChromaDB 검색
            search_results = vector_store.search(query, top_k=top_k)
            if not search_results:
                logger.info('⚠️  검색 결과 없음')
                return "<documents></documents>"
            
            # 유사도 필터링 (0.4 이상만 포함)
            filtered_results = [
                r for r in search_results
                if r.get('distance', float('inf')) <= 0.4
            ]
            
            if not filtered_results:
                logger.info('⚠️  필터링 후 결과 없음')
                return "<documents></documents>"
            
            # XML 포맷 Context 생성
            context_parts = []
            for i, result in enumerate(filtered_results, 1):
                content = result['content']
                source = result['metadata'].get('source', 'unknown')
                
                context_parts.append(f"""    <doc id="{i}">
        <source>{source}</source>
        <content>{content}</content>
    </doc>""")
            
            context = "<documents>\n" + "\n".join(context_parts) + "\n</documents>"
            
            logger.info(f'✅ {len(filtered_results)}개 문서 검색 완료')
            logger.debug(f'Context 길이: {len(context)}자')
            
            return context
            
        except Exception as e:
            logger.error(f'❌ 벡터 검색 실패: {str(e)}', exc_info=True)
            return "<documents></documents>"


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
    
    SYSTEM_PROMPT = """당신은 박민준 개발자의 프로젝트를 돕는 'DeepSeek-R1 기반 AI 엔진'입니다.
제공된 <documents> 데이터를 기반으로 논리적으로 추론하여 답변하세요.

[답변 작성 절차]
1. **Analyze**: 사용자의 질문 의도를 파악하고 <documents> 내의 정보와 대조하세요.
2. **Think**: 문서의 내용이 질문을 해결하는 데 충분한지 논리적으로 따져보세요. (생각 과정을 <think> 태그에 담으세요)
3. **Answer**: 분석된 내용을 바탕으로 개발자에게 필요한 코드나 솔루션을 구체적으로 제시하세요.

[제약 사항]
- <documents> 태그 안의 내용만 사실로 간주하세요.
- 문서에 없는 내용은 "문서에 명시되지 않았으나, 일반적인 지식으로는..."이라고 구분하여 답하세요.
- 맹목적인 친절함보다는 정확한 기술적 솔루션을 우선하세요."""
    
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
        stream: bool = False
    ) -> Any:
        """vLLM 호출"""
        global vllm_client
        
        if vllm_client is None:
            vllm_client = openai.OpenAI(
                api_key='sk-not-needed',
                base_url=VLLM_API_URL
            )
        
        try:
            logger.info('📡 vLLM 호출 중...')
            
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
                    "---",
                    "[관련 문서 없음]",
                    "<｜end of sentence｜>"
                ],
                presence_penalty=0.6,
                frequency_penalty=0.6
            )
            
            logger.info('✅ vLLM 응답 수신')
            return response
            
        except Exception as e:
            logger.error(f'❌ vLLM 호출 실패: {str(e)}', exc_info=True)
            raise
    
    @staticmethod
    def stream_response(response) -> Generator[str, None, None]:
        """스트리밍 응답 생성"""
        try:
            in_think = False
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
                        
                        # <think> 태그 필터링
                        if '<think>' in content:
                            in_think = True
                        
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
            
            logger.info(f'✅ 스트리밍 완료 ({chunk_count}개 청크)')
            
            # 종료
            yield f"data: {json.dumps({'choices': [{'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            logger.error(f'❌ 스트리밍 오류: {str(e)}', exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"


# ==================== 초기화 ====================

@app.on_event('startup')
async def startup():
    """서버 시작"""
    global vector_store
    
    logger.info('=' * 70)
    logger.info('RAG API 시작 중...')
    logger.info('=' * 70)
    
    try:
        # 벡터 DB 초기화
        logger.info('1️⃣  벡터 DB 초기화...')
        vector_store = VectorStoreManager(
            chroma_host=CHROMA_HOST,
            chroma_port=CHROMA_PORT
        )
        stats = vector_store.get_collection_stats()
        logger.info(f'✅ 벡터 DB 준비: {stats.get("document_count", 0)}개 문서')
        
        # 모델 정보 로드
        logger.info('2️⃣  모델 정보 로드...')
        models = ModelManager.get_models()
        logger.info(f'✅ {len(models)}개 모델 감지')
        
        logger.info('=' * 70)
        logger.info('🚀 RAG API 준비 완료!')
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
        return {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'vector_store': stats,
            'model': MODEL_NAME,
            'documents': stats.get('document_count', 0)
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
        # 'data': models
        "data": [{"id": "DeepSeek-R1-Distill-Qwen-14B", "object": "model"}]
    }


@app.post('/v1/chat/completions')
async def chat_completions(request: ChatCompletionRequest) -> Any:
    """
    Chat Completion 엔드포인트
    
    플로우:
    1. 사용자 질문 추출
    2. ChromaDB에서 관련 문서 검색 (XML 포맷)
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
        
        # 메시지 재구성
        vllm_messages = [
            {'role': 'system', 'content': QuestionAnsweringManager.SYSTEM_PROMPT}
        ]
        
        # 과거 대화 추가
        for msg in request.messages[:-1]:
            if msg.role in ['user', 'assistant']:
                vllm_messages.append({'role': msg.role, 'content': msg.content})
        
        # 최종 사용자 메시지
        vllm_messages.append({'role': 'user', 'content': final_user_content})
        
        logger.info(f'메시지: System + History({len(vllm_messages)-2}) + Current = {len(vllm_messages)}개')
        
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
