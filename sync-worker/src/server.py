"""
RAG Inference API Server - FastAPI 기반

OpenAI 호환 포맷의 API를 제공하며, Vector Store에서 검색한 문서를 바탕으로
vLLM 서버에서 답변을 생성합니다. Open WebUI와 통합됩니다.

엔드포인트:
- POST /v1/chat/completions (OpenAI 호환)
- GET /health (헬스 체크)
"""

import os
import sys
import logging
from typing import Optional, List, Generator
from datetime import datetime
import time

# 모듈 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
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

# FastAPI 앱 생성
app = FastAPI(
    title='RAG Inference API',
    description='Local Enterprise RAG - OpenAI Compatible API',
    version='1.0.0'
)

# 전역 설정
LLM_BACKEND = os.getenv('LLM_BACKEND', 'auto').lower()
VLLM_API_URL = os.getenv('VLLM_API_URL', 'http://localhost:8000/v1')
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
CHROMA_HOST = os.getenv('CHROMA_HOST', 'localhost')
CHROMA_PORT = int(os.getenv('CHROMA_PORT', '8000'))
SEARCH_TOP_K = int(os.getenv('SEARCH_TOP_K', '5'))

# LLM 백엔드 결정 로직
if LLM_BACKEND == 'auto':
    # 자동 선택: vLLM 먼저 시도, 실패하면 Ollama
    try:
        requests.get(VLLM_API_URL.replace('/v1', '/health'), timeout=2)
        LLM_BACKEND = 'vllm'
        LLM_API_URL = VLLM_API_URL
    except:
        LLM_BACKEND = 'ollama'
        LLM_API_URL = f'{OLLAMA_API_URL}/api'
elif LLM_BACKEND == 'vllm':
    LLM_API_URL = VLLM_API_URL
elif LLM_BACKEND == 'ollama':
    LLM_API_URL = f'{OLLAMA_API_URL}/api'
else:
    raise ValueError(f'Unknown LLM_BACKEND: {LLM_BACKEND}')

logger.info(f'LLM Backend: {LLM_BACKEND.upper()} ({LLM_API_URL})')

# 시스템 프롬프트
SYSTEM_PROMPT = """너는 기업 내부 AI 어시스턴트다. 
아래 제공된 Context(코드, 문서)를 기반으로 정확하고 도움이 되는 답변을 제공하자.
Context가 없으면 "관련 정보를 찾을 수 없습니다"라고 답변하자.
답변은 명확하고 구조화되어 있어야 한다."""

# ==================== 데이터 모델 ====================

class ChatMessage(BaseModel):
    """채팅 메시지 모델"""
    role: str = Field(..., description="메시지 역할 (system/user/assistant)")
    content: str = Field(..., description="메시지 내용")


class ChatCompletionRequest(BaseModel):
    """Chat Completion 요청 모델 (OpenAI 호환)"""
    model: str = Field(default="qwen2.5-coder", description="사용할 모델")
    messages: List[ChatMessage] = Field(..., description="메시지 목록")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=2048)
    stream: bool = Field(default=False, description="스트리밍 응답 여부")


class ChatCompletionResponse(BaseModel):
    """Chat Completion 응답 모델"""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]
    usage: dict


# ==================== 벡터 스토어 초기화 ====================

@app.on_event('startup')
async def startup_event():
    """서버 시작 시 초기화"""
    global vector_store
    
    logger.info('=' * 70)
    logger.info('RAG Inference API Server 시작')
    logger.info('=' * 70)
    
    try:
        # VectorStoreManager 초기화
        vector_store = VectorStoreManager(
            chroma_host=CHROMA_HOST,
            chroma_port=CHROMA_PORT
        )
        logger.info('✓ VectorStoreManager 초기화 완료')
        
        # ChromaDB 연결 테스트
        stats = vector_store.get_collection_stats()
        logger.info(f'✓ ChromaDB 연결 성공')
        logger.info(f'  - 저장된 문서: {stats.get("document_count", 0)}개')
        
        # vLLM 서버 연결 테스트
        try:
            response = requests.get(f'{VLLM_API_URL.rsplit("/", 1)[0]}/models', timeout=5)
            logger.info('✓ vLLM API 연결 성공')
        except Exception as e:
            logger.warning(f'⚠ vLLM API 연결 실패: {str(e)}')
        
        logger.info('=' * 70)
        logger.info(f'VLLM_API_URL: {VLLM_API_URL}')
        logger.info(f'CHROMA_HOST: {CHROMA_HOST}:{CHROMA_PORT}')
        logger.info(f'SEARCH_TOP_K: {SEARCH_TOP_K}')
        logger.info('=' * 70)
        
    except Exception as e:
        logger.error(f'서버 초기화 실패: {str(e)}', exc_info=True)
        raise


# ==================== 헬퍼 함수 ====================

def retrieve_context(query: str, top_k: int = SEARCH_TOP_K) -> str:
    """
    벡터 스토어에서 관련 문서 검색
    
    Args:
        query: 검색 쿼리
        top_k: 상위 문서 개수
    
    Returns:
        포맷팅된 Context 문자열
    """
    try:
        logger.debug(f'Context 검색 중: "{query}" (top_k={top_k})')
        
        # 유사도 검색
        search_results = vector_store.search(query, top_k=top_k)
        
        if not search_results:
            logger.info('검색 결과 없음')
            return ""
        
        # Context 포맷팅
        context_parts = []
        for i, result in enumerate(search_results, 1):
            doc_id = result['id']
            content = result['content']
            metadata = result['metadata']
            
            # 문서 출처 정보
            source = metadata.get('source', '알 수 없음')
            
            # 구조화된 Context
            context_parts.append(f"""
---
[문서 #{i}]
출처: {source}
내용:
{content}
---""")
        
        context = "".join(context_parts)
        logger.debug(f'검색된 문서: {len(search_results)}개')
        
        return context
        
    except Exception as e:
        logger.error(f'Context 검색 실패: {str(e)}', exc_info=True)
        return ""


def build_prompt(user_message: str, context: str = "") -> str:
    """
    사용자 메시지와 Context를 결합하여 최종 프롬프트 생성
    
    Args:
        user_message: 사용자 질문
        context: Context (검색 결과)
    
    Returns:
        최종 프롬프트
    """
    if context:
        prompt = f"""{SYSTEM_PROMPT}

## 제공된 Context:
{context}

## 사용자 질문:
{user_message}"""
    else:
        prompt = f"""{SYSTEM_PROMPT}

## 사용자 질문:
{user_message}"""
    
    return prompt


def call_vllm_api(
    messages: List[dict],
    temperature: float = 0.7,
    top_p: float = 0.95,
    max_tokens: Optional[int] = 2048,
    stream: bool = False
) -> dict | Generator:
    """
    vLLM API 호출
    
    Args:
        messages: 메시지 목록
        temperature: 온도 파라미터
        top_p: top-p 파라미터
        max_tokens: 최대 토큰 수
        stream: 스트리밍 여부
    
    Returns:
        API 응답 또는 스트리밍 제너레이터
    """
    try:
        if LLM_BACKEND == 'vllm':
            logger.debug(f'vLLM API 호출: stream={stream}')
            
            # OpenAI 클라이언트 생성 (vLLM은 OpenAI 호환)
            client = openai.OpenAI(
                api_key='sk-not-needed',  # vLLM은 인증 필수 아님
                base_url=VLLM_API_URL
            )
            
            # API 호출
            response = client.chat.completions.create(
                model='qwen2.5-coder',
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                stream=stream
            )
            
            return response
        
        else:  # Ollama
            logger.debug(f'Ollama API 호출: stream={stream}')
            
            # Ollama REST API 호출
            headers = {'Content-Type': 'application/json'}
            
            # 메시지 변환
            messages_for_ollama = []
            for msg in messages:
                messages_for_ollama.append({
                    'role': msg.get('role', 'user'),
                    'content': msg.get('content', '')
                })
            
            payload = {
                'model': 'qwen2.5',
                'messages': messages_for_ollama,
                'stream': stream,
                'options': {
                    'temperature': temperature,
                    'top_p': top_p,
                }
            }
            
            if max_tokens:
                payload['options']['num_predict'] = max_tokens
            
            # API 호출
            response = requests.post(
                f'{OLLAMA_API_URL}/api/chat',
                json=payload,
                headers=headers,
                stream=stream,
                timeout=300
            )
            
            if response.status_code != 200:
                raise Exception(f'Ollama API 오류: {response.text}')
            
            # 응답 형식 변환 (OpenAI 호환)
            if not stream:
                data = response.json()
                # OpenAI 형식으로 변환
                return {
                    'id': 'ollama-' + str(int(time.time())),
                    'object': 'chat.completion',
                    'created': int(time.time()),
                    'model': 'qwen2.5',
                    'choices': [
                        {
                            'index': 0,
                            'message': {
                                'role': 'assistant',
                                'content': data.get('message', {}).get('content', '')
                            },
                            'finish_reason': 'stop'
                        }
                    ],
                    'usage': {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0
                    }
                }
            else:
                # 스트리밍 응답
                return response
        
    except Exception as e:
        logger.error(f'LLM API 호출 실패 ({LLM_BACKEND}): {str(e)}', exc_info=True)
        raise


def stream_response_generator(response) -> Generator[str, None, None]:
    """
    vLLM 스트리밍 응답을 OpenAI 호환 형식으로 변환
    
    Args:
        response: vLLM 스트리밍 응답
    
    Yields:
        OpenAI 호환 포맷의 데이터 라인
    """
    try:
        for chunk in response:
            # 스트림 청크 처리
            if chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                
                if choice.delta and choice.delta.content:
                    # OpenAI 호환 포맷으로 변환
                    data = {
                        "id": f"chatcmpl-{datetime.now().timestamp()}",
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now().timestamp()),
                        "model": "qwen2.5-coder",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": choice.delta.content
                                },
                                "finish_reason": None
                            }
                        ]
                    }
                    
                    yield f"data: {str(data)}\n\n"
        
        # 종료 신호
        final_data = {
            "id": f"chatcmpl-{datetime.now().timestamp()}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": "qwen2.5-coder",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {str(final_data)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f'스트림 처리 중 오류: {str(e)}', exc_info=True)
        error_data = {"error": f"Stream processing failed: {str(e)}"}
        yield f"data: {str(error_data)}\n\n"


# ==================== API 엔드포인트 ====================

@app.get('/health', tags=['Health'])
async def health_check():
    """헬스 체크 엔드포인트"""
    try:
        stats = vector_store.get_collection_stats()
        return {
            'status': 'ok',
            'timestamp': datetime.now().isoformat(),
            'vector_store': stats,
            'vllm_api': VLLM_API_URL
        }
    except Exception as e:
        logger.error(f'헬스 체크 실패: {str(e)}')
        return {
            'status': 'unhealthy',
            'error': str(e)
        }


@app.post('/v1/chat/completions', response_model=Optional[ChatCompletionResponse], tags=['Chat'])
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI 호환 Chat Completion 엔드포인트
    
    RAG 프롬프트:
    1. 사용자 질문 추출
    2. 벡터 저장소에서 관련 문서 검색
    3. Context와 함께 프롬프트 구성
    4. vLLM에 전송 및 답변 생성
    5. 스트리밍 또는 일반 응답 반환
    """
    logger.info('=' * 70)
    logger.info(f'[Chat Completion] 요청 수신')
    logger.info(f'  - 메시지 수: {len(request.messages)}')
    logger.info(f'  - 스트리밍: {request.stream}')
    logger.info('=' * 70)
    
    try:
        # 사용자 질문 추출 (마지막 user 메시지)
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == 'user':
                user_message = msg.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="사용자 메시지를 찾을 수 없습니다")
        
        logger.info(f'사용자 질문: {user_message[:100]}...')
        
        # Step 1: Context 검색 (RAG)
        logger.info('Step 1: Vector Store에서 Context 검색')
        context = retrieve_context(user_message, top_k=SEARCH_TOP_K)
        
        if context:
            logger.info(f'  ✓ {len(context)}자 Context 검색됨')
        else:
            logger.info('  ⚠ 관련 Context 없음')
        
        # Step 2: 최종 프롬프트 구성
        logger.info('Step 2: 최종 프롬프트 구성')
        final_prompt = build_prompt(user_message, context)
        logger.debug(f'프롬프트 길이: {len(final_prompt)}자')
        
        # Step 3: vLLM 호출을 위한 메시지 구성
        vllm_messages = [
            {
                'role': 'system',
                'content': SYSTEM_PROMPT
            }
        ]
        
        # 기존 메시지 중 마지막 user 메시지 이전까지 포함 (conversation history)
        for msg in request.messages:
            if msg.role in ['user', 'assistant']:
                vllm_messages.append({
                    'role': msg.role,
                    'content': msg.content
                })
        
        # RAG Context를 마지막 user 메시지에 주입
        if context:
            vllm_messages[-1]['content'] = f"""{SYSTEM_PROMPT}

## 제공된 Context:
{context}

## 질문:
{user_message}"""
        
        logger.info('Step 3: vLLM API 호출')
        
        # Step 4: vLLM 호출
        response = call_vllm_api(
            messages=vllm_messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=request.stream
        )
        
        # Step 5: 응답 반환
        if request.stream:
            logger.info('스트리밍 응답 반환')
            return StreamingResponse(
                stream_response_generator(response),
                media_type='text/event-stream'
            )
        else:
            logger.info('일반 응답 반환')
            
            # 응답 객체 구성
            assistant_message = response.choices[0].message.content
            
            return {
                'id': f'chatcmpl-{datetime.now().timestamp()}',
                'object': 'chat.completion',
                'created': int(datetime.now().timestamp()),
                'model': 'qwen2.5-coder',
                'choices': [
                    {
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'content': assistant_message
                        },
                        'finish_reason': 'stop'
                    }
                ],
                'usage': {
                    'prompt_tokens': response.usage.prompt_tokens,
                    'completion_tokens': response.usage.completion_tokens,
                    'total_tokens': response.usage.total_tokens
                }
            }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f'Chat Completion 처리 중 오류: {str(e)}', exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f'Chat Completion 처리 실패: {str(e)}'
        )


# ==================== 서버 실행 ====================

if __name__ == '__main__':
    host = os.getenv('SERVER_HOST', '0.0.0.0')
    port = int(os.getenv('SERVER_PORT', '8010'))
    
    logger.info(f'Server will run on {host}:{port}')
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
        access_log=True
    )
