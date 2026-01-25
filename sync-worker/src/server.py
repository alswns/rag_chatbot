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
import json
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
SYSTEM_PROMPT_TEMPLATE = """### SYSTEM_PROMPT
당신은 박민준 개발자의 프로젝트를 전담하는 'Senior Full-Stack AI Engineer'입니다.
당신의 임무는 <CONTEXT>로 제공된 문서와 당신의 지식을 결합하여, 사용자의 질문에 가장 정확하고 실용적인 답변을 제공하는 것입니다.

[핵심 행동 강령]
1. **NO ROBOTIC INTRO**: "안녕하세요", "저는 AI입니다", "문서를 확인해 보니" 같은 쓸데없는 서론을 **절대 금지**합니다. 바로 핵심 답변부터 시작하세요.
2. **Context Synthesis (문맥 종합)**:
   - <CONTEXT>에 나열된 문서들을 하나씩 읽어주는 앵무새가 되지 마세요.
   - 문서들의 내용을 **종합하고 분석**하여, 사용자가 이해하기 쉬운 **하나의 완결된 문장/문단**으로 답변하세요.
   - 예: "[문서1]은 A, [문서2]는 B입니다" (X) -> "제공된 자료에 따르면 A와 B가 주요 특징입니다." (O)
3. **Hallucination Control (환각 제어)**:
   - <CONTEXT> 내용이 비어있거나 의미 없는 문자열이라면, 솔직하게 "관련 문서가 비어있어 내용을 확인할 수 없습니다."라고 말하세요.
   - 절대로 "암호화되어 있다", "보안상 알 수 없다"라고 추측해서 거짓말하지 마세요.
4. **Code First**:
   - 코드에 대한 질문이면 설명보다는 **실행 가능한 코드 블록**을 우선적으로 보여주세요.
5. **Tone & Manner**:
   - 전문적이고 건조한(Dry) 개발자 톤을 유지하세요.
   - 불확실한 내용은 "추정됩니다"라고 명확히 하세요.

---

### <CONTEXT> (RAG 검색 결과)
{retrieved_documents}

---

### <USER_QUERY> (사용자 질문)
{user_query}

### <ANSWER_GENERATION>
위 [핵심 행동 강령]을 준수하여 답변을 작성하세요:
"""

# 글로벌 변수
available_models = []
vector_store = None
vllm_client = None  # ✅ 전역 vLLM 클라이언트 (연결 최적화)

# ==================== 데이터 모델 ====================

class ChatMessage(BaseModel):
    """채팅 메시지 모델"""
    role: str = Field(..., description="메시지 역할 (system/user/assistant)")
    content: str = Field(..., description="메시지 내용")


class ChatCompletionRequest(BaseModel):
    """Chat Completion 요청 모델 (OpenAI 호환)"""
    model: str = Field(default="Qwen/Qwen2.5-Coder-14B", description="사용할 모델")
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
    global vector_store, vllm_client
    
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
        
        # vLLM 서버 연결 테스트 및 모델 목록 로드
        global available_models
        logger.info('🔄 vLLM 모델 로드 시작...')
        logger.info(f'요청 URL: {VLLM_API_URL}/models')
        
        try:
            logger.debug(f'vLLM 요청 중... (타임아웃: 5초)')
            response = requests.get(f'{VLLM_API_URL}/models', timeout=5)
            logger.debug(f'vLLM 응답 상태: {response.status_code}')
            logger.debug(f'vLLM 응답 헤더: {response.headers}')
            
            if response.status_code != 200:
                raise Exception(f'HTTP {response.status_code}: {response.text}')
            
            models_data = response.json()
            logger.debug(f'vLLM 응답 데이터: {models_data}')
            
            available_models = models_data.get('data', [])
            logger.info(f'✓ vLLM API 연결 성공 - {len(available_models)}개 모델 감지')
            for model in available_models:
                logger.info(f'  - {model.get("id")}')
        except Exception as e:
            logger.error(f'⚠ vLLM API 연결 실패: {str(e)}', exc_info=True)
            logger.warning('기본 모델 설정 사용')
            available_models = [{
                'id': 'Qwen/Qwen2.5-Coder-14B',
                'object': 'model',
                'created': int(datetime.now().timestamp()),
                'owned_by': 'vllm'
            }]
            logger.info(f'기본 모델 설정됨: {len(available_models)}개')
        
        logger.info(f'최종 available_models 상태: {len(available_models)}개 모델')
        
        logger.info('=' * 70)
        logger.info(f'VLLM_API_URL: {VLLM_API_URL}')
        logger.info(f'CHROMA_HOST: {CHROMA_HOST}:{CHROMA_PORT}')
        logger.info(f'SEARCH_TOP_K: {SEARCH_TOP_K}')
        
        # ✅ vLLM 클라이언트 미리 초기화 (연결 최적화)
        if LLM_BACKEND == 'vllm':
            vllm_client = openai.OpenAI(
                api_key='sk-not-needed',
                base_url=VLLM_API_URL
            )
            logger.info('✓ vLLM 클라이언트 초기화 완료')
        
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
        
        # 유사도 임계값 필터링 (매우 관련성 높은 문서만)
        # ChromaDB는 distance를 반환 (작을수록 유사)
        filtered_results = []
        for result in search_results:
            distance = result.get('distance', float('inf'))
            similarity_score = result.get('similarity_score', 0)
            
            logger.info(f'  검색결과: 거리={distance:.3f}, 유사도={similarity_score:.3f}, 문서={result.get("id")}')
            
            # 거리 0.4 이하만 포함 (적절한 수준의 관련성)
            if distance <= 0.4:  # ✅ 0.2 → 0.4 완화 (너무 엄격한 임계값 개선)
                filtered_results.append(result)
                logger.info(f'    ✓ 포함됨')
            else:
                logger.info(f'    ✗ 제외됨 (거리 임계값 0.2 초과)')
        
        if not filtered_results:
            logger.warning('⚠️ 임계값을 초과하는 검색 결과 - Context 제공 안함')
            return ""
        
        # Context 포맷팅
        context_parts = []
        for i, result in enumerate(filtered_results, 1):
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
        logger.info(f'✓ 검색된 관련 문서: {len(filtered_results)}개')
        
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
        최종 프롬프트 (System Prompt Template에 값 주입)
    """
    # Context가 없으면 "정보 없음" 표시
    retrieved_docs = context if context else "[관련 문서 없음]"
    
    # System Prompt Template에 값 주입
    prompt = SYSTEM_PROMPT_TEMPLATE.format(
        retrieved_documents=retrieved_docs,
        user_query=user_message
    )
    
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
    global vllm_client  # ✅ 전역 변수 선언 추가
    
    try:
        if LLM_BACKEND == 'vllm':
            logger.debug(f'vLLM API 호출: stream={stream}')
            
            # ✅ 전역 클라이언트 사용 (이미 초기화됨)
            if vllm_client is None:
                logger.warning('vLLM 클라이언트가 초기화되지 않았습니다. 지금 생성합니다.')
                vllm_client = openai.OpenAI(
                    api_key='sk-not-needed',
                    base_url=VLLM_API_URL
                )
            
            # API 호출
            response = vllm_client.chat.completions.create(
                model='Qwen/Qwen2.5-Coder-14B',
                messages=messages,
                temperature=0.3,  # ✅ 0.7 → 0.3 (정확성 강화, 창의성 감소)
                top_p=top_p,
                max_tokens=max_tokens,
                stream=stream,
                # ✅ 반복 방지 파라미터
                stop=["<|file_sep|>", "### <CONTEXT>", "---", "[관련 문서 없음]"],
                presence_penalty=0.6,  # ✅ 같은 주제 반복 방지 강화 (0.5 → 0.6)
                frequency_penalty=0.6  # ✅ 같은 단어 반복 방지 강화 (0.5 → 0.6)
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
                    'repeat_penalty': 1.15,  # 반복 억제
                    'stop': ["<|file_sep|>", "### <CONTEXT>", "---"]  # Stop sequences
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
    logger.info('스트리밍 생성 시작')
    
    try:
        chunk_count = 0
        
        for chunk in response:
            chunk_count += 1
            
            # 스트림 청크 처리
            if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                
                if hasattr(choice, 'delta') and choice.delta and hasattr(choice.delta, 'content') and choice.delta.content:
                    content = choice.delta.content
                    logger.debug(f'청크 #{chunk_count}: {len(content)}자')
                    
                    # OpenAI 호환 포맷으로 변환
                    data = {
                        "id": f"chatcmpl-{datetime.now().timestamp()}",
                        "object": "chat.completion.chunk",
                        "created": int(datetime.now().timestamp()),
                        "model": "Qwen/Qwen2.5-Coder-14B",
                        "choices": [
                            {
                                "index": 0,
                                "delta": {
                                    "content": content
                                },
                                "finish_reason": None
                            }
                        ]
                    }
                    
                    # ✅ JSON 형식으로 변환 (str() 대신 json.dumps() 사용)
                    yield f"data: {json.dumps(data)}\n\n"
        
        logger.info(f'✓ 스트리밍 완료 - 총 {chunk_count}개 청크')
        
        # 종료 신호
        final_data = {
            "id": f"chatcmpl-{datetime.now().timestamp()}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": "Qwen/Qwen2.5-Coder-14B",
            "choices": [
                {
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }
            ]
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f'스트림 처리 중 오류: {str(e)}', exc_info=True)
        error_data = {"error": f"Stream processing failed: {str(e)}"}
        yield f"data: {json.dumps(error_data)}\n\n"


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


@app.get('/v1/models', tags=['Models'])
async def list_models():
    """vLLM의 모델 목록 반환 (OpenAI 호환)"""
    global available_models
    return {
        'object': 'list',
        'data': available_models if available_models else [{
            'id': 'Qwen/Qwen2.5-Coder-14B',
            'object': 'model',
            'created': int(datetime.now().timestamp()),
            'owned_by': 'vllm'
        }]
    }


@app.post('/v1/chat/completions', tags=['Chat'])
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
        
        # Step 3: vLLM 호출을 위한 메시지 구성 (수정됨 - 순서 중요!) ✅
        logger.info('Step 3: 메시지 구조 재구성 (System + History + Current)')
        
        # ✅ 1. 순수 시스템 프롬프트 (규칙만 정의, Context나 Query는 제외)
        pure_system_prompt = """당신은 박민준 개발자의 프로젝트를 돕는 숙련된 풀스택 AI 엔지니어입니다.
아래 제공될 <CONTEXT> 데이터를 바탕으로 질문에 답하되, [운영 규칙]을 엄격히 준수하세요.

[운영 규칙]
1. 지식 소스 활용: <CONTEXT> 섹션의 내용이 질문과 관련 있으면 최우선으로 참고하세요.
2. 답변 스타일: 핵심 결론부터 말하고, 불필요한 서론은 생략하세요.
3. 중복 방지: 동일한 내용을 반복하지 말고 하나로 통합하여 요약하세요.
4. 정직성: 정보가 없으면 모른다고 답하세요."""
        
        vllm_messages = []
        vllm_messages.append({'role': 'system', 'content': pure_system_prompt})
        
        # ✅ 2. 과거 대화 내역 추가 (마지막 user 메시지는 제외)
        history_messages = [msg for msg in request.messages[:-1] if msg.role in ['user', 'assistant']]
        for msg in history_messages:
            vllm_messages.append({
                'role': msg.role,
                'content': msg.content
            })
        
        # ✅ 3. 마지막 User 메시지에 Context + Query 결합 (가장 중요!)
        # 모델이 가장 마지막에 볼 내용이므로 여기에 Context를 넣어 집중력을 높입니다.
        final_user_content = f"""### <CONTEXT>
{context if context else "[관련 문서 없음]"}

---

### <USER_QUERY>
{user_message}
"""
        vllm_messages.append({'role': 'user', 'content': final_user_content})
        
        logger.info(f'최종 메시지 구성: System(1) + History({len(history_messages)}) + Current(1) = {len(vllm_messages)}개')
        
        logger.info('Step 3: vLLM API 호출')
        
        # Step 4: vLLM 호출
        logger.info(f'vLLM 호출 중... (메시지 {len(vllm_messages)}개)')
        response = call_vllm_api(
            messages=vllm_messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=request.stream
        )
        logger.info(f'vLLM 응답 수신: stream={request.stream}')
        logger.debug(f'응답 타입: {type(response)}')
        
        # Step 5: 응답 반환
        if request.stream:
            logger.info('스트리밍 응답 반환')
            return StreamingResponse(
                stream_response_generator(response),
                media_type='text/event-stream',
                headers={
                    'Cache-Control': 'no-cache',
                    'X-Accel-Buffering': 'no',
                    'Connection': 'keep-alive'
                }
            )
        else:
            logger.info('일반 응답 반환')
            
            try:
                # 응답 객체 구성
                logger.debug('응답 처리 중...')
                logger.debug(f'response.choices: {response.choices}')
                
                if not response.choices or len(response.choices) == 0:
                    logger.error('응답에 choices가 없습니다')
                    raise HTTPException(status_code=500, detail='vLLM 응답 형식 오류')
                
                assistant_message = response.choices[0].message.content
                logger.debug(f'생성된 응답 길이: {len(assistant_message)}자')
                logger.info(f'✓ 응답 생성 완료: {assistant_message[:100]}...')
            except Exception as e:
                logger.error(f'응답 처리 중 오류: {str(e)}', exc_info=True)
                raise
            
            return {
                'id': f'chatcmpl-{datetime.now().timestamp()}',
                'object': 'chat.completion',
                'created': int(datetime.now().timestamp()),
                'model': 'Qwen/Qwen2.5-Coder-14B',
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
