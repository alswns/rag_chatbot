"""
RAG Inference API Server - FastAPI 기반

OpenAI 호환 포맷의 API를 제공하며, Vector Store에서 검색한 문서를 바탕으로
vLLM 서버에서 답변을 생성합니다. Open WebUI와 통합됩니다.

주요 기능:
- DeepSeek-R1 추론 모델 기반 RAG
- <think> 태그 처리 (추론 과정 숨김)
- ChromaDB Vector Store 통합
- OpenAI API 호환성

엔드포인트:
- POST /v1/chat/completions (OpenAI 호환)
- GET /health (헬스 체크)
"""

import os
import sys
import logging
import json
import re
from typing import Optional, List, Generator
from datetime import datetime
import time

# 모듈 경로 추가
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

# FastAPI 앱 생성
app = FastAPI(
    title='RAG Inference API',
    description='Enterprise RAG - OpenAI Compatible API (DeepSeek-R1)',
    version='1.0.0'
)

# ✅ [Chief Cloud Architect] 모델명 중앙화 - docker-compose 환경변수 반영
# 기본값: DeepSeek-R1-Distill-Qwen-14B-AWQ
# 환경변수 LLM_MODEL_ID로 오버라이딩 가능
MODEL_NAME = os.getenv('LLM_MODEL_ID', 'casperhansen/deepseek-r1-distill-qwen-14b-awq')
logger.info(f'✅ [Chief Cloud Architect] 사용 모델: {MODEL_NAME}')

# 전역 설정
LLM_BACKEND = os.getenv('LLM_BACKEND', 'vllm').lower()
VLLM_API_URL = os.getenv('VLLM_API_URL', 'http://localhost:8000/v1')
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')
CHROMA_HOST = os.getenv('CHROMA_HOST', 'localhost')
CHROMA_PORT = int(os.getenv('CHROMA_PORT', '8000'))
SEARCH_TOP_K = int(os.getenv('SEARCH_TOP_K', '5'))

logger.info(f'LLM Backend: {LLM_BACKEND.upper()} ({VLLM_API_URL})')

# 시스템 프롬프트
SYSTEM_PROMPT_TEMPLATE = """### SYSTEM_PROMPT
당신은 박민준 개발자의 프로젝트를 전담하는 'Senior Full-Stack AI Engineer'입니다.
당신의 임무는 <CONTEXT>로 제공된 문서와 당신의 지식을 결합하여, 사용자의 질문에 가장 정확하고 실용적인 답변을 제공하는 것입니다.

[핵심 행동 강령]
1. **NO ROBOTIC INTRO**: 쓸데없는 서론을 절대 금지하고, 바로 핵심 답변부터 시작하세요.
2. **Context Synthesis**: 문서를 종합하여 하나의 완결된 문장/문단으로 답변하세요.
3. **Hallucination Control**: 정보가 없으면 솔직하게 모른다고 답하세요.
4. **Code First**: 코드에 대한 질문이면 설명보다는 실행 가능한 코드를 우선적으로 보여주세요.
5. **Tone & Manner**: 전문적이고 건조한 개발자 톤을 유지하세요.

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
vllm_client = None


# ==================== 헬퍼 함수: <think> 태그 처리 ====================

def extract_think_and_response(text: str) -> tuple[str, str]:
    """
    ✅ [Chief Cloud Architect] <think> 태그 처리
    
    DeepSeek-R1의 추론 과정(<think>...</think>)을 분리합니다.
    - 추론 내용은 로그에만 기록
    - 사용자에게는 최종 답변만 반환
    
    Args:
        text: 모델이 생성한 전체 텍스트
    
    Returns:
        (think_content, response_content) 튜플
    """
    # <think> 태그 패턴 (비탐욕적 매칭)
    think_pattern = r'<think>(.*?)</think>'
    
    match = re.search(think_pattern, text, re.DOTALL)
    
    if match:
        think_content = match.group(1).strip()
        # <think> 태그를 제거한 최종 응답
        response_content = re.sub(think_pattern, '', text, flags=re.DOTALL).strip()
        
        logger.debug(f'[<think> 추론 과정] ({len(think_content)}자)')
        logger.debug(f'추론 내용:\n{think_content[:200]}...')
        
        return think_content, response_content
    
    # <think> 태그가 없으면 전체 텍스트가 응답
    return "", text


def process_streaming_chunk(content: str) -> tuple[str, Optional[str]]:
    """
    ✅ 스트리밍 중 <think> 태그 처리
    
    스트리밍 청크에서 <think> 태그를 감지하고 제거합니다.
    
    Args:
        content: 스트리밍 청크 텍스트
    
    Returns:
        (정제된_텍스트, think_시작_여부)
    """
    # <think> 태그 시작 감지
    if '<think>' in content:
        # <think>부터 </think>까지만 필터링
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        return cleaned, True
    
    if '</think>' in content:
        # </think> 이후만 반환
        cleaned = re.sub(r'<think>.*?</think>', '', content, flags=re.DOTALL)
        return cleaned, False
    
    return content, None


# ==================== 데이터 모델 ====================

class ChatMessage(BaseModel):
    role: str = Field(..., description="메시지 역할 (system/user/assistant)")
    content: str = Field(..., description="메시지 내용")


class ChatCompletionRequest(BaseModel):
    # ✅ 기본 모델명을 MODEL_NAME 상수로 설정
    model: str = Field(default=MODEL_NAME, description="사용할 모델")
    messages: List[ChatMessage] = Field(..., description="메시지 목록")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(default=2048)
    stream: bool = Field(default=False, description="스트리밍 응답 여부")


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str = MODEL_NAME
    choices: List[dict]
    usage: dict


# ==================== 벡터 스토어 초기화 ====================

@app.on_event('startup')
async def startup_event():
    """서버 시작 시 초기화"""
    global vector_store, vllm_client, available_models
    
    logger.info('=' * 70)
    logger.info('RAG Inference API Server 시작')
    logger.info('=' * 70)
    
    try:
        vector_store = VectorStoreManager(
            chroma_host=CHROMA_HOST,
            chroma_port=CHROMA_PORT
        )
        logger.info('✓ VectorStoreManager 초기화 완료')
        
        stats = vector_store.get_collection_stats()
        logger.info(f'✓ ChromaDB 연결 성공 - {stats.get("document_count", 0)}개 문서')
        
        logger.info('🔄 vLLM 모델 로드 중...')
        
        try:
            response = requests.get(f'{VLLM_API_URL}/models', timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                available_models = models_data.get('data', [])
                logger.info(f'✓ vLLM API 연결 성공 - {len(available_models)}개 모델 감지')
            else:
                raise Exception(f'HTTP {response.status_code}')
        except Exception as e:
            logger.error(f'⚠ vLLM API 연결 실패: {str(e)}')
            available_models = [{
                'id': MODEL_NAME,
                'object': 'model',
                'created': int(datetime.now().timestamp()),
                'owned_by': 'vllm'
            }]
        
        logger.info(f'✅ 사용 모델: {MODEL_NAME}')
        
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
    """벡터 스토어에서 관련 문서 검색"""
    try:
        logger.debug(f'Context 검색: "{query}" (top_k={top_k})')
        
        search_results = vector_store.search(query, top_k=top_k)
        
        if not search_results:
            return ""
        
        filtered_results = []
        for result in search_results:
            distance = result.get('distance', float('inf'))
            if distance <= 0.4:
                filtered_results.append(result)
        
        if not filtered_results:
            return ""
        
        # ✅ [DeepSeek-R1 최적화] XML 포맷팅으로 문서 경계 명확화
        # Prompt Injection 방지 및 모델의 정확한 데이터 인식 지원
        context_parts = []
        for i, result in enumerate(filtered_results, 1):
            content = result['content']
            metadata = result['metadata']
            source = metadata.get('source', '알 수 없음')
            
            # XML 태그를 사용하여 문서 경계를 명확히 함
            context_parts.append(f"""    <doc id="{i}">
        <source>{source}</source>
        <content>{content}</content>
    </doc>""")
        
        # 전체를 documents 태그로 감싸서 구조화
        context = "<documents>\n" + "\n".join(context_parts) + "\n</documents>"
        logger.info(f'✓ {len(filtered_results)}개 문서 (XML 포맷) 검색')
        return context
        
    except Exception as e:
        logger.error(f'Context 검색 실패: {str(e)}')
        return ""


def build_prompt(user_message: str, context: str = "") -> str:
    """사용자 메시지와 Context를 결합하여 최종 프롬프트 생성"""
    retrieved_docs = context if context else "[관련 문서 없음]"
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
    ✅ [Chief Cloud Architect] vLLM API 호출 (DeepSeek-R1 최적화)
    
    Args:
        messages: 메시지 목록
        temperature: 온도 파라미터 (0.6 권장 - 추론 강화)
        top_p: top-p 파라미터
        max_tokens: 최대 토큰 수 (8192 권장 - RAG Context)
        stream: 스트리밍 여부
    
    Returns:
        API 응답 또는 스트리밍 제너레이터
    """
    global vllm_client
    
    try:
        if LLM_BACKEND == 'vllm':
            if vllm_client is None:
                vllm_client = openai.OpenAI(
                    api_key='sk-not-needed',
                    base_url=VLLM_API_URL
                )
            
            # ✅ [Chief Cloud Architect] DeepSeek-R1 최적화 옵션
            response = vllm_client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.6,  # ✅ R1 추론 능력 강화 (0.7 → 0.6)
                top_p=top_p,
                max_tokens=max_tokens,
                stream=stream,
                # ✅ Stop 토큰 (R1 특화)
                stop=["<|file_sep|>", "### <CONTEXT>", "---", "[관련 문서 없음]", "<｜end of sentence｜>"],
                presence_penalty=0.6,
                frequency_penalty=0.6
            )
            return response
        
        else:  # Ollama
            headers = {'Content-Type': 'application/json'}
            
            messages_for_ollama = []
            for msg in messages:
                messages_for_ollama.append({
                    'role': msg.get('role', 'user'),
                    'content': msg.get('content', '')
                })
            
            payload = {
                'model': MODEL_NAME,
                'messages': messages_for_ollama,
                'stream': stream,
                'options': {
                    'temperature': 0.6,
                    'top_p': top_p,
                    'repeat_penalty': 1.15,
                    'stop': ["<|file_sep|>", "### <CONTEXT>", "---", "<｜end of sentence｜>"]
                }
            }
            
            if max_tokens:
                payload['options']['num_predict'] = max_tokens
            
            response = requests.post(
                f'{OLLAMA_API_URL}/api/chat',
                json=payload,
                headers=headers,
                stream=stream,
                timeout=300
            )
            
            if response.status_code != 200:
                raise Exception(f'Ollama API 오류: {response.text}')
            
            if not stream:
                data = response.json()
                return {
                    'id': 'ollama-' + str(int(time.time())),
                    'object': 'chat.completion',
                    'created': int(time.time()),
                    'model': MODEL_NAME,
                    'choices': [{
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'content': data.get('message', {}).get('content', '')
                        },
                        'finish_reason': 'stop'
                    }],
                    'usage': {
                        'prompt_tokens': 0,
                        'completion_tokens': 0,
                        'total_tokens': 0
                    }
                }
            else:
                return response
        
    except Exception as e:
        logger.error(f'LLM API 호출 실패: {str(e)}')
        raise


def stream_response_generator(response) -> Generator[str, None, None]:
    """
    ✅ [Chief Cloud Architect] vLLM 스트리밍 응답을 OpenAI 호환 형식으로 변환
    
    <think> 태그를 필터링하며 스트리밍합니다.
    """
    logger.info('스트리밍 생성 시작')
    
    try:
        chunk_count = 0
        in_think_tag = False
        
        for chunk in response:
            chunk_count += 1
            
            if hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0:
                choice = chunk.choices[0]
                
                if hasattr(choice, 'delta') and choice.delta and hasattr(choice.delta, 'content') and choice.delta.content:
                    content = choice.delta.content
                    
                    # ✅ <think> 태그 필터링
                    if '<think>' in content:
                        in_think_tag = True
                    
                    if not in_think_tag:
                        # <think> 태그 밖의 콘텐츠만 전송
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
                        in_think_tag = False
        
        logger.info(f'✓ 스트리밍 완료 - {chunk_count}개 청크')
        
        # 종료 신호
        final_data = {
            "id": f"chatcmpl-{datetime.now().timestamp()}",
            "object": "chat.completion.chunk",
            "created": int(datetime.now().timestamp()),
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_data)}\n\n"
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        logger.error(f'스트림 처리 오류: {str(e)}')
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
            'vllm_api': VLLM_API_URL,
            'current_model': MODEL_NAME
        }
    except Exception as e:
        logger.error(f'헬스 체크 실패: {str(e)}')
        return {'status': 'unhealthy', 'error': str(e)}


@app.get('/v1/models', tags=['Models'])
async def list_models():
    """
    ✅ [Open WebUI 통합] vLLM의 모델 목록 반환 (OpenAI 호환)
    
    Open WebUI가 이 엔드포인트를 호출하여 사용 가능한 모델을 감지합니다.
    available_models가 비어있으면 기본값으로 MODEL_NAME을 반환합니다.
    """
    global available_models
    
    # 기본 모델 정보 구성
    model_data = available_models if available_models else [{
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
    
    logger.info(f'✅ /v1/models 요청 - {len(model_data)}개 모델 반환')
    logger.debug(f'모델 데이터: {model_data}')
    
    return {
        'object': 'list',
        'data': model_data
    }


@app.post('/v1/chat/completions', tags=['Chat'])
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI 호환 Chat Completion 엔드포인트
    
    ✅ [Chief Cloud Architect] 기능:
    1. Vector Store에서 관련 문서 검색 (RAG)
    2. Context + Query를 포함한 프롬프트 구성
    3. vLLM (DeepSeek-R1)로 답변 생성
    4. <think> 태그 필터링 (추론 과정 숨김)
    5. 스트리밍 또는 일반 응답 반환
    """
    logger.info('=' * 70)
    logger.info(f'Chat Completion 요청: {len(request.messages)}개 메시지, stream={request.stream}')
    logger.info('=' * 70)
    
    try:
        # 사용자 질문 추출
        user_message = None
        for msg in reversed(request.messages):
            if msg.role == 'user':
                user_message = msg.content
                break
        
        if not user_message:
            raise HTTPException(status_code=400, detail="사용자 메시지 없음")
        
        logger.info(f'질문: {user_message[:100]}...')
        
        # Step 1: Context 검색 (RAG)
        logger.info('Step 1: Context 검색')
        context = retrieve_context(user_message, top_k=SEARCH_TOP_K)
        
        if context:
            logger.info(f'  ✓ {len(context)}자 Context 검색')
        else:
            logger.info('  ⚠ 관련 Context 없음')
        
        # Step 2: 프롬프트 구성
        logger.info('Step 2: 프롬프트 구성')
        final_prompt = build_prompt(user_message, context)
        
        # Step 3: 메시지 재구성
        logger.info('Step 3: 메시지 재구성')
        
        # ✅ [DeepSeek-R1 최적화] CoT 유도 기반 System Prompt
        # R1의 추론 능력을 극대화하기 위해 "생각하는 과정"을 명시적으로 유도
        pure_system_prompt = """당신은 박민준 개발자의 프로젝트를 돕는 'DeepSeek-R1 기반 AI 엔진'입니다.
제공된 <documents> 데이터를 기반으로 논리적으로 추론하여 답변하세요.

[답변 작성 절차]
1. **Analyze**: 사용자의 질문 의도를 파악하고 <documents> 내의 정보와 대조하세요.
2. **Think**: 문서의 내용이 질문을 해결하는 데 충분한지 논리적으로 따져보세요. (생각 과정을 <think> 태그에 담으세요)
3. **Answer**: 분석된 내용을 바탕으로 개발자에게 필요한 코드나 솔루션을 구체적으로 제시하세요.

[제약 사항]
- <documents> 태그 안의 내용만 사실로 간주하세요.
- 문서에 없는 내용은 "문서에 명시되지 않았으나, 일반적인 지식으로는..."이라고 구분하여 답하세요.
- 맹목적인 친절함보다는 정확한 기술적 솔루션을 우선하세요."""
        
        vllm_messages = []
        vllm_messages.append({'role': 'system', 'content': pure_system_prompt})
        
        # 과거 대화 추가
        history_messages = [msg for msg in request.messages[:-1] if msg.role in ['user', 'assistant']]
        for msg in history_messages:
            vllm_messages.append({'role': msg.role, 'content': msg.content})
        
        # ✅ [DeepSeek-R1 최적화] XML 구조 기반 최종 사용자 메시지
        # 명확한 구분으로 모델이 참고 자료와 실제 질문을 정확히 인식
        final_user_content = f"""다음은 검색된 참고 문서입니다:
{context if context else "<documents></documents>"}

---

사용자 질문:
{user_message}

위 문서를 참고하여 질문에 답하세요."""
        vllm_messages.append({'role': 'user', 'content': final_user_content})
        
        logger.info(f'메시지: System(1) + History({len(history_messages)}) + Current(1) = {len(vllm_messages)}개')
        
        # Step 4: vLLM 호출
        logger.info('Step 4: vLLM 호출')
        
        response = call_vllm_api(
            messages=vllm_messages,
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
            stream=request.stream
        )
        
        logger.info(f'vLLM 응답 수신: stream={request.stream}')
        
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
            
            if not response.choices or len(response.choices) == 0:
                raise HTTPException(status_code=500, detail='vLLM 응답 형식 오류')
            
            assistant_message = response.choices[0].message.content
            
            # ✅ <think> 태그 제거
            _, clean_message = extract_think_and_response(assistant_message)
            
            logger.info(f'✓ 응답 생성: {clean_message[:100]}...')
            
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
        logger.error(f'Chat Completion 오류: {str(e)}')
        raise HTTPException(status_code=500, detail=f'처리 실패: {str(e)}')


# ==================== 서버 실행 ====================

if __name__ == '__main__':
    host = os.getenv('SERVER_HOST', '0.0.0.0')
    port = int(os.getenv('SERVER_PORT', '8010'))
    
    logger.info(f'🚀 Server starting on {host}:{port}')
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level.lower(),
        access_log=True
    )
