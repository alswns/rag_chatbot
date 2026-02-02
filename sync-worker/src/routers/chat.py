"""Chat completions router"""
import logging
import json
import re
from typing import Any, Dict, Optional, List
from datetime import datetime
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from managers.token_manager import TokenManager, ChatMessage
from managers.qa_manager import QuestionAnsweringManager
from services.search_service import VectorSearchManager
from core.config import MAX_CONTEXT_TOKENS, MAX_MODEL_LEN, MODEL_NAME
import core.dependencies as deps

logger = logging.getLogger(__name__)
router = APIRouter()


# === 검색 재개 헬퍼 함수 ===
def _check_search_approval(message: str) -> bool:
    """사용자가 웹 검색을 허용했는지 확인"""
    approval_patterns = [
        '검색해', '찾아', '진행해', '해줘', '알아봐',
        '네', '응', '좋아', '그래', 'yes', 'ok', '부탁',
        '검색 해', '찾아봐', '검색해줘', '찾아줘',
        '검색 진행', '웹 검색', '인터넷'
    ]
    message_lower = message.lower().strip()
    
    # 짧은 긍정 응답 (예: "네", "응", "해줘")
    if len(message_lower) <= 10:
        for pattern in approval_patterns:
            if pattern in message_lower:
                return True
    
    # 명시적 검색 요청
    for pattern in ['검색해', '찾아', '진행해', '알아봐']:
        if pattern in message_lower:
            return True
    
    return False


def _extract_previous_context(messages: List[ChatMessage]) -> Optional[Dict[str, Any]]:
    """이전 대화에서 컨텍스트 추출"""
    context = {
        'original_query': None,
        'internal_data': None,
        'pending_task': None
    }
    
    # 최근 메시지에서 역순으로 탐색
    for i in range(len(messages) - 2, -1, -1):  # 마지막 메시지 제외
        msg = messages[i]
        
        if msg.role == 'user' and context['original_query'] is None:
            # 검색 허용 요청이 아닌 원래 질문 찾기
            if not _check_search_approval(msg.content):
                context['original_query'] = msg.content
                logger.info(f'🔍 원래 질문 복구: {msg.content[:50]}...')
        
        if msg.role == 'assistant':
            content = msg.content
            
            # 내부 검색 결과 추출
            if '[내부 검색 결과]' in content or '내부 문서' in content:
                # 내부 데이터 부분 추출
                context['internal_data'] = content
                logger.info(f'📄 내부 데이터 복구: {len(content)}자')
            
            # 검색 허용 요청 감지
            if '웹 검색을 진행할까요' in content or '웹 검색을 승인' in content:
                context['pending_task'] = 'web_search'
                logger.info('⏸️ 보류된 웹 검색 작업 감지')
    
    # 원래 질문이 있어야 유효한 컨텍스트
    if context['original_query']:
        return context
    
    return None


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=MODEL_NAME)
    messages: list[ChatMessage]
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=2048)
    stream: bool = Field(default=False)


def is_complex_query(query: str) -> bool:
    """복합 질문 여부 판단"""
    # 복합 질문 패턴
    complex_patterns = [
        '그리고', '또한', '그 다음', '이어서',
        '비교', '차이', '각각',
        '~와 ~', '~랑 ~',
        '총장', '학장', '대표', '창업자',  # 2단계 정보 필요
        '의 ~는', '의 ~가',  # 소유격 체인
    ]
    
    for pattern in complex_patterns:
        if pattern in query:
            return True
    
    # 접속사로 연결된 질문
    if query.count('?') > 1:
        return True
    
    return False


class ChatCompletionRequest(BaseModel):
    model: str = Field(default=MODEL_NAME)
    messages: list[ChatMessage]
    temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    max_tokens: int | None = Field(default=2048)
    stream: bool = Field(default=False)


@router.post('/v1/chat/completions')
async def chat_completions(request: ChatCompletionRequest) -> Any:
    """Chat Completion 엔드포인트"""
    logger.info('=' * 70)
    logger.info(f'💬 Chat: {len(request.messages)}개 메시지, stream={request.stream}')
    logger.info('=' * 70)
    
    try:
        user_message = QuestionAnsweringManager.extract_user_message(request.messages)
        if not user_message:
            raise HTTPException(status_code=400, detail="사용자 메시지 없음")
        
        logger.info(f'질문: {user_message[:100]}...')
        
        # Internal task 감지
        is_internal_task = user_message.strip().startswith('### Task:')
        
        # follow-up questions 차단
        if 'follow-up questions' in user_message.lower():
            logger.info('🚫 Follow-up questions 요청 차단')
            return {
                'id': f'chatcmpl-{datetime.now().timestamp()}',
                'object': 'chat.completion',
                'created': int(datetime.now().timestamp()),
                'model': MODEL_NAME,
                'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': ''}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0}
            }
        
        if is_internal_task:
            logger.info('🔧 Open WebUI 내부 Task 감지 → RAG 스킵')
            vllm_messages = [
                {'role': 'system', 'content': 'You are a helpful assistant.'},
                {'role': 'user', 'content': user_message}
            ]
        # ✅ 항상 Hierarchical Agent 사용 (스트리밍 모드일 때)
        elif request.stream:
            logger.info('🧠 Hierarchical Agent 실행')
            return await handle_hierarchical_agent(request, user_message)
        else:
            # 비스트리밍 → 기존 RAG 파이프라인
            logger.info('Step 1: RAG 검색...')
            context = await VectorSearchManager.search(user_message)
            logger.info(f'검색 완료: {len(context)}자')
            
            # 조건부 웹 검색
            web_context = ""
            
            # ✅ Pre-check: 내부 검색 품질 평가 (문서 길이만 확인)
            force_web_search = False
            context_length = len(context)
            
            # 문서 길이 기반 판단만 유지 (유사도 판단 제거)
            if context_length == 0:
                logger.info('🔍 Pre-check: 내부 문서 없음 → 강제 웹 검색')
                force_web_search = True
            elif context_length < 200:
                logger.info(f'🔍 Pre-check: 내부 문서 부족 ({context_length}자 < 200자) → 강제 웹 검색')
                force_web_search = True
            else:
                logger.info(f'✅ Pre-check: 내부 문서 충분 ({context_length}자) → LLM 판단 요청')
            
            try:
                from utils.web_search import get_web_search_service
                
                web_service = get_web_search_service()
                
                # 대화 히스토리 준비 (최근 3턴)
                history_for_web = [msg for msg in request.messages[:-1] if msg.role in ['user', 'assistant']]
                history_context = [{'role': msg.role, 'content': msg.content} for msg in history_for_web[-6:]]
                
                # 웹 검색 수행
                web_context = await web_service.search_if_needed(
                    user_query=user_message,
                    internal_context=context,
                    force_search=force_web_search,
                    history=history_context
                )
                
                if web_context:
                    logger.info(f'🌐 웹 검색 결과 획득: {len(web_context)}자')
                else:
                    logger.info('ℹ️  웹 검색 스킵')
                    
            except Exception as e:
                logger.error(f'❌ 웹 검색 실패: {str(e)}')
            
            # Intent 분류
            detected_intent = 'explanation'
            if deps.semantic_router:
                try:
                    detected_intent, confidence = deps.semantic_router.classify(user_message)
                    logger.info(f'🎯 Intent: {detected_intent} (confidence={confidence:.2f})')
                except Exception as e:
                    logger.warning(f'⚠️ Intent 분류 실패: {str(e)}')
            
            # 프롬프트 구성
            logger.info('Step 2: 토큰 관리 & 프롬프트 구성...')
            
            history = [msg for msg in request.messages[:-1] if msg.role in ['user', 'assistant']]
            dynamic_prompt = QuestionAnsweringManager.get_dynamic_prompt(detected_intent)
            
            combined_context = context
            if web_context:
                combined_context = f"{context}\n\n{web_context}"
            
            vllm_messages = TokenManager.manage_context_window(
                system_prompt=dynamic_prompt,
                context=combined_context,
                current_query=user_message,
                history=history,
                max_tokens=MAX_CONTEXT_TOKENS
            )
        
        # LLM 호출
        logger.info('Step 3: LLM 호출...')
        
        estimated_input_tokens = TokenManager.estimate_messages_tokens(vllm_messages)
        available_output_tokens = MAX_MODEL_LEN - estimated_input_tokens - 100
        
        dynamic_max_tokens = min(
            request.max_tokens or 2048,
            available_output_tokens,
            1024
        )
        
        if dynamic_max_tokens < 50:
            dynamic_max_tokens = 50
        
        logger.info(f'📊 토큰 계산: 입력 ~{estimated_input_tokens}, 출력 {dynamic_max_tokens}')
        
        response = await QuestionAnsweringManager.call_llm_async(
            messages=vllm_messages,
            temperature=request.temperature,
            max_tokens=dynamic_max_tokens,
            stream=request.stream
        )
        
        if request.stream:
            logger.info('스트리밍 응답 반환')
            return StreamingResponse(
                QuestionAnsweringManager.stream_response(response),
                media_type='text/event-stream'
            )
        else:
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


# === Hierarchical Agent Handler ===
async def handle_hierarchical_agent(request: ChatCompletionRequest, user_message: str):
    """복합 질문을 Hierarchical Agent로 처리 (스트리밍)"""
    from managers.hierarchical_agent import get_hierarchical_agent
    from utils.web_search import get_web_search_service
    
    agent = get_hierarchical_agent()
    web_service = get_web_search_service()
    
    # ✅ 검색 재개 의도 확인
    is_search_approval = _check_search_approval(user_message)
    previous_context = None
    
    if is_search_approval:
        logger.info('🔄 웹 검색 재개 의도 감지')
        # 이전 대화에서 컨텍스트 복구
        previous_context = _extract_previous_context(request.messages)
        if previous_context:
            logger.info(f'📋 이전 컨텍스트 복구: {previous_context.get("original_query", "N/A")}')
    
    # Internal Search Function
    async def internal_search_fn(query: str) -> str:
        return await VectorSearchManager.search(query)
    
    # Web Search Function
    async def web_search_fn(query: str) -> str:
        # 대화 히스토리 준비
        history = [{'role': msg.role, 'content': msg.content} 
                   for msg in request.messages[-6:] if msg.role in ['user', 'assistant']]
        
        return await web_service.search_if_needed(
            user_query=query,
            internal_context="",
            force_search=True,
            history=history
        )
    
    async def generate_stream():
        """OpenAI 호환 SSE 스트림 생성 (실시간 증분)"""
        created = int(datetime.now().timestamp())
        
        # ✅ 검색 재개 모드 처리
        if is_search_approval and previous_context:
            async for event in agent.resume_with_web_search(
                previous_context=previous_context,
                internal_search_fn=internal_search_fn,
                web_search_fn=web_search_fn
            ):
                if event["type"] == "thinking_start":
                    chunk = {
                        "id": f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{
                            "index": 0,
                            "delta": {
                                "role": "assistant",
                                "content": "\n\n<details open>\n<summary>thought: 검색 재개 중...</summary>\n\n"
                            },
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif event["type"] == "thinking":
                    chunk = {
                        "id": f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": event["content"]},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif event["type"] == "thinking_end":
                    chunk = {
                        "id": f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": "\n\n</details>\n\n---\n\n"},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
                elif event["type"] == "result":
                    content = event["content"]
                    chunk_size = 50
                    for i in range(0, len(content), chunk_size):
                        text_chunk = content[i:i+chunk_size]
                        chunk = {
                            "id": f"chatcmpl-{created}",
                            "object": "chat.completion.chunk",
                            "created": created,
                            "model": MODEL_NAME,
                            "choices": [{
                                "index": 0,
                                "delta": {"content": text_chunk},
                                "finish_reason": None
                            }]
                        }
                        yield f"data: {json.dumps(chunk)}\n\n"
            
            # 종료 청크
            final_chunk = {
                "id": f"chatcmpl-{created}",
                "object": "chat.completion.chunk",
                "created": created,
                "model": MODEL_NAME,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(final_chunk)}\n\n"
            yield "data: [DONE]\n\n"
            return
        
        # 일반 모드
        async for event in agent.run(user_message, internal_search_fn, web_search_fn):
            if event["type"] == "thinking_start":
                # details 태그 시작 (빈 줄 포함)
                chunk = {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "delta": {
                            "role": "assistant",
                            "content": "\n\n<details open>\n<summary>thought: 생각 중...</summary>\n\n"
                        },
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                
            elif event["type"] == "thinking":
                # 실시간 사고 과정 추가 (리스트 형식)
                chunk = {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": event["content"]},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            elif event["type"] == "permission_needed":
                # 검색 허용 요청 - details 내부에 표시하고 종료
                permission_msg = "\n\n⚠️ **알림**: 내부 자료가 부족합니다. 웹 검색을 승인하시겠습니까?\n\n"
                chunk = {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": permission_msg},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
                
                # details 닫기
                close_chunk = {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": "\n\n</details>\n\n"},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(close_chunk)}\n\n"
                
                # 스트림 종료
                final_chunk = {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }]
                }
                yield f"data: {json.dumps(final_chunk)}\n\n"
                yield "data: [DONE]\n\n"
                return  # 여기서 종료
            
            elif event["type"] == "thinking_end":
                # details 태그 종료 (빈 줄 포함)
                chunk = {
                    "id": f"chatcmpl-{created}",
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": MODEL_NAME,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": "\n\n</details>\n\n---\n\n"},
                        "finish_reason": None
                    }]
                }
                yield f"data: {json.dumps(chunk)}\n\n"
            
            elif event["type"] == "result":
                # 최종 답변
                content = event["content"]
                
                # 청크 단위로 스트리밍
                chunk_size = 50
                for i in range(0, len(content), chunk_size):
                    text_chunk = content[i:i+chunk_size]
                    chunk = {
                        "id": f"chatcmpl-{created}",
                        "object": "chat.completion.chunk",
                        "created": created,
                        "model": MODEL_NAME,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": text_chunk},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk)}\n\n"
        
        # 종료 청크
        final_chunk = {
            "id": f"chatcmpl-{created}",
            "object": "chat.completion.chunk",
            "created": created,
            "model": MODEL_NAME,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop"
            }]
        }
        yield f"data: {json.dumps(final_chunk)}\n\n"
        yield "data: [DONE]\n\n"
    
    logger.info('🧠 Hierarchical Agent 스트리밍 응답 시작')
    return StreamingResponse(
        generate_stream(),
        media_type='text/event-stream'
    )
