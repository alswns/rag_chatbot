"""Chat completions router"""
import logging
from typing import Any
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
        else:
            # RAG 검색
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
                
                # 웹 검색 수행 (force_search 플래그 전달)
                web_context = await web_service.search_if_needed(
                    user_query=user_message,
                    internal_context=context,
                    force_search=force_web_search
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
