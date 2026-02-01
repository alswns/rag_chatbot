"""Token Management for LLM context window"""
import re
import logging
from typing import List, Dict
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class ChatMessage(BaseModel):
    role: str
    content: str


class TokenManager:
    """Smart Token Management with dynamic context window"""
    
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
        
        korean_chars = len(re.findall(r'[가-힣]', text))
        english_chars = len(re.findall(r'[a-zA-Z0-9]', text))
        other_chars = len(text) - korean_chars - english_chars
        
        korean_tokens = korean_chars / 2.0
        english_tokens = english_chars / 3.5
        other_tokens = other_chars / 1.5
        
        total = int(korean_tokens + english_tokens + other_tokens)
        return max(total, len(text.split()) if text.strip() else 0)
    
    @staticmethod
    def estimate_messages_tokens(messages: List[Dict[str, str]]) -> int:
        """메시지 리스트의 총 토큰 수 추정"""
        total = 0
        for msg in messages:
            total += 4  # role 토큰
            total += TokenManager.estimate_tokens(msg.get('content', ''))
        return total
    
    @staticmethod
    def manage_context_window(
        system_prompt: str,
        context: str,
        current_query: str,
        history: List[ChatMessage],
        max_tokens: int = 8192
    ) -> List[Dict[str, str]]:
        """동적 컨텍스트 윈도우 관리"""
        RESERVED_OUTPUT = 2048
        effective_limit = max_tokens - RESERVED_OUTPUT
        
        # 시스템 프롬프트 조립
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
        
        system_tokens = TokenManager.estimate_tokens(full_system)
        query_tokens = TokenManager.estimate_tokens(current_query)
        fixed_tokens = system_tokens + query_tokens + 50
        
        # Fix: Context 강제 트리밍
        if fixed_tokens > effective_limit:
            excess_tokens = fixed_tokens - effective_limit
            chars_to_remove = int(excess_tokens * 3)
            
            if context and len(context) > chars_to_remove:
                trimmed_context = context[:-chars_to_remove]
                logger.warning(f"⚠️ Context 강제 트리밍: {len(context)} → {len(trimmed_context)}자")
                
                full_system = f"""{system_prompt}
            ---
            ### 제공된 참고 문서 (Reference Context)
            <context>
            {trimmed_context}
            </context>
            ---
            """
                system_tokens = TokenManager.estimate_tokens(full_system)
                fixed_tokens = system_tokens + query_tokens + 50
            else:
                logger.error(f"❌ Context 트리밍 실패")
                fixed_tokens = effective_limit
        
        remaining_tokens = effective_limit - fixed_tokens
        
        # 히스토리 동적 포함
        selected_history = []
        history_tokens = 0
        max_history_count = 10
        
        for msg in reversed(history[-max_history_count:]):
            msg_tokens = TokenManager.estimate_tokens(msg.content) + 10
            if history_tokens + msg_tokens <= remaining_tokens:
                selected_history.insert(0, {'role': msg.role, 'content': msg.content})
                history_tokens += msg_tokens
            else:
                break
        
        messages = [{'role': 'system', 'content': full_system}]
        messages.extend(selected_history)
        messages.append({'role': 'user', 'content': current_query})
        
        final_input_tokens = TokenManager.estimate_messages_tokens(messages)
        logger.info(f'✅ 토큰 관리 완료: 히스토리 {len(selected_history)}개, 예측 총합 ~{final_input_tokens}')
        
        return messages
