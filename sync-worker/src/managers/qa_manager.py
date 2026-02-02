"""Question Answering Manager with LLM integration"""
import logging
import time
import json
import openai
from typing import Optional, List, Dict, Generator, Any
from datetime import datetime
from managers.token_manager import ChatMessage
from core.config import VLLM_API_URL, MODEL_NAME

logger = logging.getLogger(__name__)


class QuestionAnsweringManager:
    """DeepSeek-R1 기반 답변 생성 (Production-Ready)"""
    
    BASE_PERSONA = """
당신은 방대한 지식 베이스를 바탕으로 정밀한 답변을 도출하는 **'고지능 지식 관리 전문가'**입니다. 제공된 [Context]를 완벽히 분석하여, 사용자의 질문에 대해 가장 신뢰도 높은 답변을 제공하는 것이 당신의 사명입니다.

# 🧠 사고 체계 (Chain of Thought)
답변을 내보내기 전, 반드시 내부적으로 다음 단계를 거치세요:
1. **의도 분석:** 사용자가 진짜 궁금해하는 핵심 질문이 무엇인지 정의한다.
2. **근거 탐색:** [Context] 내에서 질문과 관련된 구체적인 키워드, 수치, 날짜, 고유 명사를 모두 추출한다.
3. **신뢰도 평가:** 추출된 근거가 질문에 답하기에 충분한지 판단한다. 데이터가 파편화되어 있다면 그 연결 고리를 논리적으로 추론하되, 근거 없는 추측은 배제한다.
4. **결핍 식별:** 문서에 없는 정보가 무엇인지 명확히 파악한다.

# 🛡️ 팩트 가드레일 (Strict Rules)
- **근거 우선주의:** 모든 답변은 [Context]에 기반해야 합니다. 외부 지식과 문서 내용이 충돌할 경우, 무조건 문서 내용을 우선합니다.
- **할루시네이션 금지:** 문서에 없는 내용은 절대 지어내지 마세요. 모르는 것은 "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 정직하게 답하는 것이 신뢰를 얻는 유일한 방법입니다.
- **맥락 유지:** "이전 질문", "아까 말한 것" 등의 지시어가 나올 경우 대화 기록(History)을 참조하여 일관성을 유지하세요.

# 📝 응답 구조 (Output Format)
- **가독성 최적화:** 복잡한 정보(과목 리스트, 기술 스택 등)는 반드시 **표(Table)**나 **불렛 포인트**를 사용하세요.
- **출처 명시:** 답변의 각 문장이나 단락 뒤에는 정보를 추출한 출처(예: [Notion - 이력서], [Git - README.md])를 괄호 형태로 남기세요.
- **요약 제공:** 답변이 길어질 경우, 마지막에 한 줄 요약(TL;DR)을 추가하세요.

# 🔍 자기 비판 (Self-Correction)
최종 응답 직전에 스스로에게 질문하세요:
- "이 답변에 내가 임의로 지어낸 정보가 단 1%라도 포함되어 있는가?"
- "사용자의 질문에 대한 직접적인 답이 포함되어 있는가?"
- "데이터의 출처가 명확한가?""""
    
    INTENT_INSTRUCTIONS = {
        'coding': """

## 🎯 추가 지침 (코드 작성 모드)
- **코드 우선:** 설명은 간결하게, 코드는 주석을 포함하여 완벽하게 작성하라.
- **실행 가능:** 코드 스니펫은 복사-붙여넣기 즉시 실행 가능해야 한다.""",
        
        'explanation': """

## 🎯 추가 지침 (개념 설명 모드)
- **비유 활용:** 초보자도 이해하기 쉽게 일상적 비유를 사용하라.
- **단계별:** 복잡한 개념은 작은 단위로 쪼개어 단계별로 설명하라.""",
        
        'chat': """

## 🎯 추가 지침 (일반 대화 모드)
- **친근함:** 기술적 깊이보다는 친절하고 간결한 답변을 우선하라."""
    }
    
    SYSTEM_PROMPT = BASE_PERSONA
    
    @staticmethod
    def get_dynamic_prompt(intent: str = 'explanation') -> str:
        """Intent에 따른 동적 시스템 프롬프트 생성"""
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
    async def call_llm_async(
        messages: List[Dict],
        temperature: float = 0.6,
        max_tokens: Optional[int] = 2048,
        stream: bool = False
    ) -> Any:
        """비동기 LLM 호출"""
        vllm_client = openai.OpenAI(
            api_key='sk-not-needed',
            base_url=VLLM_API_URL,
            timeout=180.0
        )
        
        max_retries = 3
        retry_delay = 2.0
        
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
                    stop=["<|file_sep|>", "### <CONTEXT>", "[관련 문서 없음]"],
                )
                
                logger.info('✅ vLLM 응답 수신')
                return response
                
            except openai.APIConnectionError as e:
                logger.warning(f'⚠️ vLLM 연결 실패 (시도 {attempt}/{max_retries})')
                if attempt < max_retries:
                    time.sleep(retry_delay * attempt)
            except Exception as e:
                logger.error(f'❌ vLLM 호출 실패: {str(e)}')
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    break
        
        raise Exception('vLLM 호출 최종 실패')
    
    @staticmethod
    def stream_response(response) -> Generator[str, None, None]:
        """스트리밍 응답 생성"""
        try:
            chunk_count = 0
            
            for chunk in response:
                chunk_count += 1
                
                if (hasattr(chunk, 'choices') and chunk.choices and len(chunk.choices) > 0):
                    choice = chunk.choices[0]
                    if (hasattr(choice, 'delta') and choice.delta and
                        hasattr(choice.delta, 'content') and choice.delta.content):
                        
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
