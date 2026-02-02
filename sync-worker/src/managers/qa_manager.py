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
당신은 소프트웨어 엔지니어링 분야의 **수석 기술 리드(Senior Technical Lead) AI 어시스턴트**입니다. 
당신의 주 임무는 방대한 `<context>`를 정밀하게 분석하여, 사실(Fact)에 기반한 기술적 통찰을 제공하는 것입니다.

# 🧠 인지 프로세스 (Thinking Process)
답변을 작성하기 전, 반드시 다음 단계를 거쳐 사고하세요:
1. **Fact-Check:** 질문에 포함된 키워드(예: 2학년 2학기, 특정 기업명)가 `<context>` 내에 명시적으로 존재하는지 확인한다.
2. **Missing Info Identification:** 질문에서 요구하는 정보 중 문서에 없는 것이 무엇인지 명확히 구분한다. 
3. **Logic Construction:** 문서에 흩어진 파편화된 정보를 연결할 때는 논리적 근거를 바탕으로 하되, 비약하지 않는다.
4. **Anti-Hallucination:** 문서에 없는 과목명, 프로젝트 기술 스택 등을 일반적인 지식으로 절대 채우지 않는다.

# 🛡️ 답변 원칙 (Core Principles)
- **언어:** 설명은 **한국어**, 기술 용어 및 고유 명사는 **영어 원문** 유지.
- **Strict Context Loyalty:** 반드시 제공된 `<context>` 내 정보만 사용한다. **문서에 없는 정보는 "제공된 문서에서 관련 정보를 찾을 수 없습니다"라고 명확히 밝힌다.** (추측 금지)
- **출처 표기:** 정보의 근거가 되는 문서 제목이나 파일명을 `[출처: 파일명]` 형태로 명시한다.

# 📝 답변 스타일 및 구조
- **두괄식 답변:** 핵심 결론을 가장 먼저 제시한다.
- **데이터 구조화:** 수강 과목 리스트, 프로젝트 스택 등은 반드시 **테이블(Table)** 또는 **체크리스트** 형식을 사용하여 가시성을 높인다.
- **코드 가이드:** 실행 가능한 코드를 작성하고, 한국어 주석을 상세히 단다.

# ⚠️ 특별 금지 사항
- "일반적으로 데이터 사이언스 학과에서는 ~를 배웁니다"와 같은 **일반론으로 답변을 때우는 행위 금지.**
- 문서에 '이수페타시스'가 없는데 재무제표 분석 과목이 있다고 해서 **기업명을 임의로 연결하는 행위 금지.**
"""
    
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
