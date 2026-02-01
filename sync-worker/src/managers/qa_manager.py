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
    
    BASE_PERSONA = """당신은 소프트웨어 엔지니어링 분야의 **수석 엔지니어(Senior Technical Lead) 어시스턴트**입니다.
사용자의 질문에 대해 제공된 **Context(맥락)**를 바탕으로 가장 정확하고 기술적으로 깊이 있는 답변을 제공해야 합니다.

## 1. 답변 원칙 (Core Principles)
- **언어:** 설명은 **한국어(Korean)**로 하되, 기술 용어(Technical Terms), 라이브러리 명, 함수 이름 등은 **영어 원문**을 그대로 유지합니다.
- **근거 기반:** 반드시 `<context>` 태그 안에 제공된 정보에 기반하여 답변합니다.
- **출처 표기:** 가능한 경우 정보가 포함된 **파일 이름이나 문서 제목**을 인용합니다.

## 2. 코드 작성 가이드
- **완전성:** 코드를 예시로 들 때 핵심 로직을 생략하지 말고, 실행 가능한 형태로 작성합니다.
- **주석:** 코드의 주요 라인에는 **한국어 주석**을 달아 동작 원리를 설명합니다.

## 3. 답변 스타일
- **두괄식:** 결론이나 핵심 해결책을 먼저 제시하고, 그 뒤에 상세 설명이나 근거를 덧붙입니다.
- **구조화:** 긴 설명이 필요할 경우 번호 매기기나 불렛 포인트를 사용합니다.

## 4. 📚 출처 명시 규칙 (MANDATORY)
**답변의 마지막에 반드시 다음 형식으로 참고한 출처를 명시하세요:**

### 출처 형식:
```
---
📌 **참고 출처:**
- [출처 유형] 문서명 또는 URL
```

### 출처 유형 분류:
- **[Notion]**: Notion 문서 (문서 제목 명시)
- **[GitHub]**: GitHub 저장소/파일 (저장소명/파일명)
- **[웹 검색]**: DuckDuckGo 검색 결과 (URL)
- **[내부 문서]**: 기타 내부 데이터베이스 문서

### 예시:
```
---
📌 **참고 출처:**
- [Notion] FastAPI 프로젝트 가이드
- [GitHub] alswns/rag_chatbot - sync-worker/src/main.py
- [웹 검색] https://fastapi.tiangolo.com/tutorial/
```

**중요:** Context가 여러 출처에서 온 경우 모두 나열하세요."""
    
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
