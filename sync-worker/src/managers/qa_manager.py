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

## 🎯 핵심 역할: 정보 통합 및 출처 구분
당신은 **내부 지식(Internal Knowledge)**과 **외부 정보(Web Search)**를 통합하되, 출처를 엄격히 구분하는 전문가입니다.

## 📋 Input Data Structure
1. **<internal_context>**: 사용자의 로컬 문서/코드 (Notion, GitHub, 내부 DB) - **최우선 신뢰**
2. **<external_context>**: 웹 검색 결과 (DuckDuckGo) - **보조 정보**

## ⚠️ Critical Rules (할루시네이션 방지 원칙)

### 1️⃣ Source Priority (출처 우선순위)
- **내부 프로젝트, 코드, 설정값, 인물 프로필**에 대한 내용은 **오직 <internal_context>만 신뢰**하세요.
- **<external_context>**가 내부 정보와 충돌하면, 외부 정보를 **무시**하고 내부 정보를 따르세요.
- 예시: 내부는 Python 3.8인데 웹은 3.12라고 하면 → "내부 규정에 따라 3.8을 사용합니다"

### 2️⃣ Context Separation (맥락 분리)
- 정보의 출처를 명확히 인지하고 **섞지 마세요**.
- ❌ Bad: "박민준님의 프로젝트는 (내부내용)이고, GitHub 스타는 100개입니다(외부 엉뚱한 사람 정보)."
- ✅ Good: "박민준님의 프로젝트는 (내부내용)입니다. 참고로 웹에서 검색된 유사 프로젝트들은..."

### 3️⃣ "I Don't Know" Policy
- 내부/외부 모두에 명확한 답이 없다면 솔직하게 **"제공된 정보 내에서는 알 수 없습니다"**라고 답하세요.
- **절대 추측하지 마세요.**

## 📝 답변 원칙 (Core Principles)
- **언어:** 설명은 **한국어(Korean)**로 하되, 기술 용어, 라이브러리명, 함수명은 **영어 원문** 유지
- **근거 기반:** 반드시 Context에 기반하여 답변
- **완전성:** 코드는 실행 가능한 형태로, 주석은 한국어로

## 📚 출처 명시 규칙 (MANDATORY)
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
- **[내부 문서]**: 기타 내부 데이터베이스 문서
- **[웹 검색]**: DuckDuckGo 검색 결과 (URL)

### 예시:
```
---
📌 **참고 출처:**
- [Notion] FastAPI 프로젝트 가이드
- [GitHub] alswns/rag_chatbot - sync-worker/src/main.py
- [웹 검색] https://fastapi.tiangolo.com/tutorial/
```

**중요:** 
1. Context가 여러 출처에서 온 경우 모두 나열하세요.
2. 내부 문서는 문서 제목을, 웹 검색은 URL을 명시하세요.
3. 출처를 알 수 없는 경우 "출처 불명"이라고 명시하세요."""
    
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
