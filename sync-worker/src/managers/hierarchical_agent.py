"""
Hierarchical Agent System (2-Layer Architecture)

Manager (Planner): 복잡한 질문을 하위 작업으로 분해
Worker (ReAct Executor): 각 작업을 Internal/Web 검색으로 수행
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional, AsyncGenerator
import openai
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SubTask:
    """하위 작업 정의"""
    id: int
    task: str
    depends_on: Optional[int] = None  # 이전 작업 의존성
    result: Optional[str] = None
    status: str = "pending"  # pending, running, completed, failed


@dataclass
class AgentContext:
    """에이전트 실행 컨텍스트"""
    original_query: str
    plan: List[SubTask] = field(default_factory=list)
    context_memory: Dict[int, str] = field(default_factory=dict)
    thinking_logs: List[str] = field(default_factory=list)


class ManagerAgent:
    """
    The Manager (Planner)
    복잡한 질문을 해결 가능한 하위 작업 리스트로 분해
    """
    
    SYSTEM_PROMPT = """당신은 **프로젝트 매니저**입니다.

## 역할
사용자의 복잡한 질문을 해결하기 위해 필요한 **정보 수집 단계**를 계획하세요.

## 규칙
1. 각 단계는 **하나의 명확한 정보 수집 목표**여야 합니다.
2. 순차적 의존성이 있는 경우 `depends_on` 필드로 표시하세요.
3. 단순한 질문은 1개의 작업으로 충분합니다.
4. 최대 5개의 하위 작업까지만 생성하세요.

## 예시

### 예시 1: 복합 질문
**질문:** "박민준의 대학과 그 대학 총장을 알려줘"
**계획:**
```json
[
  {"id": 1, "task": "박민준의 출신 대학교 확인", "depends_on": null},
  {"id": 2, "task": "1번에서 확인된 대학교의 현재 총장 확인", "depends_on": 1}
]
```

### 예시 2: 단순 질문
**질문:** "FastAPI 비동기 처리 방법 알려줘"
**계획:**
```json
[
  {"id": 1, "task": "FastAPI 비동기 처리 방법 검색", "depends_on": null}
]
```

### 예시 3: 비교 질문
**질문:** "선린인터넷고와 한성과학고 비교해줘"
**계획:**
```json
[
  {"id": 1, "task": "선린인터넷고등학교 정보 수집", "depends_on": null},
  {"id": 2, "task": "한성과학고등학교 정보 수집", "depends_on": null},
  {"id": 3, "task": "두 학교 비교 분석", "depends_on": 2}
]
```

## 출력 형식
**오직 JSON 배열만 출력하세요. 설명 없이 JSON만!**

```json
[{"id": 1, "task": "...", "depends_on": null}, ...]
```"""

    def __init__(self, vllm_api_url: str = None):
        self.vllm_api_url = vllm_api_url or os.getenv('VLLM_API_URL', 'http://localhost:8000/v1')
        self.client = openai.OpenAI(
            api_key='sk-not-needed',
            base_url=self.vllm_api_url,
            timeout=60.0
        )
    
    async def create_plan(self, query: str) -> List[SubTask]:
        """질문을 하위 작업으로 분해"""
        try:
            logger.info(f'📋 Manager: 계획 수립 중... "{query[:50]}..."')
            
            response = self.client.chat.completions.create(
                model=os.getenv('LLM_MODEL_ID', 'DeepSeek-R1'),
                messages=[
                    {'role': 'system', 'content': self.SYSTEM_PROMPT},
                    {'role': 'user', 'content': f'다음 질문에 대한 정보 수집 계획을 세우세요:\n\n{query}'}
                ],
                temperature=0.1,
                max_tokens=500,
                stream=False
            )
            
            raw_output = response.choices[0].message.content.strip()
            
            # JSON 추출
            plan_data = self._parse_json_plan(raw_output)
            
            if not plan_data:
                # Fallback: 단일 작업
                logger.warning('⚠️ 계획 파싱 실패, 단일 작업으로 대체')
                return [SubTask(id=1, task=query, depends_on=None)]
            
            # SubTask 객체로 변환
            tasks = []
            for item in plan_data[:5]:  # 최대 5개
                task = SubTask(
                    id=item.get('id', len(tasks) + 1),
                    task=item.get('task', ''),
                    depends_on=item.get('depends_on')
                )
                tasks.append(task)
            
            logger.info(f'✅ Manager: {len(tasks)}개 하위 작업 생성')
            for t in tasks:
                logger.debug(f'  └─ [{t.id}] {t.task}')
            
            return tasks
            
        except Exception as e:
            logger.error(f'❌ Manager 계획 실패: {str(e)}')
            return [SubTask(id=1, task=query, depends_on=None)]
    
    def _parse_json_plan(self, raw: str) -> Optional[List[Dict]]:
        """LLM 출력에서 JSON 추출"""
        try:
            # 코드 블록 제거
            cleaned = re.sub(r'```json\s*', '', raw)
            cleaned = re.sub(r'```\s*', '', cleaned)
            
            # JSON 배열 찾기
            match = re.search(r'\[[\s\S]*\]', cleaned)
            if match:
                return json.loads(match.group())
            
            return None
        except json.JSONDecodeError:
            return None


class WorkerAgent:
    """
    The Worker (ReAct Executor)
    하나의 Task를 수행하며 Internal → Web 순서로 도구 사용
    """
    
    # Note: {{, }}로 escape하여 .format()과 충돌 방지
    SYSTEM_PROMPT = """당신은 **현장 요원**입니다.

## 역할
주어진 'Task'를 해결하기 위해 정보를 수집하세요.

## 도구
1. **internal_search**: 내부 문서 검색 (Notion, GitHub, 내부 DB)
2. **web_search**: 외부 웹 검색 (DuckDuckGo)

## 실행 규칙
1. **Internal Search** 우선 시도
2. 결과가 부족하면 **Web Search** 시도
3. 그래도 없으면 "정보 없음" 보고

## 이전 작업 결과
{previous_results}

## 출력 형식
**오직 JSON만 출력하세요!**

```json
{{
  "thought": "현재 상황과 다음 행동에 대한 추론",
  "action": "internal_search 또는 web_search 또는 finish",
  "query": "검색할 쿼리 (finish인 경우 최종 결과)"
}}
```"""

    def __init__(self, vllm_api_url: str = None):
        self.vllm_api_url = vllm_api_url or os.getenv('VLLM_API_URL', 'http://localhost:8000/v1')
        self.client = openai.OpenAI(
            api_key='sk-not-needed',
            base_url=self.vllm_api_url,
            timeout=60.0
        )
        self.max_steps = 3
    
    async def execute_task(
        self,
        task: SubTask,
        context_memory: Dict[int, str],
        internal_search_fn,
        web_search_fn
    ) -> str:
        """단일 작업 실행 (ReAct Loop)"""
        logger.info(f'🔧 Worker: [{task.id}] {task.task}')
        
        # 이전 결과 컨텍스트 구성
        previous_results = ""
        if task.depends_on and task.depends_on in context_memory:
            previous_results = f"[작업 {task.depends_on} 결과]\n{context_memory[task.depends_on]}"
        
        collected_info = []
        
        for step in range(self.max_steps):
            logger.debug(f'  Step {step + 1}/{self.max_steps}')
            
            try:
                # LLM에게 다음 행동 결정 요청
                action = await self._decide_action(task.task, previous_results, collected_info)
                
                if action['action'] == 'finish':
                    result = action.get('query', '정보 수집 완료')
                    logger.info(f'  ✅ 작업 완료: {result[:50]}...')
                    return result
                
                elif action['action'] == 'internal_search':
                    logger.debug(f'  🔍 Internal Search: {action["query"]}')
                    search_result = await internal_search_fn(action['query'])
                    if search_result:
                        collected_info.append(f"[내부 검색 결과]\n{search_result[:1000]}")
                    else:
                        collected_info.append("[내부 검색 결과] 관련 정보 없음")
                
                elif action['action'] == 'web_search':
                    logger.debug(f'  🌐 Web Search: {action["query"]}')
                    search_result = await web_search_fn(action['query'])
                    if search_result:
                        collected_info.append(f"[웹 검색 결과]\n{search_result[:1000]}")
                    else:
                        collected_info.append("[웹 검색 결과] 관련 정보 없음")
                
            except Exception as e:
                logger.warning(f'  ⚠️ Step {step + 1} 실패: {str(e)}')
                continue
        
        # 최대 스텝 도달
        if collected_info:
            return '\n'.join(collected_info)
        return "정보를 찾을 수 없습니다."
    
    async def _decide_action(
        self,
        task: str,
        previous_results: str,
        collected_info: List[str]
    ) -> Dict[str, str]:
        """다음 행동 결정"""
        prompt = self.SYSTEM_PROMPT.format(previous_results=previous_results or "없음")
        
        # 수집된 정보 추가
        user_message = f"## 현재 작업\n{task}\n\n"
        if collected_info:
            user_message += f"## 지금까지 수집된 정보\n" + '\n'.join(collected_info[-2:]) + "\n\n"
        user_message += "다음 행동을 결정하세요."
        
        response = self.client.chat.completions.create(
            model=os.getenv('LLM_MODEL_ID', 'DeepSeek-R1'),
            messages=[
                {'role': 'system', 'content': prompt},
                {'role': 'user', 'content': user_message}
            ],
            temperature=0.1,
            max_tokens=200,
            stream=False
        )
        
        raw = response.choices[0].message.content.strip()
        
        # JSON 파싱
        try:
            cleaned = re.sub(r'```json\s*', '', raw)
            cleaned = re.sub(r'```\s*', '', cleaned)
            match = re.search(r'\{[\s\S]*\}', cleaned)
            if match:
                return json.loads(match.group())
        except:
            pass
        
        # Fallback
        return {'thought': raw, 'action': 'finish', 'query': raw}


class HierarchicalAgent:
    """
    Hierarchical Agent System
    Manager + Worker 통합 실행
    """
    
    def __init__(self, vllm_api_url: str = None):
        self.manager = ManagerAgent(vllm_api_url)
        self.worker = WorkerAgent(vllm_api_url)
        self.vllm_api_url = vllm_api_url or os.getenv('VLLM_API_URL', 'http://localhost:8000/v1')
        self.client = openai.OpenAI(
            api_key='sk-not-needed',
            base_url=self.vllm_api_url,
            timeout=120.0
        )
    
    async def run(
        self,
        query: str,
        internal_search_fn,
        web_search_fn
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        전체 워크플로우 실행 (스트리밍)
        
        Yields:
            {"type": "thinking", "content": "..."} - 진행 상황
            {"type": "result", "content": "..."} - 최종 결과
        """
        context = AgentContext(original_query=query)
        
        # === Step 1: Planning ===
        yield {"type": "thinking", "step": "planning", "content": "🧠 질문 분석 중..."}
        
        context.plan = await self.manager.create_plan(query)
        
        plan_summary = "\n".join([f"  {t.id}. {t.task}" for t in context.plan])
        yield {
            "type": "thinking",
            "step": "plan_created",
            "content": f"📋 계획 수립 완료:\n{plan_summary}"
        }
        
        # === Step 2: Execute Tasks ===
        for task in context.plan:
            # 의존성 체크
            if task.depends_on and task.depends_on not in context.context_memory:
                # 의존하는 작업이 아직 완료되지 않음 - 스킵하지 않고 진행
                pass
            
            task.status = "running"
            yield {
                "type": "thinking",
                "step": "task_start",
                "task_id": task.id,
                "content": f"🔧 작업 {task.id} 실행: {task.task}"
            }
            
            # Worker 실행
            result = await self.worker.execute_task(
                task=task,
                context_memory=context.context_memory,
                internal_search_fn=internal_search_fn,
                web_search_fn=web_search_fn
            )
            
            task.result = result
            task.status = "completed"
            context.context_memory[task.id] = result
            
            yield {
                "type": "thinking",
                "step": "task_complete",
                "task_id": task.id,
                "content": f"✅ 작업 {task.id} 완료"
            }
        
        # === Step 3: Generate Final Answer ===
        yield {"type": "thinking", "step": "synthesizing", "content": "📝 최종 답변 생성 중..."}
        
        final_answer = await self._generate_final_answer(context)
        
        yield {"type": "result", "content": final_answer}
    
    async def run_simple(
        self,
        query: str,
        internal_search_fn,
        web_search_fn
    ) -> str:
        """단순 실행 (스트리밍 없이 결과만 반환)"""
        result = ""
        async for event in self.run(query, internal_search_fn, web_search_fn):
            if event["type"] == "result":
                result = event["content"]
        return result
    
    async def _generate_final_answer(self, context: AgentContext) -> str:
        """수집된 정보를 바탕으로 최종 답변 생성"""
        
        # 수집된 모든 정보 조합
        collected_data = []
        for task in context.plan:
            if task.result:
                collected_data.append(f"[작업 {task.id}: {task.task}]\n{task.result}")
        
        synthesis_prompt = f"""당신은 **정보 검증 및 통합 전문가**입니다.

## 원래 질문
{context.original_query}

## 수집된 정보
{chr(10).join(collected_data)}

## ⚠️ 크로스 체크 지침 (CRITICAL)

### 1. 출처 간 교차 검증
- **내부 문서**와 **웹 검색** 결과를 비교하세요.
- 두 출처의 정보가 **일치**하면 → 신뢰도 높음 ✅
- 두 출처의 정보가 **충돌**하면 → 내부 문서 우선, 웹 정보는 참고용으로 표시

### 2. 할루시네이션 방지
- 수집된 정보에 **명시적으로 언급되지 않은 내용**은 절대 추가하지 마세요.
- 웹 검색에서 **다른 사람/기관의 정보**가 섞여 있을 수 있으므로 주의하세요.
- 예: "박민준" 검색 시 동명이인 정보가 포함될 수 있음

### 3. 불확실성 표시
- 확인되지 않은 정보: `(확인 필요)` 표시
- 출처가 웹만인 경우: `(웹 검색 기반, 검증 권장)` 표시
- 정보가 없는 경우: "해당 정보를 찾을 수 없습니다" 명시

### 4. 답변 형식
- 한국어로 자연스럽고 구조화된 답변
- 핵심 정보를 먼저, 부가 정보는 뒤에
- 마지막에 출처 명시

### 5. 출처 표기
```
---
📌 **참고 출처:**
- [내부 문서] 문서명
- [웹 검색] URL (검증 권장)
```"""

        response = self.client.chat.completions.create(
            model=os.getenv('LLM_MODEL_ID', 'DeepSeek-R1'),
            messages=[
                {'role': 'system', 'content': '정보 검증 및 통합 전문가입니다. 크로스 체크를 통해 할루시네이션을 방지합니다.'},
                {'role': 'user', 'content': synthesis_prompt}
            ],
            temperature=0.2,  # 더 낮은 temperature로 일관성 확보
            max_tokens=2048,
            stream=False
        )
        
        return response.choices[0].message.content.strip()


# === Singleton Instance ===
_hierarchical_agent: Optional[HierarchicalAgent] = None

def get_hierarchical_agent() -> HierarchicalAgent:
    """HierarchicalAgent 싱글톤 반환"""
    global _hierarchical_agent
    if _hierarchical_agent is None:
        _hierarchical_agent = HierarchicalAgent()
        logger.info('✅ HierarchicalAgent 초기화 완료')
    return _hierarchical_agent
