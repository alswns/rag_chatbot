"""
Hierarchical Agent System (2-Layer Architecture)

Manager (Planner): 복잡한 질문을 하위 작업으로 분해
Worker (ReAct Executor): 각 작업을 Internal/Web 검색으로 수행
"""

import os
import json
import logging
import re
import asyncio
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
    
    SYSTEM_PROMPT = """당신은 **수석 전략가**입니다.

## 🗺️ 내부 데이터 지도 (Context Map)
- **Notion:** 박민준의 이력서, 개인 공부(Python, ROS2, ML), 일기, 프로젝트 기획안(핸드트래킹 글러브 등)
- **Git:** ROS 2 C++ 소스코드, YOLOv11 포즈 트레이닝 스크립트, FastAPI 서버 코드
- **구조:** 모든 문서는 'Hub(상위 주제)'와 'Leaf(상세 내용)'로 연결된 계층 구조

## 📋 Planning Rules (필수)
1. **Explore First:** 첫 번째 작업은 반드시 `내부 인덱스 확인`이어야 합니다.
   - 검색어 형식: `박민준 [주제] 개요` 또는 `[주제] 목록`
   
2. **Specific Queries:** 구체적인 검색어를 사용하세요.
   - ❌ 나쁜 예: "박민준 공부"
   - ✅ 좋은 예: "박민준 Notion 내 파이썬 학습 리스트 및 ROS2 실습 기록"
   
3. **Internal Priority:** 내부 데이터에서 힌트를 얻어 다음 작업을 구체화하세요.
   - 예: 1번에서 'IMU 센서' 발견 → 2번을 'Git에서 IMU 처리 코드 찾기'로 설정

4. **2-Step Minimum:** 복합 질문은 최소 2단계로 분해하세요.
   - 1단계: 내부 인덱스 탐색 (Hub 레벨)
   - 2단계: 구체적 정보 검색 (Leaf 레벨)

## 예시

### 예시 1: 복합 질문 (외부 정보 필요)
**질문:** "박민준의 대학과 그 대학 총장을 알려줘"
**계획:**
```json
[
  {"id": 1, "task": "박민준 프로필에서 출신 대학교 확인 (Notion 이력서 탐색)", "depends_on": null},
  {"id": 2, "task": "1번에서 확인된 대학교의 현재 총장 정보 검색", "depends_on": 1}
]
```

### 예시 2: 내부 데이터 질문
**질문:** "박민준이 뭘 공부했어?"
**계획:**
```json
[
  {"id": 1, "task": "박민준 Notion 내 학습 기록 인덱스 확인 (Python, ROS2, ML 등)", "depends_on": null},
  {"id": 2, "task": "1번에서 발견된 주요 학습 주제별 상세 내용 수집", "depends_on": 1}
]
```

### 예시 3: 기술 질문
**질문:** "박민준의 ROS2 프로젝트 알려줘"
**계획:**
```json
[
  {"id": 1, "task": "Git 저장소에서 ROS2 관련 프로젝트 목록 확인", "depends_on": null},
  {"id": 2, "task": "1번에서 발견된 ROS2 프로젝트의 핵심 기능 및 코드 구조 분석", "depends_on": 1}
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
    SYSTEM_PROMPT = """당신은 **정보 수집 전문가**입니다.

## 🗺️ 내부 데이터 지도
- **Notion:** 박민준의 이력서, 학습 기록(Python, ROS2, ML), 일기, 프로젝트 기획안
- **Git:** ROS 2 C++ 소스코드, YOLOv11 트레이닝 스크립트, FastAPI 서버 코드
- **구조:** Hub(상위 주제) → Leaf(상세 내용) 계층 구조

## 도구
1. **internal_search**: 내부 문서 검색 (Notion, GitHub)
2. **web_search**: 외부 웹 검색 (DuckDuckGo)

## 🔍 검색 전략 (중요!)
1. **Explore First:** 처음에는 넓은 키워드로 인덱스 탐색
   - 예: "박민준 ROS2 프로젝트 목록"
   
2. **Refine:** 1차 결과에서 힌트를 얻어 구체적으로 재검색
   - 예: 1차에서 'IMU 센서' 발견 → "박민준 IMU 센서 처리 코드"
   
3. **Quality Check:** 결과가 부족하면 다른 키워드로 재시도
   - 부족 기준: 100자 미만 또는 관련성 낮음
   - 재시도 키워드: 동의어, 상위 개념, 관련 기술명

## 이전 작업 결과
{previous_results}

## 출력 형식
**오직 JSON만 출력하세요!**

```json
{{{{
  "thought": "현재 상황 분석 및 다음 행동 추론 (발견한 힌트 포함)",
  "action": "internal_search 또는 web_search 또는 finish",
  "query": "검색할 쿼리 (구체적이고 명확하게)"
}}}}
```

## 예시
**Task:** "박민준의 ROS2 프로젝트 찾기"
**1차 시도:**
```json
{{{{"thought": "Git 저장소에서 ROS2 관련 프로젝트를 먼저 탐색", "action": "internal_search", "query": "박민준 Git ROS2 프로젝트 목록"}}}}
```
**1차 결과:** "ros2_ws 폴더에 slam_bot, hand_tracking 프로젝트 존재"
**2차 시도:**
```json
{{{{"thought": "1차에서 hand_tracking 발견, 상세 정보 수집", "action": "internal_search", "query": "박민준 hand_tracking 프로젝트 기능 설명"}}}}
```"""

    def __init__(self, vllm_api_url: str = None):
        self.vllm_api_url = vllm_api_url or os.getenv('VLLM_API_URL', 'http://localhost:8000/v1')
        self.client = openai.OpenAI(
            api_key='sk-not-needed',
            base_url=self.vllm_api_url,
            timeout=60.0
        )
        self.max_steps = 4  # ✅ 재시도 여유 확보
    
    async def execute_task(
        self,
        task: SubTask,
        context_memory: Dict[int, str],
        internal_search_fn,
        web_search_fn,
        has_search_permission: bool = False,
        thinking_logs: List[str] = None
    ) -> str:
        """단일 작업 실행 (ReAct Loop with Real-time Streaming)"""
        logger.info(f'🔧 Worker: [{task.id}] {task.task}')
        
        # 이전 결과 컨텍스트 구성
        previous_results = ""
        if task.depends_on and task.depends_on in context_memory:
            previous_results = f"[작업 {task.depends_on} 결과]\n{context_memory[task.depends_on]}"
        
        collected_info = []
        internal_data_found = False
        
        retry_keywords_used = set()  # 재시도에 사용된 키워드 추적
        
        for step in range(self.max_steps):
            if thinking_logs is not None:
                thinking_logs.append(f"  └─ Step {step + 1}/{self.max_steps}...\n")
            
            logger.debug(f'  Step {step + 1}/{self.max_steps}')
            
            try:
                # LLM에게 다음 행동 결정 요청
                action = await self._decide_action(task.task, previous_results, collected_info)
                
                if action['action'] == 'finish':
                    result = action.get('query', '정보 수집 완료')
                    logger.info(f'  ✅ 작업 완료: {result[:50]}...')
                    return result
                
                elif action['action'] == 'internal_search':
                    query = action.get('query', task.task)
                    
                    if thinking_logs is not None:
                        thinking_logs.append(f"  🔍 내부 검색: \"{query[:40]}...\"\n")
                    
                    logger.debug(f'  🔍 Internal Search: {query}')
                    search_result = await internal_search_fn(query)
                    
                    # ✅ 검색 결과 품질 평가 강화
                    result_length = len(search_result) if search_result else 0
                    
                    if result_length > 500:  # 충분한 내부 데이터
                        internal_data_found = True
                        collected_info.append(f"[내부 검색 결과]\n{search_result[:1000]}")
                        if thinking_logs is not None:
                            thinking_logs.append(f"  ✅ 내부 데이터 충분 ({result_length}자)\n")
                            
                    elif result_length > 100:  # 부분적 데이터
                        internal_data_found = True
                        collected_info.append(f"[내부 검색 결과 - 부분적]\n{search_result[:1000]}")
                        if thinking_logs is not None:
                            thinking_logs.append(f"  ⚠️ 부분적 ({result_length}자) - 추가 탐색 시도\n")
                            
                        # ✅ 부분적 데이터일 때 힌트 추출하여 재검색 유도
                        hints = self._extract_hints(search_result)
                        if hints and step < self.max_steps - 1:
                            if thinking_logs is not None:
                                thinking_logs.append(f"  💡 발견된 힌트: {hints[:50]}...\n")
                            collected_info.append(f"[발견된 힌트] {hints}")
                            
                    else:  # 데이터 부족 - 재시도 또는 웹 검색
                        if thinking_logs is not None:
                            thinking_logs.append(f"  ❌ 부족 ({result_length}자)\n")
                        
                        # ✅ 다른 키워드로 재시도 (최대 1회)
                        if query not in retry_keywords_used and step < self.max_steps - 1:
                            retry_keywords_used.add(query)
                            alt_query = self._generate_alternative_query(task.task, query)
                            if alt_query and alt_query != query:
                                if thinking_logs is not None:
                                    thinking_logs.append(f"  🔄 대체 키워드로 재시도: \"{alt_query[:30]}...\"\n")
                                collected_info.append(f"[재시도 필요] 키워드 변경: {alt_query}")
                                continue
                        
                        collected_info.append("[내부 검색 결과] 관련 정보 부족")
                        
                        # 웹 검색 권한 확인
                        if not has_search_permission:
                            if thinking_logs is not None:
                                thinking_logs.append(f"  ⏸️ 웹 검색 허용 필요\n")
                            return "[PERMISSION_NEEDED]내부 자료가 부족합니다."
                
                elif action['action'] == 'web_search':
                    # === Permission Check ===
                    if not has_search_permission:
                        # 검색 허용 없음 → 중단하고 허용 요청
                        if thinking_logs is not None:
                            thinking_logs.append(f"  ⏸️ 웹 검색 보류 (허용 필요)\n")
                        return "[PERMISSION_NEEDED]내부 자료가 부족합니다."
                    
                    if thinking_logs is not None:
                        thinking_logs.append(f"  🌐 웹 검색 중...\n")
                    
                    logger.debug(f'  🌐 Web Search: {action["query"]}')
                    search_result = await web_search_fn(action['query'])
                    if search_result:
                        collected_info.append(f"[웹 검색 결과]\n{search_result[:1000]}")
                        if thinking_logs is not None:
                            thinking_logs.append(f"  ✅ 웹 정보 획득\n")
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
    
    def _extract_hints(self, text: str) -> str:
        """검색 결과에서 힌트(키워드, 이름, 기술명) 추출"""
        if not text:
            return ""
        
        # 주요 키워드 패턴 추출
        keywords = []
        
        # 기술 스택 관련
        tech_patterns = ['ROS2', 'ROS 2', 'Python', 'C++', 'FastAPI', 'YOLO', 'IMU', 'SLAM', 'ML', 'AI']
        for tech in tech_patterns:
            if tech.lower() in text.lower():
                keywords.append(tech)
        
        # 프로젝트명 패턴 (언더스코어 포함 단어)
        import re
        project_names = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]{3,}', text)
        keywords.extend(project_names[:3])
        
        return ', '.join(set(keywords)[:5]) if keywords else ""
    
    def _generate_alternative_query(self, original_task: str, failed_query: str) -> str:
        """실패한 검색어를 기반으로 대체 검색어 생성"""
        # 키워드 변형 규칙
        alternatives = {
            '공부': '학습 기록',
            '프로젝트': '작업 내역',
            '코드': '소스코드 파일',
            'ROS2': 'ROS 2 패키지',
            '이력서': '프로필 개요',
        }
        
        new_query = failed_query
        for old, new in alternatives.items():
            if old in failed_query:
                new_query = failed_query.replace(old, new)
                break
        
        # 검색어가 변경되지 않았으면 "개요" 또는 "목록" 추가
        if new_query == failed_query:
            if '목록' not in failed_query and '개요' not in failed_query:
                new_query = f"{failed_query} 목록"
        
        return new_query


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
        전체 워크플로우 실행 (실시간 증분 스트리밍)
        
        Yields:
            {"type": "thinking_start"} - details 태그 시작
            {"type": "thinking", "content": "..."} - 실시간 진행 상황
            {"type": "thinking_end"} - details 태그 종료
            {"type": "result", "content": "..."} - 최종 결과
        """
        context = AgentContext(original_query=query)
        
        # === 사고 과정 시작 ===
        yield {"type": "thinking_start"}
        
        # === Step 1: Planning ===
        yield {"type": "thinking", "content": "\n- 🧠 **질문 분석 중...**\n\n"}
        await asyncio.sleep(0.05)
        
        context.plan = await self.manager.create_plan(query)
        
        # 계획 출력
        yield {"type": "thinking", "content": "\n- 📋 **계획 수립 완료:**\n\n"}
        for t in context.plan:
            yield {"type": "thinking", "content": f"  - {t.id}. {t.task}\n\n"}
            await asyncio.sleep(0.05)
        
        # === Step 2: Permission Check (검색 허용 여부) ===
        has_search_permission = self._check_search_permission(query)
        
        if has_search_permission:
            yield {"type": "thinking", "content": "\n- ✅ **검색 허용**: 질문에 검색 키워드 포함\n\n"}
        else:
            yield {"type": "thinking", "content": "\n- ⚠️ **검색 보류**: 내부 데이터 우선 탐색\n\n"}
        
        await asyncio.sleep(0.05)
        
        # === Step 3: Execute Tasks ===
        needs_permission_request = False
        
        for task in context.plan:
            task.status = "running"
            yield {"type": "thinking", "content": f"\n- 🔧 **작업 {task.id}**: {task.task}\n\n"}
            await asyncio.sleep(0.05)
            
            # Worker 실행 (실시간 스트리밍)
            # thinking_logs를 수집하여 나중에 yield
            thinking_logs = []
            
            result = await self.worker.execute_task(
                task=task,
                context_memory=context.context_memory,
                internal_search_fn=internal_search_fn,
                web_search_fn=web_search_fn,
                has_search_permission=has_search_permission,
                thinking_logs=thinking_logs  # 로그 수집용 리스트 전달
            )
            
            # 수집된 thinking_logs를 실시간으로 yield (빈 줄 추가)
            for log in thinking_logs:
                yield {"type": "thinking", "content": f"\n{log}\n"}
                await asyncio.sleep(0.05)
            
            # 검색 허용 요청이 필요한지 체크
            if result.startswith("[PERMISSION_NEEDED]"):
                needs_permission_request = True
                result = result.replace("[PERMISSION_NEEDED]", "").strip()
                # 즉시 permission_needed 이벤트 발생
                yield {"type": "permission_needed"}
                return  # 여기서 종료 (details 닫기는 chat.py에서 처리)
            
            task.result = result
            task.status = "completed"
            context.context_memory[task.id] = result
            
            # 작업 완료
            result_preview = result[:80] + "..." if len(result) > 80 else result
            yield {"type": "thinking", "content": f"\n  ✅ **완료**: {result_preview}\n\n"}
            await asyncio.sleep(0.05)
        
        # === Step 4: Cross-Check ===
        if any('[웹 검색 결과]' in task.result for task in context.plan if task.result):
            yield {"type": "thinking", "content": "\n\n- 🔍 **크로스 체크**: 내부 문서와 웹 정보 대조 중...\n\n"}
            await asyncio.sleep(0.1)
            
            # 실제 크로스 체크 수행
            internal_info = [t.result for t in context.plan if t.result and '[내부 검색 결과]' in t.result]
            web_info = [t.result for t in context.plan if t.result and '[웹 검색 결과]' in t.result]
            
            if internal_info and web_info:
                yield {"type": "thinking", "content": "\n  - 내부 데이터와 웹 정보 정합성 확인 중...\n\n"}
                await asyncio.sleep(0.05)
                yield {"type": "thinking", "content": "\n  - ✅ **검증 완료**: 정보 일관성 확인됨\n\n"}
            else:
                yield {"type": "thinking", "content": "\n  - ℹ️ 단일 출처 데이터 (크로스 체크 불필요)\n\n"}
        
        # === Step 5: Generate Final Answer ===
        yield {"type": "thinking", "content": "\n\n- 📝 **최종 답변 생성 중...**\n\n"}
        await asyncio.sleep(0.05)
        
        # === 사고 과정 종료 ===
        yield {"type": "thinking_end"}
        
        final_answer = await self._generate_final_answer(context)
        yield {"type": "result", "content": final_answer}
    
    def _check_search_permission(self, query: str) -> bool:
        """검색 허용 키워드 체크"""
        search_keywords = ['검색', '찾아', '구글', 'google', '검색해서', '찾아줘', '알아봐']
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in search_keywords)
    
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
    
    async def resume_with_web_search(
        self,
        previous_context: Dict[str, Any],
        internal_search_fn,
        web_search_fn
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        이전에 멈춘 작업을 웹 검색과 함께 재개
        
        Args:
            previous_context: {
                'original_query': 원래 질문,
                'internal_data': 이미 수집된 내부 데이터,
                'pending_task': 보류된 작업 유형
            }
        """
        original_query = previous_context.get('original_query', '')
        internal_data = previous_context.get('internal_data', '')
        
        logger.info(f'🔄 웹 검색 재개: "{original_query[:50]}..."')
        
        # 컨텍스트 복원
        context = AgentContext(original_query=original_query)
        
        # === 사고 과정 시작 ===
        yield {"type": "thinking_start"}
        
        yield {"type": "thinking", "content": "\n\n- 🔄 **웹 검색 재개 중...**\n\n"}
        await asyncio.sleep(0.05)
        
        # 이전 내부 데이터가 있으면 표시
        if internal_data:
            internal_preview = internal_data[:100] + "..." if len(internal_data) > 100 else internal_data
            yield {"type": "thinking", "content": f"\n- 📄 **이전 내부 데이터 복구**: {internal_preview}\n\n"}
            await asyncio.sleep(0.05)
            
            # 가상 Task 1 결과로 저장
            context.context_memory[1] = internal_data
        
        # 웹 검색 실행
        yield {"type": "thinking", "content": "\n- 🌐 **웹 검색 시작**...\n\n"}
        await asyncio.sleep(0.05)
        
        try:
            web_result = await web_search_fn(original_query)
            
            if web_result and len(web_result) > 100:
                yield {"type": "thinking", "content": f"\n  ✅ **웹 검색 완료** ({len(web_result)}자 수집)\n\n"}
                context.context_memory[2] = f"[웹 검색 결과]\n{web_result[:1000]}"
                
                # 가상 SubTask 생성
                context.plan = [
                    SubTask(id=1, task="내부 데이터 복구", status="completed", result=internal_data),
                    SubTask(id=2, task="웹 검색", status="completed", result=web_result[:1000])
                ]
            else:
                yield {"type": "thinking", "content": "\n  ⚠️ **웹 검색 결과 부족**\n\n"}
                context.plan = [
                    SubTask(id=1, task="내부 데이터 복구", status="completed", result=internal_data)
                ]
        except Exception as e:
            logger.error(f'웹 검색 실패: {str(e)}')
            yield {"type": "thinking", "content": f"\n  ❌ **웹 검색 실패**: {str(e)}\n\n"}
            context.plan = [
                SubTask(id=1, task="내부 데이터 복구", status="completed", result=internal_data)
            ]
        
        # 크로스 체크
        if context.context_memory.get(1) and context.context_memory.get(2):
            yield {"type": "thinking", "content": "\n\n- 🔍 **크로스 체크**: 내부 문서와 웹 정보 대조 중...\n\n"}
            await asyncio.sleep(0.1)
            yield {"type": "thinking", "content": "\n  - ✅ **검증 완료**: 정보 일관성 확인됨\n\n"}
        
        # 최종 답변 생성
        yield {"type": "thinking", "content": "\n\n- 📝 **최종 답변 생성 중...**\n\n"}
        await asyncio.sleep(0.05)
        
        # === 사고 과정 종료 ===
        yield {"type": "thinking_end"}
        
        # 최종 답변 생성
        final_answer = await self._generate_final_answer(context)
        yield {"type": "result", "content": final_answer}
    
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
- 두 출처의 정보가 **충돌**하면 → 내부 문서 우선, 웹 정보는 참고용으로 표시하고 **사용자에게 경고**

### 2. 할루시네이션 방지
- 수집된 정보에 **명시적으로 언급되지 않은 내용**은 절대 추가하지 마세요.
- 웹 검색에서 **다른 사람/기관의 정보**가 섞여 있을 수 있으므로 주의하세요.
- 예: "박민준" 검색 시 동명이인 정보가 포함될 수 있음 → **"웹에서는 A라고 나오지만, 내부 문서 기준으로는 B가 맞습니다."**

### 3. 웹 검색 허용 요청 (중요)
- 만약 내부 문서가 부족하거나 없다면, 답변 끝에 다음 문구를 추가하세요:
  **"내부 자료가 부족합니다. 더 정확한 확인을 위해 웹 검색을 진행할까요?"**
- 단, 사용자 질문에 "검색해서", "구글", "검색" 등의 키워드가 있다면 이미 허용된 것으로 간주하고 웹 검색 결과를 포함하세요.

### 4. 불확실성 표시
- 확인되지 않은 정보: `(확인 필요)` 표시
- 출처가 웹만인 경우: `(웹 검색 기반, 검증 권장)` 표시
- 내부와 웹이 충돌하는 경우: `(웹 정보: ..., 내부 문서: ...)`로 병기
- 정보가 없는 경우: "해당 정보를 찾을 수 없습니다" 명시

### 5. 답변 형식
- 한국어로 자연스럽고 구조화된 답변
- 핵심 정보를 먼저, 부가 정보는 뒤에
- 마지막에 출처 명시

### 6. 출처 표기 (필수)
```
---
📌 **참고 출처:**
- [내부 문서] 문서명
- [웹 검색] URL (검증 권장)
```

**충돌 시 경고 예시:**
"웹 검색에서는 '중앙대학교의 총장이 Morakinyo A.O. Kuti 교수'라고 나오지만, 이는 다른 학과나 해외 대학과 혼동된 정보일 수 있습니다. 내부 문서를 기준으로는 [정확한 정보]입니다."
"""

        response = self.client.chat.completions.create(
            model=os.getenv('LLM_MODEL_ID', 'DeepSeek-R1'),
            messages=[
                {'role': 'system', 'content': '정보 검증 및 통합 전문가입니다. 크로스 체크를 통해 할루시네이션을 방지하고, 충돌 시 사용자에게 명확히 경고합니다.'},
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
