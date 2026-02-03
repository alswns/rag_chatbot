"""
Query Expansion Engine - LLM을 활용한 다중 쿼리 생성

사용자의 단순한 질문을 의미론적으로 다른 3가지 버전으로 확장하여
다양한 관점에서의 검색을 동시에 수행합니다.
"""

import logging
import asyncio
from typing import List, Dict, Optional
import openai
from core.config import VLLM_API_URL, MODEL_NAME

logger = logging.getLogger(__name__)


class MultiQueryExpander:
    """LLM 기반 다중 쿼리 확장 엔진"""
    
    EXPANSION_PROMPT = """당신은 정보 검색 최적화 전문가입니다.

사용자의 질문(Query)을 받았을 때, 다양한 관점에서 검색할 수 있도록 의미론적으로 다른 3가지 버전으로 확장해야 합니다.

# 요구사항:
1. **Original**: 사용자 질문을 다시 쓰되, 더 명확하고 상세하게 작성
2. **Paraphrase**: 다른 표현으로 바꿔 같은 의미를 전달 (유의어 사용)
3. **Focused**: 질문의 핵심 키워드에 더 집중하여 작성

# 규칙:
- 질문의 의미를 바꾸지 마세요. 단지 표현 방식만 다양화하세요.
- 각 쿼리는 한 줄의 명확한 질문이어야 합니다.
- 응답 형식: JSON으로 정확히 다음과 같이 작성하세요.

```json
{{
    "original": "확장된 원본 질문",
    "paraphrase": "다르게 표현한 질문",
    "focused": "핵심 키워드 집중 질문"
}}
```

사용자 질문: {query}

JSON만 반환하세요. 다른 설명이나 마크다운은 포함하지 마세요."""
    
    @staticmethod
    async def expand_query(query: str, max_retries: int = 3) -> Dict[str, List[str]]:
        """
        사용자 질문을 다중 버전으로 확장
        
        Args:
            query: 사용자 질문
            max_retries: LLM 호출 재시도 횟수
        
        Returns:
            {
                "queries": ["확장된_쿼리_1", "확장된_쿼리_2", "확장된_쿼리_3"],
                "original": "원본 쿼리"
            }
        """
        try:
            logger.info(f'🔄 Query Expansion 시작: "{query[:50]}..."')
            
            # LLM 호출
            client = openai.OpenAI(
                api_key='sk-not-needed',
                base_url=VLLM_API_URL,
                timeout=60.0
            )
            
            prompt = MultiQueryExpander.EXPANSION_PROMPT.format(query=query)
            
            for attempt in range(1, max_retries + 1):
                try:
                    response = client.chat.completions.create(
                        model=MODEL_NAME,
                        messages=[
                            {
                                "role": "system",
                                "content": "당신은 정보 검색 최적화 전문가입니다. JSON 형식으로만 응답하세요."
                            },
                            {
                                "role": "user",
                                "content": prompt
                            }
                        ],
                        temperature=0.7,
                        max_tokens=256,
                        timeout=60.0
                    )
                    
                    result_text = response.choices[0].message.content.strip()
                    logger.debug(f'🔍 LLM 응답: {result_text}')
                    
                    # JSON 파싱
                    import json
                    
                    # JSON 블록 추출 (마크다운 코드 블록 제거)
                    if "```json" in result_text:
                        result_text = result_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in result_text:
                        result_text = result_text.split("```")[1].split("```")[0].strip()
                    
                    expanded = json.loads(result_text)
                    
                    queries = [
                        expanded.get("original", query),
                        expanded.get("paraphrase", query),
                        expanded.get("focused", query)
                    ]
                    
                    # 중복 제거
                    queries = list(dict.fromkeys(queries))
                    
                    logger.info(f'✅ Query Expansion 완료: {len(queries)}개 쿼리 생성')
                    for i, q in enumerate(queries, 1):
                        logger.debug(f'   [{i}] {q}')
                    
                    return {
                        "queries": queries,
                        "original": query
                    }
                    
                except json.JSONDecodeError as e:
                    logger.warning(f'⚠️ JSON 파싱 실패 (시도 {attempt}/{max_retries}): {str(e)}')
                    if attempt == max_retries:
                        logger.error('❌ Query Expansion 최종 실패 - 원본 쿼리 사용')
                        return {
                            "queries": [query],
                            "original": query
                        }
                except Exception as e:
                    logger.warning(f'⚠️ LLM 호출 실패 (시도 {attempt}/{max_retries}): {str(e)}')
                    if attempt < max_retries:
                        await asyncio.sleep(2 ** attempt)  # 지수 백오프
                    else:
                        logger.error('❌ Query Expansion 최종 실패 - 원본 쿼리 사용')
                        return {
                            "queries": [query],
                            "original": query
                        }
                        
        except Exception as e:
            logger.error(f'❌ Query Expansion 프로세스 실패: {str(e)}')
            return {
                "queries": [query],
                "original": query
            }
    
    @staticmethod
    async def batch_expand_queries(queries: List[str]) -> Dict[str, List[str]]:
        """
        여러 쿼리를 동시에 확장
        
        Args:
            queries: 원본 쿼리 리스트
        
        Returns:
            확장된 모든 쿼리의 합집합
        """
        try:
            logger.info(f'🔄 배치 Query Expansion: {len(queries)}개 쿼리')
            
            # 동시 실행
            tasks = [
                MultiQueryExpander.expand_query(query)
                for query in queries
            ]
            results = await asyncio.gather(*tasks)
            
            # 모든 확장 쿼리 수집 (중복 제거)
            all_expanded = []
            for result in results:
                all_expanded.extend(result.get("queries", []))
            
            # 중복 제거 (순서 유지)
            unique_queries = list(dict.fromkeys(all_expanded))
            
            logger.info(f'✅ 배치 확장 완료: {len(unique_queries)}개 고유 쿼리')
            
            return {
                "queries": unique_queries,
                "original": queries
            }
            
        except Exception as e:
            logger.error(f'❌ 배치 확장 실패: {str(e)}')
            return {
                "queries": queries,
                "original": queries
            }
