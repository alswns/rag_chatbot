"""Global dependencies and state"""
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor
import openai

from core.config import THREAD_POOL_SIZE

# Global executor
executor = ThreadPoolExecutor(max_workers=THREAD_POOL_SIZE)

# Global state
vector_store: Optional[Any] = None
graph_processor: Optional[Any] = None
drill_down_retriever: Optional[Any] = None
intent_router: Optional[Any] = None
semantic_router: Optional[Any] = None
vllm_client: Optional[openai.OpenAI] = None
available_models: List[Dict[str, Any]] = []
