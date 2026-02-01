"""Debug endpoints for development"""
from fastapi import APIRouter, HTTPException
from typing import Dict
import logging

from services.search_service import VectorSearchManager
import core.dependencies as deps

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post('/v1/search')
async def search_documents(query: str, top_k: int = 5) -> Dict:
    """검색 결과 확인 엔드포인트"""
    logger.info(f'🔍 /v1/search 요청: "{query[:50]}..."')
    
    context = await VectorSearchManager.search(query, top_k=top_k)
    documents = VectorSearchManager._last_search_results
    
    return {
        'query': query,
        'document_count': len(documents),
        'documents': documents[:3],
        'context_length': len(context)
    }


@router.post('/v1/intent')
async def analyze_intent(query: str) -> Dict:
    """Intent 분석 엔드포인트"""
    if deps.intent_router is None:
        raise HTTPException(status_code=503, detail='Intent Router 미초기화')
    
    result = deps.intent_router.route(query)
    return result.to_dict()
