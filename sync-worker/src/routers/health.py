"""Health check and system status router"""
from datetime import datetime
from fastapi import APIRouter
from typing import Dict
import logging

from managers.model_manager import ModelManager
import core.dependencies as deps
from core.config import ENABLE_RERANKING

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get('/health')
async def health_check() -> Dict:
    """헬스 체크"""
    try:
        if deps.vector_store is None:
            return {'status': 'unhealthy', 'error': 'Vector store not initialized'}
        
        stats = deps.vector_store.get_collection_stats()
        graph_stats = None
        if deps.graph_processor:
            graph_stats = {
                'nodes': deps.graph_processor.graph.number_of_nodes(),
                'edges': deps.graph_processor.graph.number_of_edges()
            }
        
        return {
            'status': 'ok',
            'version': '2.0.0',
            'timestamp': datetime.now().isoformat(),
            'vector_store': stats,
            'graph': graph_stats,
            'components': {
                'vector_store': deps.vector_store is not None,
                'graph': deps.graph_processor is not None,
                'intent_router': deps.intent_router is not None,
                'drill_down_retriever': deps.drill_down_retriever is not None,
                'semantic_router': deps.semantic_router is not None,
                'reranking': ENABLE_RERANKING
            }
        }
    except Exception as e:
        logger.error(f'❌ 헬스 체크 실패: {str(e)}')
        return {'status': 'unhealthy', 'error': str(e)}


@router.get('/v1/models')
async def list_models() -> Dict:
    """모델 목록 반환"""
    logger.info('📊 /v1/models 요청')
    models = ModelManager.get_models()
    return {'object': 'list', 'data': models}
