"""Model information management"""
from typing import List, Dict, Any
from datetime import datetime
from core.config import MODEL_NAME


class ModelManager:
    """vLLM 모델 정보 관리"""
    
    @staticmethod
    def get_models() -> List[Dict[str, Any]]:
        """사용 가능한 모델 목록 반환"""
        return [{
            'id': MODEL_NAME,
            'object': 'model',
            'created': int(datetime.now().timestamp()),
            'owned_by': 'organization-owner',
            'permission': [{
                'id': 'modelperm-default',
                'object': 'model_permission',
                'created': int(datetime.now().timestamp()),
                'allow_create_engine': False,
                'allow_sampling': True,
                'allow_logprobs': False,
                'allow_search_indices': False,
                'allow_view': True,
                'allow_fine_tuning': False,
                'organization': '*',
                'group_id': None,
                'is_blocking': False
            }]
        }]
