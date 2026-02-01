import logging
from typing import Optional
import numpy as np
import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingService:
    """싱글톤 임베딩 서비스"""
    
    _instance = None
    _model = None
    _model_name = None
    
    def __new__(cls, model_name='BAAI/bge-m3'):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._model_name = model_name
        return cls._instance
    
    def __init__(self, model_name='BAAI/bge-m3'):
        """
        초기화 (싱글톤이므로 처음 한 번만 실행)
        
        Args:
            model_name: 임베딩 모델명
        """
        if self._model is None:
            logger.info(f'🔄 임베딩 모델 로드 중: {model_name}')
            try:
                # self._model = SentenceTransformer(model_name)
                self._model = SentenceTransformer(
                    model_name,
                    device='cuda',
                    model_kwargs={"torch_dtype": torch.float16} # FP16으로 메모리 50% 절감
                )
                logger.info(f'✅ 임베딩 모델 로드 완료: {model_name}')
            except Exception as e:
                logger.error(f'❌ 임베딩 모델 로드 실패: {str(e)}')
                raise
    
    def encode(self, texts: list) -> np.ndarray:
        """
        텍스트를 임베딩으로 변환
        
        Args:
            texts: 임베딩할 텍스트 리스트
            
        Returns:
            임베딩 벡터 (numpy array)
        """
        if self._model is None:
            raise RuntimeError('임베딩 모델이 로드되지 않았습니다')
        
        return self._model.encode(texts, convert_to_numpy=True)
    
    def get_model(self) -> SentenceTransformer:
        """로드된 모델 객체 반환"""
        if self._model is None:
            raise RuntimeError('임베딩 모델이 로드되지 않았습니다')
        return self._model
    
    @property
    def model_name(self) -> str:
        """모델 이름 반환"""
        return self._model_name


# 싱글톤 인스턴스
def get_embedding_service(model_name: str = 'BAAI/bge-m3') -> EmbeddingService:
    """임베딩 서비스 싱글톤 인스턴스 획득"""
    return EmbeddingService(model_name)
