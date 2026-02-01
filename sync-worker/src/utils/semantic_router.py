"""Semantic Intent Router (Embedding-based, No LLM)"""
import logging
from typing import Tuple

logger = logging.getLogger(__name__)


class SemanticIntentRouter:
    """LLM 없이 Embedding 기반 의도 분류"""
    
    INTENT_ANCHORS = {
        'coding': ["코드 작성해줘", "에러 수정해줘", "함수 구현", "디버깅 도와줘", "코드 리뷰", "버그 찾아줘"],
        'explanation': ["설명해줘", "이게 무슨 뜻이야", "개념 알려줘", "어떻게 작동하는지", "원리가 뭐야", "차이점이 뭐야"],
        'chat': ["안녕", "반가워", "너 누구니", "고마워", "괜찮아", "알겠어"]
    }
    
    def __init__(self, embedding_service):
        """기존 embedding_service 재사용"""
        self.embedding_service = embedding_service
        self.anchor_embeddings = {}
        self._build_anchors()
    
    def _build_anchors(self) -> None:
        """Anchor 문장들의 평균 임베딩 생성"""
        for intent, anchors in self.INTENT_ANCHORS.items():
            embeddings = self.embedding_service.encode(anchors)
            avg_embedding = embeddings.mean(axis=0)
            self.anchor_embeddings[intent] = avg_embedding
        
        logger.info(f"✅ SemanticIntentRouter 초기화: {len(self.anchor_embeddings)}개 Intent")
    
    def classify(self, query: str, threshold: float = 0.5) -> Tuple[str, float]:
        """쿼리를 분류하여 Intent와 신뢰도 반환"""
        from sklearn.metrics.pairwise import cosine_similarity
        import numpy as np
        
        query_embedding = self.embedding_service.encode([query])[0]
        
        max_similarity = -1.0
        best_intent = 'explanation'
        
        for intent, anchor_emb in self.anchor_embeddings.items():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                anchor_emb.reshape(1, -1)
            )[0][0]
            
            if similarity > max_similarity:
                max_similarity = similarity
                best_intent = intent
        
        if max_similarity < threshold:
            return 'explanation', max_similarity
        
        return best_intent, float(max_similarity)
