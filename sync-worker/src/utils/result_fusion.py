"""
Result Fusion & Deduplication Engine
다중 쿼리/경로의 검색 결과를 효과적으로 통합합니다.
- Reciprocal Rank Fusion (RRF) 알고리즘
- Document ID 기반 중복 제거
- 점수 정규화 및 재정렬
"""

import logging
from typing import List, Dict, Any, Set
import numpy as np

logger = logging.getLogger(__name__)


class ResultFusionManager:
    """검색 결과 통합 및 중복 제거 매니저"""
    
    @staticmethod
    def fuse_results(
        result_sets: List[List[Dict[str, Any]]],
        method: str = 'rrf',
        k: int = 10,
        remove_duplicates: bool = True
    ) -> List[Dict[str, Any]]:
        """
        여러 검색 결과 세트를 통합
        
        Args:
            result_sets: 여러 검색의 결과 리스트
                [
                    [{"id": "doc1", "content": "...", "score": 0.9}, ...],
                    [{"id": "doc2", "content": "...", "score": 0.85}, ...],
                    ...
                ]
            method: 통합 방식 ('rrf' 또는 'score_sum')
            k: 최종 반환 문서 개수
            remove_duplicates: Document ID 기반 중복 제거 여부
        
        Returns:
            통합된 결과 리스트 (상위 k개)
        """
        if not result_sets:
            return []
        
        # 유효한 결과만 필터링
        valid_results = [r for r in result_sets if r]
        if not valid_results:
            return []
        
        logger.info(f'🔀 Result Fusion 시작: {len(result_sets)}개 결과 세트 (method={method})')
        
        if method == 'rrf':
            fused = ResultFusionManager._fuse_with_rrf(valid_results, remove_duplicates)
        else:  # score_sum
            fused = ResultFusionManager._fuse_with_score_sum(valid_results, remove_duplicates)
        
        # 상위 k개 선택
        result = fused[:k]
        logger.info(f'✅ Result Fusion 완료: {len(result)}개 반환')
        
        return result
    
    @staticmethod
    def _fuse_with_rrf(
        result_sets: List[List[Dict[str, Any]]],
        remove_duplicates: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Reciprocal Rank Fusion (RRF) 알고리즘
        
        각 검색 결과에서의 순위에 따라 점수를 계산하고 합산:
        RRF_score = Σ(1 / (k + rank))
        
        k는 보통 60 (논문: Croft et al., 2009)
        """
        K = 60  # RRF 하이퍼파라미터
        
        # doc_id -> {metadata, rrf_score, source_count, ranks}
        doc_scores: Dict[str, Dict[str, Any]] = {}
        
        for set_idx, result_set in enumerate(result_sets):
            for rank, doc in enumerate(result_set, 1):
                doc_id = doc.get('id', doc.get('document_id', ''))
                
                if not doc_id:
                    continue
                
                # 중복 제거 활성화 시 정확한 ID로, 아니면 콘텐츠로도 비교
                if remove_duplicates:
                    key = doc_id
                else:
                    # 콘텐츠의 해시로 비교 (정확한 중복만)
                    content = doc.get('content', '')
                    key = hash(content) if content else doc_id
                
                if key not in doc_scores:
                    doc_scores[key] = {
                        'doc': doc,
                        'rrf_score': 0.0,
                        'source_count': 0,
                        'ranks': [],
                        'original_scores': []
                    }
                
                # RRF 점수 계산
                rrf_contribution = 1 / (K + rank)
                doc_scores[key]['rrf_score'] += rrf_contribution
                doc_scores[key]['source_count'] += 1
                doc_scores[key]['ranks'].append(rank)
                doc_scores[key]['original_scores'].append(
                    doc.get('score', doc.get('final_score', 0.0))
                )
        
        # 점수 기준 정렬
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: (
                x[1]['source_count'],  # 여러 경로에서 등장한 문서 우선
                x[1]['rrf_score']       # RRF 점수
            ),
            reverse=True
        )
        
        # 최종 결과 구성
        results = []
        for key, metadata in sorted_docs:
            doc = metadata['doc'].copy()
            
            # 점수 업데이트
            doc['rrf_score'] = metadata['rrf_score']
            doc['source_count'] = metadata['source_count']
            doc['score'] = metadata['rrf_score']  # 최종 점수로 설정
            doc['ranks'] = metadata['ranks']
            doc['fusion_method'] = 'rrf'
            
            results.append(doc)
        
        logger.debug(f'[RRF] {len(results)}개 문서 통합 (평균 {np.mean([m["source_count"] for m in doc_scores.values()]):.2f}개 경로)')
        
        return results
    
    @staticmethod
    def _fuse_with_score_sum(
        result_sets: List[List[Dict[str, Any]]],
        remove_duplicates: bool = True
    ) -> List[Dict[str, Any]]:
        """
        점수 합산 기반 통합
        
        각 결과의 원본 점수를 정규화 후 합산:
        final_score = Σ(normalized_score_i)
        """
        # doc_id -> {metadata, combined_score, source_count}
        doc_scores: Dict[str, Dict[str, Any]] = {}
        
        for set_idx, result_set in enumerate(result_sets):
            # 점수 정규화 (Min-Max)
            if result_set:
                scores = [doc.get('score', doc.get('final_score', 0.0)) for doc in result_set]
                min_score = min(scores)
                max_score = max(scores)
                score_range = max_score - min_score if max_score > min_score else 1
            else:
                score_range = 1
            
            for doc in result_set:
                doc_id = doc.get('id', doc.get('document_id', ''))
                
                if not doc_id:
                    continue
                
                key = doc_id if remove_duplicates else hash(doc.get('content', ''))
                
                if key not in doc_scores:
                    doc_scores[key] = {
                        'doc': doc,
                        'combined_score': 0.0,
                        'source_count': 0
                    }
                
                # 점수 정규화 및 누적
                original_score = doc.get('score', doc.get('final_score', 0.0))
                normalized = (original_score - min_score) / score_range if score_range > 0 else 0.5
                
                doc_scores[key]['combined_score'] += normalized
                doc_scores[key]['source_count'] += 1
        
        # 점수 기준 정렬
        sorted_docs = sorted(
            doc_scores.items(),
            key=lambda x: (
                x[1]['source_count'],     # 여러 경로에서 등장한 문서 우선
                x[1]['combined_score']    # 누적 점수
            ),
            reverse=True
        )
        
        # 최종 결과 구성
        results = []
        for key, metadata in sorted_docs:
            doc = metadata['doc'].copy()
            
            # 점수 업데이트
            doc['combined_score'] = metadata['combined_score']
            doc['source_count'] = metadata['source_count']
            doc['score'] = metadata['combined_score'] / metadata['source_count']  # 평균화
            doc['fusion_method'] = 'score_sum'
            
            results.append(doc)
        
        logger.debug(f'[Score Sum] {len(results)}개 문서 통합')
        
        return results
    
    @staticmethod
    def deduplicate_by_document_id(
        documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Document ID 기반 중복 제거
        
        같은 document_id를 가진 여러 청크 중 점수가 높은 것만 유지
        """
        # doc_id -> best_document
        best_docs: Dict[str, Dict[str, Any]] = {}
        
        for doc in documents:
            doc_id = doc.get('document_id', doc.get('id', ''))
            
            if not doc_id:
                continue
            
            score = doc.get('score', doc.get('final_score', 0.0))
            
            if doc_id not in best_docs or score > best_docs[doc_id].get('score', 0):
                best_docs[doc_id] = doc
        
        # 점수 기준 정렬 후 반환
        result = list(best_docs.values())
        result.sort(key=lambda x: x.get('score', 0), reverse=True)
        
        logger.info(f'🔄 중복 제거: {len(documents)} → {len(result)}개')
        
        return result
    
    @staticmethod
    def deduplicate_by_content_hash(
        documents: List[Dict[str, Any]],
        similarity_threshold: float = 0.95
    ) -> List[Dict[str, Any]]:
        """
        콘텐츠 해시 기반 중복 제거
        
        완전히 동일한 콘텐츠는 제거
        """
        seen_hashes: Set[str] = set()
        unique_docs = []
        
        for doc in documents:
            content = doc.get('content', '')
            
            # 콘텐츠 해시 계산
            content_hash = str(hash(content))
            
            if content_hash not in seen_hashes:
                seen_hashes.add(content_hash)
                unique_docs.append(doc)
        
        logger.info(f'🔄 콘텐츠 중복 제거: {len(documents)} → {len(unique_docs)}개')
        
        return unique_docs
