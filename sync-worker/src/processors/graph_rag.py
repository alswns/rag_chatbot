"""
GraphRAG Refactored - NetworkX 기반 경량 그래프 RAG

개선 사항:
1. NetworkX 도입: nx.DiGraph() + nx.ego_graph() 사용
2. 명시적 관계 중심: "작업" 컬럼(관계형 속성) + parent 필드 기반
3. 메모리 최적화: text 필드 제거, Lazy Loading (Chroma에서만 로드)
4. 타입 힌팅 & 로깅 강화

처리 파이프라인:
1. Notion API → 모든 문서 추출
2. 문서 → Node 변환 (메타데이터만, text 제외)
3. 임베딩 생성 (문서 제목 + 메타데이터 기반)
4. NetworkX 그래프 구축 ("작업" 관계 + parent 계층)
5. Chroma DB 저장 (전체 text 포함)
6. 질문 처리 시 ego_graph로 이웃 노드 추출, Chroma에서 lazy load

작성자: RAG Chatbot Team
"""

import logging
from typing import List, Dict, Set, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
import json
import pickle
import os

import numpy as np
import networkx as nx

from utils.embedding_service import get_embedding_service

# 로깅 설정
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class GraphNode:
    """
    메모리 효율적인 그래프 노드
    
    텍스트(text)는 저장하지 않음 → Lazy Loading으로 Chroma에서만 로드
    """
    
    # 기본 정보
    node_id: str                        # Notion document ID
    title: str                          # 문서 제목
    
    # 메타데이터
    source_url: str                     # Notion URL
    created_at: str                     # 생성 날짜 (ISO 8601)
    updated_at: str                     # 수정 날짜 (ISO 8601)
    
    # 임베딩
    embedding: Optional[np.ndarray] = None  # 384차원 벡터 (메모리 최적화)
    
    # 추가 정보
    source: str = 'notion'
    parent_id: Optional[str] = None     # 상위 노드 ID (계층 구조)
    
    def to_metadata_dict(self) -> Dict[str, Any]:
        """Chroma DB 저장용 메타데이터 생성"""
        return {
            'node_id': self.node_id,
            'title': self.title,
            'source_url': self.source_url,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'source': self.source,
            'parent_id': self.parent_id
        }


class GraphRAGProcessor:
    """
    NetworkX 기반 경량 그래프 RAG 프로세서
    
    - NetworkX DiGraph로 그래프 관리
    - 명시적 관계("작업" 컬럼) 기반의 엣지 생성
    - parent 필드를 통한 계층 엣지 추가
    - 메모리 최적화 (Lazy Loading)
    """
    
    def __init__(
        self,
        embedding_model: str = 'BAAI/bge-m3', 
        max_chunk_tokens: int = 512,
        chunk_overlap_tokens: int = 50,
        traversal_depth: int = 2,
        relation_column_name: str = '작업',
    ) -> None:
        """
        초기화
        
        Args:
            embedding_model: 임베딩 모델 경로 (기본값: BAAI/bge-m3)
            max_chunk_tokens: 청크 최대 토큰 수
            chunk_overlap_tokens: 청크 간 오버랩
            traversal_depth: ego_graph 탐색 깊이
            relation_column_name: 관계형 컬럼명 (기본값: '작업')
        """
        self.max_chunk_tokens: int = max_chunk_tokens
        self.chunk_overlap_tokens: int = chunk_overlap_tokens
        self.traversal_depth: int = traversal_depth
        self.relation_column_name: str = relation_column_name  # ✅ 관계형 컬럼명 저장
        
        # NetworkX 그래프 (노드: id, 엣지: 관계)
        self.graph: nx.DiGraph = nx.DiGraph()
        
        # 메모리 저장소 (text 제외)
        self.nodes: Dict[str, GraphNode] = {}
        self.node_embeddings: Dict[str, np.ndarray] = {}
        
        # 싱글톤 임베딩 서비스 (중복 로드 방지)
        logger.info(f'🔄 싱글톤 임베딩 서비스 초기화: {embedding_model}')
        self.embedding_service = get_embedding_service(embedding_model)
        logger.info(f'✅ 싱글톤 임베딩 서비스 준비 완료')
    
    # ========================================================================
    # 참고: Notion 문서 추출은 NotionConnector에서 수행
    # 본 클래스는 추출된 문서를 처리하는 역할을 수행합니다.
    # ========================================================================
    
    # ========================================================================
    # 1단계: 문서 → Node 변환 (text 제외)
    # ========================================================================
    
    def process_document_to_nodes(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[GraphNode]:
        """
        Notion 문서를 메모리 효율적인 Node로 변환
        
        text 필드는 저장하지 않음 (Chroma DB에만 저장)
        
        Args:
            documents: Notion 문서 리스트
        
        Returns:
            GraphNode 리스트
        """
        logger.info(f'{len(documents)}개 문서를 Node로 변환 중')
        
        nodes: List[GraphNode] = []
        
        for doc in documents:
            try:
                # 문서 ID
                node_id: str = doc.get('id')
                title: str = doc.get('title', 'Untitled')
                content: str = doc.get('content', '')
                
                # 내용이 없는 문서는 스킵
                if not content or not content.strip():
                    logger.warning(f'내용 없음 (스킵): {title}')
                    continue
                
                # Node 생성 (text 제외)
                node: GraphNode = GraphNode(
                    node_id=node_id,
                    title=title,
                    source_url=doc.get('url', ''),
                    created_at=doc.get('created_at', datetime.now().isoformat()),
                    updated_at=doc.get('updated_at', datetime.now().isoformat()),
                    parent_id=doc.get('parent_id'),  # 계층 정보 포함
                    source='notion'
                )
                
                nodes.append(node)
                logger.debug(f'Node 생성: {title} ({node_id})')
                
            except Exception as e:
                logger.warning(f'Node 변환 실패: {str(e)}')
                continue
        
        # 메모리에 저장
        for node in nodes:
            self.nodes[node.node_id] = node
        
        logger.info(f'✓ {len(nodes)}개 Node 생성 완료')
        return nodes
    
    # ========================================================================
    # 2단계: 임베딩 생성
    # ========================================================================
    
    def generate_embeddings(
        self,
        nodes: List[GraphNode],
        documents: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, np.ndarray]:
        """
        ✅ [리팩토링] Title + Parent ID + Content Snippet(500자) 임베딩
        
        임베딩 품질 향상:
        - 제목만으로는 의미 부족 → 본문 앞 500자 포함
        - Parent ID로 계층 구조 반영
        - 인자로 받은 nodes 객체에 직접 embedding 주입 (버그 수정)
        
        Args:
            nodes: GraphNode 리스트 (embedding 필드에 직접 주입)
            documents: Notion 문서 리스트 (content 포함)
        
        Returns:
            {node_id: embedding_vector} 딕셔너리
        """
        if not nodes:
            logger.warning('임베딩할 노드가 없습니다')
            return {}
        
        logger.info(f'{len(nodes)}개 노드의 임베딩 생성 중')
        
        # 문서 내용 맵 생성 (빠른 조회용)
        doc_map: Dict[str, str] = {}
        if documents:
            doc_map = {doc.get('id'): doc.get('content', '') for doc in documents}
        
        texts: List[str] = []
        
        # ✅ [핵심] 임베딩 텍스트 준비: Title + Parent + Content Snippet
        for node in nodes:
            text_parts = []
            
            # 1️⃣ 제목 (필수)
            text_parts.append(f"Title: {node.title}")
            
            # 2️⃣ 계층 정보 (Parent ID)
            if node.parent_id:
                text_parts.append(f"Parent: {node.parent_id}")
            
            # 3️⃣ 본문 앞부분 (500자) - 의미 있는 정보 추출
            content = doc_map.get(node.node_id, '')
            if content:
                content_snippet = content.strip().replace('\n\n', '\n')[:500]
                text_parts.append(f"Content: {content_snippet}")
            
            text_for_embedding = "\n".join(text_parts)
            texts.append(text_for_embedding)
            
            logger.debug(f'임베딩 텍스트 준비: {node.title} ({len(text_for_embedding)}자)')
        
        try:
            # 배치 임베딩 생성 (싱글톤 서비스 사용)
            logger.debug(f'싱글톤 임베딩 서비스로 {len(texts)}개 텍스트 임베딩 중...')
            embedding_vectors: np.ndarray = self.embedding_service.encode(texts)
            
            embeddings: Dict[str, np.ndarray] = {}
            
            # ✅ [버그 수정] 인자로 받은 nodes에 직접 embedding 주입
            for node, embedding in zip(nodes, embedding_vectors):
                node.embedding = embedding  # 직접 주입
                embeddings[node.node_id] = embedding
                
                # 내부 저장소에도 저장
                self.nodes[node.node_id] = node
            
            # 메모리 인덱스 업데이트
            self.node_embeddings.update(embeddings)
            
            logger.info(f'✓ {len(embeddings)}개 임베딩 생성 완료')
            return embeddings
            
        except Exception as e:
            logger.error(f'임베딩 생성 실패: {str(e)}', exc_info=True)
            return {}
    
    # ========================================================================
    # 3단계: 그래프 구축 (명시적 관계 + parent 계층 + 본문 링크)
    # ========================================================================
    
    # 가상 루트 노드 ID (모든 최상위 문서의 부모)
    VIRTUAL_ROOT_ID: str = 'ROOT_UNIVERSE'
    
    def build_graph(
        self,
        documents: List[Dict[str, Any]],
        nodes: List[GraphNode],
        clear_existing_edges: bool = True  # Delta Sync 시 기존 엣지 정리 옵션
    ) -> None:
        """
        ✅ [리팩토링] NetworkX 그래프 구축 - 연결성 극대화
        
        개선 사항:
        1. 가상 루트(Virtual Root): ROOT_UNIVERSE 노드로 최상위 문서 연결
        2. 유령 노드(Ghost Node): 없는 노드 참조 시 껍데기 노드 자동 생성
        3. 본문 링크(Mentions): 본문 내 Notion UUID 추출하여 엣지 생성
        4. Delta Sync 지원: 기존 엣지 정리 후 재연결
        
        엣지 타입:
        - hierarchy: parent_id 기반 계층 관계
        - references: "작업" 컬럼 등 명시적 관계
        - mention: 본문 내 링크
        - virtual_root: 가상 루트 연결
        
        Args:
            documents: 원본 Notion 문서 리스트 (properties, content 포함)
            nodes: GraphNode 리스트
            clear_existing_edges: Delta Sync 시 기존 엣지 삭제 여부
        """
        import re
        
        logger.info('=' * 60)
        logger.info(f'🔗 그래프 구축 시작 (노드: {len(nodes)}개, 엣지 초기화: {clear_existing_edges})')
        logger.info('=' * 60)
        
        # 현재 배치의 노드 ID 집합
        current_node_ids: Set[str] = {node.node_id for node in nodes}
        
        # ========================================
        # Step 0: 가상 루트 노드 생성
        # ========================================
        if self.VIRTUAL_ROOT_ID not in self.graph:
            self.graph.add_node(
                self.VIRTUAL_ROOT_ID,
                title='Virtual Root',
                node_type='virtual_root'
            )
            logger.info(f'📍 가상 루트 노드 생성: {self.VIRTUAL_ROOT_ID}')
        
        # ========================================
        # Step 1: Delta Sync - 기존 엣지 정리
        # ========================================
        if clear_existing_edges:
            edges_removed = 0
            for node_id in current_node_ids:
                if node_id in self.graph:
                    # 해당 노드에서 나가는 모든 엣지 삭제 (Stale Edge 방지)
                    out_edges = list(self.graph.out_edges(node_id))
                    for edge in out_edges:
                        self.graph.remove_edge(*edge)
                        edges_removed += 1
            if edges_removed > 0:
                logger.info(f'🧹 기존 엣지 정리: {edges_removed}개 삭제')
        
        # ========================================
        # Step 2: 노드 추가/업데이트
        # ========================================
        for node in nodes:
            self.graph.add_node(
                node.node_id,
                title=node.title,
                node_type='document',  # 실제 문서
                source=node.source,
                parent_id=node.parent_id
            )
        logger.info(f'📄 노드 추가/업데이트: {len(nodes)}개')
        
        # 문서 ID → 문서 맵 생성
        doc_map: Dict[str, Dict[str, Any]] = {doc.get('id'): doc for doc in documents}
        
        # 통계 변수
        stats = {
            'hierarchy': 0,
            'references': 0,
            'mention': 0,
            'virtual_root': 0,
            'ghost_nodes': 0
        }
        
        # ========================================
        # Step 3: 각 노드에 대해 엣지 생성
        # ========================================
        for node in nodes:
            node_id = node.node_id
            doc = doc_map.get(node_id, {})
            
            # ----------------------------------
            # 3-1. Hierarchy 엣지 (parent_id)
            # ----------------------------------
            parent_id = node.parent_id
            
            if parent_id:
                # 부모 노드가 그래프에 없으면 Ghost Node 생성
                if parent_id not in self.graph:
                    self.graph.add_node(
                        parent_id,
                        title=f'Ghost: {parent_id[:8]}...',
                        node_type='ghost'
                    )
                    stats['ghost_nodes'] += 1
                    logger.debug(f'👻 Ghost Node 생성: {parent_id[:12]}...')
                
                # parent → child 엣지
                self.graph.add_edge(
                    parent_id,
                    node_id,
                    edge_type='hierarchy',
                    weight=1.0
                )
                stats['hierarchy'] += 1
            else:
                # 부모가 없으면 → 가상 루트에 연결
                self.graph.add_edge(
                    self.VIRTUAL_ROOT_ID,
                    node_id,
                    edge_type='virtual_root',
                    weight=0.5  # 가상 연결은 낮은 가중치
                )
                stats['virtual_root'] += 1
            
            # ----------------------------------
            # 3-2. References 엣지 (관계형 속성 확장)
            # ----------------------------------
            properties: Dict = doc.get('properties', {})
            
            # ✅ 확장된 관계형 컬럼명 목록 (대소문자 무시)
            RELATION_KEYWORDS = [
                # 기본 설정
                self.relation_column_name.lower(),
                # 영문
                'work', 'related', 'reference', 'references', 'link', 'links',
                'parent', 'child', 'children', 'depends', 'dependency', 'dependencies',
                'related to', 'linked to', 'see also', 'associated',
                # 한글
                '작업', '관련', '참고', '관련 문서', '참고 문서', '연결', '링크',
                '상위', '하위', '부모', '자식', '의존성', '연관'
            ]
            
            for prop_name, prop_value in properties.items():
                prop_name_lower = prop_name.lower().strip()
                
                # 방법 1: 키워드 매칭
                is_relation_by_keyword = any(
                    keyword in prop_name_lower for keyword in RELATION_KEYWORDS
                )
                
                # 방법 2: Notion relation 타입 자동 감지
                is_relation_by_type = (
                    isinstance(prop_value, dict) and 
                    prop_value.get('type') == 'relation'
                )
                
                # 방법 3: relation 키 존재 확인
                has_relation_key = (
                    isinstance(prop_value, dict) and 
                    'relation' in prop_value
                )
                
                if is_relation_by_keyword or is_relation_by_type or has_relation_key:
                    if isinstance(prop_value, dict):
                        relation_list = prop_value.get('relation', [])
                        
                        if isinstance(relation_list, list):
                            for target in relation_list:
                                target_id = target.get('id') if isinstance(target, dict) else target
                                
                                if not target_id or target_id == node_id:
                                    continue  # 자기 참조 제외
                                
                                # 타겟 노드가 없으면 Ghost Node 생성
                                if target_id not in self.graph:
                                    self.graph.add_node(
                                        target_id,
                                        title=f'Ghost: {target_id[:8]}...',
                                        node_type='ghost'
                                    )
                                    stats['ghost_nodes'] += 1
                                
                                # 중복 엣지 방지
                                if not self.graph.has_edge(node_id, target_id):
                                    self.graph.add_edge(
                                        node_id,
                                        target_id,
                                        edge_type='references',
                                        relation_name=prop_name,  # 원본 컬럼명 저장
                                        weight=1.0
                                    )
                                    stats['references'] += 1
                                    logger.debug(f'🔗 references 엣지: {node_id[:8]}... → {target_id[:8]}... (컬럼: {prop_name})')
            
            # ----------------------------------
            # 3-3. Mention 엣지 (본문 내 링크)
            # ----------------------------------
            content: str = doc.get('content', '')
            
            if content:
                # Notion UUID 패턴: 32자리 16진수 (하이픈 제거된 형태)
                uuid_pattern = r'([a-fA-F0-9]{32})'
                # 하이픈 포함 UUID 패턴
                uuid_with_hyphen = r'([a-fA-F0-9]{8}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{4}-[a-fA-F0-9]{12})'
                
                # 두 패턴 모두 검색
                found_uuids: Set[str] = set()
                
                for match in re.finditer(uuid_pattern, content):
                    found_uuids.add(match.group(1))
                
                for match in re.finditer(uuid_with_hyphen, content):
                    # 하이픈 제거하여 통일
                    uuid_normalized = match.group(1).replace('-', '')
                    found_uuids.add(uuid_normalized)
                
                # 자기 자신 제외
                node_id_normalized = node_id.replace('-', '')
                found_uuids.discard(node_id_normalized)
                
                for target_uuid in found_uuids:
                    # 하이픈 있는 형식과 없는 형식 모두 확인
                    target_id = None
                    
                    # 그래프에서 매칭되는 노드 찾기
                    for existing_node in self.graph.nodes():
                        if existing_node.replace('-', '') == target_uuid:
                            target_id = existing_node
                            break
                    
                    if not target_id:
                        # Ghost Node 생성 (하이픈 포함 형식으로)
                        target_id = f'{target_uuid[:8]}-{target_uuid[8:12]}-{target_uuid[12:16]}-{target_uuid[16:20]}-{target_uuid[20:]}'
                        self.graph.add_node(
                            target_id,
                            title=f'Ghost: {target_uuid[:8]}...',
                            node_type='ghost'
                        )
                        stats['ghost_nodes'] += 1
                    
                    # Mention 엣지 추가 (중복 방지)
                    if not self.graph.has_edge(node_id, target_id):
                        self.graph.add_edge(
                            node_id,
                            target_id,
                            edge_type='mention',
                            weight=0.8  # 본문 링크는 약간 낮은 가중치
                        )
                        stats['mention'] += 1
        
        # ========================================
        # Step 4: 고립된 Ghost Node 정리 (선택적)
        # ========================================
        # Ghost Node 중 어떤 엣지도 없는 노드는 제거
        isolated_ghosts = []
        for node_id in list(self.graph.nodes()):
            node_data = self.graph.nodes[node_id]
            if node_data.get('node_type') == 'ghost':
                if self.graph.in_degree(node_id) == 0 and self.graph.out_degree(node_id) == 0:
                    isolated_ghosts.append(node_id)
        
        for ghost_id in isolated_ghosts:
            self.graph.remove_node(ghost_id)
            stats['ghost_nodes'] -= 1
        
        if isolated_ghosts:
            logger.debug(f'🧹 고립된 Ghost Node 제거: {len(isolated_ghosts)}개')
        
        # ========================================
        # 통계 로그
        # ========================================
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()
        ghost_count = len([n for n in self.graph.nodes() if self.graph.nodes[n].get('node_type') == 'ghost'])
        
        logger.info('=' * 60)
        logger.info('✅ 그래프 구축 완료')
        logger.info(f'  📊 노드: {total_nodes}개 (실제: {total_nodes - ghost_count - 1}, Ghost: {ghost_count}, Root: 1)')
        logger.info(f'  🔗 엣지: {total_edges}개')
        logger.info(f'     ├─ hierarchy: {stats["hierarchy"]}개')
        logger.info(f'     ├─ references: {stats["references"]}개')
        logger.info(f'     ├─ mention: {stats["mention"]}개')
        logger.info(f'     └─ virtual_root: {stats["virtual_root"]}개')
        logger.info('=' * 60)
    
    # ========================================================================
    # [LEGACY] 기존 메서드 - 새 build_graph에 통합됨
    # ========================================================================
    
    def _extract_relation_edges_legacy(
        self,
        documents: List[Dict[str, Any]]
    ) -> None:
        """
        [DEPRECATED] 새 build_graph에 통합됨
        환경변수로 설정된 관계형 컬럼명에서 ID 기반 엣지 추출
        
        Args:
            documents: Notion 문서 리스트 (properties 포함)
        """
        logger.info(f'"{self.relation_column_name}" 관계형 컬럼에서 엣지 추출 중')
        
        edge_count: int = 0
        
        for doc in documents:
            source_id: str = doc.get('id')
            if source_id not in self.graph:
                continue
            
            try:
                # properties에서 관계형 컬럼 찾기
                properties: Dict = doc.get('properties', {})
                
                for prop_name, prop_value in properties.items():
                    # 관계형 컬럼명과 매칭 (환경변수 기반, case-insensitive)
                    if prop_name.lower() == self.relation_column_name.lower() or prop_name.lower() == 'work':
                        # 관계형 속성의 값에서 ID 추출
                        if isinstance(prop_value, dict):
                            relation_ids: List[str] = prop_value.get('relation', [])
                            
                            if isinstance(relation_ids, list):
                                for target_id in relation_ids:
                                    if isinstance(target_id, dict):
                                        target_id = target_id.get('id')
                                    
                                    # 대상 ID가 그래프에 존재하는지 확인
                                    if target_id and target_id in self.graph:
                                        # references 엣지 추가 (가중치: 1.0)
                                        self.graph.add_edge(
                                            source_id,
                                            target_id,
                                            edge_type='references',
                                            weight=1.0
                                        )
                                        edge_count += 1
                                        logger.debug(f'references 엣지 추가: {source_id} → {target_id}')
                
            except Exception as e:
                logger.warning(f'관계형 엣지 추출 실패 ({source_id}): {str(e)}')
        
        logger.info(f'✓ {edge_count}개의 references 엣지 추가')
    
    def _add_hierarchy_edges_legacy(
        self,
        nodes: List[GraphNode]
    ) -> None:
        """
        [DEPRECATED] 새 build_graph에 통합됨
        parent_id 필드를 기반으로 계층 엣지 추가
        
        Args:
            nodes: GraphNode 리스트
        """
        logger.info('parent_id 기반 계층 엣지 추가 중')
        
        hierarchy_count: int = 0
        
        for node in nodes:
            if node.parent_id and node.parent_id in self.graph:
                # parent → child 방향으로 엣지 추가
                self.graph.add_edge(
                    node.parent_id,
                    node.node_id,
                    edge_type='hierarchy',
                    weight=1.0
                )
                hierarchy_count += 1
                logger.debug(f'hierarchy 엣지 추가: {node.parent_id} → {node.node_id}')
        
        logger.info(f'✓ {hierarchy_count}개의 hierarchy 엣지 추가')
    
    # ========================================================================
    # 4단계: Chroma DB 저장 (text 포함)
    # ========================================================================
    
    def save_nodes_to_chroma(
        self,
        vector_store_manager,
        documents: List[Dict[str, Any]]
    ) -> int:
        """
        노드와 전체 텍스트를 Chroma DB에 저장
        
        Args:
            vector_store_manager: VectorStoreManager 인스턴스
            documents: Notion 문서 리스트 (content 포함)
        
        Returns:
            저장된 항목 개수
        """
        logger.info(f'{len(documents)}개 문서를 Chroma DB에 저장 중')
        
        documents_to_add: List[Dict[str, Any]] = []
        
        for doc in documents:
            node_id: str = doc.get('id')
            
            # 그래프에 있는 노드만 저장
            if node_id not in self.nodes:
                continue
            
            node: GraphNode = self.nodes[node_id]
            
            # Chroma DB 문서 형식으로 변환
            chroma_doc: Dict[str, Any] = {
                'id': node_id,
                'content': doc.get('content', ''),  # 전체 텍스트
                'metadata': {
                    **node.to_metadata_dict(),
                    # 그래프 정보 추가
                    'in_degree': self.graph.in_degree(node_id),
                    'out_degree': self.graph.out_degree(node_id),
                    'edges': json.dumps([
                        {
                            'source': u,
                            'target': v,
                            'type': self.graph[u][v].get('edge_type', 'unknown'),
                            'weight': self.graph[u][v].get('weight', 1.0)
                        }
                        for u, v in self.graph.in_edges(node_id)
                    ])
                }
            }
            
            documents_to_add.append(chroma_doc)
        
        try:
            # Chroma DB에 저장
            stored_count: int = vector_store_manager.add_documents(
                documents_to_add,
                batch_size=100
            )
            
            logger.info(f'✓ {stored_count}개 문서 저장 완료')
            return stored_count
            
        except Exception as e:
            logger.error(f'Chroma DB 저장 실패: {str(e)}', exc_info=True)
            return 0
    
    # ========================================================================
    # 5단계: 그래프 탐색 (nx.ego_graph 사용)
    # ========================================================================
    
    def graph_traversal(
        self,
        question_embedding: np.ndarray,
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        질문 임베딩을 기반으로 관련 노드 탐색
        
        처리:
        1. 질문과 유사한 Top-K 노드 검색
        2. 각 Top-K 노드에서 ego_graph로 이웃 노드 추출
        3. 탐색 깊이 제한으로 토큰 폭발 방지
        
        Args:
            question_embedding: 질문의 임베딩 벡터
            top_k: 초기 후보 노드 개수
        
        Returns:
            {
                'candidate_nodes': [노드 ID],
                'traversed_nodes': {노드 ID: {거리, 경로}},
                'subgraph': nx.DiGraph (탐색 결과 서브그래프)
            }
        """
        logger.info(f'그래프 탐색 시작 (top_k={top_k}, depth={self.traversal_depth})')
        
        # 1. Top-K 유사 노드 검색
        candidate_nodes: List[str] = self._search_similar_nodes(question_embedding, top_k)
        logger.info(f'Top-{top_k} 후보 노드: {candidate_nodes}')
        
        if not candidate_nodes:
            logger.warning('유사한 노드를 찾을 수 없습니다')
            return {
                'candidate_nodes': [],
                'traversed_nodes': {},
                'subgraph': nx.DiGraph()
            }
        
        # 2. ego_graph로 이웃 노드 추출 (NetworkX)
        traversed_nodes: Dict[str, Dict[str, Any]] = {}
        subgraph: nx.DiGraph = nx.DiGraph()
        
        # 각 후보 노드에 대해 ego_graph 생성
        for i, node_id in enumerate(candidate_nodes):
            try:
                # ego_graph: 지정된 깊이 내의 모든 이웃 노드 포함
                ego: nx.DiGraph = nx.ego_graph(
                    self.graph,
                    node_id,
                    radius=self.traversal_depth,
                    undirected=False  # 방향성 그래프
                )
                
                # 서브그래프에 통합
                subgraph.add_nodes_from(ego.nodes(data=True))
                subgraph.add_edges_from(ego.edges(data=True))
                
                # 노드 정보 기록 (거리 = 초기 TOP-K 순위)
                for node in ego.nodes():
                    if node not in traversed_nodes:
                        traversed_nodes[node] = {
                            'distance': nx.shortest_path_length(ego, node_id, node) if node != node_id else 0,
                            'from_candidate': node_id,
                            'rank': i
                        }
                
                logger.debug(f'ego_graph 생성: {node_id} (노드: {len(ego.nodes())}, 엣지: {len(ego.edges())})')
                
            except Exception as e:
                logger.warning(f'ego_graph 생성 실패 ({node_id}): {str(e)}')
        
        logger.info(f'✓ 그래프 탐색 완료: {len(traversed_nodes)}개 노드 탐색됨')
        
        return {
            'candidate_nodes': candidate_nodes,
            'traversed_nodes': traversed_nodes,
            'subgraph': subgraph
        }
    
    def _search_similar_nodes(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[str]:
        """
        질문 임베딩과 유사한 노드 검색
        
        Args:
            query_embedding: 질문의 임베딩 벡터
            top_k: 반환할 상위 노드 개수
        
        Returns:
            노드 ID 리스트 (유사도 순)
        """
        if not self.node_embeddings:
            logger.warning('임베딩이 없습니다')
            return []
        
        # 모든 노드와의 유사도 계산
        similarities: Dict[str, float] = {}
        for node_id, node_embedding in self.node_embeddings.items():
            # Cosine 유사도
            similarity: float = float(
                np.dot(query_embedding, node_embedding) / 
                (np.linalg.norm(query_embedding) * np.linalg.norm(node_embedding) + 1e-8)
            )
            similarities[node_id] = similarity
        
        # 상위 K개 선택
        top_nodes: List[Tuple[str, float]] = sorted(
            similarities.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        logger.debug(f'Top-{top_k} 유사 노드: {[n[0] for n in top_nodes]}')
        
        return [node_id for node_id, _ in top_nodes]
    
    # ========================================================================
    # 6단계: Lazy Loading을 통한 컨텍스트 생성
    # ========================================================================
    
    def prepare_context_from_traversal(
        self,
        vector_store_manager,
        traversal_result: Dict[str, Any],
        include_depth_limit: int = 2
    ) -> str:
        """
        그래프 탐색 결과에서 LLM 입력용 컨텍스트 생성
        
        Lazy Loading: 탐색된 노드 ID들을 Chroma에서 한 번에 로드
        
        Args:
            vector_store_manager: VectorStoreManager 인스턴스
            traversal_result: graph_traversal() 반환값
            include_depth_limit: 포함할 최대 깊이
        
        Returns:
            컨텍스트 문자열
        """
        logger.info('Lazy Loading으로 컨텍스트 생성 중')
        
        traversed_nodes: Dict[str, Dict] = traversal_result.get('traversed_nodes', {})
        
        # 포함할 노드 ID 필터링
        node_ids_to_load: List[str] = [
            node_id for node_id, info in traversed_nodes.items()
            if info.get('distance', 0) <= include_depth_limit
        ]
        
        logger.debug(f'Chroma에서 {len(node_ids_to_load)}개 문서 로드 중')
        
        if not node_ids_to_load:
            logger.warning('포함할 노드가 없습니다')
            return ""
        
        try:
            # Chroma DB에서 한 번에 로드 (Lazy Loading)
            documents: Dict[str, str] = vector_store_manager.get_documents(
                ids=node_ids_to_load
            )
            
            # 깊이순으로 정렬
            sorted_nodes: List[Tuple[str, Dict]] = sorted(
                traversed_nodes.items(),
                key=lambda x: x[1].get('distance', 0)
            )
            
            # 컨텍스트 생성
            context_parts: List[str] = []
            for node_id, info in sorted_nodes:
                distance: int = info.get('distance', 0)
                
                if distance > include_depth_limit:
                    continue
                
                if node_id not in documents:
                    continue
                
                content: str = documents[node_id]
                node: GraphNode = self.nodes.get(node_id)
                
                if not node:
                    continue
                
                # 깊이별 섹션 제목
                if distance == 0:
                    section: str = f"[Primary Document: {node.title}]\n{content}\n"
                else:
                    section: str = f"[Related Document (depth={distance}): {node.title}]\n{content}\n"
                
                context_parts.append(section)
            
            context: str = '\n---\n'.join(context_parts)
            logger.info(f'✓ 컨텍스트 생성 완료 ({len(context)} 자)')
            
            return context
            
        except Exception as e:
            logger.error(f'Lazy Loading 실패: {str(e)}', exc_info=True)
            return ""
    
    # ========================================================================
    # 7단계: End-to-End 질문 처리
    # ========================================================================
    
    def query_with_graph(
        self,
        question: str,
        vector_store_manager,
        top_k_candidates: int = 5,
        include_depth_limit: int = 2
    ) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성을 위한 컨텍스트 준비 (End-to-End)
        
        처리:
        1. 질문 임베딩
        2. Top-K 노드 검색
        3. ego_graph로 이웃 노드 탐색
        4. Lazy Loading으로 컨텍스트 생성
        
        Args:
            question: 사용자 질문
            vector_store_manager: VectorStoreManager 인스턴스
            top_k_candidates: 초기 후보 노드 개수
            include_depth_limit: 컨텍스트에 포함할 최대 깊이
        
        Returns:
            {
                'question': 질문,
                'context': LLM 입력용 컨텍스트,
                'used_documents': 사용된 문서 ID,
                'traversal_info': 그래프 탐색 정보
            }
        """
        logger.info(f'질문 처리 시작: "{question}"')
        
        try:
            # 1. 질문 임베딩
            logger.debug('질문 임베딩 생성 중...')
            question_embedding: np.ndarray = self.embedding_model.encode(
                question,
                convert_to_numpy=True
            )
            
            # 2. 그래프 탐색
            logger.debug('그래프 탐색 중...')
            traversal_result: Dict[str, Any] = self.graph_traversal(
                question_embedding,
                top_k=top_k_candidates
            )
            
            # 3. Lazy Loading으로 컨텍스트 생성
            logger.debug('Lazy Loading으로 컨텍스트 생성 중...')
            context: str = self.prepare_context_from_traversal(
                vector_store_manager,
                traversal_result,
                include_depth_limit=include_depth_limit
            )
            
            # 4. 사용된 문서 추적
            used_documents: List[str] = list(traversal_result.get('traversed_nodes', {}).keys())
            
            logger.info(f'✓ 질문 처리 완료 (사용된 문서: {len(used_documents)}개)')
            
            return {
                'question': question,
                'context': context,
                'used_documents': used_documents,
                'traversal_info': traversal_result
            }
            
        except Exception as e:
            logger.error(f'질문 처리 실패: {str(e)}', exc_info=True)
            return {
                'question': question,
                'context': '',
                'used_documents': [],
                'traversal_info': {}
            }
    
    # ========================================================================
    # 그래프 영속화: Pickle 저장/로드
    # ========================================================================
    
    def save_graph(self, filepath: str) -> bool:
        """
        그래프와 노드 데이터를 pickle 파일로 저장
        
        저장 내용:
        - NetworkX 그래프 객체
        - 노드 메타데이터 (nodes dict)
        - 노드 임베딩 (node_embeddings dict)
        
        Args:
            filepath: 저장할 파일 경로 (.pkl)
        
        Returns:
            성공 여부
        """
        try:
            # 디렉토리 생성
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # 저장할 데이터 구성
            data = {
                'graph': self.graph,
                'nodes': {
                    node_id: {
                        'node_id': node.node_id,
                        'title': node.title,
                        'source_url': node.source_url,
                        'created_at': node.created_at,
                        'updated_at': node.updated_at,
                        'parent_id': node.parent_id,
                        'source': node.source
                    }
                    for node_id, node in self.nodes.items()
                },
                'embeddings': {
                    node_id: emb.tolist() if isinstance(emb, np.ndarray) else emb
                    for node_id, emb in self.node_embeddings.items()
                },
                'metadata': {
                    'saved_at': datetime.now().isoformat(),
                    'node_count': len(self.nodes),
                    'edge_count': self.graph.number_of_edges(),
                    'traversal_depth': self.traversal_depth
                }
            }
            
            # Pickle로 저장
            with open(filepath, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            logger.info(f'✓ 그래프 저장 완료: {filepath} (노드: {len(self.nodes)}, 엣지: {self.graph.number_of_edges()})')
            return True
            
        except Exception as e:
            logger.error(f'그래프 저장 실패: {str(e)}', exc_info=True)
            return False
    
    def load_graph(self, filepath: str) -> bool:
        """
        pickle 파일에서 그래프와 노드 데이터 로드
        
        Args:
            filepath: 로드할 파일 경로 (.pkl)
        
        Returns:
            성공 여부
        """
        try:
            if not os.path.exists(filepath):
                logger.warning(f'그래프 파일 없음: {filepath}')
                return False
            
            # Pickle에서 로드
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            # 그래프 복원
            self.graph = data.get('graph', nx.DiGraph())
            
            # 노드 복원
            nodes_data = data.get('nodes', {})
            self.nodes = {}
            for node_id, node_dict in nodes_data.items():
                self.nodes[node_id] = GraphNode(
                    node_id=node_dict['node_id'],
                    title=node_dict['title'],
                    source_url=node_dict.get('source_url', ''),
                    created_at=node_dict.get('created_at', ''),
                    updated_at=node_dict.get('updated_at', ''),
                    parent_id=node_dict.get('parent_id'),
                    source=node_dict.get('source', 'notion')
                )
            
            # 임베딩 복원
            embeddings_data = data.get('embeddings', {})
            self.node_embeddings = {
                node_id: np.array(emb) if isinstance(emb, list) else emb
                for node_id, emb in embeddings_data.items()
            }
            
            # 노드 객체에 임베딩 연결
            for node_id, embedding in self.node_embeddings.items():
                if node_id in self.nodes:
                    self.nodes[node_id].embedding = embedding
            
            # 메타데이터 로깅
            metadata = data.get('metadata', {})
            logger.info(f'✓ 그래프 로드 완료: {filepath}')
            logger.info(f'  - 저장 시간: {metadata.get("saved_at", "unknown")}')
            logger.info(f'  - 노드: {len(self.nodes)}개')
            logger.info(f'  - 엣지: {self.graph.number_of_edges()}개')
            
            return True
            
        except Exception as e:
            logger.error(f'그래프 로드 실패: {str(e)}', exc_info=True)
            return False
    
    @classmethod
    def from_file(
        cls,
        filepath: str,
        embedding_model: str = 'sentence-transformers/all-MiniLM-L6-v2',
        traversal_depth: int = 2
    ) -> Optional['GraphRAGProcessor']:
        """
        파일에서 GraphRAGProcessor 인스턴스 생성
        
        Args:
            filepath: 그래프 파일 경로
            embedding_model: 임베딩 모델
            traversal_depth: 탐색 깊이
        
        Returns:
            GraphRAGProcessor 인스턴스 또는 None
        """
        try:
            processor = cls(
                embedding_model=embedding_model,
                traversal_depth=traversal_depth
            )
            
            if processor.load_graph(filepath):
                return processor
            else:
                return None
                
        except Exception as e:
            logger.error(f'GraphRAGProcessor 생성 실패: {str(e)}')
            return None
