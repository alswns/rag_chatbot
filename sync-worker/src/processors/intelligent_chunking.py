"""
Intelligent Chunking with Metadata Enrichment

메타데이터 추출, 키워드 추출, 중첩 청킹을 통해 검색 최적화된 청크를 생성합니다.

특징:
1. 메타데이터 자동 추출 (Temporal, Entities, Document Type)
2. TF-IDF 기반 키워드 추출
3. 중첩 청킹 (800-1000자, 200자 overlap)
4. Metadata Header 자동 삽입
"""

import logging
import re
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
from collections import Counter
import math

logger = logging.getLogger(__name__)


@dataclass
class EnrichedChunk:
    """메타데이터를 포함한 청크"""
    id: str
    content: str
    metadata: Dict[str, Any]
    keywords: List[str]
    formatted_content: str  # Metadata Header가 포함된 최종 콘텐츠


class MetadataExtractor:
    """메타데이터 추출기"""
    
    # 시간 관련 패턴
    TEMPORAL_PATTERNS = {
        'semester': r'(\d학년\s*\d학기|[1-4]학년\s*[1-2]학기)',
        'year': r'(202\d|201\d)',
        'date': r'(\d{1,2}월\s*\d{1,2}일|[01]\d[/\-][0-3]\d)',
        'period': r'(1학기|2학기|겨울|여름)',
    }
    
    # 엔티티 패턴 (대소문자 혼합, 숫자 포함)
    ENTITY_PATTERNS = {
        'university': r'(중앙대학교|서울대학교|연세대학교|고려대학교|KAIST|포항공대)',
        'library': r'(이수페타시스|ROS\s*2?|FastAPI|React|Vue|Django)',
        'tech_keyword': r'(API|REST|JSON|HTTP|GPU|CPU|IMU|센서)',
        'company': r'(삼성|SK|LG|현대|카카오|네이버)',
        'code': r'([A-Z0-9]{6,}|[0-9]{6,})',  # ID, 코드
    }
    
    # 문서 타입 키워드
    DOCUMENT_TYPE_PATTERNS = {
        'lecture_note': r'(강의노트|수업 자료|강의 내용|강의안)',
        'report': r'(과제|보고서|리포트|결과보고)',
        'grade_sheet': r'(성적표|성적|학점|GPA)',
        'project': r'(프로젝트|프로젝트 계획|README)',
        'plan': r'(계획|일정|로드맵|방안)',
        'guide': r'(가이드|매뉴얼|설명서|instructions)',
        'summary': r'(요약|정리|개요|총정리)',
    }
    
    @staticmethod
    def extract_temporal_info(text: str) -> Dict[str, List[str]]:
        """시간 정보 추출"""
        temporal_info = {}
        
        for key, pattern in MetadataExtractor.TEMPORAL_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                temporal_info[key] = list(set(matches))  # 중복 제거
        
        return temporal_info
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """엔티티 추출"""
        entities = {}
        
        for entity_type, pattern in MetadataExtractor.ENTITY_PATTERNS.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                entities[entity_type] = list(set(matches))  # 중복 제거
        
        return entities
    
    @staticmethod
    def detect_document_type(text: str, title: str = "") -> str:
        """문서 타입 감지"""
        combined_text = (title + " " + text).lower()
        
        scores = {}
        for doc_type, pattern in MetadataExtractor.DOCUMENT_TYPE_PATTERNS.items():
            matches = len(re.findall(pattern, combined_text, re.IGNORECASE))
            if matches > 0:
                scores[doc_type] = matches
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return "general"
    
    @staticmethod
    def extract_subject(text: str, title: str = "") -> Optional[str]:
        """전공 과목명 추출"""
        # 과목명 패턴 (숫자 + 과목명)
        patterns = [
            r'(\w+\s+\d+)\s*[-:]?\s*([^(\n]+)',  # "COURSE 101: Introduction"
            r'([A-Z0-9]+)[-]?(\d+)',  # "CS101", "EE-201"
            r'([\w\s]+)\s*\([0-9]+학년',  # "데이터베이스(3학년"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, title or text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None


class KeywordExtractor:
    """TF-IDF 기반 키워드 추출"""
    
    # 불용어 (한국어, 영어)
    STOPWORDS = {
        '이', '그', '저', '것', '수', '등', '들', '및', '또는', '그리고', '만약', '안', '없',
        '있', '있다', '되', '된', '나', '우리', '저희', '따라', '때문', '함께', '같', '같다',
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
        'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'this', 'that',
        'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who',
        'when', 'where', 'why', 'how'
    }
    
    @staticmethod
    def extract_keywords(
        text: str,
        top_k: int = 8,
        min_length: int = 3
    ) -> List[str]:
        """
        TF-IDF 기반 키워드 추출
        
        Args:
            text: 텍스트
            top_k: 추출할 키워드 개수
            min_length: 최소 단어 길이
        
        Returns:
            키워드 리스트 (점수순)
        """
        # 텍스트 전처리
        text_lower = text.lower()
        
        # 단어 분리 (공백, 구두점 기준)
        words = re.findall(r'\b\w+\b', text_lower)
        
        # 불용어 제거
        filtered_words = [
            w for w in words
            if w not in KeywordExtractor.STOPWORDS and len(w) >= min_length
        ]
        
        if not filtered_words:
            return []
        
        # 단어 빈도 계산 (TF)
        word_freq = Counter(filtered_words)
        
        # IDF 계산 (간단 버전: 고유 단어 기준)
        unique_words = set(filtered_words)
        total_words = len(unique_words)
        
        # TF-IDF 점수
        scores = {}
        for word, freq in word_freq.items():
            tf = freq / len(filtered_words)
            # 간단한 IDF (실제로는 문서 빈도 사용)
            idf = math.log(total_words / (1 + word_freq[word]))
            tfidf = tf * idf
            scores[word] = tfidf
        
        # 상위 k개 추출
        top_keywords = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [kw[0] for kw in top_keywords]
    
    @staticmethod
    def extract_named_entities(
        text: str,
        top_k: int = 5
    ) -> List[str]:
        """
        명명된 엔티티 추출 (대문자, 숫자 조합)
        
        Args:
            text: 텍스트
            top_k: 추출할 엔티티 개수
        
        Returns:
            엔티티 리스트
        """
        # 대문자 시작, 숫자 포함 단어 추출
        patterns = [
            r'\b[A-Z][a-zA-Z0-9]*\b',  # CamelCase, 대문자 시작
            r'\b[A-Z0-9]{2,}\b',        # 모두 대문자 또는 숫자
            r'\b[0-9]{6,}\b',            # 6자리 이상 숫자 (ID, 코드)
        ]
        
        entities = set()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entities.update(matches)
        
        # 빈도 기반 정렬
        entity_freq = Counter()
        for pattern in patterns:
            matches = re.findall(pattern, text)
            entity_freq.update(matches)
        
        top_entities = sorted(
            entity_freq.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]
        
        return [ent[0] for ent in top_entities]


class IntelligentChunkingEngine:
    """지능형 청킹 엔진"""
    
    def __init__(
        self,
        chunk_size: int = 900,           # 800-1000자
        chunk_overlap: int = 200,
        min_chunk_size: int = 300
    ):
        """
        Args:
            chunk_size: 청크 최대 크기
            chunk_overlap: 청크 간 오버랩
            min_chunk_size: 최소 청크 크기
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        
        logger.info(f'IntelligentChunkingEngine 초기화 (size={chunk_size}, overlap={chunk_overlap})')
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any] = None
    ) -> List[Dict[str, Any]]:
        """
        텍스트를 청크로 분할 (오버랩 포함)
        
        Args:
            text: 분할할 텍스트
            metadata: 청크 메타데이터
        
        Returns:
            청크 리스트
        """
        if not text or len(text) < self.min_chunk_size:
            return []
        
        chunks = []
        
        # 문단 단위로 먼저 분할
        paragraphs = text.split('\n\n')
        
        current_chunk = ""
        chunk_count = 0
        
        for para in paragraphs:
            # 너무 긴 문단은 문장 단위로 재분할
            if len(para) > self.chunk_size:
                sentences = re.split(r'(?<=[.!?])\s+', para)
                para = '\n'.join(sentences)
            
            # 현재 청크에 문단 추가
            potential_chunk = current_chunk + '\n\n' + para if current_chunk else para
            
            if len(potential_chunk) <= self.chunk_size:
                current_chunk = potential_chunk
            else:
                # 현재 청크가 일정 크기 이상이면 저장
                if len(current_chunk) >= self.min_chunk_size:
                    chunk_data = {
                        'chunk_id': f"{metadata.get('document_id', 'doc')}_{chunk_count}",
                        'text': current_chunk.strip(),
                        'length': len(current_chunk),
                        'offset': len('\n\n'.join(chunks))
                    }
                    chunks.append(chunk_data)
                    chunk_count += 1
                
                # 오버랩 처리: 이전 청크의 마지막 부분 + 새 문단
                overlap_text = current_chunk[-(self.chunk_overlap):] if len(current_chunk) > self.chunk_overlap else current_chunk[-100:]
                current_chunk = overlap_text + '\n\n' + para if overlap_text else para
        
        # 마지막 청크
        if len(current_chunk) >= self.min_chunk_size:
            chunk_data = {
                'chunk_id': f"{metadata.get('document_id', 'doc')}_{chunk_count}",
                'text': current_chunk.strip(),
                'length': len(current_chunk),
                'offset': len('\n\n'.join(chunks))
            }
            chunks.append(chunk_data)
        
        logger.info(f'✅ 청킹 완료: {len(chunks)}개 청크')
        
        return chunks
    
    def create_enriched_chunks(
        self,
        text: str,
        title: str,
        document_id: str,
        breadcrumb_path: str = "",
        metadata: Dict[str, Any] = None
    ) -> List[EnrichedChunk]:
        """
        전체 메타데이터 추출 + 청킹
        
        Args:
            text: 문서 텍스트
            title: 문서 제목
            document_id: 문서 ID
            breadcrumb_path: 경로 (예: "학사팀 > 근로장학생 > 업무")
            metadata: 추가 메타데이터
        
        Returns:
            EnrichedChunk 리스트
        """
        if metadata is None:
            metadata = {}
        
        logger.info(f'🔍 지능형 청킹 시작: "{title}" (ID: {document_id})')
        
        # =====================================================
        # 1. 메타데이터 추출
        # =====================================================
        temporal_info = MetadataExtractor.extract_temporal_info(text)
        entities = MetadataExtractor.extract_entities(text)
        doc_type = MetadataExtractor.detect_document_type(text, title)
        subject = MetadataExtractor.extract_subject(text, title)
        
        # 전역 키워드 추출
        global_keywords = KeywordExtractor.extract_keywords(text, top_k=8)
        named_entities = KeywordExtractor.extract_named_entities(text, top_k=5)
        all_keywords = list(set(global_keywords + named_entities))[:8]
        
        base_metadata = {
            'document_id': document_id,
            'title': title,
            'breadcrumb_path': breadcrumb_path,
            'document_type': doc_type,
            'temporal': temporal_info,
            'entities': entities,
            'subject': subject,
            'keywords': all_keywords,
            **metadata
        }
        
        logger.info(f'   📊 메타데이터:')
        logger.info(f'      - Type: {doc_type}')
        logger.info(f'      - Subject: {subject}')
        logger.info(f'      - Keywords: {", ".join(all_keywords)}')
        logger.info(f'      - Entities: {", ".join(entities.get("library", [])[:3])}')
        
        # =====================================================
        # 2. 청킹
        # =====================================================
        chunk_data_list = self.chunk_text(text, base_metadata)
        
        # =====================================================
        # 3. 각 청크에 대한 로컬 메타데이터 + Metadata Header 생성
        # =====================================================
        enriched_chunks = []
        
        for i, chunk_data in enumerate(chunk_data_list):
            chunk_text = chunk_data['text']
            
            # 청크별 키워드 추출 (로컬)
            local_keywords = KeywordExtractor.extract_keywords(chunk_text, top_k=5)
            
            # 청크 요약 생성 (이전과 현재의 연결고리)
            if i > 0 and chunk_data_list[i-1]:
                summary = self._create_summary(chunk_data_list[i-1]['text'], chunk_text)
            else:
                summary = chunk_text[:100].replace('\n', ' ').strip() + '...'
            
            # Metadata Header 생성
            metadata_header = self._format_metadata_header(
                breadcrumb_path=breadcrumb_path,
                temporal=temporal_info,
                entities=entities,
                keywords=local_keywords,
                summary=summary
            )
            
            # Metadata Header + Content 결합
            formatted_content = f"{metadata_header}\n\n{chunk_text}"
            
            enriched_chunk = EnrichedChunk(
                id=chunk_data['chunk_id'],
                content=chunk_text,  # 원본 콘텐츠
                metadata={
                    **base_metadata,
                    'chunk_index': i,
                    'chunk_count': len(chunk_data_list),
                    'local_keywords': local_keywords,
                    'summary': summary
                },
                keywords=local_keywords,
                formatted_content=formatted_content
            )
            
            enriched_chunks.append(enriched_chunk)
        
        logger.info(f'✅ 지능형 청킹 완료: {len(enriched_chunks)}개 EnrichedChunk')
        
        return enriched_chunks
    
    @staticmethod
    def _create_summary(prev_text: str, current_text: str) -> str:
        """이전 청크와 현재 청크를 잇는 요약 생성"""
        prev_end = prev_text[-150:].replace('\n', ' ').strip()
        curr_start = current_text[:150].replace('\n', ' ').strip()
        
        return f"{prev_end}... → ...{curr_start}"
    
    @staticmethod
    def _format_metadata_header(
        breadcrumb_path: str = "",
        temporal: Dict[str, Any] = None,
        entities: Dict[str, Any] = None,
        keywords: List[str] = None,
        summary: str = ""
    ) -> str:
        """메타데이터 헤더 포맷팅"""
        lines = ["---[METADATA HEADER]---"]
        
        if breadcrumb_path:
            lines.append(f"**Path:** {breadcrumb_path}")
        
        if temporal:
            temporal_str = ", ".join(
                f"{k}: {v}" for k, v in temporal.items()
                if v
            )
            if temporal_str:
                lines.append(f"**Temporal:** {temporal_str}")
        
        if entities:
            entity_str = ", ".join(
                v for vals in entities.values()
                for v in vals
            )[:100]  # 길이 제한
            if entity_str:
                lines.append(f"**Entities:** {entity_str}")
        
        if keywords:
            lines.append(f"**Keywords:** {', '.join(keywords)}")
        
        if summary:
            lines.append(f"**Context:** {summary[:200]}")
        
        lines.append("---[END METADATA]---")
        
        return "\n".join(lines)


# 편의 함수
def create_intelligent_chunks(
    text: str,
    title: str,
    document_id: str,
    breadcrumb_path: str = "",
    chunk_size: int = 900,
    chunk_overlap: int = 200
) -> List[EnrichedChunk]:
    """
    간단한 인터페이스로 지능형 청킹 수행
    
    Args:
        text: 문서 텍스트
        title: 문서 제목
        document_id: 문서 ID
        breadcrumb_path: 경로
        chunk_size: 청크 크기
        chunk_overlap: 오버랩
    
    Returns:
        EnrichedChunk 리스트
    """
    engine = IntelligentChunkingEngine(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    
    return engine.create_enriched_chunks(
        text=text,
        title=title,
        document_id=document_id,
        breadcrumb_path=breadcrumb_path
    )
