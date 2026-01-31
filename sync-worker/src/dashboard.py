"""RAG 데이터 시각화 대시보드 - 개선 버전"""

import os
import json
import pickle
from datetime import datetime
from typing import Dict, Any, List

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import uvicorn

try:
    import chromadb
except ImportError:
    chromadb = None

import logging
logger = logging.getLogger(__name__)

CHROMA_HOST = os.getenv('CHROMA_HOST', 'localhost')
CHROMA_PORT = int(os.getenv('CHROMA_PORT', '8000'))
GRAPH_PATH = os.getenv('GRAPH_PERSIST_PATH', '/app/data/graph.pkl')
PROGRESS_FILE = os.getenv('PROGRESS_FILE', '/app/data/sync_progress.json')

app = FastAPI(title='RAG Dashboard', version='1.0')


def get_sync_progress() -> Dict[str, Any]:
    """동기화 진행률 조회"""
    try:
        if not os.path.exists(PROGRESS_FILE):
            return {
                'status': 'idle',
                'processed': 0,
                'total': 0,
                'percentage': 0,
                'message': '동기화 대기 중'
            }
        
        with open(PROGRESS_FILE, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {'status': 'error', 'message': f'진행률 조회 실패: {str(e)}'}


def get_chroma_stats() -> Dict[str, Any]:
    """ChromaDB 통계 조회"""
    if chromadb is None:
        return {'connected': False, 'error': 'chromadb 모듈 미설치', 'host': f'{CHROMA_HOST}:{CHROMA_PORT}'}
    
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        client.heartbeat()
        
        collections = client.list_collections()
        stats = {'connected': True, 'host': f'{CHROMA_HOST}:{CHROMA_PORT}', 'collections': []}
        
        for col in collections:
            try:
                collection = client.get_collection(col.name)
                stats['collections'].append({'name': col.name, 'count': collection.count()})
            except Exception:
                pass
        
        return stats
    except Exception as e:
        return {'connected': False, 'error': str(e), 'host': f'{CHROMA_HOST}:{CHROMA_PORT}'}


def get_graph_stats() -> Dict[str, Any]:
    """그래프 통계 조회"""
    try:
        if not os.path.exists(GRAPH_PATH):
            return {'loaded': False, 'message': '그래프 파일 없음'}
        
        with open(GRAPH_PATH, 'rb') as f:
            data = pickle.load(f)
        
        # ✅ 그래프 데이터 구조 처리 (dict 또는 NetworkX 그래프)
        if isinstance(data, dict):
            # 새로운 형식: {'graph': nx.DiGraph, 'nodes': {...}, ...}
            graph = data.get('graph')
            nodes_count = graph.number_of_nodes() if graph else len(data.get('nodes', {}))
            edges_count = graph.number_of_edges() if graph else 0
            
            return {
                'loaded': True,
                'nodes': nodes_count,
                'edges': edges_count,
                'embeddings': len(data.get('embeddings', {})),
                'file_size': round(os.path.getsize(GRAPH_PATH) / 1024 / 1024, 2)
            }
        else:
            # 기존 형식: nx.DiGraph 직접 저장
            return {
                'loaded': True,
                'nodes': data.number_of_nodes(),
                'edges': data.number_of_edges(),
                'file_size': round(os.path.getsize(GRAPH_PATH) / 1024 / 1024, 2)
            }
    except Exception as e:
        logger.error(f'그래프 로드 오류: {str(e)}')
        return {'loaded': False, 'error': f'오류: {str(e)}'}


def get_documents() -> Dict[str, Any]:
    """ChromaDB에서 문서 조회 (내용 포함)"""
    if chromadb is None:
        return {'documents': [], 'pages': {}, 'total': 0}
    
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        
        try:
            collection = client.get_collection('rag_documents')
        except Exception:
            return {'documents': [], 'pages': {}, 'total': 0}
        
        try:
            all_docs = collection.get(limit=500)  # 더 많은 문서 가져오기
        except Exception:
            return {'documents': [], 'pages': {}, 'total': 0}
        
        documents = []
        pages = {}  # 페이지별 그룹화
        
        for i, doc_id in enumerate(all_docs.get('ids', [])):
            metadata = all_docs.get('metadatas', [])[i] if i < len(all_docs.get('metadatas', [])) else {}
            doc_content = all_docs.get('documents', [])[i] if i < len(all_docs.get('documents', [])) else ''
            
            doc = {
                'id': doc_id,
                'title': metadata.get('title', '제목 없음'),
                'source': metadata.get('source', 'unknown'),
                'content': doc_content,
                'document_id': metadata.get('document_id', ''),
                'chunk_index': metadata.get('chunk_index', 0),
                'chunk_count': metadata.get('chunk_count', 1),
                'created_at': metadata.get('created_at', ''),
                'breadcrumb_path': metadata.get('breadcrumb_path', ''),
                'preview': doc_content[:150].replace('\n', ' ') + ('...' if len(doc_content) > 150 else '')
            }
            documents.append(doc)
            
            # 페이지별 그룹화
            page_id = metadata.get('document_id', doc_id.split('_chunk_')[0])
            if page_id not in pages:
                pages[page_id] = {
                    'title': metadata.get('title', '제목 없음'),
                    'source': metadata.get('source', 'unknown'),
                    'chunk_count': metadata.get('chunk_count', 1),
                    'breadcrumb_path': metadata.get('breadcrumb_path', ''),
                    'chunks': []
                }
            pages[page_id]['chunks'].append(doc)
        
        # 청크 정렬
        for page_id in pages:
            pages[page_id]['chunks'].sort(key=lambda x: x.get('chunk_index', 0))
        
        return {'documents': documents, 'pages': pages, 'total': collection.count(), 'page_count': len(pages)}
    except Exception:
        return {'documents': [], 'pages': {}, 'total': 0}


@app.get('/', response_class=HTMLResponse)
async def dashboard():
    """메인 대시보드"""
    chroma_stats = get_chroma_stats()
    graph_stats = get_graph_stats()
    documents = get_documents()
    progress = get_sync_progress()
    
    # 진행 상태 표시
    status_map = {
        'processing': ('🔄 진행 중', '#f0883e'),
        'completed': ('✅ 완료', '#3fb950'),
        'failed': ('❌ 실패', '#f85149'),
        'idle': ('⏸️ 대기', '#6e7681')
    }
    status_text, status_color = status_map.get(progress.get('status', 'idle'), ('❓ 알 수 없음', '#858585'))
    
    # ChromaDB 상태
    chroma_status = '✅ 연결됨' if chroma_stats.get('connected') else '❌ 연결 실패'
    
    # 그래프 상태
    graph_status = '✅ 로드됨' if graph_stats.get('loaded') else '❌ 없음'
    
    # 페이지 목록 HTML (페이지 단위로 그룹화)
    pages_html = ''
    pages_data = documents.get('pages', {})
    for page_id, page_info in pages_data.items():
        chunk_count = len(page_info.get('chunks', []))
        breadcrumb = page_info.get('breadcrumb_path', '')
        title = page_info.get('title', 'Untitled')
        
        # 청크 미리보기 (첫 번째 청크)
        first_chunk = page_info['chunks'][0] if page_info.get('chunks') else {}
        preview = first_chunk.get('preview', '')[:100]
        
        pages_html += f'''
        <div class="page-item" onclick="toggleChunks('{page_id}')">
            <div class="page-header">
                <strong>📄 {title}</strong>
                <span class="chunk-badge">{chunk_count}개 청크</span>
            </div>
            <p class="breadcrumb">{breadcrumb if breadcrumb else title}</p>
            <p class="preview">{preview}...</p>
            <div id="chunks-{page_id}" class="chunks-container" style="display:none;"></div>
        </div>
        '''
    
    if not pages_html:
        pages_html = '<p style="color:#6e7681; padding:20px;">저장된 문서가 없습니다.</p>'
    
    # 청크 데이터를 JavaScript로 전달
    import json
    pages_json = json.dumps(pages_data, ensure_ascii=False)
    
    # 컬렉션 목록
    collections_html = ''
    for col in chroma_stats.get('collections', []):
        collections_html += f'<p>📦 {col["name"]}: <strong>{col["count"]}</strong>개</p>'
    
    if not collections_html:
        collections_html = '<p style="color:#6e7681;">컬렉션이 없습니다.</p>'
    
    html = f'''
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>RAG 대시보드</title>
    <!-- vis.js CDN for graph visualization -->
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ margin:0; padding:0; box-sizing:border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background:#0d1117; color:#c9d1d9; padding:20px; }}
        .container {{ max-width:1600px; margin:0 auto; }}
        h1 {{ color:#58a6ff; margin-bottom:30px; font-size:2.5em; }}
        .grid {{ display:grid; grid-template-columns: repeat(2, 1fr); gap:20px; margin-bottom:30px; }}
        .grid-full {{ grid-column: 1 / -1; }}
        .card {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:20px; }}
        .card h2 {{ color:#8b949e; margin-bottom:15px; font-size:1.2em; }}
        .stat-row {{ display:flex; justify-content:space-between; padding:10px 0; border-bottom:1px solid #30363d; }}
        .stat-label {{ color:#8b949e; }}
        .stat-value {{ color:#58a6ff; font-weight:bold; }}
        .progress-bar {{ width:100%; height:24px; background:#30363d; border-radius:4px; overflow:hidden; margin:15px 0; }}
        .progress-fill {{ height:100%; background:linear-gradient(90deg, #238636, #2ea043); display:flex; align-items:center; justify-content:center; color:white; font-weight:bold; font-size:0.8em; }}
        .doc-item {{ padding:15px; border:1px solid #30363d; border-radius:6px; margin-bottom:12px; cursor:pointer; transition:background 0.2s; }}
        .doc-item:hover {{ background:#21262d; }}
        .doc-item strong {{ color:#58a6ff; }}
        .doc-item p {{ margin:5px 0 0 0; }}
        #modal {{ display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.7); z-index:1000; overflow-y:auto; padding:20px; }}
        .modal-content {{ background:#161b22; border:1px solid #30363d; border-radius:8px; margin:50px auto; padding:30px; max-width:900px; }}
        .modal-content h2 {{ color:#58a6ff; margin-bottom:15px; }}
        .modal-close {{ position:absolute; top:20px; right:30px; font-size:28px; color:#8b949e; cursor:pointer; }}
        .modal-meta {{ background:#0d1117; padding:15px; border-radius:6px; margin-bottom:20px; }}
        .modal-meta p {{ margin:5px 0; color:#8b949e; font-size:0.9em; }}
        .modal-text {{ white-space:pre-wrap; word-wrap:break-word; line-height:1.6; color:#c9d1d9; }}
        .status {{ padding:8px 12px; border-radius:4px; font-weight:bold; display:inline-block; }}
        .btn {{ padding:8px 16px; background:#238636; color:white; border:none; border-radius:6px; cursor:pointer; font-size:0.9em; }}
        .btn:hover {{ background:#2ea043; }}
        .legend {{ display:flex; gap:20px; font-size:0.85em; color:#8b949e; flex-wrap:wrap; margin-top:15px; padding:10px; background:#0d1117; border-radius:6px; }}
        .legend-item {{ display:flex; align-items:center; gap:6px; }}
        .legend-dot {{ width:12px; height:12px; border-radius:50%; }}
        #graph-container {{ width:100%; height:450px; border:1px solid #30363d; border-radius:8px; background:#0d1117; }}
        
        /* 페이지/청크 스타일 */
        .page-item {{ padding:15px; border:1px solid #30363d; border-radius:8px; margin-bottom:12px; cursor:pointer; transition:all 0.2s; background:#161b22; }}
        .page-item:hover {{ background:#21262d; border-color:#58a6ff; }}
        .page-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
        .page-header strong {{ color:#58a6ff; font-size:1.1em; }}
        .chunk-badge {{ background:#238636; color:white; padding:4px 10px; border-radius:12px; font-size:0.8em; }}
        .breadcrumb {{ font-size:0.85em; color:#f0883e; margin:5px 0; }}
        .preview {{ font-size:0.8em; color:#6e7681; margin:5px 0 0 0; }}
        .chunks-container {{ margin-top:15px; padding-top:15px; border-top:1px solid #30363d; }}
        .chunk-item {{ padding:12px; background:#0d1117; border:1px solid #30363d; border-radius:6px; margin-bottom:8px; cursor:pointer; transition:background 0.2s; }}
        .chunk-item:hover {{ background:#161b22; }}
        .chunk-header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; }}
        .chunk-index {{ background:#58a6ff; color:white; padding:2px 8px; border-radius:4px; font-size:0.75em; }}
        .chunk-content {{ font-size:0.85em; color:#c9d1d9; white-space:pre-wrap; word-wrap:break-word; max-height:150px; overflow-y:auto; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 RAG 대시보드</h1>
        
        <div class="grid">
            <!-- 진행 상황 -->
            <div class="card">
                <h2>🔄 동기화 상황</h2>
                <div class="stat-row">
                    <span class="stat-label">상태</span>
                    <span class="status" style="background-color:{status_color};">{status_text}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">처리됨</span>
                    <span class="stat-value">{progress.get('processed', 0)} / {progress.get('total', 0)}</span>
                </div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width:{progress.get('percentage', 0)}%;">{progress.get('percentage', 0)}%</div>
                </div>
                <p style="color:#6e7681; font-size:0.9em;">마지막 업데이트: {progress.get('timestamp', 'N/A')[:16]}</p>
            </div>
            
            
            
            <!-- 그래프 정보 -->
            <div class="card">
                <h2>🔗 그래프</h2>
                <div class="stat-row">
                    <span class="stat-label">상태</span>
                    <span class="stat-value">{graph_status}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">노드</span>
                    <span class="stat-value">{graph_stats.get('nodes', 0)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">엣지</span>
                    <span class="stat-value">{graph_stats.get('edges', 0)}</span>
                </div>
                <div class="stat-row">
                    <span class="stat-label">파일 크기</span>
                    <span class="stat-value">{graph_stats.get('file_size', 0):.2f}MB</span>
                </div>
                <button onclick="loadGraph()" class="btn" style="margin-top:15px;">🔄 그래프 시각화</button>
            </div>
            
            <!-- 그래프 시각화 -->
            <div class="card grid-full">
                <h2>🌐 그래프 시각화</h2>
                <div id="graph-container"></div>
                <div class="legend">
                    <div class="legend-item"><span class="legend-dot" style="background:#58a6ff;"></span> Notion 페이지</div>
                    <div class="legend-item"><span style="color:#3fb950;">━━</span> hierarchy (계층)</div>
                    <div class="legend-item"><span style="color:#f0883e;">- - -</span> references (참조)</div>
                </div>
            </div>
            
            <!-- 저장된 문서 (페이지 단위) -->
            <div class="card grid-full">
                <h2>📄 저장된 문서 ({documents.get('page_count', 0)}개 페이지, {documents.get('total', 0)}개 청크)</h2>
                <div style="max-height:600px; overflow-y:auto;">
                    {pages_html}
                </div>
            </div>
        </div>
    </div>
    
    <!-- 문서 상세 모달 -->
    <div id="modal">
        <span class="modal-close" onclick="closeModal()">&times;</span>
        <div class="modal-content">
            <h2 id="modalTitle"></h2>
            <div class="modal-meta">
                <p><strong>출처:</strong> <span id="modalSource"></span></p>
                <p><strong>경로:</strong> <span id="modalBreadcrumb"></span></p>
            </div>
            <div class="modal-text" id="modalContent"></div>
        </div>
    </div>
    
    <script>
        // 페이지 데이터 저장
        const pagesData = {pages_json};
        
        // 청크 토글 함수
        function toggleChunks(pageId) {{
            const container = document.getElementById('chunks-' + pageId);
            if (container.style.display === 'none') {{
                // 청크 표시
                const pageInfo = pagesData[pageId];
                if (pageInfo && pageInfo.chunks) {{
                    let chunksHtml = '';
                    pageInfo.chunks.forEach((chunk, idx) => {{
                        const content = chunk.content || '';
                        const preview = content.substring(0, 300).replace(/</g, '&lt;').replace(/>/g, '&gt;');
                        chunksHtml += `
                            <div class="chunk-item" onclick="event.stopPropagation(); showChunkDetail('${{chunk.id}}')">
                                <div class="chunk-header">
                                    <span class="chunk-index">청크 #${{idx + 1}}</span>
                                    <span style="color:#6e7681; font-size:0.8em;">${{content.length}}자</span>
                                </div>
                                <div class="chunk-content">${{preview}}...</div>
                            </div>
                        `;
                    }});
                    container.innerHTML = chunksHtml;
                }}
                container.style.display = 'block';
            }} else {{
                container.style.display = 'none';
            }}
        }}
        
        // 청크 상세 보기
        function showChunkDetail(chunkId) {{
            fetch('/api/document/' + chunkId)
                .then(r => r.json())
                .then(data => {{
                    if (data.error) {{
                        alert('오류: ' + data.error);
                        return;
                    }}
                    document.getElementById('modalTitle').textContent = data.title;
                    document.getElementById('modalSource').textContent = data.source;
                    document.getElementById('modalBreadcrumb').textContent = data.breadcrumb_path || data.title;
                    document.getElementById('modalContent').textContent = data.content;
                    document.getElementById('modal').style.display = 'block';
                }})
                .catch(e => alert('오류: ' + e));
        }}
        
        function closeModal() {{
            document.getElementById('modal').style.display = 'none';
        }}
        
        window.onclick = function(e) {{
            var modal = document.getElementById('modal');
            if (e.target == modal) {{
                modal.style.display = 'none';
            }}
        }}
        
        // 그래프 시각화 함수
        function loadGraph() {{
            fetch('/api/graph')
                .then(r => r.json())
                .then(data => {{
                    if (data.error) {{
                        alert('그래프 로드 오류: ' + data.error);
                        return;
                    }}
                    
                    // 노드 데이터 변환
                    var nodes = new vis.DataSet(
                        data.nodes.map(n => ({{
                            id: n.id,
                            label: n.title.substring(0, 20) + (n.title.length > 20 ? '...' : ''),
                            title: n.title,
                            color: {{
                                background: '#238636',
                                border: '#2ea043',
                                highlight: {{ background: '#3fb950', border: '#46d160' }}
                            }},
                            font: {{ color: '#c9d1d9', size: 12 }},
                            shape: 'box',
                            borderWidth: 2
                        }}))
                    );
                    
                    // 엣지 데이터 변환
                    var edges = new vis.DataSet(
                        data.edges.map((e, i) => ({{
                            id: i,
                            from: e.source,
                            to: e.target,
                            arrows: 'to',
                            color: {{
                                color: e.type === 'hierarchy' ? '#58a6ff' : '#f0883e',
                                highlight: '#79c0ff'
                            }},
                            title: e.type,
                            width: 2
                        }}))
                    );
                    
                    // 시각화 옵션
                    var options = {{
                        nodes: {{
                            shadow: true
                        }},
                        edges: {{
                            smooth: {{
                                type: 'cubicBezier',
                                forceDirection: 'horizontal'
                            }}
                        }},
                        physics: {{
                            enabled: true,
                            barnesHut: {{
                                gravitationalConstant: -3000,
                                centralGravity: 0.3,
                                springLength: 150,
                                springConstant: 0.04
                            }}
                        }},
                        interaction: {{
                            hover: true,
                            tooltipDelay: 100
                        }},
                        layout: {{
                            hierarchical: {{
                                enabled: false
                            }}
                        }}
                    }};
                    
                    // 그래프 렌더링
                    var container = document.getElementById('graph-container');
                    var network = new vis.Network(container, {{ nodes: nodes, edges: edges }}, options);
                    
                    // 노드 클릭 시 상세 정보 표시
                    network.on('click', function(params) {{
                        if (params.nodes.length > 0) {{
                            var nodeId = params.nodes[0];
                            var node = nodes.get(nodeId);
                            alert('📄 ' + node.title);
                        }}
                    }});
                }})
                .catch(e => alert('오류: ' + e));
        }}
        
        // 페이지 로드 시 자동으로 그래프 로드
        window.onload = function() {{
            loadGraph();
        }};
    </script>
</body>
</html>
    '''
    
    return HTMLResponse(content=html)


@app.get('/api/document/{doc_id}')
async def api_document(doc_id: str):
    """특정 문서 상세 조회"""
    if chromadb is None:
        return {'error': 'chromadb 모듈 미설치'}
    
    try:
        client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
        collection = client.get_collection('rag_documents')
        
        doc = collection.get(ids=[doc_id])
        if doc and doc.get('ids'):
            metadata = doc.get('metadatas', [{}])[0]
            content = doc.get('documents', [''])[0]
            return {
                'id': doc_id,
                'title': metadata.get('title', '제목 없음'),
                'source': metadata.get('source', 'unknown'),
                'content': content,
                'breadcrumb_path': metadata.get('breadcrumb_path', ''),
                'chunk_index': metadata.get('chunk_index', 0),
                'chunk_count': metadata.get('chunk_count', 1),
                'created_at': metadata.get('created_at', '')
            }
        
        return {'error': '문서를 찾을 수 없습니다'}
    except Exception as e:
        return {'error': f'조회 실패: {str(e)}'}


@app.get('/api/stats')
async def api_stats():
    """전체 통계"""
    return {
        'timestamp': datetime.now().isoformat(),
        'progress': get_sync_progress(),
        'chroma': get_chroma_stats(),
        'graph': get_graph_stats(),
        'documents': get_documents()
    }


@app.get('/api/graph')
async def api_graph():
    """그래프 데이터 조회 (시각화용)"""
    try:
        if not os.path.exists(GRAPH_PATH):
            return {'error': '그래프 파일 없음'}
        
        with open(GRAPH_PATH, 'rb') as f:
            data = pickle.load(f)
        
        nodes = []
        edges = []
        
        if isinstance(data, dict):
            graph = data.get('graph')
            nodes_data = data.get('nodes', {})
            
            # 노드 데이터 추출
            for node_id, node_info in nodes_data.items():
                if hasattr(node_info, 'title'):
                    nodes.append({
                        'id': node_id,
                        'title': node_info.title,
                        'source': getattr(node_info, 'source', 'notion')
                    })
                elif isinstance(node_info, dict):
                    nodes.append({
                        'id': node_id,
                        'title': node_info.get('title', 'Untitled'),
                        'source': node_info.get('source', 'notion')
                    })
            
            # 엣지 데이터 추출
            if graph:
                for source, target, edge_data in graph.edges(data=True):
                    edges.append({
                        'source': source,
                        'target': target,
                        'type': edge_data.get('edge_type', 'unknown')
                    })
        else:
            # NetworkX 그래프 직접 저장된 경우
            for node_id in data.nodes():
                node_data = data.nodes[node_id]
                nodes.append({
                    'id': node_id,
                    'title': node_data.get('title', 'Untitled'),
                    'source': node_data.get('source', 'notion')
                })
            
            for source, target, edge_data in data.edges(data=True):
                edges.append({
                    'source': source,
                    'target': target,
                    'type': edge_data.get('edge_type', 'unknown')
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'node_count': len(nodes),
            'edge_count': len(edges)
        }
    except Exception as e:
        logger.error(f'그래프 API 오류: {str(e)}')
        return {'error': str(e)}


@app.get('/health')
async def health():
    return {'status': 'ok'}


if __name__ == '__main__':
    port = int(os.getenv('DASHBOARD_PORT', '8080'))
    uvicorn.run(app, host='0.0.0.0', port=port, log_level='info')
