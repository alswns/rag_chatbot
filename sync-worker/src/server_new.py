"""
🚀 RAG Inference API Server (Refactored & Production-Ready)

간소화된 메인 서버 파일:
- FastAPI 앱 생성
- 라우터 등록
- 이벤트 핸들러 설정
"""

import logging
import sys
import os

# 모듈 경로
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
import uvicorn

from core.config import SERVER_HOST, SERVER_PORT, LOG_LEVEL, MODEL_NAME, LLM_BACKEND, ENABLE_RERANKING
from core.startup import initialize_app, cleanup_app
from routers import chat, health, debug

# 로깅 설정
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FastAPI 앱 생성
app = FastAPI(
    title='RAG Inference API',
    description='Enterprise RAG - OpenAI Compatible (Production-Ready)',
    version='2.0.0'
)

# 라우터 등록
app.include_router(chat.router, tags=['Chat'])
app.include_router(health.router, tags=['Health'])
app.include_router(debug.router, tags=['Debug'])

# 이벤트 핸들러
@app.on_event('startup')
async def startup():
    """서버 시작"""
    logger.info(f'✅ 모델: {MODEL_NAME}')
    logger.info(f'✅ LLM Backend: {LLM_BACKEND.upper()}')
    logger.info(f'✅ Reranking: {"활성화" if ENABLE_RERANKING else "비활성화"}')
    
    await initialize_app()


@app.on_event('shutdown')
async def shutdown():
    """서버 종료"""
    await cleanup_app()


# 서버 실행
if __name__ == '__main__':
    logger.info(f'🚀 서버 시작: {SERVER_HOST}:{SERVER_PORT}')
    
    uvicorn.run(
        app,
        host=SERVER_HOST,
        port=SERVER_PORT,
        log_level=LOG_LEVEL.lower()
    )
