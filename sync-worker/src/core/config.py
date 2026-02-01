"""Configuration settings"""
import os
from dotenv import load_dotenv

load_dotenv()

# Model Configuration
MODEL_NAME = os.getenv('LLM_MODEL_ID', 'DeepSeek-R1-Distill-8B')
LLM_BACKEND = os.getenv('LLM_BACKEND', 'vllm').lower()
VLLM_API_URL = os.getenv('VLLM_API_URL', 'http://localhost:8000/v1')
OLLAMA_API_URL = os.getenv('OLLAMA_API_URL', 'http://localhost:11434')

# Database Configuration
CHROMA_HOST = os.getenv('CHROMA_HOST', 'localhost')
CHROMA_PORT = int(os.getenv('CHROMA_PORT', '8000'))
GRAPH_PERSIST_PATH = os.getenv('GRAPH_PERSIST_PATH', './data/graph.pkl')
SEARCH_TOP_K = int(os.getenv('SEARCH_TOP_K', '5'))

# Reranking Configuration
ENABLE_RERANKING = os.getenv('ENABLE_RERANKING', 'true').lower() == 'true'
RERANKER_MODEL = os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')
RERANKER_TOP_K = int(os.getenv('RERANKER_TOP_K', '50'))

# Token Management
MAX_MODEL_LEN = int(os.getenv('MAX_MODEL_LEN', '8192'))
RESERVED_OUTPUT_TOKENS = int(os.getenv('RESERVED_OUTPUT_TOKENS', '1024'))
MAX_CONTEXT_TOKENS = MAX_MODEL_LEN - RESERVED_OUTPUT_TOKENS

# Server Configuration
SERVER_HOST = os.getenv('SERVER_HOST', '0.0.0.0')
SERVER_PORT = int(os.getenv('SERVER_PORT', '8010'))
LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')

# ThreadPool Configuration
THREAD_POOL_SIZE = int(os.getenv('THREAD_POOL_SIZE', '4'))
