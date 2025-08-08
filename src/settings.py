from decouple import config
import torch
from src.constants import DEFAULT_RERANKER_MODEL

# GitHub
GITHUB_TOKEN = config("GITHUB_TOKEN", default=None)
GITHUB_REPO_OWNER = config("GITHUB_REPO_OWNER", default="metabase")
GITHUB_REPO_NAME = config("GITHUB_REPO_NAME", default="metabase")
GITHUB_WORKER_TOKEN = config("GITHUB_WORKER_TOKEN", default=GITHUB_TOKEN)

# Discourse
DISCOURSE_BASE_URL = config("DISCOURSE_BASE_URL", default="https://discourse.metabase.com")
DISCOURSE_API_USERNAME = config("DISCOURSE_API_USERNAME", default=None)
DISCOURSE_API_KEY = config("DISCOURSE_API_KEY", default=None)
DISCOURSE_MAX_PAGES = config("DISCOURSE_MAX_PAGES", default=1000, cast=int)  # Safety limit

# Database
DATABASE_URL = str(config("DATABASE_URL", default="postgresql+psycopg://localhost/metabase_duplicates"))

# Ensure the URL uses the correct driver for psycopg (v3)
if DATABASE_URL.startswith("postgresql://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+psycopg://", 1)

# LLM
LITELLM_API_BASE = config("LITELLM_API_BASE", default="http://localhost:4000")
LITELLM_API_KEY = config("LITELLM_API_KEY", default="your-litellm-proxy-api-key")
LITELLM_MODEL_NAME = config("LITELLM_MODEL_NAME", default="gemini-2.5-flash-lite-preview-06-17")
LITELLM_RPM = config("LITELLM_RPM", default=15, cast=int)

# LLM Model Configurations
LITELLM_FAST_MODEL = config("LITELLM_FAST_MODEL", default="openai-fast")
LITELLM_SLOW_MODEL = config("LITELLM_SLOW_MODEL", default="openai-slow")

# API Security
API_KEY = config("API_KEY", default="a_super_secret_key_for_your_api")

# OpenAI Batch Processing Configuration
OPENAI_API_BASE = config("OPENAI_API_BASE", default="https://api.openai.com")
OPENAI_API_KEY = config("OPENAI_API_KEY", default="your_openai_api_key_here")
OPENAI_BATCH_MODEL = config("OPENAI_BATCH_MODEL", default="gpt-4.1-nano")
OPENAI_BATCH_ENTITIES_PER_BATCH = config("OPENAI_BATCH_ENTITIES_PER_BATCH", default=100, cast=int)

# Embedding configuration
# Provider can be 'local' or 'api'
EMBEDDING_PROVIDER = config("EMBEDDING_PROVIDER", default="local")
EMBEDDING_MODEL = config("EMBEDDING_MODEL", default="sentence-transformers/all-mpnet-base-v2")
EMBEDDING_DEVICE = config("EMBEDDING_DEVICE", default="cuda" if torch.cuda.is_available() else "cpu")

# External embedding API (used when EMBEDDING_PROVIDER=api)
EMBEDDING_API_BASE = config("EMBEDDING_API_BASE", default="http://localhost:8000")
EMBEDDING_API_KEY = config("EMBEDDING_API_KEY", default=API_KEY)
EMBEDDING_API_EMBEDDING_PATH = config("EMBEDDING_API_EMBEDDING_PATH", default="/embedding")

# Application URLs  
GITHUB_BASE_URL = config("GITHUB_BASE_URL", default="https://github.com/metabase/metabase")
# Note: DISCOURSE_BASE_URL already defined above

# Embedding Configuration
EMBEDDING_DIM = 768  # Dimension of the 'all-mpnet-base-v2' model

# Reranker Configuration  
RERANKER_ENABLED = config("RERANKER_ENABLED", default=True, cast=bool)
RERANKER_PROVIDER = config("RERANKER_PROVIDER", default="local")  # 'local' or 'api'
RERANKER_MODEL = str(config("RERANKER_MODEL", default=DEFAULT_RERANKER_MODEL))
# Auto-detect best available device for reranker
RERANKER_DEVICE = config("RERANKER_DEVICE", default="cuda" if torch.cuda.is_available() else "cpu")
RERANKER_MAX_CANDIDATES = config("RERANKER_MAX_CANDIDATES", default=20, cast=int)
RERANKER_BATCH_SIZE = config("RERANKER_BATCH_SIZE", default=8, cast=int)  # Batch size for processing

# External reranker API (used when RERANKER_PROVIDER=api)
RERANKER_API_BASE = config("RERANKER_API_BASE", default="http://localhost:8000")
RERANKER_API_KEY = config("RERANKER_API_KEY", default=API_KEY)
RERANKER_API_RERANK_PATH = config("RERANKER_API_RERANK_PATH", default="/rerank")
RERANKER_API_TIMEOUT = config("RERANKER_API_TIMEOUT", default=30, cast=int)

# HTTP and worker settings
HTTPX_TIMEOUT = config("HTTPX_TIMEOUT", default=30, cast=int)
WORKER_POLL_INTERVAL_SECONDS = config("WORKER_POLL_INTERVAL_SECONDS", default=5, cast=int)
WORKER_BACKOFF_SECONDS = config("WORKER_BACKOFF_SECONDS", default=60, cast=int)
WORKER_MAX_BACKOFF_SECONDS = config("WORKER_MAX_BACKOFF_SECONDS", default=600, cast=int)