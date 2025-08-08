"""
Constants to eliminate magic numbers and strings throughout the codebase.
"""

# Query limits and pagination
DEFAULT_SIMILARITY_LIMIT = 10
DEFAULT_CANDIDATE_LIMIT = 20
MAX_SIMILARITY_CANDIDATES = 50

# Rate limiting
DEFAULT_RATE_LIMIT_PER_MINUTE = 10

# Batch processing
DEFAULT_BATCH_SIZE = 300
DEFAULT_EMBEDDING_BATCH_SIZE = 8

# Database connection
MAX_RETRIES = 3

# Reranker settings
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
