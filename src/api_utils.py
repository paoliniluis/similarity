"""
Utility functions to reduce code duplication in API endpoints.
"""
from typing import List
from fastapi import HTTPException
import logging
from .embedding_service import get_embedding_service

logger = logging.getLogger(__name__)


def create_embedding_safe(text: str) -> List[float]:
    """
    Create an embedding with proper error handling.
    
    Args:
        text: The text to create an embedding for
        
    Returns:
        The embedding vector
        
    Raises:
        HTTPException: If embedding creation fails
    """
    embedding_service = get_embedding_service()
    embedding = embedding_service.create_embedding(text)
    
    if embedding is None:
        logger.error(f"Failed to create embedding for text: '{text[:50]}...'")
        raise HTTPException(status_code=500, detail="Failed to create embedding")
    
    logger.info(f"⚡ Embedding generated (dim: {len(embedding)})")
    return embedding


def format_github_issue_url(issue_number: int) -> str:
    """Format a GitHub issue URL for the metabase repository."""
    return f"https://github.com/metabase/metabase/issues/{issue_number}"


def format_discourse_url(slug: str, topic_id: int) -> str:
    """Format a Discourse post URL."""
    return f"https://discourse.metabase.com/t/{slug}/{topic_id}"


def truncate_text_for_logging(text: str, max_length: int = 50) -> str:
    """Truncate text for logging purposes."""
    if len(text) <= max_length:
        return text
    return f"{text[:max_length]}..."
