"""
Deprecated shim for EmbeddingClient. Use src.embedding_service.get_embedding_service instead.
This module keeps a minimal compatibility layer for any legacy imports.
"""
from typing import List, Optional
from .embedding_service import get_embedding_service

class EmbeddingClient:  # pragma: no cover
    def __init__(self, *_, **__):
        self._svc = get_embedding_service()

    def create_embedding(self, text: str) -> Optional[List[float]]:
        return self._svc.create_embedding(text)

    def create_embeddings_batch(self, texts: List[str]):
        return [self._svc.create_embedding(t) for t in texts]