import warnings
from .embedding_service import EmbeddingService, LocalEmbeddingProvider
from . import settings
from .utils import get_device

class SemanticAnalyzer:
    """
    Handles the creation of sentence embeddings.
    
    DEPRECATED: Use EmbeddingService directly instead.
    This class is kept for backward compatibility and will be removed in a future version.
    """

    def __init__(self, device=None):
        """Initializes the SemanticAnalyzer and loads the model."""
        warnings.warn(
            "SemanticAnalyzer is deprecated. Use EmbeddingService directly instead. "
            "This class will be removed in a future version.",
            DeprecationWarning,
            stacklevel=2
        )
        
        # Auto-detect device if not specified
        if device is None:
            device = get_device()
        
        # Use the new embedding service internally
        self._embedding_service = EmbeddingService(
            LocalEmbeddingProvider(settings.EMBEDDING_MODEL, device)
        )

    def create_embedding(self, text: str):
        """
        Creates a semantic embedding for a given text.
        
        Args:
            text (str): The text to create an embedding for
            
        Returns:
            List[float] or None: The embedding vector
        """
        warnings.warn(
            "SemanticAnalyzer.create_embedding is deprecated. Use EmbeddingService.create_embedding instead.",
            DeprecationWarning,
            stacklevel=2
        )
        return self._embedding_service.create_embedding(text) 