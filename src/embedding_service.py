"""
Unified embedding service to eliminate code duplication between SemanticAnalyzer and EmbeddingClient.
Provides a consistent interface for creating text embeddings using either local or API providers.
"""

import logging
import json
from typing import Protocol, List, Optional, Union
from abc import ABC, abstractmethod

import torch
from sentence_transformers import SentenceTransformer
import requests

from src import settings
from .utils import get_device

logger = logging.getLogger(__name__)


class EmbeddingProvider(Protocol):
    """Protocol for embedding providers."""
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """Create an embedding for the given text."""
        ...
    
    def create_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Create embeddings for multiple texts."""
        ...


class LocalEmbeddingProvider:
    """Local embedding provider using SentenceTransformers."""
    
    def __init__(self, model_name: Optional[str] = None, device: Optional[str] = None):
        """Initialize the local embedding provider."""
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self.device = device or get_device()
        self.model = SentenceTransformer(self.model_name, device=self.device)
        logger.info(f"Initialized local embedding provider with model: {self.model_name}, device: {self.device}")
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """Create an embedding for the given text."""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return None
            
            embedding = self.model.encode(text)
            return embedding.tolist()
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory creating embedding: {e}")
            return None
        except (OSError, ImportError) as e:
            logger.error(f"Model loading error creating embedding: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating embedding: {e}")
            return None
    
    def create_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Create embeddings for multiple texts in batch."""
        try:
            # Filter out empty texts but remember their positions
            text_positions = []
            valid_texts = []
            
            for i, text in enumerate(texts):
                if text and text.strip():
                    text_positions.append(i)
                    valid_texts.append(text)
            
            if not valid_texts:
                return [None] * len(texts)
            
            # Create embeddings for valid texts
            embeddings = self.model.encode(valid_texts)
            
            # Map back to original positions
            result: List[Optional[List[float]]] = [None] * len(texts)
            for i, embedding in enumerate(embeddings):
                original_position = text_positions[i]
                result[original_position] = embedding.tolist()
            
            return result
        except torch.cuda.OutOfMemoryError as e:
            logger.error(f"GPU out of memory creating batch embeddings: {e}")
            return [None] * len(texts)
        except (OSError, ImportError) as e:
            logger.error(f"Model loading error creating batch embeddings: {e}")
            return [None] * len(texts)
        except Exception as e:
            logger.error(f"Unexpected error creating batch embeddings: {e}")
            return [None] * len(texts)


class APIEmbeddingProvider:
    """API-based embedding provider using a configurable embedding endpoint."""
    
    def __init__(
        self,
        api_base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        embedding_path: str = "/embedding",
    ):
        """Initialize the API embedding provider."""
        import requests
        
        self.api_base_url = api_base_url.rstrip('/')
        self.embedding_path = embedding_path if embedding_path.startswith('/') else f"/{embedding_path}"
        self.api_key = api_key or settings.API_KEY
        self.session = requests.Session()
        self.session.headers['X-API-Key'] = str(self.api_key or '')
        self.session.headers['Content-Type'] = 'application/json'
        logger.info(
            f"Initialized API embedding provider with base URL: {self.api_base_url}, path: {self.embedding_path}"
        )
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """Create an embedding using the API endpoint."""
        try:
            if not text or not text.strip():
                logger.warning("Empty text provided for embedding")
                return None
                
            url = f"{self.api_base_url}{self.embedding_path}"
            payload = {"text": text}
            
            logger.debug(f"Making embedding request for text: '{text[:50]}...'")
            response = self.session.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                embedding = result.get('embedding')
                if embedding:
                    logger.debug(f"Successfully created embedding (dim: {len(embedding)})")
                    return embedding
                else:
                    logger.error("API returned empty embedding")
                    return None
            else:
                logger.error(f"API request failed with status {response.status_code}: {response.text}")
                return None
                
        except requests.exceptions.Timeout as e:
            logger.error(f"Timeout creating embedding via API: {e}")
            return None
        except requests.exceptions.ConnectionError as e:
            logger.error(f"Connection error creating embedding via API: {e}")
            return None
        except requests.exceptions.HTTPError as e:
            logger.error(f"HTTP error creating embedding via API: {e}")
            return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error creating embedding via API: {e}")
            return None
        except (KeyError, ValueError, json.JSONDecodeError) as e:
            logger.error(f"Invalid response format from embedding API: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating embedding via API: {e}")
            return None
    
    def create_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Create embeddings for multiple texts (sequential API calls)."""
        embeddings = []
        for text in texts:
            embedding = self.create_embedding(text)
            embeddings.append(embedding)
        return embeddings
    
    async def create_embeddings_batch_async(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Create embeddings for multiple texts using parallel HTTP requests."""
        import asyncio
        import aiohttp
        
        async def create_single_embedding_async(session: aiohttp.ClientSession, text: str) -> Optional[List[float]]:
            """Create a single embedding using async HTTP request."""
            try:
                if not text or not text.strip():
                    logger.warning("Empty text provided for embedding")
                    return None
                    
                url = f"{self.api_base_url}{self.embedding_path}"
                payload = {"text": text}
                
                logger.debug(f"Making async embedding request for text: '{text[:50]}...'")
                async with session.post(url, json=payload, timeout=30) as response:
                    if response.status == 200:
                        result = await response.json()
                        embedding = result.get('embedding')
                        if embedding:
                            logger.debug(f"Successfully created async embedding (dim: {len(embedding)})")
                            return embedding
                        else:
                            logger.error("API returned empty embedding")
                            return None
                    else:
                        error_text = await response.text()
                        logger.error(f"API request failed with status {response.status}: {error_text}")
                        return None
                        
            except Exception as e:
                logger.error(f"Error creating embedding via async API: {e}")
                return None
        
        # Create async HTTP session with proper headers
        headers = {
            'X-API-Key': str(self.api_key or ''),
            'Content-Type': 'application/json'
        }
        
        try:
            async with aiohttp.ClientSession(headers=headers) as session:
                # Create tasks for parallel execution
                tasks = [create_single_embedding_async(session, text) for text in texts]
                
                # Execute all requests in parallel
                embeddings = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Handle exceptions and convert to expected format
                result = []
                for i, embedding in enumerate(embeddings):
                    if isinstance(embedding, Exception):
                        logger.error(f"Error creating embedding for text {i}: {embedding}")
                        result.append(None)
                    else:
                        result.append(embedding)
                
                return result
                
        except Exception as e:
            logger.error(f"Error in batch async embedding creation: {e}")
            # Fallback to sequential processing
            return self.create_embeddings_batch(texts)


class EmbeddingService:
    """Unified embedding service that can use different providers."""
    
    def __init__(self, provider: Optional[EmbeddingProvider] = None):
        """Initialize with a specific provider or create default local provider."""
        if provider is None:
            provider = LocalEmbeddingProvider()
        self.provider = provider
    
    def create_embedding(self, text: str) -> Optional[List[float]]:
        """Create an embedding for the given text."""
        return self.provider.create_embedding(text)
    
    def create_embeddings_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Create embeddings for multiple texts."""
        return self.provider.create_embeddings_batch(texts)
    
    @classmethod
    def create_local(cls, model_name: Optional[str] = None, device: Optional[str] = None) -> 'EmbeddingService':
        """Create service with local provider."""
        provider = LocalEmbeddingProvider(model_name, device)
        return cls(provider)
    
    @classmethod
    def create_api(
        cls,
        api_base_url: str = "http://localhost:8000",
        api_key: Optional[str] = None,
        embedding_path: str = "/embedding",
    ) -> 'EmbeddingService':
        """Create service with API provider."""
        provider = APIEmbeddingProvider(api_base_url, api_key, embedding_path)
        return cls(provider)


# Global instance for backward compatibility
_embedding_service = None

def get_embedding_service() -> EmbeddingService:
    """Get the global embedding service instance, configured via settings."""
    global _embedding_service
    if _embedding_service is None:
        provider = str(getattr(settings, 'EMBEDDING_PROVIDER', 'local')).lower()
        if provider == 'api':
            # Prevent accidental self-recursion when the API provider points back to this same API process
            api_base_url = str(getattr(settings, 'EMBEDDING_API_BASE', 'http://localhost:8000'))
            embedding_path = str(getattr(settings, 'EMBEDDING_API_EMBEDDING_PATH', '/embedding'))

            is_localhost = (
                'localhost:8000' in api_base_url
                or '127.0.0.1:8000' in api_base_url
                or '0.0.0.0:8000' in api_base_url
            )

            if is_localhost and embedding_path.rstrip('/') == '/embedding':
                logger.warning(
                    "EMBEDDING_PROVIDER=api is configured to call this same API (" 
                    f"{api_base_url}{embedding_path}). Falling back to local provider to avoid recursion."
                )
                _embedding_service = EmbeddingService.create_local(
                    model_name=getattr(settings, 'EMBEDDING_MODEL', None),
                    device=getattr(settings, 'EMBEDDING_DEVICE', None),
                )
            else:
                _embedding_service = EmbeddingService.create_api(
                    api_base_url=api_base_url,
                    api_key=getattr(settings, 'EMBEDDING_API_KEY', None),
                    embedding_path=embedding_path,
                )
        else:
            # Default to local
            _embedding_service = EmbeddingService.create_local(
                model_name=getattr(settings, 'EMBEDDING_MODEL', None),
                device=getattr(settings, 'EMBEDDING_DEVICE', None),
            )
    return _embedding_service

def set_embedding_service(service: EmbeddingService):
    """Set the global embedding service instance."""
    global _embedding_service
    _embedding_service = service
