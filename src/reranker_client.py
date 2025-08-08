import torch
import gc
import logging
from typing import List, Dict, Any, Optional, Tuple
from sentence_transformers import CrossEncoder
from . import settings
from .utils import get_device

logger = logging.getLogger(__name__)

class RerankerClient:
    """Client for the cross-encoder/ms-marco-MiniLM-L6-v2 model to rerank search results."""
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2", device: Optional[str] = None):
        """
        Initialize the reranker client.
        
        Args:
            model_name: The Hugging Face model name
            device: Device to run the model on ('cpu' or 'cuda'). If None, auto-detects best available device.
        """
        self.model_name = model_name
        # Auto-detect device if not specified
        if device is None:
            device = get_device()
        self.device = device
        self.model = None
        
        # Check device availability and warn about performance
        self._check_device_availability()
        
        self._load_model()
    
    def _check_device_availability(self):
        """Check device availability and warn about performance implications."""
        if self.device == "cuda":
            if not torch.cuda.is_available():
                logger.warning("⚠️ CUDA requested but not available. Falling back to CPU (will be slower)")
                self.device = "cpu"
            else:
                logger.info(f"✅ Using CUDA with {torch.cuda.get_device_name()}")
                logger.info(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        elif self.device == "cpu":
            logger.info("ℹ️ Using CPU for reranker (should be reasonably fast with this model)")
        else:
            logger.warning(f"⚠️ Unknown device '{self.device}', falling back to CPU")
            self.device = "cpu"
    
    def _load_model(self):
        """Load the reranker model."""
        try:
            logger.info(f"Loading reranker model: {self.model_name}")
            
            # Load the CrossEncoder model
            self.model = CrossEncoder(
                self.model_name,
                device=self.device
            )
            
            logger.info("Reranker model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise
    
    def _optimize_memory(self):
        """Optimize memory usage."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        import gc
        gc.collect()
    
    def rerank_results(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank search results using the cross-encoder/ms-marco-MiniLM-L6-v2 model.
        
        Args:
            query: The search query
            candidates: List of candidate documents with their content
            
        Returns:
            Reranked list of candidates with similarity scores
        """
        if not candidates:
            return []
        
        try:
            # Prepare pairs for the cross-encoder
            pairs = []
            for candidate in candidates:
                # Extract document content based on candidate type
                doc_content = self._extract_document_content(candidate)
                if doc_content:
                    # CrossEncoder expects (query, document) pairs
                    pairs.append((query, doc_content))
                else:
                    # If no content, use a placeholder
                    pairs.append((query, "No content available"))
            
            if not pairs:
                logger.warning("No valid pairs to rerank")
                return candidates
            
            # Get scores from the cross-encoder
            if self.model is not None:
                scores = self.model.predict(pairs, show_progress_bar=False)
            else:
                logger.error("Model is not loaded, returning original candidates")
                return candidates

            # Combine candidates with scores
            scored_candidates = []
            for i, candidate in enumerate(candidates):
                scored_candidate = candidate.copy()
                scored_candidate['reranker_score'] = float(scores[i])
                scored_candidates.append(scored_candidate)
            
            # Sort by reranker score (descending)
            scored_candidates.sort(key=lambda x: x['reranker_score'], reverse=True)
            
            return scored_candidates
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            # Return original candidates if reranking fails
            return candidates
    
    def _extract_document_content(self, candidate: Dict[str, Any]) -> Optional[str]:
        """
        Extract document content from a candidate based on its type.
        
        Args:
            candidate: The candidate document
            
        Returns:
            Extracted content string or None
        """
        # Handle different types of candidates
        if 'title' in candidate and 'body' in candidate:
            # GitHub issue
            return f"Title: {candidate['title']}\nBody: {candidate.get('body', '')}"
        elif 'title' in candidate and 'conversation' in candidate:
            # Discourse post
            return f"Title: {candidate['title']}\nConversation: {candidate['conversation']}"
        elif 'markdown' in candidate:
            # Metabase doc
            return candidate['markdown']
        elif 'question' in candidate and 'answer' in candidate:
            # Q&A
            return f"Question: {candidate['question']}\nAnswer: {candidate['answer']}"
        elif 'keyword' in candidate and 'definition' in candidate:
            # Keyword definition
            return f"Keyword: {candidate['keyword']}\nDefinition: {candidate['definition']}"
        else:
            # Fallback: try to find any text content
            for key in ['title', 'body', 'content', 'text', 'description']:
                if key in candidate and candidate[key]:
                    return str(candidate[key])
        
        return None
    
    def rerank_similar_issues(
        self, 
        query: str, 
        similar_issues: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Rerank similar issues specifically.
        
        Args:
            query: The search query
            similar_issues: List of similar issues from the database
            
        Returns:
            Reranked list of issues with reranker scores
        """
        # Prepare candidates for reranking
        candidates = []
        for issue in similar_issues:
            candidate = {
                'number': issue.get('number'),
                'title': issue.get('title'),
                'body': issue.get('body', ''),
                'state': issue.get('state'),
                'url': issue.get('url'),
                'similarity_score': issue.get('similarity_score', 0.0)
            }
            candidates.append(candidate)
        
        # Rerank using the general rerank method
        reranked_candidates = self.rerank_results(query, candidates)
        
        # Convert back to the expected format
        reranked_issues = []
        for candidate in reranked_candidates:
            issue = {
                'number': candidate['number'],
                'title': candidate['title'],
                'state': candidate['state'],
                'url': candidate['url'],
                'similarity_score': candidate['similarity_score'],
                'reranker_score': candidate['reranker_score']
            }
            reranked_issues.append(issue)
        
        return reranked_issues 