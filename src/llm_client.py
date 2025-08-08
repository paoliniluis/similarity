"""
Centralized LiteLLM client for the project.
Provides unified access to different models with consistent configuration.
"""

import litellm
import time
import logging
from typing import List, Dict, Optional, Any
from . import settings
from src.keyword_service import KeywordService
from src.prompts import get_llm_analysis_prompts
from .db import SessionLocal

# Configure logging
logger = logging.getLogger(__name__)

class LLMClient:
    """Centralized LiteLLM client for the project."""
    
    def __init__(self):
        """Initialize the LLM client with configuration from settings."""
        self.api_base = settings.LITELLM_API_BASE
        self.api_key = settings.LITELLM_API_KEY
        self.rpm_limit = settings.LITELLM_RPM
        self.delay = 60.0 / self.rpm_limit if self.rpm_limit > 0 else 0
        self.keyword_service = KeywordService()
        
        # Model configurations
        self.models = {
            "openai-fast": settings.LITELLM_FAST_MODEL,  # Fast model for quick responses
            "openai-slow": settings.LITELLM_SLOW_MODEL,  # Slower but more capable model
            "gemini": settings.LITELLM_MODEL_NAME,  # Default Gemini model
        }
        
        logger.info(f"LLM Client initialized with {len(self.models)} model configurations")
        logger.info(f"API Base: {self.api_base}")
        logger.info(f"Rate Limit: {self.rpm_limit} requests/minute")
    
    def call_llm(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "openai-fast",
        max_retries: int = 3,
        response_format: Optional[Dict] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        inject_keywords: bool = True
    ) -> Optional[str]:
        """
        Call LLM with retry logic and consistent error handling.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use ('openai-fast', 'openai-slow', or 'gemini')
            max_retries: Maximum number of retry attempts
            response_format: Optional response format specification
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            inject_keywords: Whether to inject keyword definitions into prompts
            
        Returns:
            LLM response content or None if failed
        """
        # Map model name to actual model
        actual_model = self.models.get(model, model)
        
        # Inject keyword definitions if enabled
        if inject_keywords:
            messages = self._inject_keywords_into_messages(messages)
        
        for attempt in range(1, max_retries + 1):
            try:
                kwargs = {
                    "model": f"litellm_proxy/{actual_model}",
                    "messages": messages,
                    "api_base": self.api_base,
                    "api_key": self.api_key,
                    "temperature": temperature,
                }
                
                if response_format:
                    kwargs["response_format"] = response_format
                
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                
                logger.debug(f"Calling LLM (attempt {attempt}/{max_retries}) with model {actual_model}")
                
                response = litellm.completion(**kwargs)
                
                # Rate limiting
                if self.delay > 0:
                    time.sleep(self.delay)
                
                # Handle different response types from litellm
                content = None
                try:
                    # Try standard OpenAI-style response structure
                    content = getattr(getattr(getattr(response, 'choices', [{}])[0], 'message', {}), 'content', None)
                except (AttributeError, IndexError, TypeError):
                    pass
                
                # Fallback to string conversion if no content extracted
                if not content:
                    content = str(response)
                
                if content:
                    logger.debug(f"LLM response received (length: {len(content)})")
                    return content
                else:
                    logger.warning("LLM returned empty content")
                    return None
                
            except Exception as e:
                logger.error(f"Error calling LLM (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    logger.error(f"Failed to call LLM after {max_retries} attempts")
                    return None
                time.sleep(1)  # Brief delay before retry
        
        return None
    
    def call_llm_with_usage(
        self, 
        messages: List[Dict[str, str]], 
        model: str = "openai-fast",
        max_retries: int = 3,
        response_format: Optional[Dict] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        inject_keywords: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Call LLM with retry logic and return both content and usage information.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model to use ('openai-fast', 'openai-slow', or 'gemini')
            max_retries: Maximum number of retry attempts
            response_format: Optional response format specification
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            inject_keywords: Whether to inject keyword definitions into prompts
            
        Returns:
            Dictionary with 'content', 'tokens_sent', and 'tokens_received' or None if failed
        """
        # Map model name to actual model
        actual_model = self.models.get(model, model)
        
        # Inject keyword definitions if enabled
        if inject_keywords:
            messages = self._inject_keywords_into_messages(messages)
        
        for attempt in range(1, max_retries + 1):
            try:
                kwargs = {
                    "model": f"litellm_proxy/{actual_model}",
                    "messages": messages,
                    "api_base": self.api_base,
                    "api_key": self.api_key,
                    "temperature": temperature,
                }
                
                if response_format:
                    kwargs["response_format"] = response_format
                
                if max_tokens:
                    kwargs["max_tokens"] = max_tokens
                
                logger.debug(f"Calling LLM (attempt {attempt}/{max_retries}) with model {actual_model}")
                
                response = litellm.completion(**kwargs)
                
                # Log relevant properties for troubleshooting litellm call
                logger.info(f"🔍 LiteLLM Response Type: {type(response)}")
                logger.info(f"🔍 LiteLLM Response ID: {getattr(response, 'id', 'No ID')}")
                logger.info(f"🔍 LiteLLM Model Used: {getattr(response, 'model', 'No model info')}")
                logger.info(f"🔍 LiteLLM Usage Info: {getattr(response, 'usage', 'No usage info')}")
                logger.info(f"🔍 LiteLLM Choices Count: {len(getattr(response, 'choices', []))}")
                
                # Rate limiting
                if self.delay > 0:
                    time.sleep(self.delay)
                
                # Extract content
                content = None
                try:
                    # Try standard OpenAI-style response structure
                    content = getattr(getattr(getattr(response, 'choices', [{}])[0], 'message', {}), 'content', None)
                except (AttributeError, IndexError, TypeError):
                    pass
                
                # Fallback to string conversion if no content extracted
                if not content:
                    content = str(response)
                
                # Extract usage information
                tokens_sent = 0
                tokens_received = 0
                cache_hit = False
                
                try:
                    usage = getattr(response, 'usage', None)
                    if usage:
                        tokens_sent = getattr(usage, 'prompt_tokens', 0)
                        tokens_received = getattr(usage, 'completion_tokens', 0)
                        
                        # Debug: Log the response structure to understand LiteLLM proxy format
                        logger.debug(f"LLM Response structure - has cache_hit: {hasattr(response, 'cache_hit')}")
                        if hasattr(response, 'choices') and len(response.choices) > 0:
                            choice = response.choices[0]
                            logger.debug(f"First choice has cache_hit: {hasattr(choice, 'cache_hit')}")
                        
                        # Extract cache hit information directly from LiteLLM proxy response
                        # LiteLLM proxy includes cache hit information in the response
                        # Check if the response has cache hit information
                        if hasattr(response, 'cache_hit'):
                            cache_hit = getattr(response, 'cache_hit', False)
                            logger.debug(f"Cache hit from response.cache_hit: {cache_hit}")
                        elif hasattr(response, 'choices') and len(response.choices) > 0:
                            # Check if cache hit info is in the first choice
                            choice = response.choices[0]
                            if hasattr(choice, 'cache_hit'):
                                cache_hit = getattr(choice, 'cache_hit', False)
                                logger.debug(f"Cache hit from choice.cache_hit: {cache_hit}")
                        
                        # If no explicit cache hit info, check for cached tokens in usage details
                        if not cache_hit and hasattr(usage, 'prompt_tokens_details'):
                            prompt_tokens_details = getattr(usage, 'prompt_tokens_details', None)
                            if prompt_tokens_details and hasattr(prompt_tokens_details, 'cached_tokens'):
                                cached_tokens = getattr(prompt_tokens_details, 'cached_tokens', 0)
                                cache_hit = cached_tokens > 0
                                logger.debug(f"Cache hit from cached_tokens: {cache_hit} (cached_tokens: {cached_tokens})")
                        
                        logger.debug(f"Token usage - Sent: {tokens_sent}, Received: {tokens_received}, Cache hit: {cache_hit}")
                except (AttributeError, TypeError):
                    logger.warning("Could not extract token usage from response")
                
                if content:
                    logger.debug(f"LLM response received (length: {len(content)})")
                    
                    # Capture relevant metadata from the response
                    response_metadata = {
                        'content': content,
                        'tokens_sent': tokens_sent,
                        'tokens_received': tokens_received,
                        'cache_hit': cache_hit,
                        'response_id': getattr(response, 'id', None),
                        'response_model': getattr(response, 'model', None),
                        'response_type': str(type(response))
                    }
                    
                    return response_metadata
                else:
                    logger.warning("LLM returned empty content")
                    return None
                
            except Exception as e:
                logger.error(f"Error calling LLM (attempt {attempt}/{max_retries}): {e}")
                if attempt == max_retries:
                    logger.error(f"Failed to call LLM after {max_retries} attempts")
                    return None
                time.sleep(1)  # Brief delay before retry
        
        return None
    
    def _inject_keywords_into_messages(self, messages: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """
        Inject relevant keyword definitions into user messages for context.
        
        Args:
            messages: Original messages list
            
        Returns:
            Messages with relevant keyword context injected
        """
        db = None
        try:
            db = SessionLocal()
            
            # Find user messages and inject relevant keywords
            enhanced_messages = []
            for message in messages:
                if message['role'] == 'user':
                    # Use the new relevant keyword injection method
                    enhanced_content = self.keyword_service.inject_relevant_keywords_into_prompt(
                        message['content'], db
                    )
                    enhanced_messages.append({
                        'role': message['role'],
                        'content': enhanced_content
                    })
                else:
                    enhanced_messages.append(message)
            
            return enhanced_messages
            
        except Exception as e:
            logger.error(f"Error injecting keywords into messages: {e}")
            return messages
        finally:
            if db:
                db.close()
    
    def call_fast_model(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """Call the fast model for quick responses."""
        return self.call_llm(messages, model="openai-fast", **kwargs)
    
    def call_slow_model(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """Call the slow model for more detailed responses."""
        return self.call_llm(messages, model="openai-slow", **kwargs)
    
    def call_gemini(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """Call the Gemini model."""
        return self.call_llm(messages, model="gemini", **kwargs)
    
    def summarize_text(self, text: str, model: str = "openai-fast") -> Optional[str]:
        """
        Generate a summary of the given text.
        
        Args:
            text: Text to summarize
            model: Model to use for summarization
            
        Returns:
            Summary text or None if failed
        """
        messages = [
            {
                "role": "user",
                "content": f"Please provide a concise summary of the following text:\n\n{text}"
            }
        ]
        
        return self.call_llm(messages, model=model, max_tokens=500)
    
    def analyze_text(self, text: str, analysis_type: str, model: str = "openai-fast") -> Optional[str]:
        """
        Analyze text for specific information.
        
        Args:
            text: Text to analyze
            analysis_type: Type of analysis ('questions', 'summary', 'classification', etc.)
            model: Model to use for analysis
            
        Returns:
            Analysis result or None if failed
        """
        prompts = get_llm_analysis_prompts()
        
        prompt = prompts.get(analysis_type, f"Analyze the following text for {analysis_type}:\n\n")
        
        messages = [
            {
                "role": "user",
                "content": f"{prompt}{text}"
            }
        ]
        
        return self.call_llm(messages, model=model)

# Global instance for easy access
llm_client = LLMClient() 