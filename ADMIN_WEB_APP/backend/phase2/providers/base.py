"""
CYPEARL Phase 2 - LLM Provider Base
Abstract base class for all LLM provider implementations.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any, List
from datetime import datetime
import asyncio
import time


@dataclass
class LLMRequest:
    """Standardized request to any LLM provider."""
    system_prompt: str
    user_prompt: str
    temperature: float = 0.3
    max_tokens: int = 500
    top_p: float = 1.0  # Nucleus sampling parameter
    stop_sequences: Optional[List[str]] = None
    
    # Metadata
    model_id: str = ""
    request_id: str = ""


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    # Content
    content: str
    
    # Token usage
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    
    # Cost
    cost_usd: float = 0.0
    
    # Performance
    latency_ms: int = 0
    
    # Metadata
    model_id: str = ""
    provider: str = ""
    request_id: str = ""
    timestamp: datetime = None
    
    # Status
    success: bool = True
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


class BaseProvider(ABC):
    """
    Abstract base class for LLM providers.
    
    All provider implementations must inherit from this class
    and implement the abstract methods.
    """
    
    provider_name: str = "base"
    
    def __init__(self, api_key: Optional[str] = None, **kwargs):
        self.api_key = api_key
        self.config = kwargs
        self._rate_limit_remaining = 1000
        self._rate_limit_reset = time.time()
        self._last_request_time = 0
        self._initialized = False
    
    @abstractmethod
    async def complete(self, request: LLMRequest, model_config: Dict[str, Any]) -> LLMResponse:
        """
        Send a completion request to the LLM.
        
        Args:
            request: Standardized LLM request
            model_config: Model-specific configuration from MODEL_REGISTRY
            
        Returns:
            Standardized LLM response
        """
        pass
    
    @abstractmethod
    async def check_health(self, model_id: str) -> Dict[str, Any]:
        """
        Check if the provider and specific model are healthy.
        
        Args:
            model_id: The model to check
            
        Returns:
            Dict with 'healthy' (bool), 'latency_ms' (int), 'error' (str or None)
        """
        pass
    
    def calculate_cost(self, input_tokens: int, output_tokens: int, 
                       model_config: Dict[str, Any]) -> float:
        """
        Calculate the cost of a request.
        
        Args:
            input_tokens: Number of input tokens
            output_tokens: Number of output tokens
            model_config: Model configuration with pricing
            
        Returns:
            Cost in USD
        """
        input_cost = (input_tokens / 1000) * model_config.get('cost_per_1k_input', 0)
        output_cost = (output_tokens / 1000) * model_config.get('cost_per_1k_output', 0)
        return input_cost + output_cost
    
    async def initialize(self) -> bool:
        """
        Initialize the provider (called once before first use).
        Override in subclasses if needed.
        
        Returns:
            True if initialization successful
        """
        self._initialized = True
        return True
    
    async def close(self):
        """
        Clean up provider resources.
        Override in subclasses if needed.
        """
        pass
    
    def is_configured(self) -> bool:
        """Check if the provider has required configuration."""
        return self.api_key is not None or self.provider_name == "local"
    
    async def _apply_rate_limit(self, requests_per_minute: int):
        """Apply rate limiting between requests."""
        min_interval = 60.0 / requests_per_minute
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            await asyncio.sleep(min_interval - elapsed)
        self._last_request_time = time.time()


class ProviderError(Exception):
    """Exception raised by providers."""
    
    def __init__(self, message: str, provider: str, 
                 recoverable: bool = True, retry_after: Optional[int] = None):
        super().__init__(message)
        self.provider = provider
        self.recoverable = recoverable
        self.retry_after = retry_after


class RateLimitError(ProviderError):
    """Rate limit exceeded."""
    
    def __init__(self, provider: str, retry_after: int = 60):
        super().__init__(
            f"Rate limit exceeded for {provider}",
            provider=provider,
            recoverable=True,
            retry_after=retry_after
        )


class AuthenticationError(ProviderError):
    """Authentication failed."""
    
    def __init__(self, provider: str):
        super().__init__(
            f"Authentication failed for {provider}",
            provider=provider,
            recoverable=False
        )


class ModelNotFoundError(ProviderError):
    """Model not found or not available."""
    
    def __init__(self, provider: str, model_id: str):
        super().__init__(
            f"Model {model_id} not found on {provider}",
            provider=provider,
            recoverable=False
        )