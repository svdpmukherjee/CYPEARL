"""
CYPEARL Phase 2 - Together AI and OpenRouter Providers
Implementations for open-source model hosting services.
"""

import time
from typing import Dict, Any, Optional
import httpx

from .base import (
    BaseProvider, LLMRequest, LLMResponse,
    ProviderError, RateLimitError, AuthenticationError, ModelNotFoundError
)


class TogetherAIProvider(BaseProvider):
    """
    Together AI provider implementation.
    
    Supports Llama, Mixtral, Qwen, DeepSeek, and other open-source models.
    """
    
    provider_name = "together_ai"
    
    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.together.xyz/v1", **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = base_url
        self._client = None
    
    async def initialize(self) -> bool:
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            },
            timeout=120.0
        )
        self._initialized = True
        return True
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
    
    async def complete(self, request: LLMRequest, model_config: Dict[str, Any]) -> LLMResponse:
        """Send completion request to Together AI."""
        if not self._initialized:
            await self.initialize()
        
        model_id = model_config['provider_model_id']
        start_time = time.time()
        
        try:
            # Build request body (OpenAI-compatible format)
            body = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt}
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            
            if request.stop_sequences:
                body["stop"] = request.stop_sequences
            
            # Apply rate limiting
            await self._apply_rate_limit(model_config.get('requests_per_minute', 60))
            
            # Make API call
            response = await self._client.post("/chat/completions", json=body)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            if response.status_code == 429:
                retry_after = int(response.headers.get('retry-after', 60))
                raise RateLimitError(self.provider_name, retry_after=retry_after)
            
            if response.status_code == 401:
                raise AuthenticationError(self.provider_name)
            
            if response.status_code == 404:
                raise ModelNotFoundError(self.provider_name, model_id)
            
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            content = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            
            cost = self.calculate_cost(input_tokens, output_tokens, model_config)
            
            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                model_id=model_id,
                provider=self.provider_name,
                request_id=request.request_id,
                success=True
            )
            
        except (RateLimitError, AuthenticationError, ModelNotFoundError):
            raise
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return LLMResponse(
                content="",
                latency_ms=latency_ms,
                model_id=model_id,
                provider=self.provider_name,
                request_id=request.request_id,
                success=False,
                error_message=str(e)
            )
    
    async def check_health(self, model_id: str) -> Dict[str, Any]:
        """Check if Together AI API is healthy."""
        if not self._initialized:
            await self.initialize()
        
        try:
            start = time.time()
            response = await self._client.get("/models")
            latency = int((time.time() - start) * 1000)
            
            if response.status_code == 200:
                models = response.json()
                # Together AI returns list directly or in 'data'
                if isinstance(models, list):
                    model_ids = [m.get('id', '') for m in models]
                else:
                    model_ids = [m.get('id', '') for m in models.get('data', [])]
                
                return {
                    'healthy': model_id in model_ids or any(model_id in m for m in model_ids),
                    'latency_ms': latency,
                    'error': None
                }
            else:
                return {
                    'healthy': False,
                    'latency_ms': latency,
                    'error': f"Status {response.status_code}"
                }
        except Exception as e:
            return {
                'healthy': False,
                'latency_ms': 0,
                'error': str(e)
            }


class OpenRouterProvider(BaseProvider):
    """
    OpenRouter provider implementation.
    
    Unified API for many models - useful as fallback.
    """
    
    provider_name = "openrouter"
    
    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://openrouter.ai/api/v1", **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = base_url
        self._client = None
    
    async def initialize(self) -> bool:
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": "https://cypearl.research.edu",  # Required by OpenRouter
                "X-Title": "CYPEARL Phase 2",
                "Content-Type": "application/json"
            },
            timeout=120.0
        )
        self._initialized = True
        return True
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
    
    async def complete(self, request: LLMRequest, model_config: Dict[str, Any]) -> LLMResponse:
        """Send completion request to OpenRouter."""
        if not self._initialized:
            await self.initialize()
        
        # OpenRouter uses different model IDs
        model_id = self._get_openrouter_model_id(model_config)
        start_time = time.time()
        
        try:
            # Build request body (OpenAI-compatible format)
            body = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt}
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            
            # Apply rate limiting
            await self._apply_rate_limit(model_config.get('requests_per_minute', 60))
            
            # Make API call
            response = await self._client.post("/chat/completions", json=body)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            if response.status_code == 429:
                raise RateLimitError(self.provider_name, retry_after=60)
            
            if response.status_code == 401:
                raise AuthenticationError(self.provider_name)
            
            if response.status_code == 404:
                raise ModelNotFoundError(self.provider_name, model_id)
            
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            content = data['choices'][0]['message']['content']
            usage = data.get('usage', {})
            input_tokens = usage.get('prompt_tokens', 0)
            output_tokens = usage.get('completion_tokens', 0)
            
            cost = self.calculate_cost(input_tokens, output_tokens, model_config)
            
            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                model_id=model_id,
                provider=self.provider_name,
                request_id=request.request_id,
                success=True
            )
            
        except (RateLimitError, AuthenticationError, ModelNotFoundError):
            raise
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return LLMResponse(
                content="",
                latency_ms=latency_ms,
                model_id=model_id,
                provider=self.provider_name,
                request_id=request.request_id,
                success=False,
                error_message=str(e)
            )
    
    def _get_openrouter_model_id(self, model_config: Dict[str, Any]) -> str:
        """Map internal model ID to OpenRouter model ID."""
        # OpenRouter uses format like "anthropic/claude-3.5-sonnet"
        mapping = {
            "claude-sonnet-4-5": "anthropic/claude-sonnet-4.5",
            "claude-haiku-4-5": "anthropic/claude-haiku-4.5",
            "gpt-4o": "openai/gpt-4o",
            "gpt-4o-mini": "openai/gpt-4o-mini",
            "llama-4-maverick": "meta-llama/llama-4-maverick",
        }
        internal_id = model_config.get('model_id', '')
        return mapping.get(internal_id, model_config.get('provider_model_id', internal_id))
    
    async def check_health(self, model_id: str) -> Dict[str, Any]:
        """Check if OpenRouter API is healthy."""
        if not self._initialized:
            await self.initialize()
        
        try:
            start = time.time()
            response = await self._client.get("/models")
            latency = int((time.time() - start) * 1000)
            
            if response.status_code == 200:
                return {
                    'healthy': True,
                    'latency_ms': latency,
                    'error': None
                }
            else:
                return {
                    'healthy': False,
                    'latency_ms': latency,
                    'error': f"Status {response.status_code}"
                }
        except Exception as e:
            return {
                'healthy': False,
                'latency_ms': 0,
                'error': str(e)
            }


class LocalOllamaProvider(BaseProvider):
    """
    Local Ollama provider implementation.
    
    Supports locally running models via Ollama.
    """
    
    provider_name = "local"
    
    def __init__(self, base_url: str = "http://localhost:11434", **kwargs):
        super().__init__(**kwargs)
        self.base_url = base_url
        self._client = None
    
    async def initialize(self) -> bool:
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=300.0  # Local models may be slower
        )
        self._initialized = True
        return True
    
    async def close(self):
        """Close HTTP client."""
        if self._client:
            await self._client.aclose()
    
    async def complete(self, request: LLMRequest, model_config: Dict[str, Any]) -> LLMResponse:
        """Send completion request to local Ollama."""
        if not self._initialized:
            await self.initialize()
        
        model_id = model_config.get('provider_model_id', model_config.get('model_id', 'llama3'))
        start_time = time.time()
        
        try:
            # Build request body (Ollama format)
            body = {
                "model": model_id,
                "system": request.system_prompt,
                "prompt": request.user_prompt,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_predict": request.max_tokens,
                }
            }
            
            # Make API call
            response = await self._client.post("/api/generate", json=body)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            if response.status_code == 404:
                raise ModelNotFoundError(self.provider_name, model_id)
            
            response.raise_for_status()
            data = response.json()
            
            # Parse response
            content = data.get('response', '')
            # Ollama provides token counts
            input_tokens = data.get('prompt_eval_count', 0)
            output_tokens = data.get('eval_count', 0)
            
            # Local models have no cost
            cost = 0.0
            
            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_ms=latency_ms,
                model_id=model_id,
                provider=self.provider_name,
                request_id=request.request_id,
                success=True
            )
            
        except ModelNotFoundError:
            raise
        except Exception as e:
            latency_ms = int((time.time() - start_time) * 1000)
            return LLMResponse(
                content="",
                latency_ms=latency_ms,
                model_id=model_id,
                provider=self.provider_name,
                request_id=request.request_id,
                success=False,
                error_message=str(e)
            )
    
    async def check_health(self, model_id: str) -> Dict[str, Any]:
        """Check if Ollama is running and model is available."""
        if not self._initialized:
            await self.initialize()
        
        try:
            start = time.time()
            response = await self._client.get("/api/tags")
            latency = int((time.time() - start) * 1000)
            
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m.get('name', '') for m in models]
                
                return {
                    'healthy': any(model_id in m for m in model_names),
                    'latency_ms': latency,
                    'error': None,
                    'available_models': model_names
                }
            else:
                return {
                    'healthy': False,
                    'latency_ms': latency,
                    'error': f"Status {response.status_code}"
                }
        except Exception as e:
            return {
                'healthy': False,
                'latency_ms': 0,
                'error': f"Ollama not running? {str(e)}"
            }
    
    def is_configured(self) -> bool:
        """Local provider doesn't need API key."""
        return True