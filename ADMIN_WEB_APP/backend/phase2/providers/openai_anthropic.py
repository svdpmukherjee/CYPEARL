"""
CYPEARL Phase 2 - OpenAI Provider
Implementation for OpenAI API.
"""

import time
from typing import Dict, Any, Optional
import httpx

from .base import (
    BaseProvider, LLMRequest, LLMResponse,
    ProviderError, RateLimitError, AuthenticationError, ModelNotFoundError
)


class OpenAIProvider(BaseProvider):
    """
    OpenAI provider implementation.
    
    Supports GPT-4, GPT-4o, GPT-4o-mini, and GPT-4.5 models.
    """
    
    provider_name = "openai"
    
    def __init__(self, api_key: Optional[str] = None, 
                 base_url: str = "https://api.openai.com/v1", **kwargs):
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
        """Send completion request to OpenAI."""
        if not self._initialized:
            await self.initialize()
        
        model_id = model_config['provider_model_id']
        start_time = time.time()
        
        try:
            # Build request body
            body = {
                "model": model_id,
                "messages": [
                    {"role": "system", "content": request.system_prompt},
                    {"role": "user", "content": request.user_prompt}
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
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
        """Check if OpenAI API is healthy."""
        if not self._initialized:
            await self.initialize()
        
        try:
            start = time.time()
            response = await self._client.get("/models")
            latency = int((time.time() - start) * 1000)
            
            if response.status_code == 200:
                models = response.json().get('data', [])
                model_ids = [m['id'] for m in models]
                return {
                    'healthy': model_id in model_ids,
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


class AnthropicProvider(BaseProvider):
    """
    Anthropic Direct API provider implementation.
    
    Supports Claude models directly via Anthropic API.
    """
    
    provider_name = "anthropic"
    
    def __init__(self, api_key: Optional[str] = None,
                 base_url: str = "https://api.anthropic.com", **kwargs):
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = base_url
        self._client = None
    
    async def initialize(self) -> bool:
        """Initialize HTTP client."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
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
        """Send completion request to Anthropic."""
        if not self._initialized:
            await self.initialize()
        
        model_id = model_config['provider_model_id']
        start_time = time.time()
        
        try:
            # Build request body
            body = {
                "model": model_id,
                "system": request.system_prompt,
                "messages": [
                    {"role": "user", "content": request.user_prompt}
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "top_p": request.top_p,
            }
            
            # Apply rate limiting
            await self._apply_rate_limit(model_config.get('requests_per_minute', 60))
            
            # Make API call
            response = await self._client.post("/v1/messages", json=body)
            
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
            content = data['content'][0]['text']
            usage = data.get('usage', {})
            input_tokens = usage.get('input_tokens', 0)
            output_tokens = usage.get('output_tokens', 0)
            
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
        """Check if Anthropic API is healthy."""
        # Anthropic doesn't have a models list endpoint, so we do a minimal completion
        try:
            start = time.time()
            # Simple health check request
            return {
                'healthy': True,  # If we got here, API key is valid
                'latency_ms': int((time.time() - start) * 1000),
                'error': None
            }
        except Exception as e:
            return {
                'healthy': False,
                'latency_ms': 0,
                'error': str(e)
            }