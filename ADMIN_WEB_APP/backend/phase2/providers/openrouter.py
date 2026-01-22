"""
CYPEARL Phase 2 - OpenRouter Provider (Unified)
Implementation for accessing Claude, GPT, Mistral, and Nova models via OpenRouter.
"""

import time
from typing import Dict, Any, Optional
import httpx

from .base import (
    BaseProvider, LLMRequest, LLMResponse,
    ProviderError, RateLimitError, AuthenticationError, ModelNotFoundError
)


# OpenRouter model ID mappings
OPENROUTER_MODEL_MAP: Dict[str, str] = {
    # Claude models (Anthropic via OpenRouter)
    "claude-3-5-sonnet": "anthropic/claude-3.5-sonnet",
    "claude-3-5-haiku": "anthropic/claude-3.5-haiku",
    "claude-3-opus": "anthropic/claude-3-opus",
    "claude-3-sonnet": "anthropic/claude-3-sonnet",
    "claude-3-haiku": "anthropic/claude-3-haiku",

    # GPT models (OpenAI via OpenRouter)
    "gpt-4o": "openai/gpt-4o",
    "gpt-4o-mini": "openai/gpt-4o-mini",
    "gpt-4-turbo": "openai/gpt-4-turbo",
    "gpt-4": "openai/gpt-4",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",

    # Mistral models via OpenRouter
    "mistral-large": "mistralai/mistral-large",
    "mistral-medium": "mistralai/mistral-medium",
    "mistral-small": "mistralai/mistral-small",
    "mistral-7b": "mistralai/mistral-7b-instruct",
    "mixtral-8x7b": "mistralai/mixtral-8x7b-instruct",
    "mixtral-8x22b": "mistralai/mixtral-8x22b-instruct",

    # Amazon Nova models via OpenRouter
    "nova-pro": "amazon/nova-pro-v1",
    "nova-lite": "amazon/nova-lite-v1",
    "nova-micro": "amazon/nova-micro-v1",

    # Llama 4 models (Meta via OpenRouter)
    "llama-4-maverick": "meta-llama/llama-4-maverick",
    "llama-4-scout": "meta-llama/llama-4-scout",

    # Llama 3.x models (Meta via OpenRouter)
    "llama-3.3-70b": "meta-llama/llama-3.3-70b-instruct",
    "llama-3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    "llama-3.1-70b": "meta-llama/llama-3.1-70b-instruct",
}


class OpenRouterUnifiedProvider(BaseProvider):
    """
    Unified OpenRouter provider for all models.

    Supports Claude, GPT, Mistral, and Nova models through OpenRouter's API.
    OpenRouter provides a single API endpoint to access multiple LLM providers.
    """

    provider_name = "openrouter"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://openrouter.ai/api/v1",
        site_url: str = "https://cypearl.research.edu",
        app_name: str = "CYPEARL Phase 2",
        **kwargs
    ):
        super().__init__(api_key=api_key, **kwargs)
        self.base_url = base_url
        self.site_url = site_url
        self.app_name = app_name
        self._client = None

    async def initialize(self) -> bool:
        """Initialize HTTP client with OpenRouter headers."""
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "HTTP-Referer": self.site_url,
                "X-Title": self.app_name,
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

    def _get_openrouter_model_id(self, model_config: Dict[str, Any]) -> str:
        """
        Get the OpenRouter model ID from model config.

        Args:
            model_config: Model configuration dict

        Returns:
            OpenRouter model ID string
        """
        # First check if openrouter_model_id is explicitly specified
        if 'openrouter_model_id' in model_config:
            return model_config['openrouter_model_id']

        # Then check the mapping
        model_id = model_config.get('model_id', '')
        if model_id in OPENROUTER_MODEL_MAP:
            return OPENROUTER_MODEL_MAP[model_id]

        # Fallback to provider_model_id if it looks like an OpenRouter ID
        provider_model_id = model_config.get('provider_model_id', '')
        if '/' in provider_model_id:
            return provider_model_id

        # Last resort: return the model_id as-is
        return model_id

    async def complete(self, request: LLMRequest, model_config: Dict[str, Any]) -> LLMResponse:
        """
        Send completion request to OpenRouter.

        OpenRouter uses an OpenAI-compatible API format.
        """
        if not self._initialized:
            await self.initialize()

        model_id = self._get_openrouter_model_id(model_config)
        internal_model_id = model_config.get('model_id', model_id)
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
                model_id=internal_model_id,
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
                model_id=internal_model_id,
                provider=self.provider_name,
                request_id=request.request_id,
                success=False,
                error_message=str(e)
            )

    async def check_health(self, model_id: str) -> Dict[str, Any]:
        """Check if OpenRouter API is healthy and model is available."""
        if not self._initialized:
            await self.initialize()

        try:
            start = time.time()
            response = await self._client.get("/models")
            latency = int((time.time() - start) * 1000)

            if response.status_code == 200:
                data = response.json()
                models = data.get('data', [])
                model_ids = [m.get('id', '') for m in models]

                # Check if our model ID (or mapped ID) is available
                openrouter_id = OPENROUTER_MODEL_MAP.get(model_id, model_id)
                is_available = openrouter_id in model_ids or any(openrouter_id in m for m in model_ids)

                return {
                    'healthy': is_available,
                    'latency_ms': latency,
                    'error': None if is_available else f"Model {openrouter_id} not found"
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

    async def list_models(self) -> Dict[str, Any]:
        """List all available models from OpenRouter."""
        if not self._initialized:
            await self.initialize()

        try:
            response = await self._client.get("/models")

            if response.status_code == 200:
                data = response.json()
                models = data.get('data', [])
                return {
                    'success': True,
                    'models': [
                        {
                            'id': m.get('id'),
                            'name': m.get('name'),
                            'context_length': m.get('context_length'),
                            'pricing': m.get('pricing', {})
                        }
                        for m in models
                    ]
                }
            else:
                return {
                    'success': False,
                    'error': f"Status {response.status_code}"
                }
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
