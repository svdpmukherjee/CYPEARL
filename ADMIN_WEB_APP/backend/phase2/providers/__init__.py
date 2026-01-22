"""
CYPEARL Phase 2 - LLM Providers Package

This package provides unified access to LLM models via OpenRouter:
- Claude models (Anthropic via OpenRouter)
- GPT models (OpenAI via OpenRouter)
- Mistral models (via OpenRouter)
- Nova models (Amazon via OpenRouter)
- Local Ollama (Self-hosted fallback)

Usage:
    from providers import get_router, LLMRequest

    router = await get_router()
    await router.initialize_provider('openrouter', api_key='...')

    request = LLMRequest(
        system_prompt="You are...",
        user_prompt="What action...",
        temperature=0.3
    )

    # All models now go through OpenRouter
    response = await router.complete('gpt-4o', request)  # Uses OpenRouter
    response = await router.complete('claude-3-5-sonnet', request)  # Uses OpenRouter
    response = await router.complete('mistral-large', request)  # Uses OpenRouter
    response = await router.complete('nova-pro', request)  # Uses OpenRouter
"""

from .base import (
    BaseProvider,
    LLMRequest,
    LLMResponse,
    ProviderError,
    RateLimitError,
    AuthenticationError,
    ModelNotFoundError
)

from .router import (
    ProviderRouter,
    get_router,
    reset_router,
    PROVIDER_CLASSES
)

from .aws_bedrock import AWSBedrockProvider
from .openai_anthropic import OpenAIProvider, AnthropicProvider
from .together_openrouter import TogetherAIProvider, OpenRouterProvider, LocalOllamaProvider
from .openrouter import OpenRouterUnifiedProvider, OPENROUTER_MODEL_MAP

__all__ = [
    # Base
    'BaseProvider',
    'LLMRequest',
    'LLMResponse',
    'ProviderError',
    'RateLimitError',
    'AuthenticationError',
    'ModelNotFoundError',

    # Router
    'ProviderRouter',
    'get_router',
    'reset_router',
    'PROVIDER_CLASSES',

    # Providers
    'OpenRouterUnifiedProvider',  # Primary provider for all models
    'OPENROUTER_MODEL_MAP',
    'AWSBedrockProvider',
    'OpenAIProvider',
    'AnthropicProvider',
    'TogetherAIProvider',
    'OpenRouterProvider',
    'LocalOllamaProvider',
]