"""
CYPEARL Phase 2 - Provider Router
Intelligent routing between LLM providers with fallback, rate limiting, and cost tracking.
"""

from typing import Dict, Any, Optional, List, Type
from datetime import datetime
import asyncio
import uuid

from .base import (
    BaseProvider, LLMRequest, LLMResponse,
    ProviderError, RateLimitError, AuthenticationError, ModelNotFoundError
)

from .aws_bedrock import AWSBedrockProvider
from .openai_anthropic import OpenAIProvider, AnthropicProvider
from .together_openrouter import TogetherAIProvider, OpenRouterProvider, LocalOllamaProvider
from .openrouter import OpenRouterUnifiedProvider
from core.config import MODEL_REGISTRY, PROVIDER_CONFIGS


# Provider class mapping
# Note: "openrouter" now uses the unified OpenRouterUnifiedProvider for all models
PROVIDER_CLASSES: Dict[str, Type[BaseProvider]] = {
    "aws_bedrock": AWSBedrockProvider,
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "together_ai": TogetherAIProvider,
    "openrouter": OpenRouterUnifiedProvider,  # Use unified provider for Claude, GPT, Mistral, Nova
    "openrouter_legacy": OpenRouterProvider,  # Keep legacy for backwards compatibility
    "local": LocalOllamaProvider,
}


class ProviderRouter:
    """
    Intelligent router for LLM requests.
    
    Features:
    - Automatic provider selection based on model
    - Fallback to alternative providers
    - Rate limit management
    - Cost tracking
    - Health monitoring
    """
    
    def __init__(self):
        self._providers: Dict[str, BaseProvider] = {}
        self._provider_configs: Dict[str, Dict] = {}
        self._model_configs: Dict[str, Dict] = dict(MODEL_REGISTRY)
        
        # Tracking
        self._total_cost: float = 0.0
        self._total_requests: int = 0
        self._total_tokens: int = 0
        self._errors: List[Dict] = []
        
        # Rate limit tracking per provider
        self._rate_limits: Dict[str, Dict] = {}
        
        # Health status
        self._health_cache: Dict[str, Dict] = {}
    
    async def initialize_provider(self, provider_type: str, **config) -> bool:
        """
        Initialize a provider with configuration.
        
        Args:
            provider_type: Provider type (aws_bedrock, openai, etc.)
            **config: Provider-specific configuration (api_key, region, etc.)
        
        Returns:
            True if initialization successful
        """
        if provider_type not in PROVIDER_CLASSES:
            raise ValueError(f"Unknown provider type: {provider_type}")
        
        provider_class = PROVIDER_CLASSES[provider_type]
        provider = provider_class(**config)
        
        success = await provider.initialize()
        
        if success:
            self._providers[provider_type] = provider
            self._provider_configs[provider_type] = config
            self._rate_limits[provider_type] = {
                'remaining': 1000,
                'reset_time': datetime.now()
            }
        
        return success
    
    async def complete(
        self, 
        model_id: str, 
        request: LLMRequest,
        fallback_models: Optional[List[str]] = None
    ) -> LLMResponse:
        """
        Route a completion request to the appropriate provider.
        
        Args:
            model_id: The model to use
            request: The LLM request
            fallback_models: Optional list of fallback models if primary fails
        
        Returns:
            LLM response
        """
        if not request.request_id:
            request.request_id = str(uuid.uuid4())
        
        request.model_id = model_id
        
        # Try primary model
        response = await self._try_model(model_id, request)
        
        # If failed and fallbacks available, try them
        if not response.success and fallback_models:
            for fallback_id in fallback_models:
                response = await self._try_model(fallback_id, request)
                if response.success:
                    break
        
        # Update tracking
        self._total_requests += 1
        if response.success:
            self._total_cost += response.cost_usd
            self._total_tokens += response.total_tokens
        else:
            self._errors.append({
                'model_id': model_id,
                'error': response.error_message,
                'timestamp': datetime.now().isoformat()
            })
        
        return response
    
    async def _try_model(self, model_id: str, request: LLMRequest) -> LLMResponse:
        """Try to complete request with a specific model."""
        # Get model config
        model_config = self._model_configs.get(model_id)
        if not model_config:
            return LLMResponse(
                content="",
                model_id=model_id,
                request_id=request.request_id,
                success=False,
                error_message=f"Model {model_id} not found in registry"
            )
        
        # Get provider
        provider_type = model_config['provider']
        provider = self._providers.get(provider_type)
        
        if not provider:
            return LLMResponse(
                content="",
                model_id=model_id,
                provider=provider_type,
                request_id=request.request_id,
                success=False,
                error_message=f"Provider {provider_type} not initialized"
            )
        
        try:
            return await provider.complete(request, model_config)
        except RateLimitError as e:
            return LLMResponse(
                content="",
                model_id=model_id,
                provider=provider_type,
                request_id=request.request_id,
                success=False,
                error_message=f"Rate limited, retry after {e.retry_after}s"
            )
        except (AuthenticationError, ModelNotFoundError) as e:
            return LLMResponse(
                content="",
                model_id=model_id,
                provider=provider_type,
                request_id=request.request_id,
                success=False,
                error_message=str(e)
            )
        except Exception as e:
            return LLMResponse(
                content="",
                model_id=model_id,
                provider=provider_type,
                request_id=request.request_id,
                success=False,
                error_message=str(e)
            )
    
    async def check_health(self, model_id: str, use_cache: bool = True) -> Dict[str, Any]:
        """
        Check health of a specific model.
        
        Args:
            model_id: Model to check
            use_cache: Whether to use cached health status (5 min TTL)
        
        Returns:
            Health status dict
        """
        # Check cache
        if use_cache and model_id in self._health_cache:
            cached = self._health_cache[model_id]
            age = (datetime.now() - cached['timestamp']).seconds
            if age < 300:  # 5 minute TTL
                return cached['status']
        
        model_config = self._model_configs.get(model_id)
        if not model_config:
            return {
                'healthy': False,
                'error': f"Model {model_id} not in registry"
            }
        
        provider_type = model_config['provider']
        provider = self._providers.get(provider_type)
        
        if not provider:
            return {
                'healthy': False,
                'error': f"Provider {provider_type} not initialized"
            }
        
        status = await provider.check_health(model_config['provider_model_id'])
        
        # Cache result
        self._health_cache[model_id] = {
            'status': status,
            'timestamp': datetime.now()
        }
        
        return status
    
    async def check_all_health(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all configured models."""
        results = {}
        
        for model_id in self._model_configs:
            results[model_id] = await self.check_health(model_id, use_cache=False)
        
        return results
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of all available models with their configurations."""
        models = []
        
        for model_id, config in self._model_configs.items():
            provider_type = config['provider']
            provider_ready = provider_type in self._providers
            
            models.append({
                'model_id': model_id,
                'display_name': config['display_name'],
                'provider': provider_type,
                'tier': config['tier'],
                'provider_ready': provider_ready,
                'cost_per_1k_input': config['cost_per_1k_input'],
                'cost_per_1k_output': config['cost_per_1k_output'],
            })
        
        return models
    
    def get_models_by_tier(self, tier: str) -> List[Dict[str, Any]]:
        """Get models filtered by tier."""
        return [m for m in self.get_available_models() if m['tier'] == tier]
    
    def get_initialized_providers(self) -> List[str]:
        """Get list of initialized providers."""
        return list(self._providers.keys())
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics."""
        return {
            'total_requests': self._total_requests,
            'total_cost_usd': round(self._total_cost, 4),
            'total_tokens': self._total_tokens,
            'error_count': len(self._errors),
            'recent_errors': self._errors[-10:] if self._errors else []
        }
    
    def estimate_cost(self, model_id: str, input_tokens: int, output_tokens: int) -> float:
        """Estimate cost for a request."""
        model_config = self._model_configs.get(model_id)
        if not model_config:
            return 0.0
        
        input_cost = (input_tokens / 1000) * model_config.get('cost_per_1k_input', 0)
        output_cost = (output_tokens / 1000) * model_config.get('cost_per_1k_output', 0)
        return input_cost + output_cost
    
    def estimate_experiment_cost(
        self,
        model_ids: List[str],
        n_trials: int,
        avg_input_tokens: int = 800,
        avg_output_tokens: int = 200
    ) -> Dict[str, Any]:
        """
        Estimate total cost for an experiment.
        
        Args:
            model_ids: Models to use
            n_trials: Number of trials per model
            avg_input_tokens: Average input tokens per request
            avg_output_tokens: Average output tokens per request
        
        Returns:
            Cost breakdown by model and total
        """
        breakdown = {}
        total = 0.0
        
        for model_id in model_ids:
            cost_per_request = self.estimate_cost(model_id, avg_input_tokens, avg_output_tokens)
            model_cost = cost_per_request * n_trials
            breakdown[model_id] = {
                'cost_per_request': round(cost_per_request, 6),
                'n_trials': n_trials,
                'total_cost': round(model_cost, 4)
            }
            total += model_cost
        
        return {
            'by_model': breakdown,
            'total_cost_usd': round(total, 2),
            'estimated_tokens': (avg_input_tokens + avg_output_tokens) * n_trials * len(model_ids)
        }
    
    async def close(self):
        """Close all provider connections."""
        for provider in self._providers.values():
            await provider.close()
        self._providers.clear()


# Global router instance
_router: Optional[ProviderRouter] = None


async def get_router() -> ProviderRouter:
    """Get or create the global provider router."""
    global _router
    if _router is None:
        _router = ProviderRouter()
    return _router


async def reset_router():
    """Reset the global router (useful for testing)."""
    global _router
    if _router:
        await _router.close()
    _router = None