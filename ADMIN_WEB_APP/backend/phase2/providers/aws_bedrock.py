"""
CYPEARL Phase 2 - AWS Bedrock Provider
Implementation for AWS Bedrock LLM service.
"""

import json
import time
from typing import Dict, Any, Optional
from datetime import datetime

from .base import (
    BaseProvider, LLMRequest, LLMResponse, 
    ProviderError, RateLimitError, AuthenticationError, ModelNotFoundError
)


class AWSBedrockProvider(BaseProvider):
    """
    AWS Bedrock provider implementation.
    
    Supports Claude, Llama, Mistral, Cohere, and Amazon Nova models.
    """
    
    provider_name = "aws_bedrock"
    
    def __init__(self, region: str = "us-east-1", **kwargs):
        super().__init__(**kwargs)
        self.region = region
        self._client = None
        self._runtime_client = None
    
    async def initialize(self) -> bool:
        """Initialize AWS clients."""
        try:
            import boto3
            self._client = boto3.client('bedrock', region_name=self.region)
            self._runtime_client = boto3.client('bedrock-runtime', region_name=self.region)
            self._initialized = True
            return True
        except ImportError:
            raise ProviderError(
                "boto3 not installed. Run: pip install boto3",
                provider=self.provider_name,
                recoverable=False
            )
        except Exception as e:
            raise AuthenticationError(self.provider_name)
    
    async def complete(self, request: LLMRequest, model_config: Dict[str, Any]) -> LLMResponse:
        """Send completion request to AWS Bedrock."""
        if not self._initialized:
            await self.initialize()
        
        model_id = model_config['provider_model_id']
        start_time = time.time()
        
        try:
            # Build request body based on model family
            if 'anthropic' in model_id:
                body = self._build_anthropic_body(request)
            elif 'meta.llama' in model_id:
                body = self._build_llama_body(request)
            elif 'mistral' in model_id:
                body = self._build_mistral_body(request)
            elif 'cohere' in model_id:
                body = self._build_cohere_body(request)
            elif 'amazon.nova' in model_id:
                body = self._build_nova_body(request)
            else:
                raise ModelNotFoundError(self.provider_name, model_id)
            
            # Apply rate limiting
            await self._apply_rate_limit(model_config.get('requests_per_minute', 60))
            
            # Make API call
            response = self._runtime_client.invoke_model(
                modelId=model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            # Parse response based on model family
            response_body = json.loads(response['body'].read())
            content, input_tokens, output_tokens = self._parse_response(
                response_body, model_id
            )
            
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
            
        except self._runtime_client.exceptions.ThrottlingException:
            raise RateLimitError(self.provider_name, retry_after=60)
        except self._runtime_client.exceptions.AccessDeniedException:
            raise AuthenticationError(self.provider_name)
        except self._runtime_client.exceptions.ModelNotReadyException:
            raise ModelNotFoundError(self.provider_name, model_id)
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
        """Check if model is available on Bedrock."""
        if not self._initialized:
            await self.initialize()
        
        try:
            start = time.time()
            # List foundation models to check availability
            response = self._client.list_foundation_models()
            latency = int((time.time() - start) * 1000)
            
            model_ids = [m['modelId'] for m in response.get('modelSummaries', [])]
            
            return {
                'healthy': model_id in model_ids or any(model_id in m for m in model_ids),
                'latency_ms': latency,
                'error': None
            }
        except Exception as e:
            return {
                'healthy': False,
                'latency_ms': 0,
                'error': str(e)
            }
    
    def _build_anthropic_body(self, request: LLMRequest) -> Dict:
        """Build request body for Anthropic Claude models."""
        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "system": request.system_prompt,
            "messages": [
                {"role": "user", "content": request.user_prompt}
            ]
        }
    
    def _build_llama_body(self, request: LLMRequest) -> Dict:
        """Build request body for Meta Llama models."""
        # Combine system and user prompts for Llama
        full_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{request.system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{request.user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
        return {
            "prompt": full_prompt,
            "max_gen_len": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
    
    def _build_mistral_body(self, request: LLMRequest) -> Dict:
        """Build request body for Mistral models."""
        return {
            "prompt": f"<s>[INST] {request.system_prompt}\n\n{request.user_prompt} [/INST]",
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
        }
    
    def _build_cohere_body(self, request: LLMRequest) -> Dict:
        """Build request body for Cohere models."""
        return {
            "message": request.user_prompt,
            "preamble": request.system_prompt,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
            "p": request.top_p,  # Cohere uses 'p' instead of 'top_p'
        }
    
    def _build_nova_body(self, request: LLMRequest) -> Dict:
        """Build request body for Amazon Nova models."""
        return {
            "messages": [
                {"role": "user", "content": [{"text": request.user_prompt}]}
            ],
            "system": [{"text": request.system_prompt}],
            "inferenceConfig": {
                "maxTokens": request.max_tokens,
                "temperature": request.temperature,
                "topP": request.top_p,
            }
        }
    
    def _parse_response(self, response_body: Dict, model_id: str) -> tuple:
        """Parse response based on model family."""
        if 'anthropic' in model_id:
            content = response_body.get('content', [{}])[0].get('text', '')
            input_tokens = response_body.get('usage', {}).get('input_tokens', 0)
            output_tokens = response_body.get('usage', {}).get('output_tokens', 0)
        elif 'meta.llama' in model_id:
            content = response_body.get('generation', '')
            # Llama doesn't always return token counts
            input_tokens = response_body.get('prompt_token_count', 0)
            output_tokens = response_body.get('generation_token_count', 0)
        elif 'mistral' in model_id:
            content = response_body.get('outputs', [{}])[0].get('text', '')
            input_tokens = 0  # Mistral doesn't return token counts
            output_tokens = 0
        elif 'cohere' in model_id:
            content = response_body.get('text', '')
            input_tokens = response_body.get('meta', {}).get('billed_units', {}).get('input_tokens', 0)
            output_tokens = response_body.get('meta', {}).get('billed_units', {}).get('output_tokens', 0)
        elif 'amazon.nova' in model_id:
            content = response_body.get('output', {}).get('message', {}).get('content', [{}])[0].get('text', '')
            input_tokens = response_body.get('usage', {}).get('inputTokens', 0)
            output_tokens = response_body.get('usage', {}).get('outputTokens', 0)
        else:
            content = str(response_body)
            input_tokens = 0
            output_tokens = 0
        
        return content, input_tokens, output_tokens
    
    def is_configured(self) -> bool:
        """Check if AWS credentials are configured."""
        try:
            import boto3
            sts = boto3.client('sts', region_name=self.region)
            sts.get_caller_identity()
            return True
        except:
            return False