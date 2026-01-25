"""
CYPEARL Phase 2 - Persona Embeddings Module

Provides continuous persona representation instead of discrete clusters.
"""

from .persona_encoder import (
    PersonaEmbedding,
    PersonaEncoderNumpy,
    PersonaEmbeddingSpace,
    EmbeddingToPromptGenerator,
    get_persona_encoder,
    TRAIT_ORDER,
    N_TRAITS
)

# Import PyTorch versions if available
try:
    from .persona_encoder import PersonaEncoderTorch, PersonaEncoderNet
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

__all__ = [
    'PersonaEmbedding',
    'PersonaEncoderNumpy',
    'PersonaEmbeddingSpace',
    'EmbeddingToPromptGenerator',
    'get_persona_encoder',
    'TRAIT_ORDER',
    'N_TRAITS',
    'TORCH_AVAILABLE'
]
