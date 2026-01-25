"""
CYPEARL Phase 2 - Persona Embeddings

Implements continuous persona embedding space instead of discrete personas.
Benefits:
1. Interpolation: Create "between" personas (e.g., 70% analytical + 30% impulsive)
2. Personalization: Map specific employees to their exact position, not nearest cluster
3. Smoother transitions: No hard cluster boundaries
4. Transfer learning: Train on one population, transfer to another

Research contribution: "Continuous Persona Embeddings for High-Fidelity LLM Simulation"
"""

from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from enum import Enum
import json

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Using numpy-only implementation.")

from core.schemas import Persona


# Define standard trait ordering for consistent vectorization
TRAIT_ORDER = [
    # Cognitive traits (4)
    'crt_score', 'need_for_cognition', 'working_memory', 'impulsivity_total',
    # Big 5 personality (5)
    'big5_extraversion', 'big5_agreeableness', 'big5_conscientiousness',
    'big5_neuroticism', 'big5_openness',
    # Psychological state (6)
    'trust_propensity', 'risk_taking', 'state_anxiety', 'current_stress',
    'fatigue_level', 'sensation_seeking',
    # Security awareness (8)
    'phishing_self_efficacy', 'perceived_risk', 'security_attitudes',
    'privacy_concern', 'phishing_knowledge', 'technical_expertise',
    'prior_victimization', 'security_training',
    # Susceptibility (3)
    'authority_susceptibility', 'urgency_susceptibility', 'scarcity_susceptibility',
    # Behavioral (3)
    'link_click_tendency', 'email_volume_numeric', 'social_media_usage'
]

N_TRAITS = len(TRAIT_ORDER)  # 29 traits


@dataclass
class PersonaEmbedding:
    """
    Continuous embedding representation of a persona.

    Instead of discrete persona ID, represents persona as a point
    in continuous embedding space.
    """
    embedding: np.ndarray  # The embedding vector
    dimension: int = 64    # Embedding dimension

    # Original traits (for reference)
    original_traits: Dict[str, float] = field(default_factory=dict)

    # Reconstruction quality (if trained)
    reconstruction_error: float = 0.0

    # Metadata
    source_persona_id: Optional[str] = None
    source_persona_name: Optional[str] = None
    is_interpolated: bool = False
    interpolation_weights: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        if self.embedding is None:
            self.embedding = np.zeros(self.dimension)
        self.dimension = len(self.embedding)

    def distance_to(self, other: 'PersonaEmbedding') -> float:
        """Calculate Euclidean distance to another embedding."""
        return float(np.linalg.norm(self.embedding - other.embedding))

    def cosine_similarity(self, other: 'PersonaEmbedding') -> float:
        """Calculate cosine similarity to another embedding."""
        dot = np.dot(self.embedding, other.embedding)
        norm1 = np.linalg.norm(self.embedding)
        norm2 = np.linalg.norm(other.embedding)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(dot / (norm1 * norm2))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'embedding': self.embedding.tolist(),
            'dimension': self.dimension,
            'original_traits': self.original_traits,
            'reconstruction_error': self.reconstruction_error,
            'source_persona_id': self.source_persona_id,
            'source_persona_name': self.source_persona_name,
            'is_interpolated': self.is_interpolated,
            'interpolation_weights': self.interpolation_weights
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonaEmbedding':
        return cls(
            embedding=np.array(data['embedding']),
            dimension=data.get('dimension', 64),
            original_traits=data.get('original_traits', {}),
            reconstruction_error=data.get('reconstruction_error', 0.0),
            source_persona_id=data.get('source_persona_id'),
            source_persona_name=data.get('source_persona_name'),
            is_interpolated=data.get('is_interpolated', False),
            interpolation_weights=data.get('interpolation_weights', {})
        )


class PersonaEncoderNumpy:
    """
    Numpy-based persona encoder for environments without PyTorch.

    Uses simple linear projection for encoding and decoding.
    """

    def __init__(self, embedding_dim: int = 64, random_seed: int = 42):
        """
        Initialize numpy-based encoder.

        Args:
            embedding_dim: Dimension of embedding space
            random_seed: Random seed for reproducibility
        """
        self.embedding_dim = embedding_dim
        self.n_traits = N_TRAITS

        np.random.seed(random_seed)

        # Initialize projection matrices (simple linear encoder/decoder)
        # Using Xavier initialization
        scale = np.sqrt(2.0 / (self.n_traits + embedding_dim))
        self.encoder_weights = np.random.randn(embedding_dim, self.n_traits) * scale
        self.encoder_bias = np.zeros(embedding_dim)

        self.decoder_weights = np.random.randn(self.n_traits, embedding_dim) * scale
        self.decoder_bias = np.zeros(self.n_traits)

        # Normalization parameters (will be set during fit)
        self.trait_means = np.zeros(self.n_traits)
        self.trait_stds = np.ones(self.n_traits)

    def fit(self, personas: List[Persona], n_iterations: int = 1000, learning_rate: float = 0.01):
        """
        Fit encoder/decoder to persona data.

        Uses simple gradient descent to minimize reconstruction error.

        Args:
            personas: List of personas to train on
            n_iterations: Number of training iterations
            learning_rate: Learning rate for gradient descent
        """
        # Convert personas to trait vectors
        trait_vectors = np.array([self._persona_to_vector(p) for p in personas])

        # Calculate normalization parameters
        self.trait_means = np.mean(trait_vectors, axis=0)
        self.trait_stds = np.std(trait_vectors, axis=0)
        self.trait_stds[self.trait_stds == 0] = 1.0  # Avoid division by zero

        # Normalize
        normalized = (trait_vectors - self.trait_means) / self.trait_stds

        # Simple training loop (gradient descent)
        for iteration in range(n_iterations):
            # Forward pass
            encoded = self._encode_batch(normalized)
            decoded = self._decode_batch(encoded)

            # Reconstruction error
            error = normalized - decoded
            mse = np.mean(error ** 2)

            # Backward pass (gradient descent)
            # Gradient of decoder
            d_decoder_weights = -2 * np.dot(error.T, encoded) / len(personas)
            d_decoder_bias = -2 * np.mean(error, axis=0)

            # Gradient of encoder (through decoder)
            d_encoded = -2 * np.dot(error, self.decoder_weights)
            d_encoder_weights = np.dot(d_encoded.T, normalized) / len(personas)
            d_encoder_bias = np.mean(d_encoded, axis=0)

            # Update weights
            self.decoder_weights -= learning_rate * d_decoder_weights
            self.decoder_bias -= learning_rate * d_decoder_bias
            self.encoder_weights -= learning_rate * d_encoder_weights
            self.encoder_bias -= learning_rate * d_encoder_bias

            if (iteration + 1) % 100 == 0:
                print(f"  Iteration {iteration + 1}/{n_iterations}, MSE: {mse:.6f}")

        print(f"  Training complete. Final MSE: {mse:.6f}")

    def _encode_batch(self, normalized_traits: np.ndarray) -> np.ndarray:
        """Encode batch of normalized trait vectors."""
        return np.tanh(np.dot(normalized_traits, self.encoder_weights.T) + self.encoder_bias)

    def _decode_batch(self, embeddings: np.ndarray) -> np.ndarray:
        """Decode batch of embeddings to normalized trait vectors."""
        return np.dot(embeddings, self.decoder_weights.T) + self.decoder_bias

    def encode(self, persona: Persona) -> PersonaEmbedding:
        """
        Encode a persona into embedding space.

        Args:
            persona: Persona to encode

        Returns:
            PersonaEmbedding with the embedding vector
        """
        traits_vector = self._persona_to_vector(persona)
        normalized = (traits_vector - self.trait_means) / self.trait_stds

        embedding = np.tanh(np.dot(self.encoder_weights, normalized) + self.encoder_bias)

        return PersonaEmbedding(
            embedding=embedding,
            dimension=self.embedding_dim,
            original_traits=persona.trait_zscores or {},
            source_persona_id=persona.persona_id,
            source_persona_name=persona.name
        )

    def decode(self, embedding: PersonaEmbedding) -> Dict[str, float]:
        """
        Decode an embedding back to trait z-scores.

        Args:
            embedding: PersonaEmbedding to decode

        Returns:
            Dict mapping trait names to z-scores
        """
        decoded_normalized = np.dot(self.decoder_weights, embedding.embedding) + self.decoder_bias
        decoded_traits = decoded_normalized * self.trait_stds + self.trait_means

        return {trait: float(decoded_traits[i]) for i, trait in enumerate(TRAIT_ORDER)}

    def _persona_to_vector(self, persona: Persona) -> np.ndarray:
        """Convert persona to ordered trait vector."""
        traits = persona.trait_zscores or {}
        return np.array([traits.get(t, 0.0) for t in TRAIT_ORDER])

    def interpolate(
        self,
        persona_a: Persona,
        persona_b: Persona,
        alpha: float = 0.5
    ) -> PersonaEmbedding:
        """
        Create interpolated persona between A and B.

        Args:
            persona_a: First persona
            persona_b: Second persona
            alpha: Interpolation weight (0=A, 1=B, 0.5=midpoint)

        Returns:
            Interpolated PersonaEmbedding
        """
        emb_a = self.encode(persona_a)
        emb_b = self.encode(persona_b)

        interpolated_embedding = (1 - alpha) * emb_a.embedding + alpha * emb_b.embedding

        return PersonaEmbedding(
            embedding=interpolated_embedding,
            dimension=self.embedding_dim,
            original_traits={},  # Will be reconstructed via decode
            is_interpolated=True,
            interpolation_weights={
                persona_a.persona_id: 1 - alpha,
                persona_b.persona_id: alpha
            }
        )

    def find_nearest_persona(
        self,
        target_embedding: PersonaEmbedding,
        persona_embeddings: List[PersonaEmbedding]
    ) -> Tuple[PersonaEmbedding, float]:
        """
        Find nearest persona to target in embedding space.

        Args:
            target_embedding: Target embedding
            persona_embeddings: List of candidate embeddings

        Returns:
            Tuple of (nearest embedding, distance)
        """
        distances = [target_embedding.distance_to(emb) for emb in persona_embeddings]
        min_idx = np.argmin(distances)
        return persona_embeddings[min_idx], distances[min_idx]

    def save(self, filepath: str):
        """Save encoder parameters to file."""
        data = {
            'embedding_dim': self.embedding_dim,
            'n_traits': self.n_traits,
            'encoder_weights': self.encoder_weights.tolist(),
            'encoder_bias': self.encoder_bias.tolist(),
            'decoder_weights': self.decoder_weights.tolist(),
            'decoder_bias': self.decoder_bias.tolist(),
            'trait_means': self.trait_means.tolist(),
            'trait_stds': self.trait_stds.tolist(),
            'trait_order': TRAIT_ORDER
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)

    def load(self, filepath: str):
        """Load encoder parameters from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)

        self.embedding_dim = data['embedding_dim']
        self.n_traits = data['n_traits']
        self.encoder_weights = np.array(data['encoder_weights'])
        self.encoder_bias = np.array(data['encoder_bias'])
        self.decoder_weights = np.array(data['decoder_weights'])
        self.decoder_bias = np.array(data['decoder_bias'])
        self.trait_means = np.array(data['trait_means'])
        self.trait_stds = np.array(data['trait_stds'])


# PyTorch implementation (if available)
if TORCH_AVAILABLE:
    class PersonaEncoderNet(nn.Module):
        """
        Neural network encoder/decoder for personas.

        Architecture:
        Encoder: 29 traits -> 128 -> 64 -> embedding_dim
        Decoder: embedding_dim -> 64 -> 128 -> 29 traits
        """

        def __init__(self, n_traits: int = N_TRAITS, embedding_dim: int = 64):
            super().__init__()

            self.encoder = nn.Sequential(
                nn.Linear(n_traits, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, embedding_dim)
            )

            self.decoder = nn.Sequential(
                nn.Linear(embedding_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 128),
                nn.ReLU(),
                nn.BatchNorm1d(128),
                nn.Linear(128, n_traits)
            )

        def encode(self, traits: torch.Tensor) -> torch.Tensor:
            return self.encoder(traits)

        def decode(self, embedding: torch.Tensor) -> torch.Tensor:
            return self.decoder(embedding)

        def forward(self, traits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            embedding = self.encode(traits)
            reconstruction = self.decode(embedding)
            return embedding, reconstruction


    class PersonaEncoderTorch:
        """
        PyTorch-based persona encoder with proper training.

        Supports:
        - Autoencoder training for embedding learning
        - Variational autoencoder (VAE) for smooth interpolation
        - Contrastive learning for persona separation
        """

        def __init__(
            self,
            embedding_dim: int = 64,
            learning_rate: float = 0.001,
            device: str = 'cpu'
        ):
            """
            Initialize PyTorch encoder.

            Args:
                embedding_dim: Dimension of embedding space
                learning_rate: Learning rate for training
                device: Device to use ('cpu' or 'cuda')
            """
            self.embedding_dim = embedding_dim
            self.device = torch.device(device)

            self.model = PersonaEncoderNet(N_TRAITS, embedding_dim).to(self.device)
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

            # Normalization
            self.trait_means = torch.zeros(N_TRAITS, device=self.device)
            self.trait_stds = torch.ones(N_TRAITS, device=self.device)

        def fit(
            self,
            personas: List[Persona],
            n_epochs: int = 100,
            batch_size: int = 32,
            verbose: bool = True
        ):
            """
            Train encoder on persona data.

            Args:
                personas: List of personas
                n_epochs: Number of training epochs
                batch_size: Batch size
                verbose: Print progress
            """
            # Convert to tensor
            trait_vectors = torch.tensor(
                [self._persona_to_vector(p) for p in personas],
                dtype=torch.float32,
                device=self.device
            )

            # Calculate normalization
            self.trait_means = trait_vectors.mean(dim=0)
            self.trait_stds = trait_vectors.std(dim=0)
            self.trait_stds[self.trait_stds == 0] = 1.0

            # Normalize
            normalized = (trait_vectors - self.trait_means) / self.trait_stds

            # Training loop
            self.model.train()
            n_samples = len(normalized)

            for epoch in range(n_epochs):
                # Shuffle
                indices = torch.randperm(n_samples)
                epoch_loss = 0.0

                for i in range(0, n_samples, batch_size):
                    batch_indices = indices[i:min(i + batch_size, n_samples)]
                    batch = normalized[batch_indices]

                    self.optimizer.zero_grad()

                    embedding, reconstruction = self.model(batch)

                    # Reconstruction loss
                    loss = F.mse_loss(reconstruction, batch)

                    loss.backward()
                    self.optimizer.step()

                    epoch_loss += loss.item() * len(batch_indices)

                epoch_loss /= n_samples

                if verbose and (epoch + 1) % 10 == 0:
                    print(f"  Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss:.6f}")

            self.model.eval()
            if verbose:
                print(f"  Training complete. Final loss: {epoch_loss:.6f}")

        def encode(self, persona: Persona) -> PersonaEmbedding:
            """Encode persona to embedding."""
            self.model.eval()

            traits = torch.tensor(
                [self._persona_to_vector(persona)],
                dtype=torch.float32,
                device=self.device
            )
            normalized = (traits - self.trait_means) / self.trait_stds

            with torch.no_grad():
                embedding = self.model.encode(normalized)

            return PersonaEmbedding(
                embedding=embedding.cpu().numpy().flatten(),
                dimension=self.embedding_dim,
                original_traits=persona.trait_zscores or {},
                source_persona_id=persona.persona_id,
                source_persona_name=persona.name
            )

        def decode(self, embedding: PersonaEmbedding) -> Dict[str, float]:
            """Decode embedding to trait z-scores."""
            self.model.eval()

            emb_tensor = torch.tensor(
                [embedding.embedding],
                dtype=torch.float32,
                device=self.device
            )

            with torch.no_grad():
                decoded_normalized = self.model.decode(emb_tensor)

            decoded = decoded_normalized * self.trait_stds + self.trait_means
            decoded = decoded.cpu().numpy().flatten()

            return {trait: float(decoded[i]) for i, trait in enumerate(TRAIT_ORDER)}

        def _persona_to_vector(self, persona: Persona) -> List[float]:
            """Convert persona to ordered trait vector."""
            traits = persona.trait_zscores or {}
            return [traits.get(t, 0.0) for t in TRAIT_ORDER]

        def interpolate(
            self,
            persona_a: Persona,
            persona_b: Persona,
            alpha: float = 0.5
        ) -> PersonaEmbedding:
            """Create interpolated persona."""
            emb_a = self.encode(persona_a)
            emb_b = self.encode(persona_b)

            interpolated = (1 - alpha) * emb_a.embedding + alpha * emb_b.embedding

            return PersonaEmbedding(
                embedding=interpolated,
                dimension=self.embedding_dim,
                is_interpolated=True,
                interpolation_weights={
                    persona_a.persona_id: 1 - alpha,
                    persona_b.persona_id: alpha
                }
            )

        def save(self, filepath: str):
            """Save model to file."""
            torch.save({
                'model_state': self.model.state_dict(),
                'trait_means': self.trait_means.cpu(),
                'trait_stds': self.trait_stds.cpu(),
                'embedding_dim': self.embedding_dim
            }, filepath)

        def load(self, filepath: str):
            """Load model from file."""
            checkpoint = torch.load(filepath, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state'])
            self.trait_means = checkpoint['trait_means'].to(self.device)
            self.trait_stds = checkpoint['trait_stds'].to(self.device)
            self.embedding_dim = checkpoint['embedding_dim']


# Select best available encoder
def get_persona_encoder(embedding_dim: int = 64, use_torch: bool = True) -> Union[PersonaEncoderNumpy, 'PersonaEncoderTorch']:
    """
    Get the best available persona encoder.

    Args:
        embedding_dim: Embedding dimension
        use_torch: Prefer PyTorch if available

    Returns:
        PersonaEncoderNumpy or PersonaEncoderTorch
    """
    if use_torch and TORCH_AVAILABLE:
        return PersonaEncoderTorch(embedding_dim=embedding_dim)
    else:
        return PersonaEncoderNumpy(embedding_dim=embedding_dim)


class EmbeddingToPromptGenerator:
    """
    Generates persona prompts from embeddings instead of discrete personas.

    This allows smooth interpolation and personalization.
    """

    def __init__(self, encoder: Union[PersonaEncoderNumpy, 'PersonaEncoderTorch']):
        """
        Initialize prompt generator.

        Args:
            encoder: Trained persona encoder
        """
        self.encoder = encoder

    def generate_prompt_traits(self, embedding: PersonaEmbedding) -> str:
        """
        Generate trait description from embedding.

        Args:
            embedding: PersonaEmbedding to generate from

        Returns:
            Trait description string for prompt
        """
        # Decode embedding to traits
        traits = self.encoder.decode(embedding)

        def describe_trait(name: str, z: float) -> str:
            """Convert trait z-score to description."""
            trait_names = {
                'crt_score': 'analytical thinking',
                'need_for_cognition': 'enjoys complex thinking',
                'working_memory': 'good memory',
                'impulsivity_total': 'acts quickly without thinking',
                'trust_propensity': 'trusting of others',
                'risk_taking': 'comfortable with risks',
                'state_anxiety': 'currently anxious',
                'current_stress': 'under stress',
                'fatigue_level': 'tired',
                'sensation_seeking': 'seeks excitement',
                'phishing_self_efficacy': 'confident in spotting phishing',
                'perceived_risk': 'believes phishing is a threat',
                'security_attitudes': 'cares about security',
                'privacy_concern': 'concerned about privacy',
                'phishing_knowledge': 'knows phishing techniques',
                'technical_expertise': 'technically skilled',
                'prior_victimization': 'been phished before',
                'security_training': 'has security training',
                'authority_susceptibility': 'defers to authority',
                'urgency_susceptibility': 'responds to time pressure',
                'scarcity_susceptibility': 'responds to scarcity',
                'link_click_tendency': 'often clicks links',
                'email_volume_numeric': 'handles many emails daily',
                'social_media_usage': 'active on social media',
                'big5_extraversion': 'outgoing',
                'big5_agreeableness': 'agreeable',
                'big5_conscientiousness': 'conscientious',
                'big5_neuroticism': 'anxious/worried',
                'big5_openness': 'open to new experiences',
            }

            desc = trait_names.get(name, name.replace('_', ' '))

            if z > 0.8:
                return f"VERY {desc}"
            elif z > 0.3:
                return f"somewhat {desc}"
            elif z > -0.3:
                return f"moderate in {desc}"
            elif z > -0.8:
                return f"NOT very {desc}"
            else:
                return f"VERY LOW in {desc}"

        # Group by category
        cognitive = ['crt_score', 'need_for_cognition', 'working_memory', 'impulsivity_total']
        personality = [t for t in traits if t.startswith('big5_')]
        psychological = ['trust_propensity', 'risk_taking', 'state_anxiety', 'current_stress',
                        'fatigue_level', 'sensation_seeking']
        security = ['phishing_self_efficacy', 'perceived_risk', 'security_attitudes',
                   'privacy_concern', 'phishing_knowledge', 'technical_expertise',
                   'prior_victimization', 'security_training']
        susceptibility = ['authority_susceptibility', 'urgency_susceptibility', 'scarcity_susceptibility']

        sections = []

        # Cognitive
        cog_lines = [describe_trait(t, traits[t]) for t in cognitive if t in traits]
        if cog_lines:
            sections.append("COGNITIVE STYLE:\n" + "\n".join(f"  • {line}" for line in cog_lines))

        # Personality
        pers_lines = [describe_trait(t, traits[t]) for t in personality if t in traits]
        if pers_lines:
            sections.append("PERSONALITY:\n" + "\n".join(f"  • {line}" for line in pers_lines))

        # Psychological
        psych_lines = [describe_trait(t, traits[t]) for t in psychological if t in traits]
        if psych_lines:
            sections.append("PSYCHOLOGICAL STATE:\n" + "\n".join(f"  • {line}" for line in psych_lines))

        # Security
        sec_lines = [describe_trait(t, traits[t]) for t in security if t in traits]
        if sec_lines:
            sections.append("SECURITY AWARENESS:\n" + "\n".join(f"  • {line}" for line in sec_lines))

        # Susceptibility
        susc_lines = [describe_trait(t, traits[t]) for t in susceptibility if t in traits]
        if susc_lines:
            sections.append("SUSCEPTIBILITIES:\n" + "\n".join(f"  • {line}" for line in susc_lines))

        return "\n\n".join(sections)

    def generate_interpolated_prompt(
        self,
        persona_a: Persona,
        persona_b: Persona,
        alpha: float = 0.5,
        name: str = "Interpolated Persona"
    ) -> str:
        """
        Generate prompt for interpolated persona.

        Args:
            persona_a: First persona
            persona_b: Second persona
            alpha: Interpolation weight
            name: Name for the interpolated persona

        Returns:
            Complete prompt content
        """
        embedding = self.encoder.interpolate(persona_a, persona_b, alpha)
        traits_text = self.generate_prompt_traits(embedding)

        weight_a = int((1 - alpha) * 100)
        weight_b = int(alpha * 100)

        return f"""You are experiencing life as "{name}" for a moment.

This persona is a blend of:
- {weight_a}% {persona_a.name}
- {weight_b}% {persona_b.name}

YOUR CHARACTERISTICS:
{traits_text}

Respond to emails as this blended persona would - combining the traits from both source personas in proportion to their weights."""


@dataclass
class PersonaEmbeddingSpace:
    """
    Manages the embedding space for all personas.

    Provides operations like:
    - Nearest neighbor search
    - Cluster analysis in embedding space
    - Visualization coordinates
    """

    embeddings: Dict[str, PersonaEmbedding] = field(default_factory=dict)
    encoder: Union[PersonaEncoderNumpy, 'PersonaEncoderTorch'] = None

    def add_persona(self, persona: Persona):
        """Add persona to embedding space."""
        if self.encoder is None:
            raise ValueError("Encoder not set")

        embedding = self.encoder.encode(persona)
        self.embeddings[persona.persona_id] = embedding

    def get_embedding(self, persona_id: str) -> Optional[PersonaEmbedding]:
        """Get embedding by persona ID."""
        return self.embeddings.get(persona_id)

    def find_nearest(
        self,
        target: PersonaEmbedding,
        k: int = 5,
        exclude_self: bool = True
    ) -> List[Tuple[str, float]]:
        """
        Find k nearest neighbors to target embedding.

        Args:
            target: Target embedding
            k: Number of neighbors
            exclude_self: Exclude exact match

        Returns:
            List of (persona_id, distance) tuples
        """
        distances = []
        for pid, emb in self.embeddings.items():
            dist = target.distance_to(emb)
            if exclude_self and dist < 1e-6:
                continue
            distances.append((pid, dist))

        distances.sort(key=lambda x: x[1])
        return distances[:k]

    def get_2d_coordinates(self) -> Dict[str, Tuple[float, float]]:
        """
        Get 2D coordinates for visualization using PCA.

        Returns:
            Dict mapping persona_id to (x, y) coordinates
        """
        if len(self.embeddings) < 2:
            return {pid: (0.0, 0.0) for pid in self.embeddings}

        # Stack embeddings
        ids = list(self.embeddings.keys())
        vectors = np.array([self.embeddings[pid].embedding for pid in ids])

        # Simple PCA (first 2 components)
        centered = vectors - vectors.mean(axis=0)
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Get top 2 eigenvectors
        idx = eigenvalues.argsort()[::-1]
        top2 = eigenvectors[:, idx[:2]]

        # Project
        coords_2d = centered @ top2

        return {pid: (float(coords_2d[i, 0]), float(coords_2d[i, 1]))
                for i, pid in enumerate(ids)}

    def to_dict(self) -> Dict[str, Any]:
        return {
            'embeddings': {pid: emb.to_dict() for pid, emb in self.embeddings.items()}
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PersonaEmbeddingSpace':
        space = cls()
        space.embeddings = {
            pid: PersonaEmbedding.from_dict(emb_data)
            for pid, emb_data in data.get('embeddings', {}).items()
        }
        return space
