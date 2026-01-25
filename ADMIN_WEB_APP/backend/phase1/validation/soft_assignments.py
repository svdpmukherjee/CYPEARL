"""
Soft Cluster Assignment Analysis

Handles cluster membership uncertainty by providing probabilistic assignments
instead of hard (discrete) assignments.

Scientific Rationale:
- Hard assignment: "Person X belongs to Cluster 3" (100% confidence)
- Soft assignment: "Person X: 65% Cluster 3, 25% Cluster 1, 10% Cluster 2"

Why This Matters:
1. Some participants are on cluster boundaries (not clearly in one cluster)
2. Hard assignments hide this uncertainty
3. For LLM conditioning, boundary cases may need multi-persona prompts
4. Scientific honesty requires acknowledging uncertainty

Methods:
1. GMM Posterior Probabilities: Direct probabilistic output
2. Distance-Based: Softmax of distances to cluster centers
3. Entropy-Based Uncertainty: High entropy = uncertain assignment
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from scipy.special import softmax
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class SoftAssignment:
    """Soft cluster assignment for a participant."""
    participant_idx: int
    hard_assignment: int
    probabilities: Dict[int, float]
    entropy: float
    uncertainty_level: str  # 'low', 'moderate', 'high'
    max_probability: float
    second_max_probability: float
    margin: float  # Difference between top two probabilities

    def to_dict(self) -> Dict:
        return {
            'participant_idx': self.participant_idx,
            'hard_assignment': self.hard_assignment,
            'probabilities': self.probabilities,
            'entropy': self.entropy,
            'uncertainty_level': self.uncertainty_level,
            'max_probability': self.max_probability,
            'second_max_probability': self.second_max_probability,
            'margin': self.margin
        }


class SoftAssignmentAnalyzer:
    """
    Compute and analyze soft (probabilistic) cluster assignments.

    Provides uncertainty quantification for cluster membership,
    identifying participants who don't fit cleanly into one cluster.
    """

    def __init__(self, random_state: int = 42):
        """Initialize analyzer."""
        self.random_state = random_state

    def analyze(
        self,
        X: np.ndarray,
        k: int,
        algorithm: str = 'gmm',
        temperature: float = 1.0
    ) -> Dict:
        """
        Compute soft assignments and uncertainty analysis.

        Args:
            X: Feature matrix (n_samples, n_features)
            k: Number of clusters
            algorithm: 'gmm' (native soft), 'kmeans' (distance-based soft)
            temperature: Softmax temperature for distance-based (lower = sharper)

        Returns:
            Comprehensive soft assignment analysis
        """
        n_samples = X.shape[0]

        # Get soft assignments
        if algorithm == 'gmm':
            soft_probs, hard_labels, model = self._gmm_soft_assignments(X, k)
        else:
            soft_probs, hard_labels, model = self._distance_based_soft_assignments(
                X, k, temperature
            )

        # Analyze each participant
        participant_assignments = []
        for i in range(n_samples):
            probs = soft_probs[i]
            assignment = self._create_soft_assignment(i, probs, hard_labels[i])
            participant_assignments.append(assignment)

        # Aggregate analysis
        uncertainty_distribution = self._analyze_uncertainty_distribution(
            participant_assignments
        )

        # Identify uncertain participants
        uncertain_participants = self._identify_uncertain_participants(
            participant_assignments
        )

        # Cluster purity analysis
        cluster_purity = self._analyze_cluster_purity(
            participant_assignments, k
        )

        # Recommendation for handling uncertainty
        handling_strategy = self._recommend_handling_strategy(
            uncertainty_distribution, uncertain_participants
        )

        return {
            'algorithm': algorithm,
            'k': k,
            'soft_assignments': [a.to_dict() for a in participant_assignments],
            'soft_probability_matrix': soft_probs.tolist(),
            'hard_labels': hard_labels.tolist(),
            'uncertainty_distribution': uncertainty_distribution,
            'uncertain_participants': uncertain_participants,
            'cluster_purity': cluster_purity,
            'handling_strategy': handling_strategy,
            'summary': self._generate_summary(
                participant_assignments, uncertainty_distribution
            )
        }

    def _gmm_soft_assignments(
        self,
        X: np.ndarray,
        k: int
    ) -> Tuple[np.ndarray, np.ndarray, GaussianMixture]:
        """Get soft assignments from Gaussian Mixture Model."""
        gmm = GaussianMixture(
            n_components=k,
            random_state=self.random_state,
            n_init=10,
            covariance_type='full'
        )
        gmm.fit(X)

        # Posterior probabilities
        soft_probs = gmm.predict_proba(X)
        hard_labels = gmm.predict(X)

        return soft_probs, hard_labels, gmm

    def _distance_based_soft_assignments(
        self,
        X: np.ndarray,
        k: int,
        temperature: float
    ) -> Tuple[np.ndarray, np.ndarray, KMeans]:
        """Get soft assignments using distance-based softmax."""
        kmeans = KMeans(
            n_clusters=k,
            random_state=self.random_state,
            n_init=10
        )
        hard_labels = kmeans.fit_predict(X)

        # Compute distances to all cluster centers
        distances = cdist(X, kmeans.cluster_centers_, metric='euclidean')

        # Convert distances to probabilities using softmax
        # Negative because smaller distance = higher probability
        # Temperature controls sharpness
        soft_probs = softmax(-distances / temperature, axis=1)

        return soft_probs, hard_labels, kmeans

    def _create_soft_assignment(
        self,
        idx: int,
        probs: np.ndarray,
        hard_label: int
    ) -> SoftAssignment:
        """Create SoftAssignment object for a participant."""
        k = len(probs)

        # Probability dictionary
        prob_dict = {i: float(probs[i]) for i in range(k)}

        # Entropy (uncertainty measure)
        # Higher entropy = more uncertain
        entropy = -np.sum(probs * np.log(probs + 1e-10))
        max_entropy = np.log(k)  # Maximum possible entropy
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Uncertainty level based on normalized entropy
        if normalized_entropy < 0.3:
            uncertainty_level = 'low'
        elif normalized_entropy < 0.6:
            uncertainty_level = 'moderate'
        else:
            uncertainty_level = 'high'

        # Top two probabilities
        sorted_probs = np.sort(probs)[::-1]
        max_prob = sorted_probs[0]
        second_max = sorted_probs[1] if len(sorted_probs) > 1 else 0
        margin = max_prob - second_max

        return SoftAssignment(
            participant_idx=idx,
            hard_assignment=int(hard_label),
            probabilities=prob_dict,
            entropy=float(normalized_entropy),
            uncertainty_level=uncertainty_level,
            max_probability=float(max_prob),
            second_max_probability=float(second_max),
            margin=float(margin)
        )

    def _analyze_uncertainty_distribution(
        self,
        assignments: List[SoftAssignment]
    ) -> Dict:
        """Analyze distribution of uncertainty across participants."""
        entropies = [a.entropy for a in assignments]
        margins = [a.margin for a in assignments]

        n_total = len(assignments)
        n_low = sum(1 for a in assignments if a.uncertainty_level == 'low')
        n_moderate = sum(1 for a in assignments if a.uncertainty_level == 'moderate')
        n_high = sum(1 for a in assignments if a.uncertainty_level == 'high')

        return {
            'entropy_stats': {
                'mean': float(np.mean(entropies)),
                'std': float(np.std(entropies)),
                'min': float(np.min(entropies)),
                'max': float(np.max(entropies)),
                'median': float(np.median(entropies))
            },
            'margin_stats': {
                'mean': float(np.mean(margins)),
                'std': float(np.std(margins)),
                'min': float(np.min(margins)),
                'max': float(np.max(margins))
            },
            'uncertainty_counts': {
                'low': n_low,
                'moderate': n_moderate,
                'high': n_high
            },
            'uncertainty_percentages': {
                'low_pct': float(n_low / n_total * 100),
                'moderate_pct': float(n_moderate / n_total * 100),
                'high_pct': float(n_high / n_total * 100)
            }
        }

    def _identify_uncertain_participants(
        self,
        assignments: List[SoftAssignment],
        threshold: float = 0.5
    ) -> Dict:
        """Identify participants with high uncertainty."""
        uncertain = [a for a in assignments if a.entropy >= threshold]
        very_uncertain = [a for a in assignments if a.entropy >= 0.7]

        # Find participants who could belong to multiple clusters
        multi_cluster = []
        for a in assignments:
            # Count clusters with probability > 0.2
            viable_clusters = [c for c, p in a.probabilities.items() if p > 0.2]
            if len(viable_clusters) > 1:
                multi_cluster.append({
                    'participant_idx': a.participant_idx,
                    'viable_clusters': viable_clusters,
                    'probabilities': {c: a.probabilities[c] for c in viable_clusters}
                })

        return {
            'uncertain_count': len(uncertain),
            'uncertain_indices': [a.participant_idx for a in uncertain],
            'very_uncertain_count': len(very_uncertain),
            'very_uncertain_indices': [a.participant_idx for a in very_uncertain],
            'multi_cluster_candidates': multi_cluster,
            'multi_cluster_count': len(multi_cluster)
        }

    def _analyze_cluster_purity(
        self,
        assignments: List[SoftAssignment],
        k: int
    ) -> Dict:
        """Analyze purity of each cluster (how confident are its members?)."""
        cluster_analysis = {}

        for c in range(k):
            cluster_members = [a for a in assignments if a.hard_assignment == c]

            if not cluster_members:
                cluster_analysis[c] = {'n_members': 0}
                continue

            entropies = [a.entropy for a in cluster_members]
            max_probs = [a.max_probability for a in cluster_members]
            n_confident = sum(1 for a in cluster_members if a.max_probability > 0.8)

            cluster_analysis[c] = {
                'n_members': len(cluster_members),
                'mean_entropy': float(np.mean(entropies)),
                'mean_max_probability': float(np.mean(max_probs)),
                'pct_confident': float(n_confident / len(cluster_members) * 100),
                'purity_level': (
                    'high' if np.mean(max_probs) > 0.8 else
                    'moderate' if np.mean(max_probs) > 0.6 else
                    'low'
                )
            }

        # Overall cluster purity
        all_max_probs = [a.max_probability for a in assignments]
        overall_purity = float(np.mean(all_max_probs))

        return {
            'by_cluster': cluster_analysis,
            'overall_purity': overall_purity,
            'overall_purity_level': (
                'high' if overall_purity > 0.8 else
                'moderate' if overall_purity > 0.6 else
                'low'
            )
        }

    def _recommend_handling_strategy(
        self,
        uncertainty_dist: Dict,
        uncertain_participants: Dict
    ) -> Dict:
        """Recommend strategy for handling uncertain participants."""
        high_uncertainty_pct = uncertainty_dist['uncertainty_percentages']['high_pct']
        multi_cluster_count = uncertain_participants['multi_cluster_count']
        n_total = sum(uncertainty_dist['uncertainty_counts'].values())

        if high_uncertainty_pct < 10:
            strategy = 'ignore'
            description = (
                "Low uncertainty overall. Safe to use hard cluster assignments. "
                "Only a small minority are boundary cases."
            )
            llm_recommendation = (
                "Use standard single-persona prompts for all participants."
            )
        elif high_uncertainty_pct < 25:
            strategy = 'flag'
            description = (
                "Moderate uncertainty. Flag boundary cases for special handling. "
                f"Consider soft assignments for the {uncertain_participants['uncertain_count']} "
                "uncertain participants."
            )
            llm_recommendation = (
                "For boundary cases, consider: (1) Multi-persona averaging - average responses "
                "from top 2 personas weighted by probability, or (2) Explicit uncertainty - "
                "tell the LLM this persona has mixed characteristics."
            )
        else:
            strategy = 'soft_everywhere'
            description = (
                "High uncertainty. Many participants don't fit cleanly into clusters. "
                "Recommend using soft assignments throughout the pipeline."
            )
            llm_recommendation = (
                "Use soft persona prompts: 'This person is 60% Type A and 30% Type B'. "
                "Or increase K to create more specific clusters."
            )

        return {
            'strategy': strategy,
            'description': description,
            'llm_recommendation': llm_recommendation,
            'affected_participants': uncertain_participants['uncertain_count'],
            'affected_pct': high_uncertainty_pct
        }

    def _generate_summary(
        self,
        assignments: List[SoftAssignment],
        uncertainty_dist: Dict
    ) -> str:
        """Generate human-readable summary."""
        low_pct = uncertainty_dist['uncertainty_percentages']['low_pct']
        high_pct = uncertainty_dist['uncertainty_percentages']['high_pct']
        mean_entropy = uncertainty_dist['entropy_stats']['mean']

        return (
            f"Soft assignment analysis: {low_pct:.1f}% of participants have low uncertainty "
            f"(clear cluster membership), {high_pct:.1f}% have high uncertainty "
            f"(boundary cases). Mean normalized entropy: {mean_entropy:.2f}. "
            f"{'Clustering provides clear persona assignments.' if high_pct < 15 else 'Consider handling boundary cases explicitly.'}"
        )

    def get_multi_persona_prompts(
        self,
        soft_assignments: List[SoftAssignment],
        persona_descriptions: Dict[int, str],
        threshold: float = 0.2
    ) -> List[Dict]:
        """
        Generate multi-persona prompt suggestions for uncertain participants.

        For participants who could belong to multiple clusters, suggest
        a blended prompt that reflects their mixed characteristics.
        """
        multi_persona_prompts = []

        for assignment in soft_assignments:
            # Get clusters with probability > threshold
            relevant_clusters = [
                (c, p) for c, p in assignment.probabilities.items()
                if p >= threshold
            ]

            if len(relevant_clusters) > 1:
                # Sort by probability
                relevant_clusters.sort(key=lambda x: x[1], reverse=True)

                # Build blended prompt
                prompt_parts = []
                for cluster_id, prob in relevant_clusters:
                    desc = persona_descriptions.get(cluster_id, f'Persona {cluster_id}')
                    prompt_parts.append(f"{int(prob*100)}% {desc}")

                blended_prompt = (
                    f"This person has a mixed profile: {'; '.join(prompt_parts)}. "
                    f"When simulating their behavior, blend these characteristics "
                    f"according to the given percentages."
                )

                multi_persona_prompts.append({
                    'participant_idx': assignment.participant_idx,
                    'primary_cluster': relevant_clusters[0][0],
                    'secondary_clusters': [c for c, _ in relevant_clusters[1:]],
                    'probabilities': {c: p for c, p in relevant_clusters},
                    'blended_prompt': blended_prompt
                })

        return multi_persona_prompts
