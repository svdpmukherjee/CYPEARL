"""
Consensus Clustering for Robust Cluster Discovery

Consensus clustering runs multiple clustering iterations with different:
1. Random initializations
2. Bootstrap samples
3. Algorithms

Then aggregates results to find stable cluster structures that persist
across variations. This identifies which cluster assignments are robust
vs. which participants are "boundary cases".

Scientific Rationale:
- Single clustering run may be sensitive to initialization
- Consensus identifies core cluster membership (high agreement)
- Identifies uncertain participants (low agreement across runs)
- More defensible than single-run clustering

Reference:
    Monti et al. (2003). Consensus Clustering: A Resampling-Based Method
    for Class Discovery and Visualization of Gene Expression Microarray Data.
    Machine Learning, 52(1-2), 91-118.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, adjusted_rand_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class ParticipantConsensus:
    """Consensus information for a single participant."""
    participant_idx: int
    final_cluster: int
    consensus_strength: float  # 0-1, how consistently assigned
    is_core_member: bool  # High consensus (>0.8)
    is_boundary_case: bool  # Low consensus (<0.5)
    cluster_probabilities: Dict[int, float]

    def to_dict(self) -> Dict:
        return {
            'participant_idx': self.participant_idx,
            'final_cluster': self.final_cluster,
            'consensus_strength': self.consensus_strength,
            'is_core_member': self.is_core_member,
            'is_boundary_case': self.is_boundary_case,
            'cluster_probabilities': self.cluster_probabilities
        }


class ConsensusClusteringAnalyzer:
    """
    Build consensus clustering from multiple runs.

    Approach:
    1. Run clustering M times with different seeds/samples
    2. Build co-association matrix (how often pairs cluster together)
    3. Cluster the co-association matrix for final assignments
    4. Measure consensus strength per participant
    """

    def __init__(
        self,
        n_iterations: int = 100,
        subsample_ratio: float = 0.8,
        random_state: int = 42
    ):
        """
        Initialize consensus clustering.

        Args:
            n_iterations: Number of clustering iterations
            subsample_ratio: Fraction of data to sample each iteration
            random_state: Random seed
        """
        self.n_iterations = n_iterations
        self.subsample_ratio = subsample_ratio
        self.random_state = random_state

    def analyze(
        self,
        X: np.ndarray,
        k: int,
        algorithms: Optional[List[str]] = None
    ) -> Dict:
        """
        Perform consensus clustering analysis.

        Args:
            X: Feature matrix (n_samples, n_features)
            k: Number of clusters
            algorithms: List of algorithms to use (default: ['kmeans'])

        Returns:
            Comprehensive consensus analysis results
        """
        if algorithms is None:
            algorithms = ['kmeans']

        n_samples = X.shape[0]

        # 1. Build co-association matrix
        coassoc_matrix = self._build_coassociation_matrix(X, k, algorithms)

        # 2. Get final consensus clustering
        final_labels = self._consensus_from_coassociation(coassoc_matrix, k)

        # 3. Calculate per-participant consensus strength
        participant_consensus = self._calculate_participant_consensus(
            coassoc_matrix, final_labels
        )

        # 4. Identify core members and boundary cases
        core_members, boundary_cases = self._identify_core_and_boundary(
            participant_consensus
        )

        # 5. Cluster-level consensus metrics
        cluster_metrics = self._calculate_cluster_consensus_metrics(
            participant_consensus, final_labels
        )

        # 6. Stability metrics
        stability_metrics = self._calculate_stability_metrics(coassoc_matrix)

        # 7. Quality comparison: consensus vs. single-run
        quality_comparison = self._compare_to_single_run(
            X, final_labels, k
        )

        return {
            'final_labels': final_labels.tolist(),
            'k': k,
            'n_iterations': self.n_iterations,
            'algorithms_used': algorithms,
            'participant_consensus': [p.to_dict() for p in participant_consensus],
            'core_members': {
                'count': len(core_members),
                'pct': len(core_members) / n_samples * 100,
                'indices': core_members
            },
            'boundary_cases': {
                'count': len(boundary_cases),
                'pct': len(boundary_cases) / n_samples * 100,
                'indices': boundary_cases
            },
            'cluster_metrics': cluster_metrics,
            'stability_metrics': stability_metrics,
            'quality_comparison': quality_comparison,
            'coassociation_matrix_summary': {
                'mean': float(np.mean(coassoc_matrix)),
                'std': float(np.std(coassoc_matrix)),
                'min': float(np.min(coassoc_matrix)),
                'max': float(np.max(coassoc_matrix))
            },
            'recommendation': self._generate_recommendation(
                stability_metrics, len(boundary_cases) / n_samples
            )
        }

    def _build_coassociation_matrix(
        self,
        X: np.ndarray,
        k: int,
        algorithms: List[str]
    ) -> np.ndarray:
        """
        Build co-association matrix from multiple clustering runs.

        coassoc[i,j] = fraction of times samples i and j were in same cluster
        """
        n_samples = X.shape[0]
        coassoc = np.zeros((n_samples, n_samples))
        count_matrix = np.zeros((n_samples, n_samples))

        np.random.seed(self.random_state)

        for iteration in range(self.n_iterations):
            # Select algorithm (rotate through provided algorithms)
            algo = algorithms[iteration % len(algorithms)]

            # Subsample
            sample_size = int(n_samples * self.subsample_ratio)
            indices = np.random.choice(n_samples, size=sample_size, replace=False)
            X_sample = X[indices]

            # Cluster
            labels = self._cluster_single_run(X_sample, k, algo, iteration)

            # Update co-association for sampled pairs
            for i, idx_i in enumerate(indices):
                for j, idx_j in enumerate(indices):
                    count_matrix[idx_i, idx_j] += 1
                    if labels[i] == labels[j]:
                        coassoc[idx_i, idx_j] += 1

        # Normalize by count
        with np.errstate(divide='ignore', invalid='ignore'):
            coassoc = np.where(count_matrix > 0, coassoc / count_matrix, 0)

        return coassoc

    def _cluster_single_run(
        self,
        X: np.ndarray,
        k: int,
        algorithm: str,
        seed: int
    ) -> np.ndarray:
        """Run single clustering iteration."""
        if algorithm == 'kmeans':
            model = KMeans(n_clusters=k, random_state=self.random_state + seed, n_init=3)
            return model.fit_predict(X)
        elif algorithm == 'gmm':
            model = GaussianMixture(n_components=k, random_state=self.random_state + seed, n_init=3)
            return model.fit_predict(X)
        elif algorithm == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=k, linkage='ward')
            return model.fit_predict(X)
        else:
            # Default to K-Means
            model = KMeans(n_clusters=k, random_state=self.random_state + seed, n_init=3)
            return model.fit_predict(X)

    def _consensus_from_coassociation(
        self,
        coassoc_matrix: np.ndarray,
        k: int
    ) -> np.ndarray:
        """
        Extract final cluster assignments from co-association matrix.

        Uses hierarchical clustering on the co-association matrix
        (interpreted as similarity).
        """
        # Convert similarity to distance
        distance_matrix = 1 - coassoc_matrix

        # Ensure diagonal is 0
        np.fill_diagonal(distance_matrix, 0)

        # Make symmetric (handle numerical issues)
        distance_matrix = (distance_matrix + distance_matrix.T) / 2

        # Clip to valid range
        distance_matrix = np.clip(distance_matrix, 0, 1)

        # Convert to condensed form for scipy
        try:
            condensed = squareform(distance_matrix)
            # Hierarchical clustering
            Z = linkage(condensed, method='average')
            labels = fcluster(Z, k, criterion='maxclust') - 1  # 0-indexed
        except Exception:
            # Fallback: use K-Means on co-association features
            kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
            labels = kmeans.fit_predict(coassoc_matrix)

        return labels

    def _calculate_participant_consensus(
        self,
        coassoc_matrix: np.ndarray,
        final_labels: np.ndarray
    ) -> List[ParticipantConsensus]:
        """Calculate consensus strength for each participant."""
        n_samples = len(final_labels)
        k = len(np.unique(final_labels))
        results = []

        for i in range(n_samples):
            cluster = final_labels[i]

            # Calculate probability of belonging to each cluster
            # Based on average co-association with members of each cluster
            cluster_probs = {}
            for c in range(k):
                members = np.where(final_labels == c)[0]
                if len(members) > 0:
                    # Exclude self if in this cluster
                    other_members = members[members != i]
                    if len(other_members) > 0:
                        avg_coassoc = np.mean(coassoc_matrix[i, other_members])
                    else:
                        avg_coassoc = 1.0 if c == cluster else 0.0
                    cluster_probs[c] = float(avg_coassoc)
                else:
                    cluster_probs[c] = 0.0

            # Normalize probabilities
            total = sum(cluster_probs.values())
            if total > 0:
                cluster_probs = {c: p/total for c, p in cluster_probs.items()}

            # Consensus strength = probability of assigned cluster
            consensus_strength = cluster_probs.get(cluster, 0)

            results.append(ParticipantConsensus(
                participant_idx=i,
                final_cluster=int(cluster),
                consensus_strength=float(consensus_strength),
                is_core_member=consensus_strength >= 0.8,
                is_boundary_case=consensus_strength < 0.5,
                cluster_probabilities=cluster_probs
            ))

        return results

    def _identify_core_and_boundary(
        self,
        participant_consensus: List[ParticipantConsensus]
    ) -> Tuple[List[int], List[int]]:
        """Identify core members and boundary cases."""
        core_members = [p.participant_idx for p in participant_consensus if p.is_core_member]
        boundary_cases = [p.participant_idx for p in participant_consensus if p.is_boundary_case]
        return core_members, boundary_cases

    def _calculate_cluster_consensus_metrics(
        self,
        participant_consensus: List[ParticipantConsensus],
        final_labels: np.ndarray
    ) -> Dict:
        """Calculate consensus metrics per cluster."""
        k = len(np.unique(final_labels))
        cluster_metrics = {}

        for c in range(k):
            cluster_participants = [p for p in participant_consensus if p.final_cluster == c]

            if cluster_participants:
                strengths = [p.consensus_strength for p in cluster_participants]
                n_core = sum(1 for p in cluster_participants if p.is_core_member)
                n_boundary = sum(1 for p in cluster_participants if p.is_boundary_case)

                cluster_metrics[c] = {
                    'n_members': len(cluster_participants),
                    'mean_consensus': float(np.mean(strengths)),
                    'std_consensus': float(np.std(strengths)),
                    'min_consensus': float(np.min(strengths)),
                    'n_core_members': n_core,
                    'n_boundary_cases': n_boundary,
                    'core_pct': float(n_core / len(cluster_participants)) * 100,
                    'cohesion_level': (
                        'high' if np.mean(strengths) >= 0.7 else
                        'moderate' if np.mean(strengths) >= 0.5 else
                        'low'
                    )
                }

        return cluster_metrics

    def _calculate_stability_metrics(
        self,
        coassoc_matrix: np.ndarray
    ) -> Dict:
        """Calculate overall stability metrics."""
        # Off-diagonal values (excluding self-comparisons)
        mask = ~np.eye(coassoc_matrix.shape[0], dtype=bool)
        off_diag = coassoc_matrix[mask]

        # Proportion of strong co-associations (>0.8)
        strong_coassoc = np.mean(off_diag > 0.8)

        # Proportion of weak co-associations (<0.2)
        weak_coassoc = np.mean(off_diag < 0.2)

        # Bimodality: strong clustering should have values near 0 or 1
        # Measure as proportion NOT in middle range (0.3-0.7)
        bimodality = 1 - np.mean((off_diag > 0.3) & (off_diag < 0.7))

        return {
            'mean_coassociation': float(np.mean(off_diag)),
            'std_coassociation': float(np.std(off_diag)),
            'strong_coassoc_pct': float(strong_coassoc) * 100,
            'weak_coassoc_pct': float(weak_coassoc) * 100,
            'bimodality_index': float(bimodality),
            'stability_score': float((strong_coassoc + weak_coassoc) / 2),
            'interpretation': (
                'Highly stable' if bimodality > 0.8 else
                'Moderately stable' if bimodality > 0.6 else
                'Unstable - many ambiguous assignments'
            )
        }

    def _compare_to_single_run(
        self,
        X: np.ndarray,
        consensus_labels: np.ndarray,
        k: int
    ) -> Dict:
        """Compare consensus clustering to single-run K-Means."""
        # Single run K-Means
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        single_labels = kmeans.fit_predict(X)

        # Silhouette comparison
        consensus_sil = silhouette_score(X, consensus_labels)
        single_sil = silhouette_score(X, single_labels)

        # Agreement between consensus and single run
        ari = adjusted_rand_score(consensus_labels, single_labels)

        return {
            'consensus_silhouette': float(consensus_sil),
            'single_run_silhouette': float(single_sil),
            'silhouette_improvement': float(consensus_sil - single_sil),
            'adjusted_rand_index': float(ari),
            'agreement_interpretation': (
                'Very high agreement' if ari > 0.8 else
                'High agreement' if ari > 0.6 else
                'Moderate agreement' if ari > 0.4 else
                'Low agreement - consensus differs substantially'
            ),
            'recommendation': (
                'Consensus clustering improves quality'
                if consensus_sil > single_sil else
                'Single-run performs similarly - consensus adds robustness but not quality'
            )
        }

    def _generate_recommendation(
        self,
        stability_metrics: Dict,
        boundary_pct: float
    ) -> str:
        """Generate recommendation based on consensus analysis."""
        stability = stability_metrics['stability_score']
        bimodality = stability_metrics['bimodality_index']

        if stability > 0.7 and boundary_pct < 0.15:
            return (
                "STRONG CONSENSUS: Clustering is highly stable. "
                f"Only {boundary_pct*100:.1f}% boundary cases. "
                "Confident in cluster assignments for persona creation."
            )
        elif stability > 0.5 and boundary_pct < 0.25:
            return (
                "MODERATE CONSENSUS: Clustering is reasonably stable. "
                f"{boundary_pct*100:.1f}% are boundary cases - consider special handling. "
                "For boundary participants, use soft cluster assignments or multi-persona approach."
            )
        else:
            return (
                "WEAK CONSENSUS: Clustering shows instability. "
                f"{boundary_pct*100:.1f}% are boundary cases. "
                "Consider: increasing K, using different features, or accepting "
                "that some participants don't fit cleanly into persona categories."
            )

    def get_soft_assignments(
        self,
        X: np.ndarray,
        k: int,
        algorithms: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Get soft (probabilistic) cluster assignments based on consensus.

        Returns DataFrame with columns for each cluster probability.
        """
        if algorithms is None:
            algorithms = ['kmeans']

        coassoc = self._build_coassociation_matrix(X, k, algorithms)
        final_labels = self._consensus_from_coassociation(coassoc, k)
        participant_consensus = self._calculate_participant_consensus(coassoc, final_labels)

        # Build DataFrame
        data = []
        for p in participant_consensus:
            row = {
                'participant_idx': p.participant_idx,
                'hard_assignment': p.final_cluster,
                'consensus_strength': p.consensus_strength
            }
            for cluster, prob in p.cluster_probabilities.items():
                row[f'prob_cluster_{cluster}'] = prob
            data.append(row)

        return pd.DataFrame(data)
