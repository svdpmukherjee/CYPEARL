import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    silhouette_score, silhouette_samples,
    calinski_harabasz_score, davies_bouldin_score
)
from typing import Dict, List, Optional

class MetricsCalculator:
    """Calculate comprehensive clustering quality metrics."""
    
    def __init__(self, df: pd.DataFrame, outcome_cols: List[str]):
        self.df = df
        self.outcome_cols = [c for c in outcome_cols if c in df.columns]
    
    def calculate_all(self, X: np.ndarray, labels: np.ndarray, k: int) -> Dict:
        """Calculate all metrics for a clustering result."""
        
        sizes = pd.Series(labels).value_counts().sort_index()
        n_total = len(labels)
        
        metrics = {
            'k': k,
            'n_clusters_actual': len(sizes),
            
            # Size metrics
            'min_cluster_size': int(sizes.min()),
            'max_cluster_size': int(sizes.max()),
            'mean_cluster_size': float(sizes.mean()),
            'size_balance': float(sizes.min() / sizes.max()),
            'size_std': float(sizes.std()),
            'smallest_cluster_pct': float(sizes.min() / n_total),
            
            # Geometric metrics
            'silhouette': float(silhouette_score(X, labels)),
            'silhouette_std': float(silhouette_samples(X, labels).std()),
            'calinski_harabasz': float(calinski_harabasz_score(X, labels)),
            'davies_bouldin': float(davies_bouldin_score(X, labels)),
            
            # Behavioral validation metrics
            'eta_squared_mean': self._calculate_eta_squared_mean(labels),
            'eta_squared_by_outcome': self._calculate_eta_squared_by_outcome(labels),
            'anova_significant_outcomes': self._count_significant_outcomes(labels),
            
            # Cluster sizes
            'cluster_sizes': {int(i): int(s) for i, s in sizes.items()}
        }
        
        return metrics
    
    def _calculate_eta_squared_mean(self, labels: np.ndarray) -> float:
        """Calculate mean eta-squared across all outcomes."""
        eta2_values = []
        
        for outcome in self.outcome_cols:
            eta2 = self._calculate_eta_squared(labels, outcome)
            if eta2 is not None:
                eta2_values.append(eta2)
        
        return float(np.mean(eta2_values)) if eta2_values else 0.0
    
    def _calculate_eta_squared_by_outcome(self, labels: np.ndarray) -> Dict[str, float]:
        """Calculate eta-squared for each outcome."""
        results = {}
        for outcome in self.outcome_cols:
            eta2 = self._calculate_eta_squared(labels, outcome)
            if eta2 is not None:
                results[outcome] = float(eta2)
        return results
    
    def _calculate_eta_squared(self, labels: np.ndarray, outcome: str) -> Optional[float]:
        """Calculate eta-squared (effect size) for one outcome."""
        if outcome not in self.df.columns:
            return None
        
        values = self.df[outcome].values
        groups = pd.Series(labels)
        
        # Between-group sum of squares
        grand_mean = np.mean(values)
        group_means = self.df.groupby(groups)[outcome].mean()
        group_sizes = groups.value_counts()
        
        between_ss = sum(
            group_sizes[g] * (group_means[g] - grand_mean)**2 
            for g in group_means.index
        )
        
        # Total sum of squares
        total_ss = np.var(values) * len(values)
        
        if total_ss == 0:
            return 0.0
        
        return between_ss / total_ss
    
    def _count_significant_outcomes(self, labels: np.ndarray, alpha: float = 0.05) -> int:
        """Count outcomes with significant ANOVA results."""
        significant = 0
        
        for outcome in self.outcome_cols:
            if outcome not in self.df.columns:
                continue
            
            groups = [
                self.df[self.df.index.isin(np.where(labels == k)[0])][outcome].values
                for k in np.unique(labels)
            ]
            
            try:
                f_stat, p_value = stats.f_oneway(*groups)
                if p_value < alpha:
                    significant += 1
            except:
                pass
        
        return significant
