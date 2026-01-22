import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from typing import Dict
from core.config import config
from phase1.algorithms.clustering import ClusteringAlgorithm

class StabilityAnalyzer:
    """Analyze clustering stability via bootstrap."""
    
    def __init__(self):
        self.config = config
    
    def analyze(self, X: np.ndarray, algorithm: ClusteringAlgorithm, k: int) -> Dict:
        """Run bootstrap stability analysis."""
        
        n_samples = X.shape[0]
        sample_size = int(n_samples * self.config.bootstrap_sample_ratio)
        
        silhouette_scores = []
        size_balances = []
        
        # Start seed
        seed_start = self.config.random_state
        
        for i in range(self.config.n_bootstrap):
            # Bootstrap sample
            np.random.seed(seed_start + i)
            indices = np.random.choice(n_samples, size=sample_size, replace=True)
            X_boot = X[indices]
            
            try:
                # We need to implement fit_predict on the algorithm instance
                # Since algorithm instances are reused, we should be careful not to mutate state if we were multithreading
                # But here it's fine.
                labels = algorithm.fit_predict(X_boot, k, seed_start + i)
                
                if len(np.unique(labels)) == k:
                    sil = silhouette_score(X_boot, labels)
                    sizes = pd.Series(labels).value_counts()
                    balance = sizes.min() / sizes.max()
                    
                    silhouette_scores.append(sil)
                    size_balances.append(balance)
            except Exception as e:
                # Silent failure for bootstrap iterations
                pass
        
        if len(silhouette_scores) < 10:
            return {
                'silhouette_mean': np.nan,
                'silhouette_std': np.nan,
                'silhouette_ci_lower': np.nan,
                'silhouette_ci_upper': np.nan,
                'stability_score': 0.0,
                'n_successful': len(silhouette_scores)
            }
        
        return {
            'silhouette_mean': float(np.mean(silhouette_scores)),
            'silhouette_std': float(np.std(silhouette_scores)),
            'silhouette_ci_lower': float(np.percentile(silhouette_scores, 2.5)),
            'silhouette_ci_upper': float(np.percentile(silhouette_scores, 97.5)),
            'size_balance_mean': float(np.mean(size_balances)),
            'size_balance_std': float(np.std(size_balances)),
            'stability_score': float(1 - np.std(silhouette_scores)),
            'n_successful': len(silhouette_scores)
        }
