import numpy as np
import pandas as pd
import time
import sys
from typing import Dict, List, Any

# Import Phase 1 specific config
from core.config import portal_config
from phase1.algorithms.clustering import ALGORITHMS
from phase1.metrics.calculator import MetricsCalculator
from phase1.constants import OUTCOME_FEATURES


class ClusteringOptimizer:
    """Optimize clustering configuration by sweeping parameters."""
    
    def __init__(self, df: pd.DataFrame, preprocessor, feature_names: List[str], config=None):
        """
        Initialize optimizer.
        
        Args:
            df: Participant dataframe
            preprocessor: DataPreprocessor instance
            feature_names: List of feature names to use
            config: Configuration object. If None, uses portal_config (Phase 1 config).
        """
        self.df = df
        self.preprocessor = preprocessor
        self.feature_names = feature_names
        # Use Phase 1 config by default since this is a Phase 1 component
        self.config = config if config is not None else portal_config
        
    def optimize(
        self, 
        algorithms: List[str], 
        k_min: int, 
        k_max: int, 
        use_pca: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Sweep K values for multiple algorithms and return metrics.
        
        Args:
            algorithms: List of algorithm names to test
            k_min: Minimum number of clusters
            k_max: Maximum number of clusters
            use_pca: Whether to use PCA for dimensionality reduction
            
        Returns:
            Dictionary mapping algorithm names to lists of results
        """
        
        results = {}
        total_start = time.time()

        # 1. Preprocess (once)
        print(f"[OPTIMIZER] Preprocessing {len(self.df)} samples with {len(self.feature_names)} features...")
        sys.stdout.flush()

        preprocess_start = time.time()
        # Update the preprocessor's config if it has one
        if hasattr(self.preprocessor, 'config'):
            self.preprocessor.config.use_pca = use_pca

        X = self.preprocessor.fit_transform(self.df, self.feature_names, use_pca=use_pca)
        print(f"[OPTIMIZER] Preprocessing done in {time.time() - preprocess_start:.2f}s. Shape: {X.shape}")
        sys.stdout.flush()

        # 2. Metrics Calculator
        metrics_calc = MetricsCalculator(self.df, OUTCOME_FEATURES)

        total_iterations = len(algorithms) * (k_max - k_min + 1)
        current_iteration = 0
        print(f"[OPTIMIZER] Starting {total_iterations} iterations ({len(algorithms)} algorithms × {k_max - k_min + 1} K values)")
        sys.stdout.flush()
        
        # Get config values with defaults
        random_state = getattr(self.config, 'random_state', 42)
        min_cluster_size = getattr(self.config, 'min_cluster_size', 30)
        min_cluster_pct = getattr(self.config, 'min_cluster_pct', 0.03)
        w_behavioral = getattr(self.config, 'w_behavioral', 0.35)
        w_silhouette = getattr(self.config, 'w_silhouette', 0.25)
        w_stability = getattr(self.config, 'w_stability', 0.20)
        w_statistical = getattr(self.config, 'w_statistical', 0.20)
        
        # 3. Sweep
        for algo_name in algorithms:
            if algo_name not in ALGORITHMS:
                print(f"[OPTIMIZER] Warning: Unknown algorithm '{algo_name}', skipping")
                continue

            algo = ALGORITHMS[algo_name]
            algo_results = []
            algo_start = time.time()
            print(f"\n[OPTIMIZER] === Algorithm: {algo_name.upper()} ===")
            sys.stdout.flush()

            for k in range(k_min, k_max + 1):
                current_iteration += 1
                iter_start = time.time()

                try:
                    # Fit
                    labels = algo.fit_predict(X, k, random_state)

                    # Calculate Metrics
                    m = metrics_calc.calculate_all(X, labels, k)
                    
                    # Calculate Composite Score
                    # Behavioral
                    beh_score = min(m.get('eta_squared_mean', 0) / 0.15, 1.0)
                    
                    # Silhouette
                    sil_score = (m.get('silhouette', 0) + 1) / 2
                    
                    # Stability/Size
                    size_ok = 1.0 if m['min_cluster_size'] >= min_cluster_size else 0.5
                    pct_ok = 1.0 if m['smallest_cluster_pct'] >= min_cluster_pct else 0.5
                    stab_score = m['size_balance'] * size_ok * pct_ok
                    
                    # Statistical
                    ch = m.get('calinski_harabasz', 0)
                    ch_norm = min(np.log1p(ch) / 8, 1.0)
                    db = m.get('davies_bouldin', 2)
                    db_norm = 1 / (1 + db)
                    stat_score = (ch_norm + db_norm) / 2
                    
                    # Calculate composite score using weights
                    composite = (
                        w_behavioral * beh_score +
                        w_silhouette * sil_score +
                        w_stability * stab_score +
                        w_statistical * stat_score
                    )

                    result = {
                        'k': k,
                        'algorithm': algo_name,
                        'composite_score': float(composite),
                        'silhouette': float(m['silhouette']),
                        'calinski_harabasz': float(m['calinski_harabasz']),
                        'davies_bouldin': float(m['davies_bouldin']),
                        'eta_squared_mean': float(m['eta_squared_mean']),
                        'min_cluster_size': int(m['min_cluster_size']),
                        'size_balance': float(m['size_balance']),
                        # Component scores for debugging
                        'score_behavioral': float(beh_score),
                        'score_silhouette': float(sil_score),
                        'score_stability': float(stab_score),
                        'score_statistical': float(stat_score),
                    }
                    algo_results.append(result)

                    iter_time = time.time() - iter_start
                    progress_pct = (current_iteration / total_iterations) * 100
                    print(f"[OPTIMIZER] {algo_name} K={k}: score={composite:.4f}, η²={m['eta_squared_mean']:.3f}, sil={m['silhouette']:.3f} ({iter_time:.2f}s) [{progress_pct:.0f}%]")
                    sys.stdout.flush()

                except Exception as e:
                    print(f"[OPTIMIZER] ERROR for {algo_name} k={k}: {e}")
                    import traceback
                    traceback.print_exc()

            algo_time = time.time() - algo_start
            print(f"[OPTIMIZER] {algo_name} completed: {len(algo_results)} results in {algo_time:.2f}s")
            sys.stdout.flush()

            results[algo_name] = algo_results

        total_time = time.time() - total_start
        print(f"\n[OPTIMIZER] All algorithms completed in {total_time:.2f}s")
        sys.stdout.flush()

        return results