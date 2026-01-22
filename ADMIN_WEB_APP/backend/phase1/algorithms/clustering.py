import numpy as np
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn.mixture import GaussianMixture
from typing import Dict

class ClusteringAlgorithm:
    """Base class for clustering algorithms."""
    
    name: str = "base"
    display_name: str = "Base Algorithm"
    supports_soft: bool = False
    
    def fit_predict(self, X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
        raise NotImplementedError
    
    def get_model_info(self) -> Dict:
        return {}


class KMeansAlgorithm(ClusteringAlgorithm):
    name = "kmeans"
    display_name = "K-Means"
    supports_soft = False
    
    def fit_predict(self, X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
        self.model = KMeans(n_clusters=k, random_state=random_state, n_init=10, max_iter=300)
        labels = self.model.fit_predict(X)
        return labels
    
    def get_model_info(self) -> Dict:
        return {
            'inertia': float(self.model.inertia_),
            'n_iter': int(self.model.n_iter_)
        }


class GMMAlgorithm(ClusteringAlgorithm):
    name = "gmm"
    display_name = "Gaussian Mixture Model"
    supports_soft = True
    
    def fit_predict(self, X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
        self.model = GaussianMixture(
            n_components=k, random_state=random_state, 
            n_init=10, max_iter=200, covariance_type='full'
        )
        labels = self.model.fit_predict(X)
        self._X = X
        return labels
    
    def get_model_info(self) -> Dict:
        return {
            'bic': float(self.model.bic(self._X)),
            'aic': float(self.model.aic(self._X)),
            'converged': bool(self.model.converged_),
            'n_iter': int(self.model.n_iter_)
        }


class HierarchicalAlgorithm(ClusteringAlgorithm):
    name = "hierarchical"
    display_name = "Agglomerative (Ward)"
    supports_soft = False
    
    def fit_predict(self, X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
        self.model = AgglomerativeClustering(n_clusters=k, linkage='ward')
        labels = self.model.fit_predict(X)
        return labels
    
    def get_model_info(self) -> Dict:
        return {
            'n_leaves': int(self.model.n_leaves_),
            'n_connected_components': int(self.model.n_connected_components_)
        }


class SpectralAlgorithm(ClusteringAlgorithm):
    name = "spectral"
    display_name = "Spectral Clustering"
    supports_soft = False
    
    def fit_predict(self, X: np.ndarray, k: int, random_state: int = 42) -> np.ndarray:
        self.model = SpectralClustering(
            n_clusters=k, random_state=random_state,
            affinity='nearest_neighbors', n_neighbors=10
        )
        labels = self.model.fit_predict(X)
        return labels
    
    def get_model_info(self) -> Dict:
        return {}


ALGORITHMS = {
    'kmeans': KMeansAlgorithm(),
    'gmm': GMMAlgorithm(),
    'hierarchical': HierarchicalAlgorithm(),
    'spectral': SpectralAlgorithm()
}
