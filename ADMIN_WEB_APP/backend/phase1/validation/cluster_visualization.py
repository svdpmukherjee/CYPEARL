"""
Cluster Visualization Module

Provides 2D projections of high-dimensional clustering results for visualization.
Uses PCA for fast, interpretable projections or t-SNE for non-linear structure.

This helps users visually understand:
1. How well-separated clusters are
2. Which participants are on cluster boundaries
3. The overall structure of the persona space
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')


@dataclass
class VisualizationPoint:
    """A single point in the 2D visualization."""
    participant_idx: int
    x: float
    y: float
    cluster: int
    distance_to_centroid: float

    def to_dict(self) -> Dict:
        return {
            'participant_idx': self.participant_idx,
            'x': self.x,
            'y': self.y,
            'cluster': self.cluster,
            'distance_to_centroid': self.distance_to_centroid
        }


@dataclass
class ClusterCentroid:
    """Cluster centroid in 2D space."""
    cluster: int
    x: float
    y: float
    size: int  # Number of members

    def to_dict(self) -> Dict:
        return {
            'cluster': self.cluster,
            'x': self.x,
            'y': self.y,
            'size': self.size
        }


class ClusterVisualizationGenerator:
    """
    Generate 2D visualizations of clustering results.

    Supports:
    - PCA: Fast, linear, preserves global structure
    - t-SNE: Non-linear, preserves local structure (slower)
    """

    def __init__(self, random_state: int = 42):
        """Initialize generator."""
        self.random_state = random_state

    def generate(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        method: str = 'pca',
        perplexity: int = 30
    ) -> Dict:
        """
        Generate 2D visualization data.

        Args:
            X: Feature matrix (n_samples, n_features)
            labels: Cluster assignments
            method: 'pca' or 'tsne'
            perplexity: t-SNE perplexity (only used if method='tsne')

        Returns:
            Dictionary with points, centroids, and metadata
        """
        n_samples = X.shape[0]
        k = len(np.unique(labels))

        # Standardize features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Compute 2D projection
        if method == 'tsne':
            X_2d = self._compute_tsne(X_scaled, perplexity)
            method_description = f"t-SNE (perplexity={perplexity})"
        else:
            X_2d, explained_variance = self._compute_pca(X_scaled)
            method_description = f"PCA (explained variance: {explained_variance:.1%})"

        # Compute cluster centroids in 2D space
        centroids_2d = []
        for cluster_id in range(k):
            mask = labels == cluster_id
            if np.sum(mask) > 0:
                centroid_x = float(np.mean(X_2d[mask, 0]))
                centroid_y = float(np.mean(X_2d[mask, 1]))
                centroids_2d.append(ClusterCentroid(
                    cluster=cluster_id,
                    x=centroid_x,
                    y=centroid_y,
                    size=int(np.sum(mask))
                ))

        # Compute distances to centroids
        centroid_coords = {c.cluster: (c.x, c.y) for c in centroids_2d}

        # Create visualization points
        points = []
        for i in range(n_samples):
            cluster = int(labels[i])
            cx, cy = centroid_coords.get(cluster, (0, 0))
            dist = np.sqrt((X_2d[i, 0] - cx)**2 + (X_2d[i, 1] - cy)**2)

            points.append(VisualizationPoint(
                participant_idx=i,
                x=float(X_2d[i, 0]),
                y=float(X_2d[i, 1]),
                cluster=cluster,
                distance_to_centroid=float(dist)
            ))

        # Compute cluster statistics
        cluster_stats = self._compute_cluster_stats(points, centroids_2d)

        # Compute axis ranges for consistent display
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        padding = 0.1
        x_range = [min(x_coords) - padding * (max(x_coords) - min(x_coords)),
                   max(x_coords) + padding * (max(x_coords) - min(x_coords))]
        y_range = [min(y_coords) - padding * (max(y_coords) - min(y_coords)),
                   max(y_coords) + padding * (max(y_coords) - min(y_coords))]

        # Compute decision boundaries for visualization
        boundaries = self._compute_decision_boundaries(
            X_2d, labels, centroids_2d, x_range, y_range, k
        )

        return {
            'method': method,
            'method_description': method_description,
            'n_samples': n_samples,
            'k': k,
            'points': [p.to_dict() for p in points],
            'centroids': [c.to_dict() for c in centroids_2d],
            'cluster_stats': cluster_stats,
            'axis_ranges': {
                'x': x_range,
                'y': y_range
            },
            'decision_boundaries': boundaries,
            'interpretation': self._generate_interpretation(cluster_stats)
        }

    def _compute_decision_boundaries(
        self,
        X_2d: np.ndarray,
        labels: np.ndarray,
        centroids: List[ClusterCentroid],
        x_range: List[float],
        y_range: List[float],
        k: int,
        grid_resolution: int = 50
    ) -> Dict:
        """
        Compute decision boundaries for cluster visualization.
        
        Uses a mesh grid approach to create contour-like boundaries
        based on nearest centroid assignment (Voronoi-like regions).
        
        Args:
            X_2d: 2D projected coordinates
            labels: Cluster labels
            centroids: List of cluster centroids
            x_range: X-axis range [min, max]
            y_range: Y-axis range [min, max]
            k: Number of clusters
            grid_resolution: Number of grid points per axis
            
        Returns:
            Dictionary with boundary contour data for each cluster
        """
        # Create mesh grid
        x_min, x_max = x_range
        y_min, y_max = y_range
        
        xx = np.linspace(x_min, x_max, grid_resolution)
        yy = np.linspace(y_min, y_max, grid_resolution)
        
        # Get centroid coordinates
        centroid_coords = np.array([[c.x, c.y] for c in centroids])
        
        # Compute cluster assignments for the entire grid (Voronoi regions)
        grid_assignments = np.zeros((grid_resolution, grid_resolution), dtype=int)
        
        for i, x in enumerate(xx):
            for j, y in enumerate(yy):
                point = np.array([x, y])
                # Assign to nearest centroid
                distances = np.sqrt(np.sum((centroid_coords - point) ** 2, axis=1))
                grid_assignments[j, i] = int(np.argmin(distances))
        
        # Extract boundary contours for each cluster
        boundaries = {
            'grid_x': xx.tolist(),
            'grid_y': yy.tolist(),
            'grid_assignments': grid_assignments.tolist(),
            'contours': []
        }
        
        # Generate contour paths for each cluster boundary
        for cluster_id in range(k):
            # Create binary mask for this cluster
            mask = (grid_assignments == cluster_id).astype(int)
            
            # Find boundary points (where mask changes)
            contour_points = self._extract_contour_points(mask, xx, yy)
            
            if contour_points:
                boundaries['contours'].append({
                    'cluster': cluster_id,
                    'points': contour_points
                })
        
        # Also compute convex hull boundaries around actual data points
        hull_boundaries = self._compute_convex_hulls(X_2d, labels, k)
        boundaries['convex_hulls'] = hull_boundaries
        
        # Compute confidence ellipses (1 standard deviation = 68% for better visualization)
        # Using 68% (1σ) instead of 95% (2σ) to reduce overlap in 2D projection
        ellipse_boundaries = self._compute_confidence_ellipses(X_2d, labels, k, confidence=0.68)
        boundaries['confidence_ellipses'] = ellipse_boundaries
        
        return boundaries
    
    def _extract_contour_points(
        self,
        mask: np.ndarray,
        xx: np.ndarray,
        yy: np.ndarray
    ) -> List[Dict]:
        """
        Extract contour boundary points from a binary mask.
        
        Finds edge pixels where the cluster region meets other clusters.
        """
        from scipy import ndimage
        
        # Find edges using gradient
        edge_y, edge_x = np.gradient(mask.astype(float))
        edges = np.sqrt(edge_y**2 + edge_x**2) > 0.1
        
        # Get edge coordinates
        edge_indices = np.where(edges)
        
        if len(edge_indices[0]) == 0:
            return []
        
        # Convert to actual coordinates
        points = []
        for j, i in zip(edge_indices[0], edge_indices[1]):
            if i < len(xx) and j < len(yy):
                points.append({
                    'x': float(xx[i]),
                    'y': float(yy[j])
                })
        
        return points
    
    def _compute_confidence_ellipses(
        self,
        X_2d: np.ndarray,
        labels: np.ndarray,
        k: int,
        confidence: float = 0.95,
        n_points: int = 100
    ) -> List[Dict]:
        """
        Compute confidence ellipses around each cluster.
        
        These are 2D Gaussian confidence ellipses based on the covariance
        of each cluster's points. They provide smooth, interpretable boundaries
        like those seen in ML textbooks.
        
        Args:
            X_2d: 2D projected coordinates
            labels: Cluster labels
            k: Number of clusters
            confidence: Confidence level (0.95 = 95% of points inside)
            n_points: Number of points to generate for ellipse path
            
        Returns:
            List of ellipse dictionaries with path, center, and parameters
        """
        from scipy import stats
        
        # Chi-squared value for 2 degrees of freedom at given confidence
        chi2_val = stats.chi2.ppf(confidence, 2)
        
        ellipses = []
        
        for cluster_id in range(k):
            cluster_points = X_2d[labels == cluster_id]
            
            if len(cluster_points) < 3:
                continue
            
            try:
                # Compute mean (center)
                center = np.mean(cluster_points, axis=0)
                
                # Compute covariance matrix
                cov = np.cov(cluster_points.T)
                
                # Handle 1D case (shouldn't happen but just in case)
                if cov.ndim == 0:
                    cov = np.array([[cov, 0], [0, cov]])
                
                # Eigenvalue decomposition for ellipse orientation
                eigenvalues, eigenvectors = np.linalg.eigh(cov)
                
                # Sort eigenvalues and eigenvectors in descending order
                order = eigenvalues.argsort()[::-1]
                eigenvalues = eigenvalues[order]
                eigenvectors = eigenvectors[:, order]
                
                # Compute ellipse parameters
                # Semi-axes lengths scaled by chi-squared value
                a = np.sqrt(chi2_val * eigenvalues[0])  # Semi-major axis
                b = np.sqrt(chi2_val * eigenvalues[1])  # Semi-minor axis
                
                # Rotation angle (in radians)
                angle = np.arctan2(eigenvectors[1, 0], eigenvectors[0, 0])
                
                # Generate ellipse path points
                theta = np.linspace(0, 2 * np.pi, n_points)
                
                # Ellipse in standard position
                ellipse_x = a * np.cos(theta)
                ellipse_y = b * np.sin(theta)
                
                # Rotation matrix
                cos_angle = np.cos(angle)
                sin_angle = np.sin(angle)
                
                # Rotate and translate to actual position
                x_rotated = cos_angle * ellipse_x - sin_angle * ellipse_y + center[0]
                y_rotated = sin_angle * ellipse_x + cos_angle * ellipse_y + center[1]
                
                # Create path
                path = [{'x': float(x_rotated[i]), 'y': float(y_rotated[i])} 
                        for i in range(n_points)]
                # Close the path
                path.append({'x': float(x_rotated[0]), 'y': float(y_rotated[0])})
                
                ellipses.append({
                    'cluster': cluster_id,
                    'path': path,
                    'center': {'x': float(center[0]), 'y': float(center[1])},
                    'semi_major': float(a),
                    'semi_minor': float(b),
                    'rotation_angle': float(np.degrees(angle)),
                    'confidence': confidence,
                    'n_points_in_cluster': len(cluster_points)
                })
                
            except Exception as e:
                # Ellipse computation can fail for degenerate cases
                print(f"Warning: Could not compute ellipse for cluster {cluster_id}: {e}")
                continue
        
        return ellipses
    
    def _compute_convex_hulls(
        self,
        X_2d: np.ndarray,
        labels: np.ndarray,
        k: int
    ) -> List[Dict]:
        """
        Compute convex hull boundaries around actual cluster data points.
        
        This provides tighter boundaries that show where the actual data lies,
        complementing the Voronoi-like decision boundaries.
        """
        from scipy.spatial import ConvexHull
        
        hulls = []
        
        for cluster_id in range(k):
            cluster_points = X_2d[labels == cluster_id]
            
            if len(cluster_points) < 3:
                # Not enough points for a hull
                continue
            
            try:
                hull = ConvexHull(cluster_points)
                hull_vertices = cluster_points[hull.vertices]
                
                # Close the hull by adding first point at the end
                hull_path = []
                for point in hull_vertices:
                    hull_path.append({'x': float(point[0]), 'y': float(point[1])})
                hull_path.append({'x': float(hull_vertices[0, 0]), 'y': float(hull_vertices[0, 1])})
                
                hulls.append({
                    'cluster': cluster_id,
                    'path': hull_path,
                    'area': float(hull.volume)  # In 2D, volume is area
                })
            except Exception:
                # Hull computation can fail for degenerate cases
                continue
        
        return hulls

    def _compute_pca(self, X: np.ndarray) -> Tuple[np.ndarray, float]:
        """Compute PCA projection to 2D."""
        pca = PCA(n_components=2, random_state=self.random_state)
        X_2d = pca.fit_transform(X)
        explained_variance = sum(pca.explained_variance_ratio_)
        return X_2d, explained_variance

    def _compute_tsne(self, X: np.ndarray, perplexity: int) -> np.ndarray:
        """Compute t-SNE projection to 2D."""
        # Adjust perplexity if dataset is small
        n_samples = X.shape[0]
        actual_perplexity = min(perplexity, max(5, n_samples // 4))

        tsne = TSNE(
            n_components=2,
            perplexity=actual_perplexity,
            random_state=self.random_state,
            n_iter=1000,
            learning_rate='auto',
            init='pca'
        )
        return tsne.fit_transform(X)

    def _compute_cluster_stats(
        self,
        points: List[VisualizationPoint],
        centroids: List[ClusterCentroid]
    ) -> Dict:
        """Compute statistics for each cluster in 2D space."""
        stats = {}

        for centroid in centroids:
            cluster_points = [p for p in points if p.cluster == centroid.cluster]
            distances = [p.distance_to_centroid for p in cluster_points]

            if distances:
                stats[centroid.cluster] = {
                    'n_members': len(cluster_points),
                    'mean_distance_to_centroid': float(np.mean(distances)),
                    'max_distance_to_centroid': float(np.max(distances)),
                    'std_distance': float(np.std(distances)),
                    'compactness': 'tight' if np.std(distances) < np.mean(distances) * 0.5 else 'spread'
                }

        return stats

    def _generate_interpretation(self, cluster_stats: Dict) -> str:
        """Generate human-readable interpretation of visualization."""
        n_clusters = len(cluster_stats)

        # Check cluster compactness
        tight_clusters = sum(1 for s in cluster_stats.values() if s['compactness'] == 'tight')
        spread_clusters = n_clusters - tight_clusters

        if tight_clusters == n_clusters:
            compactness_msg = "All clusters are compact and well-defined."
        elif tight_clusters > spread_clusters:
            compactness_msg = f"Most clusters ({tight_clusters}/{n_clusters}) are compact. Some show more spread."
        else:
            compactness_msg = f"Several clusters show spread patterns, indicating within-cluster variability."

        # Check for size imbalance
        sizes = [s['n_members'] for s in cluster_stats.values()]
        size_ratio = max(sizes) / min(sizes) if min(sizes) > 0 else float('inf')

        if size_ratio < 2:
            size_msg = "Cluster sizes are relatively balanced."
        elif size_ratio < 5:
            size_msg = "Some cluster size imbalance exists."
        else:
            size_msg = "Significant cluster size imbalance detected."

        return f"{compactness_msg} {size_msg}"

    def generate_with_features(
        self,
        X: np.ndarray,
        labels: np.ndarray,
        feature_names: List[str],
        method: str = 'pca'
    ) -> Dict:
        """
        Generate visualization with feature contribution info.

        For PCA, includes which original features contribute most to each axis.
        """
        result = self.generate(X, labels, method)

        if method == 'pca':
            # Get PCA loadings
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            pca = PCA(n_components=2, random_state=self.random_state)
            pca.fit(X_scaled)

            # Top features for each component
            loadings = pca.components_

            # PC1 top features
            pc1_idx = np.argsort(np.abs(loadings[0]))[::-1][:5]
            pc1_features = [
                {'feature': feature_names[i], 'loading': float(loadings[0, i])}
                for i in pc1_idx
            ]

            # PC2 top features
            pc2_idx = np.argsort(np.abs(loadings[1]))[::-1][:5]
            pc2_features = [
                {'feature': feature_names[i], 'loading': float(loadings[1, i])}
                for i in pc2_idx
            ]

            result['axis_features'] = {
                'x_axis': {
                    'name': 'PC1',
                    'explained_variance': float(pca.explained_variance_ratio_[0]),
                    'top_features': pc1_features
                },
                'y_axis': {
                    'name': 'PC2',
                    'explained_variance': float(pca.explained_variance_ratio_[1]),
                    'top_features': pc2_features
                }
            }

        return result
