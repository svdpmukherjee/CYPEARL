import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from typing import List, Dict, Optional

# Import the Phase 1 specific config
from core.config import portal_config

# Try to import phase1 constants, with fallback
try:
    from phase1.constants import GET_ALL_CLUSTERING_FEATURES
except ImportError:
    def GET_ALL_CLUSTERING_FEATURES():
        return [
            'crt_score', 'need_for_cognition', 'working_memory',
            'big5_extraversion', 'big5_agreeableness', 'big5_conscientiousness',
            'big5_neuroticism', 'big5_openness', 'impulsivity_total',
            'sensation_seeking', 'trust_propensity', 'risk_taking',
            'state_anxiety', 'current_stress', 'fatigue_level',
            'phishing_self_efficacy', 'perceived_risk', 'security_attitudes',
            'privacy_concern', 'phishing_knowledge', 'technical_expertise',
            'prior_victimization', 'security_training', 'email_volume_numeric',
            'link_click_tendency', 'social_media_usage',
            'authority_susceptibility', 'urgency_susceptibility', 'scarcity_susceptibility'
        ]


class DataPreprocessor:
    """Preprocess data for clustering."""
    
    def __init__(self, config=None):
        """
        Initialize preprocessor.
        
        Args:
            config: Configuration object. If None, uses portal_config (Phase 1 config).
                   Must have: use_pca, pca_variance, random_state attributes.
        """
        # Use Phase 1 config by default since this is a Phase 1 component
        self.config = config if config is not None else portal_config
        self.scaler = None
        self.pca = None
        self.imputer = None
        self.feature_names = None
        self.preprocessing_stats = {}
        self.is_fitted = False
    
    def fit_transform(
        self, 
        df: pd.DataFrame, 
        features: List[str] = None,
        use_pca: Optional[bool] = None,
        pca_variance: Optional[float] = None
    ) -> np.ndarray:
        """
        Preprocess data for clustering.
        
        Args:
            df: Input dataframe
            features: List of feature names to use. If None, uses all clustering features.
            use_pca: Whether to apply PCA. If None, uses config value.
            pca_variance: Variance to retain with PCA. If None, uses config value.
        
        Returns:
            Preprocessed numpy array
        """
        
        # Get features
        if features is None:
            all_features = GET_ALL_CLUSTERING_FEATURES()
            features = [f for f in all_features if f in df.columns]

        # Filter to only numeric columns
        numeric_features = [
            f for f in features
            if f in df.columns and pd.api.types.is_numeric_dtype(df[f])
        ]

        # Log any skipped non-numeric columns
        skipped = [f for f in features if f not in numeric_features]
        if skipped:
            print(f"[PREPROCESSOR] Skipped non-numeric columns: {skipped}")

        self.feature_names = numeric_features

        if not numeric_features:
            raise ValueError("No numeric features available for preprocessing")

        X = df[numeric_features].copy()
        
        # Impute missing values
        self.imputer = SimpleImputer(strategy='median')
        X_imputed = pd.DataFrame(
            self.imputer.fit_transform(X),
            columns=features
        )
        
        # Winsorize extreme values (1st-99th percentile)
        for col in features:
            lower, upper = X_imputed[col].quantile([0.01, 0.99])
            X_imputed[col] = X_imputed[col].clip(lower, upper)
        
        # Scale using RobustScaler (less sensitive to outliers)
        self.scaler = RobustScaler()
        X_scaled = self.scaler.fit_transform(X_imputed)
        
        self.preprocessing_stats['n_features_original'] = len(features)
        
        # Determine PCA settings (allow override via parameters)
        apply_pca = use_pca if use_pca is not None else self.config.use_pca
        variance = pca_variance if pca_variance is not None else self.config.pca_variance
        random_state = getattr(self.config, 'random_state', 42)
        
        # Optional PCA
        if apply_pca:
            self.pca = PCA(n_components=variance, random_state=random_state)
            X_pca = self.pca.fit_transform(X_scaled)
            
            self.preprocessing_stats['n_components_pca'] = X_pca.shape[1]
            self.preprocessing_stats['variance_explained'] = float(self.pca.explained_variance_ratio_.sum())
            self.preprocessing_stats['use_pca'] = True
            
            # Save stats
            print(f"PCA: {X_scaled.shape[1]} features -> {X_pca.shape[1]} components")
            
            self.is_fitted = True
            return X_pca
        
        self.preprocessing_stats['use_pca'] = False
        self.is_fitted = True
        return X_scaled
    
    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted preprocessors."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor not fitted")
            
        X = df[self.feature_names].copy()
        X_imputed = self.imputer.transform(X)
        X_scaled = self.scaler.transform(X_imputed)
        
        if self.preprocessing_stats.get('use_pca', False) and self.pca is not None:
            return self.pca.transform(X_scaled)
        return X_scaled

    def get_stats(self) -> Dict:
        """Get preprocessing statistics."""
        return self.preprocessing_stats