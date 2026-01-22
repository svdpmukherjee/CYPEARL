import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, List, Optional
from phase1.constants import OUTCOME_FEATURES, DEMOGRAPHIC_FEATURES

class ClusterCharacterizer:
    """Generate detailed cluster characterizations."""
    
    def __init__(self, df: pd.DataFrame, feature_names: List[str]):
        self.df = df
        self.feature_names = [f for f in feature_names if f in df.columns]
        self.outcome_cols = [c for c in OUTCOME_FEATURES if c in df.columns]
    
    def characterize(self, labels: np.ndarray) -> Dict[int, Dict]:
        """Generate characterization for each cluster."""
        
        df = self.df.copy()
        df['cluster'] = labels
        
        clusters = {}
        
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            
            # Basic stats
            n = len(cluster_data)
            pct = n / len(df) * 100
            
            # Trait z-scores (relative to full population)
            trait_zscores = {}
            trait_rankings = {}
            
            for feat in self.feature_names:
                cluster_mean = cluster_data[feat].mean()
                pop_mean = df[feat].mean()
                pop_std = df[feat].std()
                
                if pop_std > 0:
                    z = (cluster_mean - pop_mean) / pop_std
                    trait_zscores[feat] = float(z)
                    
                    # Percentile of cluster mean in population
                    percentile = stats.percentileofscore(df[feat], cluster_mean)
                    trait_rankings[feat] = float(percentile)
                else:
                    trait_zscores[feat] = 0.0
                    trait_rankings[feat] = 50.0
            
            # Behavioral outcomes
            outcomes = {}
            for outcome in self.outcome_cols:
                outcomes[outcome] = {
                    'mean': float(cluster_data[outcome].mean()),
                    'std': float(cluster_data[outcome].std()),
                    'median': float(cluster_data[outcome].median())
                }
            
            # Risk level based on phishing click rate
            click_rate = float(cluster_data['phishing_click_rate'].mean()) if 'phishing_click_rate' in df.columns else 0.3
            if click_rate >= 0.35:
                risk_level = 'CRITICAL'
            elif click_rate >= 0.28:
                risk_level = 'HIGH'
            elif click_rate >= 0.22:
                risk_level = 'MEDIUM'
            else:
                risk_level = 'LOW'
            
            # Top distinguishing traits
            sorted_traits = sorted(trait_zscores.items(), key=lambda x: abs(x[1]), reverse=True)
            top_high = [(k, v) for k, v in sorted_traits if v > 0.5][:3]
            top_low = [(k, v) for k, v in sorted_traits if v < -0.5][:3]
            
            # Generate description
            description = self._generate_description(cluster_id, n, pct, click_rate, 
                                                     risk_level, top_high, top_low)
            
            # Demographics (if available)
            demographics = {}
            for demo in DEMOGRAPHIC_FEATURES:
                if demo in df.columns:
                    if df[demo].dtype == 'object':
                        demographics[demo] = cluster_data[demo].value_counts(normalize=True).head(3).to_dict()
                    else:
                        demographics[demo] = {
                            'mean': float(cluster_data[demo].mean()),
                            'std': float(cluster_data[demo].std())
                        }
            
            clusters[int(cluster_id)] = {
                'cluster_id': int(cluster_id),
                'n_participants': n,
                'pct_of_population': float(pct),
                'risk_level': risk_level,
                'phishing_click_rate': float(click_rate),
                'trait_zscores': trait_zscores,
                'trait_percentiles': trait_rankings,
                'top_high_traits': top_high,
                'top_low_traits': top_low,
                'behavioral_outcomes': outcomes,
                'demographics': demographics,
                'description': description
            }
        
        return clusters
    
    def _generate_description(self, cluster_id: int, n: int, pct: float,
                              click_rate: float, risk_level: str,
                              top_high: List, top_low: List) -> str:
        """Generate human-readable cluster description."""
        
        trait_labels = {
            'crt_score': 'analytical thinking',
            'impulsivity_total': 'impulsivity',
            'trust_propensity': 'trust in others',
            'phishing_knowledge': 'phishing awareness',
            'technical_expertise': 'technical expertise',
            'big5_conscientiousness': 'conscientiousness',
            'big5_neuroticism': 'anxiety/neuroticism',
            'big5_extraversion': 'extraversion',
            'big5_agreeableness': 'agreeableness',
            'big5_openness': 'openness',
            'state_anxiety': 'current anxiety',
            'urgency_susceptibility': 'urgency susceptibility',
            'link_click_tendency': 'link clicking habit',
            'security_attitudes': 'security mindset'
        }
        
        high_desc = []
        for trait, z in top_high[:2]:
            label = trait_labels.get(trait, trait.replace('_', ' '))
            high_desc.append(f"high {label}")
        
        low_desc = []
        for trait, z in top_low[:2]:
            label = trait_labels.get(trait, trait.replace('_', ' '))
            low_desc.append(f"low {label}")
        
        traits_desc = " and ".join(high_desc + low_desc) if high_desc or low_desc else "average traits"
        
        return f"{traits_desc}"
