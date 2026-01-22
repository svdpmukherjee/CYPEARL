import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats

class EmailInteractionAnalyzer:
    """Analyze cluster × email type interactions with comprehensive metrics."""
    
    def __init__(self, responses_df: pd.DataFrame):
        self.responses = responses_df.copy()
    
    def analyze(self, participant_df: pd.DataFrame, labels: np.ndarray) -> Dict[str, Any]:
        """
        Analyze how clusters respond to different email types.
        
        Returns a comprehensive analysis including:
        - Click rates by cluster × urgency
        - Click rates by cluster × familiarity
        - Click rates by cluster × framing
        - Click rates by cluster × email
        - Interaction effect sizes
        - Statistical significance tests
        """
        
        # Add cluster labels to participants
        participants = participant_df.copy()
        participants['cluster'] = labels
        
        # Merge with responses
        if 'participant_id' not in self.responses.columns or 'participant_id' not in participants.columns:
            return {"error": "Missing participant_id column"}
        
        responses = self.responses.merge(
            participants[['participant_id', 'cluster']],
            on='participant_id'
        )
        
        # Filter to phishing emails if such column exists
        if 'email_type' in responses.columns:
            phishing = responses[responses['email_type'] == 'phishing'].copy()
            legitimate = responses[responses['email_type'] == 'legitimate'].copy()
        else:
            phishing = responses.copy()
            legitimate = pd.DataFrame()
        
        results = {
            'by_urgency': {},
            'by_familiarity': {},
            'by_framing': {},
            'by_email': {},
            'by_quality': {},
            'interaction_effects': {},
            'statistics': {},
            'summary': {}
        }
        
        n_clusters = len(np.unique(labels))
        
        # ====================================================================
        # Click rates by cluster × urgency
        # ====================================================================
        if 'urgency_level' in phishing.columns and 'clicked' in phishing.columns:
            try:
                pivot = phishing.groupby(['cluster', 'urgency_level'])['clicked'].mean().unstack()
                results['by_urgency'] = self._pivot_to_nested_dict(pivot)
                
                # Calculate urgency effect (High - Low)
                if 'high' in pivot.columns and 'low' in pivot.columns:
                    urgency_effect = pivot['high'] - pivot['low']
                    results['interaction_effects']['urgency_effect'] = urgency_effect.to_dict()
                    
                    # Statistical test for urgency × cluster interaction
                    results['statistics']['urgency_interaction'] = self._test_interaction(
                        phishing, 'urgency_level', 'cluster', 'clicked'
                    )
            except Exception as e:
                results['by_urgency'] = {"error": str(e)}
        
        # ====================================================================
        # Click rates by cluster × familiarity
        # ====================================================================
        if 'sender_familiarity' in phishing.columns and 'clicked' in phishing.columns:
            try:
                pivot = phishing.groupby(['cluster', 'sender_familiarity'])['clicked'].mean().unstack()
                results['by_familiarity'] = self._pivot_to_nested_dict(pivot)
                
                # Calculate familiarity effect (Familiar - Unfamiliar)
                if 'familiar' in pivot.columns and 'unfamiliar' in pivot.columns:
                    fam_effect = pivot['familiar'] - pivot['unfamiliar']
                    results['interaction_effects']['familiarity_effect'] = fam_effect.to_dict()
                    
                    results['statistics']['familiarity_interaction'] = self._test_interaction(
                        phishing, 'sender_familiarity', 'cluster', 'clicked'
                    )
            except Exception as e:
                results['by_familiarity'] = {"error": str(e)}
        
        # ====================================================================
        # Click rates by cluster × framing
        # ====================================================================
        if 'framing_type' in phishing.columns and 'clicked' in phishing.columns:
            try:
                pivot = phishing.groupby(['cluster', 'framing_type'])['clicked'].mean().unstack()
                results['by_framing'] = self._pivot_to_nested_dict(pivot)
                
                # Calculate framing effect (Threat - Reward or vice versa)
                if 'threat' in pivot.columns and 'reward' in pivot.columns:
                    framing_effect = pivot['threat'] - pivot['reward']
                    results['interaction_effects']['framing_effect'] = framing_effect.to_dict()
                    
                    results['statistics']['framing_interaction'] = self._test_interaction(
                        phishing, 'framing_type', 'cluster', 'clicked'
                    )
            except Exception as e:
                results['by_framing'] = {"error": str(e)}
        
        # ====================================================================
        # Click rates by cluster × email (complete matrix)
        # ====================================================================
        if 'email_id' in phishing.columns and 'clicked' in phishing.columns:
            try:
                email_pivot = phishing.groupby(['cluster', 'email_id'])['clicked'].mean().unstack()
                # Convert to {email_id: {cluster: rate}} format
                results['by_email'] = {
                    str(email): {int(cluster): float(rate) for cluster, rate in email_pivot[email].items() if pd.notna(rate)}
                    for email in email_pivot.columns
                }
            except Exception as e:
                results['by_email'] = {"error": str(e)}
        
        # ====================================================================
        # Click rates by cluster × phishing quality (if available)
        # ====================================================================
        if 'phishing_quality' in phishing.columns and 'clicked' in phishing.columns:
            try:
                pivot = phishing.groupby(['cluster', 'phishing_quality'])['clicked'].mean().unstack()
                results['by_quality'] = self._pivot_to_nested_dict(pivot)
                
                # Effect of quality
                if 'high' in pivot.columns and 'low' in pivot.columns:
                    quality_effect = pivot['high'] - pivot['low']
                    results['interaction_effects']['quality_effect'] = quality_effect.to_dict()
            except Exception as e:
                results['by_quality'] = {"error": str(e)}
        
        # ====================================================================
        # Summary statistics per cluster
        # ====================================================================
        cluster_summary = []
        for cluster_id in sorted(phishing['cluster'].unique()):
            cluster_data = phishing[phishing['cluster'] == cluster_id]
            
            summary = {
                'cluster': int(cluster_id),
                'n_responses': len(cluster_data),
                'overall_click_rate': float(cluster_data['clicked'].mean()),
                'click_rate_std': float(cluster_data['clicked'].std())
            }
            
            # Add response time if available
            if 'response_latency_ms' in cluster_data.columns:
                summary['mean_response_time'] = float(cluster_data['response_latency_ms'].mean())
                summary['response_time_std'] = float(cluster_data['response_latency_ms'].std())
            
            # Add behavioral indicators
            if 'hovered_link' in cluster_data.columns:
                summary['hover_rate'] = float(cluster_data['hovered_link'].mean())
            if 'inspected_sender' in cluster_data.columns:
                summary['sender_inspection_rate'] = float(cluster_data['inspected_sender'].mean())
            if 'confidence_rating' in cluster_data.columns:
                summary['mean_confidence'] = float(cluster_data['confidence_rating'].mean())
            if 'suspicion_rating' in cluster_data.columns:
                summary['mean_suspicion'] = float(cluster_data['suspicion_rating'].mean())
            
            cluster_summary.append(summary)
        
        results['summary'] = {
            'n_clusters': n_clusters,
            'n_phishing_emails': phishing['email_id'].nunique() if 'email_id' in phishing.columns else 0,
            'total_responses': len(phishing),
            'cluster_summary': cluster_summary
        }
        
        # ====================================================================
        # Vulnerability profiles per cluster
        # ====================================================================
        results['vulnerability_profiles'] = self._compute_vulnerability_profiles(
            results['interaction_effects'],
            n_clusters
        )
        
        return results
    
    def _pivot_to_nested_dict(self, pivot: pd.DataFrame) -> Dict:
        """Convert pivot table to nested dictionary format."""
        result = {}
        for col in pivot.columns:
            result[col] = {}
            for idx in pivot.index:
                val = pivot.loc[idx, col]
                if pd.notna(val):
                    result[col][int(idx)] = float(val)
        return result
    
    def _test_interaction(self, data: pd.DataFrame, factor: str, cluster_col: str, outcome: str) -> Dict:
        """Test for interaction effect using two-way ANOVA approximation."""
        try:
            from scipy import stats
            
            # Simple approach: compare effect sizes across clusters
            clusters = data[cluster_col].unique()
            factor_levels = data[factor].unique()
            
            if len(factor_levels) < 2:
                return {"significant": False, "p_value": 1.0, "note": "Insufficient factor levels"}
            
            # Calculate effect per cluster
            effects = []
            for c in clusters:
                cluster_data = data[data[cluster_col] == c]
                level_means = cluster_data.groupby(factor)[outcome].mean()
                if len(level_means) >= 2:
                    effect = level_means.max() - level_means.min()
                    effects.append(effect)
            
            # Test if effects vary significantly across clusters
            if len(effects) >= 3:
                # Kruskal-Wallis test as non-parametric alternative
                stat, p = stats.f_oneway(*[
                    data[(data[cluster_col] == c)][outcome].values
                    for c in clusters
                ])
                return {
                    "significant": p < 0.05,
                    "p_value": float(p),
                    "f_statistic": float(stat),
                    "effect_variation": float(np.std(effects))
                }
            
            return {"significant": False, "p_value": 1.0, "note": "Insufficient data"}
            
        except Exception as e:
            return {"error": str(e)}
    
    def _compute_vulnerability_profiles(self, effects: Dict, n_clusters: int) -> List[Dict]:
        """Compute vulnerability profiles for each cluster."""
        profiles = []
        
        urgency_effect = effects.get('urgency_effect', {})
        familiarity_effect = effects.get('familiarity_effect', {})
        framing_effect = effects.get('framing_effect', {})
        
        for cluster_id in range(n_clusters):
            profile = {
                'cluster': cluster_id,
                'vulnerabilities': [],
                'strengths': []
            }
            
            # Check urgency vulnerability
            urg = urgency_effect.get(cluster_id, 0)
            if urg > 0.1:
                profile['vulnerabilities'].append({
                    'type': 'urgency',
                    'severity': 'high' if urg > 0.2 else 'medium',
                    'effect_size': float(urg)
                })
            elif urg < -0.05:
                profile['strengths'].append({
                    'type': 'urgency_resistance',
                    'effect_size': float(abs(urg))
                })
            
            # Check familiarity vulnerability
            fam = familiarity_effect.get(cluster_id, 0)
            if fam > 0.1:
                profile['vulnerabilities'].append({
                    'type': 'trust_familiar_senders',
                    'severity': 'high' if fam > 0.2 else 'medium',
                    'effect_size': float(fam)
                })
            elif fam < -0.05:
                profile['strengths'].append({
                    'type': 'skeptical_of_familiar',
                    'effect_size': float(abs(fam))
                })
            
            # Check framing vulnerability
            fram = framing_effect.get(cluster_id, 0)
            if abs(fram) > 0.1:
                profile['vulnerabilities'].append({
                    'type': 'threat_framing' if fram > 0 else 'reward_framing',
                    'severity': 'high' if abs(fram) > 0.2 else 'medium',
                    'effect_size': float(abs(fram))
                })
            
            profiles.append(profile)
        
        return profiles
    
    def get_email_effectiveness(self, participant_df: pd.DataFrame, labels: np.ndarray) -> Dict:
        """
        Analyze which emails are most effective against each cluster.
        Useful for understanding persona-specific vulnerabilities.
        """
        participants = participant_df.copy()
        participants['cluster'] = labels
        
        responses = self.responses.merge(
            participants[['participant_id', 'cluster']],
            on='participant_id'
        )
        
        if 'email_type' in responses.columns:
            phishing = responses[responses['email_type'] == 'phishing']
        else:
            phishing = responses
        
        if 'email_id' not in phishing.columns or 'clicked' not in phishing.columns:
            return {"error": "Required columns not found"}
        
        # For each cluster, find most and least effective emails
        results = {}
        
        for cluster_id in sorted(phishing['cluster'].unique()):
            cluster_data = phishing[phishing['cluster'] == cluster_id]
            email_rates = cluster_data.groupby('email_id')['clicked'].mean().sort_values(ascending=False)
            
            results[int(cluster_id)] = {
                'most_effective': email_rates.head(3).to_dict(),
                'least_effective': email_rates.tail(3).to_dict(),
                'overall_vulnerability': float(cluster_data['clicked'].mean())
            }
        
        return results