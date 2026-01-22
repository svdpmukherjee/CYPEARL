from typing import List, Dict

# Cognitive and psychological traits for clustering (NO outcomes!)
CLUSTERING_FEATURES = {
    'cognitive': [
        'crt_score',              # Cognitive Reflection Test
        'need_for_cognition',     # Enjoys thinking
        'working_memory'          # Working memory capacity
    ],
    'personality': [
        'big5_extraversion',
        'big5_agreeableness', 
        'big5_conscientiousness',
        'big5_neuroticism',
        'big5_openness',
        'impulsivity_total',
        'sensation_seeking',
        'trust_propensity',
        'risk_taking'
    ],
    'state': [
        'state_anxiety',
        'current_stress',
        'fatigue_level'
    ],
    'attitudes': [
        'phishing_self_efficacy',
        'perceived_risk',
        'security_attitudes',
        'privacy_concern'
    ],
    'experience': [
        'phishing_knowledge',
        'technical_expertise',
        'prior_victimization',
        'security_training'
    ],
    'habits': [
        'email_volume_numeric',
        'link_click_tendency',
        'social_media_usage'
    ],
    'susceptibility': [
        'authority_susceptibility',
        'urgency_susceptibility',
        'scarcity_susceptibility'
    ]
}

# Behavioral outcomes for VALIDATION (NOT used for clustering)
OUTCOME_FEATURES = [
    'overall_accuracy',
    'phishing_detection_rate',
    'phishing_click_rate',      # Primary outcome
    'false_positive_rate',
    'report_rate',
    'mean_response_latency',
    'hover_rate',
    'sender_inspection_rate'
]

# Demographic features for characterization
DEMOGRAPHIC_FEATURES = [
    'age', 'gender', 'education', 'technical_field', 
    'employment', 'industry'
]

def GET_ALL_CLUSTERING_FEATURES() -> List[str]:
    """Get flat list of all clustering features."""
    features = []
    for group in CLUSTERING_FEATURES.values():
        features.extend(group)
    return features
