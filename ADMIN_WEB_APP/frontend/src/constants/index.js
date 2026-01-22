/**
 * CYPEARL Phase 1 Dashboard - Shared Constants
 */

// ============================================================================
// ALGORITHM CONSTANTS
// ============================================================================

export const ALGORITHM_NAMES = {
    kmeans: 'K-Means',
    gmm: 'Gaussian Mixture',
    hierarchical: 'Agglomerative (Ward)',
    spectral: 'Spectral Clustering'
};

export const ALGORITHM_COLORS = {
    kmeans: '#4f46e5',
    gmm: '#059669',
    hierarchical: '#d97706',
    spectral: '#dc2626'
};

export const DEFAULT_WEIGHTS = {
    behavioral: 0.35,
    silhouette: 0.25,
    stability: 0.20,
    statistical: 0.20
};

// ============================================================================
// CLUSTER COLORS
// ============================================================================

export const CLUSTER_COLORS = [
    '#4f46e5', '#059669', '#d97706', '#dc2626', '#7c3aed',
    '#0891b2', '#db2777', '#65a30d', '#ea580c', '#0d9488'
];

export const RISK_COLORS = {
    CRITICAL: { bg: '#fef2f2', border: '#ef4444', text: '#dc2626' },
    HIGH: { bg: '#fff7ed', border: '#f97316', text: '#ea580c' },
    MEDIUM: { bg: '#fefce8', border: '#eab308', text: '#ca8a04' },
    LOW: { bg: '#f0fdf4', border: '#22c55e', text: '#16a34a' }
};

// ============================================================================
// TRAIT CONSTANTS
// ============================================================================

export const TRAIT_CATEGORIES = {
    cognitive: ['crt_score', 'need_for_cognition', 'working_memory'],
    personality: ['big5_extraversion', 'big5_agreeableness', 'big5_conscientiousness', 'big5_neuroticism', 'big5_openness', 'impulsivity_total', 'sensation_seeking', 'trust_propensity', 'risk_taking'],
    state: ['state_anxiety', 'current_stress', 'fatigue_level'],
    attitudes: ['phishing_self_efficacy', 'perceived_risk', 'security_attitudes', 'privacy_concern'],
    experience: ['phishing_knowledge', 'technical_expertise', 'prior_victimization', 'security_training'],
    habits: ['email_volume_numeric', 'link_click_tendency', 'social_media_usage'],
    susceptibility: ['authority_susceptibility', 'urgency_susceptibility', 'scarcity_susceptibility']
};

export const TRAIT_LABELS = {
    crt_score: 'Analytical Thinking',
    need_for_cognition: 'Need for Cognition',
    working_memory: 'Working Memory',
    big5_extraversion: 'Extraversion',
    big5_agreeableness: 'Agreeableness',
    big5_conscientiousness: 'Conscientiousness',
    big5_neuroticism: 'Neuroticism',
    big5_openness: 'Openness',
    impulsivity_total: 'Impulsivity',
    sensation_seeking: 'Sensation Seeking',
    trust_propensity: 'Trust Propensity',
    risk_taking: 'Risk Taking',
    state_anxiety: 'State Anxiety',
    current_stress: 'Current Stress',
    fatigue_level: 'Fatigue Level',
    phishing_self_efficacy: 'Self-Efficacy',
    perceived_risk: 'Perceived Risk',
    security_attitudes: 'Security Attitudes',
    privacy_concern: 'Privacy Concern',
    phishing_knowledge: 'Phishing Knowledge',
    technical_expertise: 'Technical Expertise',
    prior_victimization: 'Prior Victimization',
    security_training: 'Security Training',
    email_volume_numeric: 'Email Volume',
    link_click_tendency: 'Link Click Tendency',
    social_media_usage: 'Social Media Usage',
    authority_susceptibility: 'Authority Susceptibility',
    urgency_susceptibility: 'Urgency Susceptibility',
    scarcity_susceptibility: 'Scarcity Susceptibility'
};

export const CATEGORY_LABELS = {
    cognitive: 'Cognitive Abilities',
    personality: 'Personality Traits',
    state: 'Current State',
    attitudes: 'Security Attitudes',
    experience: 'Prior Experience',
    habits: 'Digital Habits',
    susceptibility: 'Influence Susceptibility'
};

// ============================================================================
// OUTCOME CONSTANTS
// ============================================================================

export const OUTCOME_FEATURES = [
    'overall_accuracy', 'phishing_detection_rate', 'phishing_click_rate',
    'false_positive_rate', 'report_rate', 'mean_response_latency',
    'hover_rate', 'sender_inspection_rate'
];

export const OUTCOME_LABELS = {
    overall_accuracy: 'Overall Accuracy',
    phishing_detection_rate: 'Phishing Detection Rate',
    phishing_click_rate: 'Phishing Click Rate',
    false_positive_rate: 'False Positive Rate',
    report_rate: 'Report Rate',
    mean_response_latency: 'Average Response Time',
    hover_rate: 'Link Hover Rate',
    sender_inspection_rate: 'Sender Inspection Rate'
};

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

export const calculateCompositeScore = (metrics, weights) => {
    const w = weights || DEFAULT_WEIGHTS;
    const behavioral = Math.min((metrics.eta_squared_mean || 0) / 0.15, 1.0);
    const silhouette = ((metrics.silhouette || 0) + 1) / 2;
    const stability = metrics.size_balance || 0;
    const statistical = (Math.min(Math.log1p(metrics.calinski_harabasz || 0) / 8, 1.0) +
        1 / (1 + (metrics.davies_bouldin || 2))) / 2;

    return (
        w.behavioral * behavioral +
        w.silhouette * silhouette +
        w.stability * stability +
        w.statistical * statistical
    );
};