/**
 * CYPEARL Phase 1 API Service
 * 
 * API client for Phase 1 Persona Discovery backend
 */

// Base URL for API requests
// In production: use VITE_API_URL environment variable
// In development with Vite proxy: use relative path '/api/phase1'
const API_BASE = import.meta.env.VITE_API_URL
    ? `${import.meta.env.VITE_API_URL}/api/phase1`
    : '/api/phase1';

// Fallback to direct backend URL if proxy doesn't work
const BACKEND_URL = import.meta.env.VITE_API_URL
    ? `${import.meta.env.VITE_API_URL}/api/phase1`
    : 'http://localhost:8000/api/phase1';

// Helper to determine which base URL to use
let useDirectBackend = false;

/**
 * Make an API request with automatic fallback
 * @param {string} endpoint - API endpoint
 * @param {object} options - Fetch options
 * @param {AbortSignal} signal - Optional AbortSignal for cancellation
 */
async function apiRequest(endpoint, options = {}, signal = null) {
    const url = useDirectBackend ? `${BACKEND_URL}${endpoint}` : `${API_BASE}${endpoint}`;

    const defaultOptions = {
        headers: {
            'Content-Type': 'application/json',
        },
    };

    const mergedOptions = {
        ...defaultOptions,
        ...options,
        headers: {
            ...defaultOptions.headers,
            ...options.headers,
        },
    };

    // Add abort signal if provided
    if (signal) {
        mergedOptions.signal = signal;
    }

    try {
        const response = await fetch(url, mergedOptions);

        if (!response.ok) {
            // If we get a 404 and haven't tried direct backend yet, try it
            if (response.status === 404 && !useDirectBackend) {
                console.log('Proxy failed, trying direct backend connection...');
                useDirectBackend = true;
                return apiRequest(endpoint, options, signal);
            }

            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText || response.statusText}`);
        }

        return await response.json();
    } catch (error) {
        // Re-throw AbortError as-is
        if (error.name === 'AbortError') {
            throw error;
        }
        // If fetch itself fails (network error) and we haven't tried direct backend
        if (!useDirectBackend && error.name === 'TypeError') {
            console.log('Proxy connection failed, trying direct backend...');
            useDirectBackend = true;
            return apiRequest(endpoint, options, signal);
        }
        throw error;
    }
}

// =============================================================================
// DATA ENDPOINTS
// =============================================================================

/**
 * Get data summary
 */
export async function getSummary() {
    return apiRequest('/summary');
}

/**
 * Get data quality metrics
 */
export async function getDataQuality() {
    return apiRequest('/data-quality');
}

/**
 * Get email statistics
 */
export async function getEmailStats() {
    return apiRequest('/email-stats');
}

// =============================================================================
// CLUSTERING ENDPOINTS
// =============================================================================

/**
 * Run clustering with specified configuration
 * @param {object} config - Clustering configuration
 * @param {AbortSignal} signal - Optional AbortSignal for cancellation
 */
export async function runClustering(config, signal = null) {
    console.log('[API] runClustering called with:', config);
    return apiRequest('/run', {
        method: 'POST',
        body: JSON.stringify(config),
    }, signal);
}

/**
 * Optimize clustering parameters (K-Sweep)
 * @param {object} config - Optimization configuration
 * @param {AbortSignal} signal - Optional AbortSignal for cancellation
 */
export async function optimizeClustering(config, signal = null) {
    console.log('[API] optimizeClustering called with:', config);
    return apiRequest('/optimize', {
        method: 'POST',
        body: JSON.stringify(config),
    }, signal);
}

/**
 * Get cluster profiles
 * @param {number} k - Number of clusters (optional)
 */
export async function getProfiles(k = null) {
    const endpoint = k ? `/profiles?k=${k}` : '/profiles';
    return apiRequest(endpoint);
}

/**
 * Rerun clustering with optimal parameters
 * @param {object} config - Clustering configuration
 */
export async function rerunWithOptimal(config) {
    return apiRequest('/rerun-optimal', {
        method: 'POST',
        body: JSON.stringify(config),
    });
}

// =============================================================================
// ANALYSIS ENDPOINTS
// =============================================================================

/**
 * Get validation metrics for clustering
 */
export async function getValidation() {
    return apiRequest('/validation');
}

/**
 * Get stability analysis results
 */
export async function getStability() {
    return apiRequest('/stability');
}

/**
 * Get cross-industry analysis
 */
export async function getIndustryAnalysis() {
    return apiRequest('/industry-analysis');
}

/**
 * Get email interaction analysis
 */
export async function getInteractionAnalysis() {
    return apiRequest('/analyze/interactions', { method: 'POST' });
}

// =============================================================================
// PERSONA ENDPOINTS
// =============================================================================

/**
 * Get persona labels
 */
export async function getPersonaLabels() {
    return apiRequest('/persona/labels');
}

/**
 * Set label for a specific persona/cluster
 * @param {number} clusterId - Cluster ID
 * @param {string} label - New label
 */
export async function setPersonaLabel(clusterId, label) {
    return apiRequest(`/persona/${clusterId}/label`, {
        method: 'POST',
        body: JSON.stringify({ label }),
    });
}

/**
 * Generate AI-powered persona names using Claude 3.5 Sonnet
 * @param {Array} clusters - Array of cluster data with traits and behavioral outcomes
 */
export async function generatePersonaNames(clusters) {
    return apiRequest('/persona/generate-names', {
        method: 'POST',
        body: JSON.stringify({ clusters }),
    });
}

/**
 * Export personas for Phase 2
 */
export async function exportForPhase2() {
    return apiRequest('/export/phase2');
}

// =============================================================================
// UTILITY ENDPOINTS
// =============================================================================

/**
 * Get feature importance for clustering
 */
export async function getFeatureImportance() {
    return apiRequest('/feature-importance');
}

/**
 * Get cluster comparison data
 */
export async function getClusterComparison() {
    return apiRequest('/cluster-comparison');
}

// =============================================================================
// DEFAULT EXPORT
// =============================================================================

export default {
    // Data
    getSummary,
    getDataQuality,
    getEmailStats,

    // Clustering
    runClustering,
    optimizeClustering,
    getProfiles,
    rerunWithOptimal,

    // Analysis
    getValidation,
    getStability,
    getIndustryAnalysis,
    getInteractionAnalysis,

    // Personas
    getPersonaLabels,
    setPersonaLabel,
    generatePersonaNames,
    exportForPhase2,

    // Utility
    getFeatureImportance,
    getClusterComparison,
};