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
  : "/api/phase1";

// Fallback to direct backend URL if proxy doesn't work
const BACKEND_URL = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api/phase1`
  : "http://localhost:8000/api/phase1";

// Helper to determine which base URL to use
let useDirectBackend = false;

/**
 * Make an API request with automatic fallback
 * @param {string} endpoint - API endpoint
 * @param {object} options - Fetch options
 * @param {AbortSignal} signal - Optional AbortSignal for cancellation
 */
async function apiRequest(endpoint, options = {}, signal = null) {
  const url = useDirectBackend
    ? `${BACKEND_URL}${endpoint}`
    : `${API_BASE}${endpoint}`;

  const defaultOptions = {
    headers: {
      "Content-Type": "application/json",
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
        console.log("Proxy failed, trying direct backend connection...");
        useDirectBackend = true;
        return apiRequest(endpoint, options, signal);
      }

      const errorText = await response.text();
      throw new Error(
        `HTTP ${response.status}: ${errorText || response.statusText}`,
      );
    }

    return await response.json();
  } catch (error) {
    // Re-throw AbortError as-is
    if (error.name === "AbortError") {
      throw error;
    }
    // If fetch itself fails (network error) and we haven't tried direct backend
    if (!useDirectBackend && error.name === "TypeError") {
      console.log("Proxy connection failed, trying direct backend...");
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
  return apiRequest("/summary");
}

/**
 * Get data quality metrics
 */
export async function getDataQuality() {
  return apiRequest("/data-quality");
}

/**
 * Get email statistics
 */
export async function getEmailStats() {
  return apiRequest("/email-stats");
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
  console.log("[API] runClustering called with:", config);
  return apiRequest(
    "/run",
    {
      method: "POST",
      body: JSON.stringify(config),
    },
    signal,
  );
}

/**
 * Optimize clustering parameters (K-Sweep)
 * @param {object} config - Optimization configuration
 * @param {AbortSignal} signal - Optional AbortSignal for cancellation
 */
export async function optimizeClustering(config, signal = null) {
  console.log("[API] optimizeClustering called with:", config);
  return apiRequest(
    "/optimize",
    {
      method: "POST",
      body: JSON.stringify(config),
    },
    signal,
  );
}

/**
 * Get cluster profiles
 * @param {number} k - Number of clusters (optional)
 */
export async function getProfiles(k = null) {
  const endpoint = k ? `/profiles?k=${k}` : "/profiles";
  return apiRequest(endpoint);
}

/**
 * Rerun clustering with optimal parameters
 * @param {object} config - Clustering configuration
 */
export async function rerunWithOptimal(config) {
  return apiRequest("/rerun-optimal", {
    method: "POST",
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
  return apiRequest("/validation");
}

/**
 * Get stability analysis results
 */
export async function getStability() {
  return apiRequest("/stability");
}

/**
 * Get cross-industry analysis
 */
export async function getIndustryAnalysis() {
  return apiRequest("/industry-analysis");
}

/**
 * Get email interaction analysis
 */
export async function getInteractionAnalysis() {
  return apiRequest("/analyze/interactions", { method: "POST" });
}

// =============================================================================
// PERSONA ENDPOINTS
// =============================================================================

/**
 * Get persona labels
 */
export async function getPersonaLabels() {
  return apiRequest("/persona/labels");
}

/**
 * Set label for a specific persona/cluster
 * @param {number} clusterId - Cluster ID
 * @param {string} label - New label
 */
export async function setPersonaLabel(clusterId, label) {
  return apiRequest(`/persona/${clusterId}/label`, {
    method: "POST",
    body: JSON.stringify({ label }),
  });
}

/**
 * Generate AI-powered persona names using Claude 3.5 Sonnet
 * @param {Array} clusters - Array of cluster data with traits and behavioral outcomes
 */
export async function generatePersonaNames(clusters) {
  return apiRequest("/persona/generate-names", {
    method: "POST",
    body: JSON.stringify({ clusters }),
  });
}

/**
 * Export personas for Phase 2
 */
export async function exportForPhase2() {
  return apiRequest("/export/phase2");
}

// =============================================================================
// HIERARCHICAL TAXONOMY ENDPOINTS
// =============================================================================

/**
 * Get full hierarchical taxonomy of personas
 *
 * Returns a tree structure:
 * - Level 1: Meta-types (Analytical vs Intuitive vs Balanced)
 * - Level 2: Risk profiles (Critical/High/Medium/Low)
 * - Level 3: Individual personas
 */
export async function getPersonaTaxonomy() {
  return apiRequest("/taxonomy");
}

/**
 * Get flattened taxonomy for tree UI rendering
 *
 * Returns flat list with depth and parent_id for each node
 */
export async function getFlatTaxonomy() {
  return apiRequest("/taxonomy/flat");
}

/**
 * Get taxonomy summary
 *
 * Quick overview of persona distribution by cognitive style and risk level
 */
export async function getTaxonomySummary() {
  return apiRequest("/taxonomy/summary");
}

// =============================================================================
// SCIENTIFIC VALIDATION ENDPOINTS
// =============================================================================

/**
 * Run quick validation (silhouette, eta-squared, basic metrics)
 * @param {object} config - Validation configuration
 */
export async function runQuickValidation(config = {}) {
  return apiRequest("/validate/quick", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/**
 * Run comprehensive validation with all scientific methods
 * @param {object} config - Validation configuration
 */
export async function runFullValidation(config = {}) {
  return apiRequest("/validate", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/**
 * Get gap statistic analysis for optimal K
 * @param {object} config - Gap statistic configuration
 */
export async function getGapStatistic(config = {}) {
  return apiRequest("/validate/gap-statistic", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/**
 * Get feature importance analysis
 * @param {object} config - Feature importance configuration (k, include_shap)
 */
export async function getFeatureImportanceAnalysis(config = {}) {
  return apiRequest("/validate/feature-importance", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/**
 * Get cross-validation analysis
 * @param {object} config - Cross-validation configuration (k, n_folds)
 */
export async function getCrossValidation(config = {}) {
  return apiRequest("/validate/cross-validation", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/**
 * Get prediction error analysis
 * @param {object} config - Prediction error configuration (k)
 */
export async function getPredictionError(config = {}) {
  return apiRequest("/validate/prediction-error", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/**
 * Get consensus clustering analysis
 * @param {object} config - Consensus configuration (k, n_iterations)
 */
export async function getConsensusAnalysis(config = {}) {
  return apiRequest("/validate/consensus", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/**
 * Get algorithm comparison analysis
 * @param {object} config - Algorithm comparison configuration (k, n_bootstrap)
 */
export async function getAlgorithmComparison(config = {}) {
  return apiRequest("/validate/algorithm-comparison", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/**
 * Get soft cluster assignments (uncertainty analysis)
 * @param {object} config - Soft assignment configuration (k, algorithm)
 */
export async function getSoftAssignments(config = {}) {
  return apiRequest("/validate/soft-assignments", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/**
 * Get LLM readiness check
 * @param {object} config - LLM readiness configuration (k)
 */
export async function getLLMReadiness(config = {}) {
  return apiRequest("/validate/llm-readiness", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/**
 * Get cluster visualization data (2D projection)
 * @param {object} config - Visualization configuration (method: 'pca' | 'tsne', perplexity: number)
 */
export async function getClusterVisualization(config = {}) {
  return apiRequest("/validate/cluster-visualization", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

/**
 * Get validation summary
 * @param {object} config - Summary configuration (k)
 */
export async function getValidationSummary(config = {}) {
  return apiRequest("/validate/summary", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

// =============================================================================
// UTILITY ENDPOINTS
// =============================================================================

/**
 * Get feature importance for clustering (legacy endpoint)
 */
export async function getFeatureImportance() {
  return apiRequest("/feature-importance");
}

/**
 * Get cluster comparison data
 */
export async function getClusterComparison() {
  return apiRequest("/cluster-comparison");
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

  // Scientific Validation
  runQuickValidation,
  runFullValidation,
  getGapStatistic,
  getFeatureImportanceAnalysis,
  getCrossValidation,
  getPredictionError,
  getConsensusAnalysis,
  getAlgorithmComparison,
  getSoftAssignments,
  getLLMReadiness,
  getClusterVisualization,
  getValidationSummary,

  // Personas
  getPersonaLabels,
  setPersonaLabel,
  generatePersonaNames,
  exportForPhase2,

  // Hierarchical Taxonomy
  getPersonaTaxonomy,
  getFlatTaxonomy,
  getTaxonomySummary,

  // Utility
  getFeatureImportance,
  getClusterComparison,
};
