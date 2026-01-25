/**
 * CYPEARL Phase 2 API Service - Enhanced
 *
 * API client for Phase 2 AI Persona Simulation backend.
 * Includes comprehensive analysis endpoints for fidelity metrics,
 * model comparison, boundary conditions, and recommendations.
 */

// Base URL for API requests
// In production: use VITE_API_URL environment variable
// In development with Vite proxy: use relative path '/api/phase2'
const API_BASE = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api/phase2`
  : "/api/phase2";

const BACKEND_URL = import.meta.env.VITE_API_URL
  ? `${import.meta.env.VITE_API_URL}/api/phase2`
  : "http://localhost:8000/api/phase2";

let useDirectBackend = false;

/**
 * Make an API request with automatic fallback
 */
async function apiRequest(endpoint, options = {}) {
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

  try {
    const response = await fetch(url, mergedOptions);

    if (!response.ok) {
      if (response.status === 404 && !useDirectBackend) {
        console.log("Proxy failed, trying direct backend connection...");
        useDirectBackend = true;
        return apiRequest(endpoint, options);
      }

      const errorText = await response.text();
      throw new Error(
        `HTTP ${response.status}: ${errorText || response.statusText}`,
      );
    }

    return await response.json();
  } catch (error) {
    if (!useDirectBackend && error.name === "TypeError") {
      console.log("Proxy connection failed, trying direct backend...");
      useDirectBackend = true;
      return apiRequest(endpoint, options);
    }
    throw error;
  }
}

// =============================================================================
// ENVIRONMENT CONFIG ENDPOINTS
// =============================================================================

/**
 * Get environment configuration including pre-configured API keys.
 * Used to auto-populate provider setup forms.
 */
export async function getEnvConfig() {
  return apiRequest("/env-config");
}

// =============================================================================
// PROVIDER ENDPOINTS
// =============================================================================

export async function getProviders() {
  const response = await apiRequest("/providers");
  const providers = response.providers || [];
  const providersMap = {};
  for (const p of providers) {
    providersMap[p.provider_type] = {
      configured: p.initialized,
      initialized: p.initialized,
      display_name: p.display_name,
      auth_type: p.auth_type,
      requires_region: p.requires_region,
    };
  }
  return providersMap;
}

export async function setupProvider(providerType, config) {
  const result = await apiRequest(`/providers/${providerType}/setup`, {
    method: "POST",
    body: JSON.stringify(config),
  });

  return {
    configured: result.initialized,
    initialized: result.initialized,
    message: result.message,
  };
}

export async function checkProviderHealth(providerType) {
  return apiRequest(`/providers/${providerType}/health`);
}

// =============================================================================
// MODEL ENDPOINTS
// =============================================================================

export async function getModels() {
  const response = await apiRequest("/models");
  return response.models || [];
}

export async function getModel(modelId) {
  return apiRequest(`/models/${modelId}`);
}

export async function testModel(modelId) {
  return apiRequest(`/models/${modelId}/test`, {
    method: "POST",
  });
}

export async function checkAllModelsHealth() {
  return apiRequest("/models/health");
}

// =============================================================================
// PERSONA ENDPOINTS
// =============================================================================

export async function importPersonas(data) {
  const requestBody = data.phase1_export ? data : { phase1_export: data };

  const result = await apiRequest("/personas/import", {
    method: "POST",
    body: JSON.stringify(requestBody),
  });

  if (result.success) {
    const personas = await getPersonas();
    return { ...result, personas };
  }

  return result;
}

export async function getPersonas() {
  const response = await apiRequest("/personas");
  return response.personas || [];
}

export async function getPersona(personaId) {
  return apiRequest(`/personas/${personaId}`);
}

// =============================================================================
// EMAIL ENDPOINTS
// =============================================================================

export async function loadEmails(data) {
  const emails = Array.isArray(data) ? data : data.emails || data;

  if (!emails || emails.length === 0) {
    return { status: "error", message: "No emails provided" };
  }

  try {
    const result = await apiRequest("/emails/import", {
      method: "POST",
      body: JSON.stringify(emails),
    });
    return result;
  } catch (error) {
    console.error("Failed to upload emails to backend:", error);
    return {
      status: "local_only",
      emails: emails,
      loaded_count: emails.length,
      message: "Emails loaded locally but not synced to backend",
    };
  }
}

export async function getEmails() {
  const response = await apiRequest("/emails");
  return response.emails || [];
}

export async function getEmail(emailId) {
  return apiRequest(`/emails/${emailId}`);
}

// =============================================================================
// EXPERIMENT ENDPOINTS
// =============================================================================

export async function createExperiment(config) {
  console.log("Creating experiment with config:", config);

  const cleanConfig = {
    ...config,
    persona_ids: (config.persona_ids || []).filter(
      (id) => id != null && id !== "",
    ),
    model_ids: (config.model_ids || []).filter((id) => id != null && id !== ""),
    email_ids: (config.email_ids || []).filter((id) => id != null && id !== ""),
    prompt_configs: (config.prompt_configs || ["baseline"]).filter(
      (p) => p != null && p !== "",
    ),
  };

  console.log("Clean config:", cleanConfig);

  return apiRequest("/experiments/create", {
    method: "POST",
    body: JSON.stringify(cleanConfig),
  });
}

export async function getExperiments() {
  const response = await apiRequest("/experiments");
  return response.experiments || [];
}

export async function getExperiment(experimentId) {
  return apiRequest(`/experiments/${experimentId}`);
}

export async function runExperiment(experimentId) {
  return apiRequest(`/experiments/${experimentId}/run`, {
    method: "POST",
  });
}

export async function getExperimentProgress(experimentId) {
  return apiRequest(`/experiments/${experimentId}/progress`);
}

export async function stopExperiment(experimentId) {
  return apiRequest(`/experiments/${experimentId}/stop`, {
    method: "POST",
  });
}

// =============================================================================
// RESULTS ENDPOINTS
// =============================================================================

export async function getResults(experimentId) {
  return apiRequest(`/results/${experimentId}`);
}

// =============================================================================
// ENHANCED ANALYSIS ENDPOINTS
// =============================================================================

/**
 * Analyze fidelity for an experiment with comprehensive metrics.
 *
 * Returns:
 * - Per-condition fidelity metrics with CIs
 * - Cohen's d effect sizes
 * - Effect preservation scores
 * - Decision agreement rates
 * - Summary statistics
 *
 * @param {string} experimentId - Experiment ID
 * @returns {Promise<Object>} Comprehensive fidelity analysis
 */
export async function analyzeFidelity(experimentId) {
  const response = await apiRequest(`/analysis/${experimentId}/fidelity`);

  // Transform response for ResultsTab compatibility
  return {
    experiment_id: response.experiment_id,
    fidelity: response.fidelity_results || [],
    summary: response.summary || {},
    thresholds: response.thresholds || {
      accuracy: 0.85,
      decision_agreement: 0.8,
      effect_preservation: 0.8,
    },
    total_cost:
      response.fidelity_results?.reduce((sum, r) => sum + (r.cost || 0), 0) ||
      0,
    total_trials:
      response.fidelity_results?.reduce(
        (sum, r) => sum + (r.n_trials || 0),
        0,
      ) || 0,
  };
}

/**
 * Compare models with enhanced metrics including Pareto frontier.
 *
 * Returns:
 * - Model rankings by fidelity
 * - Cost-performance analysis
 * - Pareto frontier identification
 * - Latency statistics (p50, p95, p99)
 *
 * @param {string} experimentId - Experiment ID
 * @returns {Promise<Object>} Model comparison results
 */
export async function compareModels(experimentId) {
  const response = await apiRequest(
    `/analysis/${experimentId}/model-comparison`,
  );

  return {
    experiment_id: response.experiment_id,
    model_comparison: response.model_comparison || {},
    ranking: response.ranking || [],
    pareto_frontier: response.pareto_frontier || [],
    best_model: response.best_model,
    best_value_model: response.best_value_model,
  };
}

/**
 * Find boundary conditions with severity classification.
 *
 * Returns:
 * - Detailed boundary condition analysis
 * - Severity classification (high/medium/low)
 * - Per-persona recommendations
 * - Grouped by type and severity
 *
 * @param {string} experimentId - Experiment ID
 * @returns {Promise<Object>} Boundary condition analysis
 */
export async function findBoundaries(experimentId) {
  const response = await apiRequest(
    `/analysis/${experimentId}/boundary-conditions`,
  );

  return {
    experiment_id: response.experiment_id,
    boundary_conditions: response.boundary_conditions || [],
    by_severity: response.by_severity || {},
    by_type: response.by_type || {},
    summary: response.summary || {
      total_found: 0,
      high_severity: 0,
      medium_severity: 0,
      low_severity: 0,
    },
  };
}

/**
 * Get deployment recommendations based on analysis.
 *
 * Returns:
 * - Best model recommendations
 * - Cost-optimal recommendations
 * - Personas requiring human testing
 * - Overall deployment guidance
 *
 * @param {string} experimentId - Experiment ID
 * @returns {Promise<Object>} Deployment recommendations
 */
export async function getRecommendations(experimentId) {
  return apiRequest(`/analysis/${experimentId}/recommendations`);
}

/**
 * Analyze effect preservation across personas.
 *
 * Returns:
 * - Per-persona effect preservation scores
 * - Urgency effect comparison (AI vs Human)
 * - Familiarity effect comparison
 * - Framing effect comparison (if available)
 *
 * @param {string} experimentId - Experiment ID
 * @returns {Promise<Object>} Effect preservation analysis
 */
export async function analyzeEffectPreservation(experimentId) {
  return apiRequest(`/analysis/${experimentId}/effect-preservation`);
}

/**
 * Get comprehensive results for an experiment.
 * Combines fidelity, model comparison, and boundary conditions.
 *
 * @param {string} experimentId - Experiment ID
 * @returns {Promise<Object>} Combined analysis results
 */
export async function getComprehensiveResults(experimentId) {
  try {
    // Fetch all analysis data in parallel
    const [fidelity, modelComparison, boundaries, recommendations] =
      await Promise.all([
        analyzeFidelity(experimentId),
        compareModels(experimentId),
        findBoundaries(experimentId),
        getRecommendations(experimentId).catch(() => null),
      ]);

    return {
      experiment_id: experimentId,
      fidelity: fidelity.fidelity,
      summary: fidelity.summary,
      thresholds: fidelity.thresholds,
      model_comparison: modelComparison.model_comparison,
      pareto_frontier: modelComparison.pareto_frontier,
      ranking: modelComparison.ranking,
      boundary_conditions: boundaries.boundary_conditions,
      boundary_summary: boundaries.summary,
      recommendations: recommendations?.recommendations || null,
      total_cost: fidelity.total_cost,
      total_trials: fidelity.total_trials,
    };
  } catch (error) {
    console.error("Failed to get comprehensive results:", error);
    throw error;
  }
}

// =============================================================================
// EXPERIMENT LOGS ENDPOINTS
// =============================================================================

/**
 * Get experiment execution logs.
 *
 * @param {string} experimentId - Experiment ID
 * @returns {Promise<Object>} Log content and metadata
 */
export async function getExperimentLogs(experimentId) {
  return apiRequest(`/experiments/${experimentId}/logs`);
}

/**
 * List all available experiment logs.
 *
 * @returns {Promise<Object>} List of log files
 */
export async function listExperimentLogs() {
  return apiRequest("/experiments/logs/list");
}

/**
 * Generate experiment log for a completed experiment.
 *
 * @param {string} experimentId - Experiment ID
 * @returns {Promise<Object>} Generated log info
 */
export async function generateExperimentLog(experimentId) {
  return apiRequest(`/experiments/${experimentId}/logs/generate`, {
    method: "POST",
  });
}

// =============================================================================
// DETAILED COMBINATIONS ANALYSIS
// =============================================================================

/**
 * Get full 4-dimensional analysis: LLM × Persona × Email × Prompt
 * For business comparison of all combinations with fidelity and cost.
 *
 * @param {string} experimentId - Experiment ID
 * @returns {Promise<Object>} Detailed combinations analysis
 */
export async function getDetailedCombinations(experimentId) {
  return apiRequest(`/analysis/${experimentId}/detailed-combinations`);
}

// =============================================================================
// COST & USAGE ENDPOINTS
// =============================================================================

export async function estimateCost(params) {
  const queryString = new URLSearchParams(params).toString();
  return apiRequest(`/cost-estimate?${queryString}`);
}

export async function getUsage() {
  return apiRequest("/usage");
}

// =============================================================================
// PROMPT TESTING ENDPOINTS
// =============================================================================

export async function testPrompt(personaId, emailId, promptConfig = "cot") {
  const queryParams = new URLSearchParams({
    persona_id: personaId,
    email_id: emailId,
    prompt_config: promptConfig,
  }).toString();

  return apiRequest(`/test-prompt?${queryParams}`, {
    method: "POST",
  });
}

// =============================================================================
// DEBUG ENDPOINTS
// =============================================================================

export async function getDebugState() {
  return apiRequest("/debug/state");
}

// =============================================================================
// PROMPT ABLATION STUDY ENDPOINTS
// =============================================================================

/**
 * Run factorial prompt ablation study.
 * Tests all 8 combinations of prompt components (Traits, Stats, CoT).
 *
 * @param {Object} params - { persona_id, model_id, include_icl, test_sample_size }
 * @returns {Promise<Object>} Ablation study results with component importance
 */
export async function runAblationStudy(params) {
  return apiRequest("/ablation/run", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

/**
 * List all ablation study results.
 */
export async function getAblationResults() {
  return apiRequest("/ablation/results");
}

/**
 * Get detailed ablation study result.
 */
export async function getAblationResult(ablationKey) {
  return apiRequest(`/ablation/${ablationKey}`);
}

// =============================================================================
// UNCERTAINTY QUANTIFICATION ENDPOINTS
// =============================================================================

/**
 * Run uncertainty analysis on LLM persona simulation.
 * Runs multiple samples per trial to estimate confidence.
 *
 * @param {Object} params - { persona_id, model_id, prompt_config, n_samples, test_sample_size }
 * @returns {Promise<Object>} Uncertainty analysis results
 */
export async function runUncertaintyAnalysis(params) {
  return apiRequest("/uncertainty/analyze", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

/**
 * List all uncertainty analysis results.
 */
export async function getUncertaintyResults() {
  return apiRequest("/uncertainty/results");
}

/**
 * Get detailed uncertainty analysis result.
 */
export async function getUncertaintyResult(uncertaintyKey) {
  return apiRequest(`/uncertainty/${uncertaintyKey}`);
}

// =============================================================================
// BEHAVIORAL SIGNATURE MATCHING ENDPOINTS
// =============================================================================

/**
 * Run behavioral signature matching analysis.
 * Compares causal factors beyond just action matching.
 *
 * @param {Object} params - { persona_id, model_id, prompt_config }
 * @returns {Promise<Object>} Signature matching results
 */
export async function runSignatureMatching(params) {
  return apiRequest("/signatures/analyze", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

/**
 * List all signature matching results.
 */
export async function getSignatureResults() {
  return apiRequest("/signatures/results");
}

/**
 * Get detailed signature matching result.
 */
export async function getSignatureResult(signatureKey) {
  return apiRequest(`/signatures/${signatureKey}`);
}

// =============================================================================
// PERSONA EMBEDDINGS ENDPOINTS
// =============================================================================

/**
 * Train persona encoder on available personas.
 * Creates continuous embedding space.
 *
 * @param {Object} params - { embedding_dim, n_iterations }
 * @returns {Promise<Object>} Training result with 2D coordinates
 */
export async function trainPersonaEncoder(params = {}) {
  return apiRequest("/embeddings/train", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

/**
 * Get persona embedding space visualization data.
 */
export async function getEmbeddingSpace() {
  return apiRequest("/embeddings/space");
}

/**
 * Create interpolated persona between two existing personas.
 *
 * @param {Object} params - { persona_a_id, persona_b_id, alpha, name }
 * @returns {Promise<Object>} Interpolated persona with traits and prompt
 */
export async function createInterpolatedPersona(params) {
  return apiRequest("/embeddings/interpolate", {
    method: "POST",
    body: JSON.stringify(params),
  });
}

/**
 * Find nearest personas in embedding space.
 *
 * @param {string} personaId - Persona ID
 * @param {number} k - Number of neighbors (default 5)
 */
export async function findNearestPersonas(personaId, k = 5) {
  return apiRequest(`/embeddings/${personaId}/nearest?k=${k}`);
}

// =============================================================================
// DEFAULT EXPORT
// =============================================================================

export default {
  // Environment Config
  getEnvConfig,

  // Providers
  getProviders,
  setupProvider,
  checkProviderHealth,

  // Models
  getModels,
  getModel,
  testModel,
  checkAllModelsHealth,

  // Personas
  importPersonas,
  getPersonas,
  getPersona,

  // Emails
  loadEmails,
  getEmails,
  getEmail,

  // Experiments
  createExperiment,
  getExperiments,
  getExperiment,
  runExperiment,
  getExperimentProgress,
  stopExperiment,

  // Experiment Logs
  getExperimentLogs,
  listExperimentLogs,
  generateExperimentLog,

  // Results
  getResults,

  // Analysis (Enhanced)
  analyzeFidelity,
  compareModels,
  findBoundaries,
  getRecommendations,
  analyzeEffectPreservation,
  getComprehensiveResults,
  getDetailedCombinations,

  // Cost & Usage
  estimateCost,
  getUsage,

  // Prompt Testing
  testPrompt,

  // Debug
  getDebugState,

  // Ablation Study
  runAblationStudy,
  getAblationResults,
  getAblationResult,

  // Uncertainty Quantification
  runUncertaintyAnalysis,
  getUncertaintyResults,
  getUncertaintyResult,

  // Behavioral Signatures
  runSignatureMatching,
  getSignatureResults,
  getSignatureResult,

  // Persona Embeddings
  trainPersonaEncoder,
  getEmbeddingSpace,
  createInterpolatedPersona,
  findNearestPersonas,
};
