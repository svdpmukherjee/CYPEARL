/**
 * CYPEARL Phase 1 Dashboard - Main Component (FIXED - Export Button Logic)
 *
 * Tab Order (Information Flow):
 * 1. Dataset Description - Overview of dataset characteristics
 * 2. Clustering - Decision support + Run clustering + Results
 * 3. Persona Profiles - Detailed persona characterization
 * 4. Behavioral Validation - η² analysis and behavioral predictions
 * 5. Email Interactions - Cluster × email type interactions
 * 6. Cross-Industry - Domain transferability analysis
 * 7. Expert Validation - Delphi method
 * 8. AI Export - Phase 2 preparation
 */

import React, {
  useState,
  useEffect,
  useMemo,
  useCallback,
  useRef,
} from "react";
import {
  RefreshCw,
  Database,
  Layers,
  Users,
  Target,
  Building2,
  UserCheck,
  Sparkles,
  Grid3X3,
  ArrowRight,
  CheckCircle,
} from "lucide-react";

// Tab components
import { DatasetDescriptionTab } from "./tabs/DatasetDescriptionTab";
import { ClusteringTab } from "./tabs/ClusteringTab";
import { ProfilesTab } from "./tabs/ProfilesTab";
import { ValidationTab } from "./tabs/ValidationTab";
import { EmailInteractionsTab } from "./tabs/EmailInteractionsTab";
import { CrossIndustryTab } from "./tabs/CrossIndustryTab";
import { ExpertValidationTab } from "./tabs/ExpertValidationTab";
import { AIExportTab } from "./tabs/AIExportTab";

// Guidance component
import { StepGuide } from "./common/StepGuide";

// Constants
import { DEFAULT_WEIGHTS } from "../constants";

// API services
import {
  getSummary,
  getDataQuality,
  runClustering,
  optimizeClustering,
  getIndustryAnalysis,
  getInteractionAnalysis,
  setPersonaLabel,
  getPersonaLabels,
  generatePersonaNames,
  // Scientific validation
  runQuickValidation,
  getGapStatistic,
  getFeatureImportanceAnalysis,
  getCrossValidation,
  getPredictionError,
  getSoftAssignments,
  getLLMReadiness,
  getClusterVisualization,
} from "../services/api";

// ============================================================================
// TAB CONFIGURATION
// ============================================================================

// Main workflow tabs (required steps)
const MAIN_TABS = [
  { id: "dataset", label: "Dataset", icon: Database, step: 1 },
  { id: "clustering", label: "Clustering", icon: Layers, step: 2 },
  { id: "profiles", label: "Personas", icon: Users, step: 3 },
  { id: "export", label: "Export", icon: Sparkles, step: 4 },
];

// Optional exploration tabs
const OPTIONAL_TABS = [
  { id: "validation", label: "Validation", icon: Target, description: "η² analysis" },
  { id: "interactions", label: "Interactions", icon: Grid3X3, description: "Email patterns" },
  { id: "industry", label: "Cross-Industry", icon: Building2, description: "Transferability" },
];

// Expert review tab (special - for refining personas)
const EXPERT_TAB = { id: "expert", label: "Expert Review", icon: UserCheck, description: "Refine personas" };

// Combined for compatibility
const TABS = [...MAIN_TABS, ...OPTIONAL_TABS, EXPERT_TAB];

// ============================================================================
// HELPER FUNCTION - Determine Cognitive Style
// ============================================================================

const determineCognitiveStyle = (cluster) => {
  const crt = cluster.trait_zscores?.crt_score || 0;
  const nfc = cluster.trait_zscores?.need_for_cognition || 0;
  const impulsivity = cluster.trait_zscores?.impulsivity_total || 0;

  if (crt > 0.5 && nfc > 0.5) return "analytical";
  if (impulsivity > 0.5) return "impulsive";
  return "balanced";
};

// ============================================================================
// MAIN DASHBOARD COMPONENT
// ============================================================================

// ============================================================================
// URL & STORAGE UTILITIES FOR PHASE 1
// ============================================================================

const PHASE1_STORAGE_KEY = 'cypearl_phase1_state';

const getPhase1TabFromHash = () => {
  const hash = window.location.hash.slice(1);
  const parts = hash.split('/').filter(Boolean);
  if (parts[0] === 'phase1' && parts[1]) {
    const validTabs = ['dataset', 'clustering', 'profiles', 'validation', 'interactions', 'industry', 'expert', 'export'];
    if (validTabs.includes(parts[1])) {
      return parts[1];
    }
  }
  return null;
};

const setPhase1TabInHash = (tab) => {
  const hash = `/phase1/${tab}`;
  if (window.location.hash !== `#${hash}`) {
    window.history.pushState(null, '', `#${hash}`);
  }
};

const Phase1Dashboard = ({ onExportToPhase2, onPhaseComplete }) => {
  // ========================================================================
  // STATE
  // ========================================================================

  // Initialize activeTab from URL hash or localStorage
  const [activeTab, setActiveTab] = useState(() => {
    const hashTab = getPhase1TabFromHash();
    if (hashTab) return hashTab;
    try {
      const saved = localStorage.getItem(PHASE1_STORAGE_KEY);
      if (saved) {
        const state = JSON.parse(saved);
        return state.activeTab || 'dataset';
      }
    } catch (e) {}
    return 'dataset';
  });
  const [loading, setLoading] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);

  // Core data
  const [summary, setSummary] = useState(null);
  const [dataQuality, setDataQuality] = useState(null);

  // Clustering state
  const [clusteringResult, setClusteringResult] = useState(null);
  const [optimizationResult, setOptimizationResult] = useState(null);
  const [config, setConfig] = useState({
    algorithm: "kmeans",
    k: 5,
    use_pca: true,
    pca_variance: 0.9,
    random_state: 42,
  });
  const [optConfig, setOptConfig] = useState({
    algorithm: "all",
    k_min: 2,
    k_max: 12,
    use_pca: true,
    pca_variance: 0.9,
  });

  // Weights state
  const [weights, setWeights] = useState({ ...DEFAULT_WEIGHTS });
  const [minClusterSize, setMinClusterSize] = useState(30);

  // Analysis states
  const [industryAnalysis, setIndustryAnalysis] = useState(null);
  const [industryError, setIndustryError] = useState(null);
  const [interactionResult, setInteractionResult] = useState(null);
  const [interactionError, setInteractionError] = useState(null);
  const [personaLabels, setPersonaLabels] = useState({});
  const [expertRatings, setExpertRatings] = useState([]);
  const [delphiRound, setDelphiRound] = useState(1);

  // Validation progress tracking state
  const [validationProgress, setValidationProgress] = useState({
    currentStep: 0,
    steps: [
      { id: 1, name: "Quick Validation", endpoint: "quick", status: "pending" },
      {
        id: 2,
        name: "Gap Statistic",
        endpoint: "gap_statistic",
        status: "pending",
      },
      {
        id: 3,
        name: "Feature Importance",
        endpoint: "feature_importance",
        status: "pending",
      },
      {
        id: 4,
        name: "Cross-Validation",
        endpoint: "cross_validation",
        status: "pending",
      },
      {
        id: 5,
        name: "Prediction Error",
        endpoint: "prediction_error",
        status: "pending",
      },
      {
        id: 6,
        name: "Soft Assignments",
        endpoint: "soft_assignments",
        status: "pending",
      },
      {
        id: 7,
        name: "LLM Readiness",
        endpoint: "llm_readiness",
        status: "pending",
      },
      {
        id: 8,
        name: "Cluster Visualization",
        endpoint: "cluster_visualization",
        status: "pending",
      },
    ],
  });

  // AI Persona Naming states
  const [useAiNaming, setUseAiNaming] = useState(true);
  const [isGeneratingNames, setIsGeneratingNames] = useState(false);

  // Scientific validation states
  const [validationResult, setValidationResult] = useState(null);
  const [validationLoading, setValidationLoading] = useState(false);

  // Phase completion state
  const [isPhaseComplete, setIsPhaseComplete] = useState(false);

  // Abort controller for cancelling requests
  const abortControllerRef = useRef(null);
  const [operationType, setOperationType] = useState(null); // 'optimize' | 'cluster' | null

  // ========================================================================
  // COMPUTED VALUES
  // ========================================================================

  const weightsTotal = useMemo(
    () => Object.values(weights).reduce((a, b) => a + b, 0),
    [weights],
  );

  const normalizedWeights = useMemo(() => {
    const total = weightsTotal || 1;
    return {
      behavioral: weights.behavioral / total,
      silhouette: weights.silhouette / total,
      stability: weights.stability / total,
      statistical: weights.statistical / total,
    };
  }, [weights, weightsTotal]);

  // Check if phase 1 is complete enough for phase 2
  const canProceedToPhase2 = useMemo(() => {
    return (
      clusteringResult !== null &&
      Object.keys(clusteringResult.clusters || {}).length >= 3
    );
  }, [clusteringResult]);

  // ========================================================================
  // URL & STATE PERSISTENCE
  // ========================================================================

  // Sync activeTab with URL hash
  useEffect(() => {
    setPhase1TabInHash(activeTab);
  }, [activeTab]);

  // Handle browser back/forward for tab changes
  useEffect(() => {
    const handleHashChange = () => {
      const hashTab = getPhase1TabFromHash();
      if (hashTab && hashTab !== activeTab) {
        setActiveTab(hashTab);
      }
    };

    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, [activeTab]);

  // Persist important state to localStorage
  useEffect(() => {
    // Don't save during initial loading
    if (initialLoading) return;

    try {
      const stateToSave = {
        activeTab,
        clusteringResult,
        optimizationResult,
        config,
        optConfig,
        weights,
        minClusterSize,
        personaLabels,
        validationResult,
        interactionResult,
        industryAnalysis,
        expertRatings,
        delphiRound,
        useAiNaming,
      };
      localStorage.setItem(PHASE1_STORAGE_KEY, JSON.stringify(stateToSave));
    } catch (e) {
      console.warn('Failed to save Phase 1 state:', e);
    }
  }, [
    activeTab, clusteringResult, optimizationResult, config, optConfig,
    weights, minClusterSize, personaLabels, validationResult,
    interactionResult, industryAnalysis, expertRatings, delphiRound,
    useAiNaming, initialLoading
  ]);

  // Restore state from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(PHASE1_STORAGE_KEY);
      if (saved) {
        const state = JSON.parse(saved);
        // Restore clustering results and related state
        if (state.clusteringResult) setClusteringResult(state.clusteringResult);
        if (state.optimizationResult) setOptimizationResult(state.optimizationResult);
        if (state.config) setConfig(state.config);
        if (state.optConfig) setOptConfig(state.optConfig);
        if (state.weights) setWeights(state.weights);
        if (state.minClusterSize) setMinClusterSize(state.minClusterSize);
        if (state.personaLabels) setPersonaLabels(state.personaLabels);
        if (state.validationResult) setValidationResult(state.validationResult);
        if (state.interactionResult) setInteractionResult(state.interactionResult);
        if (state.industryAnalysis) setIndustryAnalysis(state.industryAnalysis);
        if (state.expertRatings) setExpertRatings(state.expertRatings);
        if (state.delphiRound) setDelphiRound(state.delphiRound);
        if (state.useAiNaming !== undefined) setUseAiNaming(state.useAiNaming);
      }
    } catch (e) {
      console.warn('Failed to restore Phase 1 state:', e);
    }
  }, []);

  // ========================================================================
  // DATA LOADING
  // ========================================================================

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    setInitialLoading(true);
    try {
      const [summaryData, qualityData, labels] = await Promise.all([
        getSummary().catch((e) => {
          console.error("Failed to load summary:", e);
          return null;
        }),
        getDataQuality().catch((e) => {
          console.error("Failed to load data quality:", e);
          return null;
        }),
        getPersonaLabels().catch((e) => {
          console.error("Failed to load persona labels:", e);
          return {};
        }),
      ]);

      if (summaryData) setSummary(summaryData);
      if (qualityData) setDataQuality(qualityData);
      if (labels?.labels) setPersonaLabels(labels.labels);
    } catch (error) {
      console.error("Failed to load initial data:", error);
    } finally {
      setInitialLoading(false);
    }
  };

  // ========================================================================
  // WEIGHT HANDLERS
  // ========================================================================

  const updateWeight = useCallback((key, value) => {
    const numValue = parseFloat(value) || 0;
    setWeights((prev) => ({
      ...prev,
      [key]: Math.max(0, Math.min(1, numValue)),
    }));
  }, []);

  const resetToEqualWeights = useCallback(() => {
    setWeights({
      behavioral: 0.25,
      silhouette: 0.25,
      stability: 0.25,
      statistical: 0.25,
    });
  }, []);

  const resetToDefaultWeights = useCallback(() => {
    setWeights({ ...DEFAULT_WEIGHTS });
  }, []);

  // ========================================================================
  // CLUSTERING HANDLERS
  // ========================================================================

  const handleRunClustering = async () => {
    // Cancel any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setLoading(true);
    setOperationType("cluster");
    setIsGeneratingNames(false);
    console.log("[Clustering] Starting with config:", config);

    try {
      const result = await runClustering(
        {
          ...config,
          min_cluster_size: minClusterSize,
        },
        abortControllerRef.current.signal,
      );

      if (!result || !result.clusters) {
        throw new Error("Invalid clustering result received");
      }

      console.log(
        "[Clustering] Complete:",
        Object.keys(result.clusters).length,
        "clusters",
      );
      setClusteringResult(result);
      setLoading(false);
      setOperationType(null);

      // If AI naming is enabled, generate names after clustering
      if (useAiNaming && result.clusters) {
        setIsGeneratingNames(true);
        try {
          const clusters = Object.values(result.clusters);
          console.log(
            "[AI Naming] Generating names for",
            clusters.length,
            "clusters",
          );
          const namesResponse = await generatePersonaNames(clusters);
          console.log("[AI Naming] Response:", namesResponse);

          if (namesResponse?.status === "success" && namesResponse?.labels) {
            setPersonaLabels(namesResponse.labels);
          } else if (namesResponse?.error) {
            console.error("[AI Naming] Error:", namesResponse.error);
            alert(
              `AI naming failed: ${namesResponse.error}. Clustering results are still available.`,
            );
          }
        } catch (nameError) {
          console.error("[AI Naming] Failed:", nameError);
          alert(
            `Failed to generate AI names: ${nameError.message || "Unknown error"}. Clustering results are still available.`,
          );
        } finally {
          setIsGeneratingNames(false);
        }
      }
    } catch (error) {
      if (error.name === "AbortError" || error.message === "canceled") {
        console.log("[Clustering] Cancelled by user");
        return;
      }
      console.error("[Clustering] Failed:", error);
      alert(
        `Clustering failed: ${error.message || "Unknown error"}. Check console for details.`,
      );
    } finally {
      setLoading(false);
      setOperationType(null);
      abortControllerRef.current = null;
    }
  };

  const handleOptimize = async () => {
    // Cancel any existing request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();

    setLoading(true);
    setOperationType("optimize");
    console.log("[K-Sweep] Starting optimization with config:", optConfig);

    try {
      const result = await optimizeClustering(
        {
          algorithm: optConfig.algorithm,
          k_min: parseInt(optConfig.k_min),
          k_max: parseInt(optConfig.k_max),
          use_pca: optConfig.use_pca,
          pca_variance: optConfig.pca_variance,
          weights: normalizedWeights,
          min_cluster_size: minClusterSize,
        },
        abortControllerRef.current.signal,
      );

      console.log("[K-Sweep] Complete:", result);
      setOptimizationResult(result);
    } catch (error) {
      if (error.name === "AbortError" || error.message === "canceled") {
        console.log("[K-Sweep] Cancelled by user");
        return;
      }
      console.error("[K-Sweep] Failed:", error);
      alert(
        `Optimization failed: ${error.message || "Unknown error"}. Check console for details.`,
      );
    } finally {
      setLoading(false);
      setOperationType(null);
      abortControllerRef.current = null;
    }
  };

  const handleCancelOperation = () => {
    if (abortControllerRef.current) {
      console.log("[Cancel] Aborting current operation:", operationType);
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
      setLoading(false);
      setOperationType(null);
    }
  };

  // ========================================================================
  // ANALYSIS HANDLERS
  // ========================================================================

  const handleLoadIndustryAnalysis = async () => {
    setLoading(true);
    setIndustryError(null);
    try {
      const result = await getIndustryAnalysis();
      setIndustryAnalysis(result);
    } catch (error) {
      console.error("Failed to load industry analysis:", error);
      setIndustryError(
        error.response?.data?.detail ||
          error.message ||
          "Failed to load industry analysis.",
      );
    } finally {
      setLoading(false);
    }
  };

  const handleAnalyzeInteractions = async () => {
    setLoading(true);
    setInteractionError(null);
    try {
      const result = await getInteractionAnalysis();
      if (result.error) {
        setInteractionError(result.error);
      } else {
        setInteractionResult(result);
      }
    } catch (error) {
      console.error("Interaction analysis failed:", error);
      setInteractionError(
        error.response?.data?.detail || error.message || "Analysis failed.",
      );
    } finally {
      setLoading(false);
    }
  };

  const handleSavePersonaLabel = async (clusterId, label) => {
    try {
      await setPersonaLabel(clusterId, label);
      setPersonaLabels((prev) => ({ ...prev, [clusterId]: label }));
    } catch (error) {
      console.error("Failed to save persona label:", error);
      setPersonaLabels((prev) => ({ ...prev, [clusterId]: label }));
    }
  };

  // ========================================================================
  // SCIENTIFIC VALIDATION HANDLERS
  // ========================================================================

  const handleRunValidation = async () => {
    if (!clusteringResult) {
      alert("Please run clustering first before validation.");
      return;
    }

    setValidationLoading(true);
    setValidationResult(null);
    console.log(
      "[Validation] Starting scientific validation for K =",
      clusteringResult.k,
    );

    // Reset progress
    const initialSteps = [
      { id: 1, name: "Quick Validation", endpoint: "quick", status: "pending" },
      {
        id: 2,
        name: "Gap Statistic",
        endpoint: "gap_statistic",
        status: "pending",
      },
      {
        id: 3,
        name: "Feature Importance",
        endpoint: "feature_importance",
        status: "pending",
      },
      {
        id: 4,
        name: "Cross-Validation",
        endpoint: "cross_validation",
        status: "pending",
      },
      {
        id: 5,
        name: "Prediction Error",
        endpoint: "prediction_error",
        status: "pending",
      },
      {
        id: 6,
        name: "Soft Assignments",
        endpoint: "soft_assignments",
        status: "pending",
      },
      {
        id: 7,
        name: "LLM Readiness",
        endpoint: "llm_readiness",
        status: "pending",
      },
      {
        id: 8,
        name: "Cluster Visualization",
        endpoint: "cluster_visualization",
        status: "pending",
      },
    ];
    setValidationProgress({ currentStep: 0, steps: initialSteps });

    const results = {};

    // Helper to update step status
    const updateStep = (stepId, status) => {
      setValidationProgress((prev) => ({
        ...prev,
        currentStep: status === "running" ? stepId : prev.currentStep,
        steps: prev.steps.map((s) => (s.id === stepId ? { ...s, status } : s)),
      }));
    };

    try {
      // Step 1: Quick Validation
      updateStep(1, "running");
      try {
        results.quick = await runQuickValidation({ k: clusteringResult.k });
        updateStep(1, "completed");
      } catch (e) {
        console.warn("[Validation] Quick validation failed:", e);
        results.quick = null;
        updateStep(1, "failed");
      }

      // Step 2: Gap Statistic
      updateStep(2, "running");
      try {
        results.gap_statistic = await getGapStatistic({
          k_min: optConfig.k_min,
          k_max: optConfig.k_max,
        });
        updateStep(2, "completed");
      } catch (e) {
        console.warn("[Validation] Gap statistic failed:", e);
        results.gap_statistic = null;
        updateStep(2, "failed");
      }

      // Step 3: Feature Importance
      updateStep(3, "running");
      try {
        results.feature_importance = await getFeatureImportanceAnalysis({
          k: clusteringResult.k,
        });
        updateStep(3, "completed");
      } catch (e) {
        console.warn("[Validation] Feature importance failed:", e);
        results.feature_importance = null;
        updateStep(3, "failed");
      }

      // Step 4: Cross-Validation
      updateStep(4, "running");
      try {
        results.cross_validation = await getCrossValidation({
          k: clusteringResult.k,
          n_folds: 5,
        });
        updateStep(4, "completed");
      } catch (e) {
        console.warn("[Validation] Cross-validation failed:", e);
        results.cross_validation = null;
        updateStep(4, "failed");
      }

      // Step 5: Prediction Error
      updateStep(5, "running");
      try {
        results.prediction_error = await getPredictionError({
          k: clusteringResult.k,
        });
        updateStep(5, "completed");
      } catch (e) {
        console.warn("[Validation] Prediction error failed:", e);
        results.prediction_error = null;
        updateStep(5, "failed");
      }

      // Step 6: Soft Assignments
      updateStep(6, "running");
      try {
        results.soft_assignments = await getSoftAssignments({
          k: clusteringResult.k,
        });
        updateStep(6, "completed");
      } catch (e) {
        console.warn("[Validation] Soft assignments failed:", e);
        results.soft_assignments = null;
        updateStep(6, "failed");
      }

      // Step 7: LLM Readiness
      updateStep(7, "running");
      try {
        results.llm_readiness = await getLLMReadiness({
          k: clusteringResult.k,
        });
        updateStep(7, "completed");
      } catch (e) {
        console.warn("[Validation] LLM readiness failed:", e);
        results.llm_readiness = null;
        updateStep(7, "failed");
      }

      // Step 8: Cluster Visualization
      updateStep(8, "running");
      try {
        results.cluster_visualization = await getClusterVisualization({
          method: "pca",
        });
        updateStep(8, "completed");
      } catch (e) {
        console.warn("[Validation] Cluster visualization failed:", e);
        results.cluster_visualization = null;
        updateStep(8, "failed");
      }

      // Combine all results
      const combinedResult = {
        ...results,
        timestamp: new Date().toISOString(),
        k: clusteringResult.k,
      };

      console.log("[Validation] Complete:", combinedResult);
      setValidationResult(combinedResult);
    } catch (error) {
      console.error("[Validation] Failed:", error);
      alert(
        `Validation failed: ${error.message || "Unknown error"}. Check console for details.`,
      );
    } finally {
      setValidationLoading(false);
    }
  };

  // ========================================================================
  // EXPERT VALIDATION HANDLERS
  // ========================================================================

  const handleSubmitExpertRating = async (rating) => {
    setExpertRatings((prev) => [...prev, rating]);
  };

  const handleAdvanceDelphiRound = async () => {
    setDelphiRound((r) => r + 1);
  };

  // ========================================================================
  // PHASE 2 EXPORT HANDLERS (FIXED)
  // ========================================================================

  const handleExportToPhase2 = (exportData) => {
    console.log("Exporting to Phase 2:", exportData);
    setIsPhaseComplete(true);

    // Notify parent component
    if (onExportToPhase2) {
      onExportToPhase2(exportData);
    }
    if (onPhaseComplete) {
      onPhaseComplete();
    }
  };

  // NEW: Function to generate personas and proceed to Phase 2
  // Now async to fetch AI-generated persona labels
  const handleProceedToPhase2 = async () => {
    if (!clusteringResult || !clusteringResult.clusters) {
      alert("Please run clustering first");
      return;
    }

    // Fetch AI-generated persona labels from backend
    let personaLabels = {};
    try {
      const labelsResponse = await getPersonaLabels();
      if (labelsResponse?.labels) {
        personaLabels = labelsResponse.labels;
      }
    } catch (err) {
      console.warn("Could not fetch persona labels:", err);
    }

    const clusters = Object.values(clusteringResult.clusters || {});
    const format = "full"; // Use full format with all stats

    // Generate persona definitions with AI-generated names (same logic as AIExportTab)
    const allPersonaDefinitions = clusters.map((cluster) => {
      const topTraits = [
        ...(cluster.top_high_traits || []).map(
          ([t, z]) => `high ${t.replace(/_/g, " ")} (+${z.toFixed(1)}σ)`,
        ),
        ...(cluster.top_low_traits || []).map(
          ([t, z]) => `low ${t.replace(/_/g, " ")} (${z.toFixed(1)}σ)`,
        ),
      ].slice(0, 5);

      const displayClusterId = cluster.cluster_id + 1;
      const clusterId = cluster.cluster_id;

      // Use AI-generated name if available, otherwise fall back to default
      const personaName =
        personaLabels[clusterId]?.name ||
        cluster.label ||
        `Persona ${displayClusterId}`;
      const personaArchetype =
        personaLabels[clusterId]?.archetype ||
        cluster.archetype ||
        cluster.label ||
        `Persona ${displayClusterId}`;

      return {
        persona_id: `PERSONA_${displayClusterId}`,
        cluster_id: displayClusterId,
        name: personaName,
        archetype: personaArchetype,
        risk_level: cluster.risk_level || "MEDIUM",
        n_participants: cluster.n_participants || cluster.size || 0,
        pct_of_population: cluster.pct_of_population || 0,
        description:
          personaLabels[clusterId]?.description || cluster.description || "",
        trait_zscores: cluster.trait_zscores || {},
        distinguishing_high_traits: (cluster.top_high_traits || []).map(
          ([t]) => t,
        ),
        distinguishing_low_traits: (cluster.top_low_traits || []).map(
          ([t]) => t,
        ),
        cognitive_style: determineCognitiveStyle(cluster),
        behavioral_statistics: {
          phishing_click_rate: cluster.phishing_click_rate || 0,
          overall_accuracy:
            cluster.behavioral_outcomes?.overall_accuracy?.mean || 0,
          report_rate: cluster.behavioral_outcomes?.report_rate?.mean || 0,
          mean_response_latency_ms:
            cluster.behavioral_outcomes?.mean_response_latency?.mean || 0,
          hover_rate: cluster.behavioral_outcomes?.hover_rate?.mean || 0,
          sender_inspection_rate:
            cluster.behavioral_outcomes?.sender_inspection_rate?.mean || 0,
        },
        email_interaction_effects: {
          urgency_effect:
            cluster.email_interaction_effects?.urgency_effect || 0,
          familiarity_effect:
            cluster.email_interaction_effects?.familiarity_effect || 0,
          framing_effect:
            cluster.email_interaction_effects?.framing_effect || 0,
        },
        target_accuracy: 0.85,
        acceptance_range: [0.8, 0.9],
      };
    });

    const exportData = {
      format: format,
      n_personas: allPersonaDefinitions.length,
      personas: allPersonaDefinitions,
      export_timestamp: new Date().toISOString(),
      clustering_config: {
        algorithm: clusteringResult.algorithm,
        k: clusteringResult.k,
      },
    };

    // Call the export handler
    handleExportToPhase2(exportData);
  };

  // ========================================================================
  // RENDER - LOADING STATE
  // ========================================================================

  if (initialLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw
            className="animate-spin text-indigo-600 mx-auto mb-4"
            size={48}
          />
          <p className="text-gray-600">Loading Phase 1 Dashboard...</p>
        </div>
      </div>
    );
  }

  // ========================================================================
  // RENDER - MAIN
  // ========================================================================

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b px-8 py-6">
        <div className="max-w-7xl mx-auto">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Phase 1: Persona Discovery
              </h1>
              <p className="text-gray-500 mt-1">
                Discover and validate behavioral personas for AI phishing
                simulation
              </p>
            </div>

            {/* Phase 2 Navigation - FIXED LOGIC */}
            <div className="flex items-center gap-3">
              {canProceedToPhase2 ? (
                <button
                  onClick={() => {
                    if (activeTab === "export") {
                      // On export tab - generate personas and proceed to Phase 2
                      handleProceedToPhase2();
                    } else {
                      // Not on export tab - go to export tab first
                      setActiveTab("export");
                    }
                  }}
                  className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 flex items-center gap-2 text-sm font-medium"
                >
                  {activeTab === "export"
                    ? "Proceed to Phase 2"
                    : "Review & Proceed to Phase 2"}
                  <ArrowRight size={16} />
                </button>
              ) : (
                <div className="flex items-center gap-2 px-4 py-2 bg-gray-100 text-gray-500 rounded-lg text-sm">
                  <span>Complete clustering to unlock Phase 2</span>
                  <ArrowRight size={16} className="opacity-50" />
                </div>
              )}
            </div>
          </div>

          {/* Tab Navigation - Redesigned with workflow indicators */}
          <nav className="mt-6 overflow-x-auto pb-2">
            <div className="flex items-center gap-6">
              {/* Main Workflow Tabs */}
              <div className="flex items-center">
                {MAIN_TABS.map((tab, index) => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;
                  const isExportTab = tab.id === "export";
                  const isCompleted = tab.id === "dataset" ? true :
                                     tab.id === "clustering" ? clusteringResult !== null :
                                     tab.id === "profiles" ? clusteringResult !== null :
                                     tab.id === "export" ? false : false;

                  return (
                    <div key={tab.id} className="flex items-center">
                      <button
                        onClick={() => setActiveTab(tab.id)}
                        className={`flex items-center gap-2 px-4 py-2.5 text-sm font-medium rounded-lg transition-all whitespace-nowrap ${
                          isActive
                            ? isExportTab
                              ? "bg-purple-100 text-purple-700 ring-2 ring-purple-300"
                              : "bg-indigo-100 text-indigo-700 ring-2 ring-indigo-300"
                            : isCompleted
                              ? "bg-green-50 text-green-700 hover:bg-green-100"
                              : "text-gray-500 hover:bg-gray-100 hover:text-gray-700"
                        }`}
                      >
                        <span className={`flex items-center justify-center w-5 h-5 rounded-full text-xs font-bold ${
                          isActive
                            ? isExportTab ? "bg-purple-600 text-white" : "bg-indigo-600 text-white"
                            : isCompleted
                              ? "bg-green-500 text-white"
                              : "bg-gray-300 text-gray-600"
                        }`}>
                          {isCompleted && !isActive ? "✓" : tab.step}
                        </span>
                        <Icon size={16} />
                        {tab.label}
                        {isExportTab && canProceedToPhase2 && (
                          <span className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
                        )}
                      </button>
                      {/* Arrow between main tabs */}
                      {index < MAIN_TABS.length - 1 && (
                        <ArrowRight size={16} className="mx-2 text-gray-300" />
                      )}
                    </div>
                  );
                })}
              </div>

              {/* Separator */}
              <div className="h-8 w-px bg-gray-200" />

              {/* Optional Tabs */}
              <div className="flex items-center gap-1">
                <span className="text-xs text-gray-400 mr-2 uppercase tracking-wide">Optional:</span>
                {OPTIONAL_TABS.map((tab) => {
                  const Icon = tab.icon;
                  const isActive = activeTab === tab.id;

                  return (
                    <button
                      key={tab.id}
                      onClick={() => setActiveTab(tab.id)}
                      title={tab.description}
                      className={`flex items-center gap-1.5 px-3 py-2 text-sm rounded-lg transition-all whitespace-nowrap ${
                        isActive
                          ? "bg-gray-100 text-gray-900 ring-1 ring-gray-300"
                          : "text-gray-500 hover:bg-gray-50 hover:text-gray-700"
                      }`}
                    >
                      <Icon size={14} />
                      {tab.label}
                    </button>
                  );
                })}

                {/* Expert Review Tab - Special styling */}
                <button
                  onClick={() => setActiveTab(EXPERT_TAB.id)}
                  title={EXPERT_TAB.description}
                  className={`flex items-center gap-1.5 px-3 py-2 text-sm rounded-lg transition-all whitespace-nowrap border ${
                    activeTab === EXPERT_TAB.id
                      ? "bg-orange-50 text-orange-700 border-orange-200 ring-1 ring-orange-300"
                      : "text-orange-600 border-orange-200 hover:bg-orange-50"
                  }`}
                >
                  <EXPERT_TAB.icon size={14} />
                  {EXPERT_TAB.label}
                </button>
              </div>
            </div>
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-8 py-6">
        {/* Step-by-step guidance for current tab */}
        <StepGuide
          phase={1}
          tab={activeTab}
          collapsed={true}
          className="mb-6"
        />

        {activeTab === "dataset" && (
          <DatasetDescriptionTab summary={summary} dataQuality={dataQuality} />
        )}

        {activeTab === "clustering" && (
          <ClusteringTab
            config={config}
            setConfig={setConfig}
            optConfig={optConfig}
            setOptConfig={setOptConfig}
            weights={weights}
            setWeights={setWeights}
            updateWeight={updateWeight}
            resetToEqualWeights={resetToEqualWeights}
            resetToDefaultWeights={resetToDefaultWeights}
            minClusterSize={minClusterSize}
            setMinClusterSize={setMinClusterSize}
            loading={loading}
            onRunClustering={handleRunClustering}
            onOptimize={handleOptimize}
            onCancel={handleCancelOperation}
            operationType={operationType}
            clusteringResult={clusteringResult}
            optimizationResult={optimizationResult}
            normalizedWeights={normalizedWeights}
            weightsTotal={weightsTotal}
            // AI Persona Naming props
            useAiNaming={useAiNaming}
            setUseAiNaming={setUseAiNaming}
            isGeneratingNames={isGeneratingNames}
            personaLabels={personaLabels}
            // Scientific validation props
            validationResult={validationResult}
            validationLoading={validationLoading}
            validationProgress={validationProgress}
            onRunValidation={handleRunValidation}
          />
        )}

        {activeTab === "profiles" && (
          <ProfilesTab
            clusteringResult={clusteringResult}
            personaLabels={personaLabels}
            onSaveLabel={handleSavePersonaLabel}
          />
        )}

        {activeTab === "validation" && (
          <ValidationTab
            clusteringResult={clusteringResult}
            minClusterSize={minClusterSize}
            personaLabels={personaLabels}
          />
        )}

        {activeTab === "interactions" && (
          <EmailInteractionsTab
            clusteringResult={clusteringResult}
            interactionResult={interactionResult}
            interactionError={interactionError}
            loading={loading}
            onAnalyzeInteractions={handleAnalyzeInteractions}
            personaLabels={personaLabels}
          />
        )}

        {activeTab === "industry" && (
          <CrossIndustryTab
            clusteringResult={clusteringResult}
            industryAnalysis={industryAnalysis}
            loading={loading}
            error={industryError}
            onLoadAnalysis={handleLoadIndustryAnalysis}
            personaLabels={personaLabels}
          />
        )}

        {activeTab === "expert" && (
          <ExpertValidationTab
            clusteringResult={clusteringResult}
            expertRatings={expertRatings}
            delphiRound={delphiRound}
            onSubmitRating={handleSubmitExpertRating}
            onAdvanceRound={handleAdvanceDelphiRound}
            personaLabels={personaLabels}
          />
        )}

        {activeTab === "export" && (
          <AIExportTab
            clusteringResult={clusteringResult}
            personaLabels={personaLabels}
          />
        )}
      </main>
    </div>
  );
};

export default Phase1Dashboard;
