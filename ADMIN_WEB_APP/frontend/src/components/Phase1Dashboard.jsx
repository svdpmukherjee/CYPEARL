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

import React, { useState, useEffect, useMemo, useCallback } from "react";
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
} from "../services/api";

// ============================================================================
// TAB CONFIGURATION
// ============================================================================

const TABS = [
  { id: "dataset", label: "Dataset", icon: Database },
  { id: "clustering", label: "Clustering", icon: Layers },
  { id: "profiles", label: "Personas", icon: Users },
  { id: "validation", label: "Validation", icon: Target },
  { id: "interactions", label: "Interactions", icon: Grid3X3 },
  { id: "industry", label: "Cross-Industry", icon: Building2 },
  { id: "expert", label: "Expert", icon: UserCheck },
  { id: "export", label: "AI Export", icon: Sparkles },
];

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

const Phase1Dashboard = ({ onExportToPhase2, onPhaseComplete }) => {
  // ========================================================================
  // STATE
  // ========================================================================

  const [activeTab, setActiveTab] = useState("dataset");
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

  // AI Persona Naming states
  const [useAiNaming, setUseAiNaming] = useState(true);
  const [isGeneratingNames, setIsGeneratingNames] = useState(false);

  // Phase completion state
  const [isPhaseComplete, setIsPhaseComplete] = useState(false);

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
    setLoading(true);
    setIsGeneratingNames(false); // Reset naming state
    try {
      const result = await runClustering({
        ...config,
        min_cluster_size: minClusterSize,
      });

      if (!result || !result.clusters) {
        throw new Error("Invalid clustering result received");
      }

      setClusteringResult(result);
      setLoading(false); // Release loading before AI naming

      // If AI naming is enabled, generate names after clustering
      if (useAiNaming && result.clusters) {
        setIsGeneratingNames(true);
        try {
          const clusters = Object.values(result.clusters);
          console.log("Sending clusters for AI naming:", clusters.length);
          const namesResponse = await generatePersonaNames(clusters);
          console.log("AI naming response:", namesResponse);

          if (namesResponse?.status === "success" && namesResponse?.labels) {
            setPersonaLabels(namesResponse.labels);
          } else if (namesResponse?.error) {
            console.error("AI naming error:", namesResponse.error);
            alert(
              `AI naming failed: ${namesResponse.error}. Clustering results are still available.`,
            );
          }
        } catch (nameError) {
          console.error("Failed to generate persona names:", nameError);
          alert(
            `Failed to generate AI names: ${nameError.message || "Unknown error"}. Clustering results are still available.`,
          );
        } finally {
          setIsGeneratingNames(false);
        }
      }
    } catch (error) {
      console.error("Clustering failed:", error);
      alert(
        `Clustering failed: ${error.message || "Unknown error"}. Check console for details.`,
      );
      setLoading(false);
    }
  };

  const handleOptimize = async () => {
    setLoading(true);
    try {
      const result = await optimizeClustering({
        algorithm: optConfig.algorithm,
        k_min: parseInt(optConfig.k_min),
        k_max: parseInt(optConfig.k_max),
        use_pca: optConfig.use_pca,
        pca_variance: optConfig.pca_variance,
        weights: normalizedWeights,
        min_cluster_size: minClusterSize,
      });
      setOptimizationResult(result);
    } catch (error) {
      console.error("Optimization failed:", error);
      alert("Optimization failed. Check console for details.");
    } finally {
      setLoading(false);
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

          {/* Tab Navigation */}
          <nav className="flex gap-1 mt-6 overflow-x-auto pb-2">
            {TABS.map((tab) => {
              const Icon = tab.icon;
              const isExportTab = tab.id === "export";
              return (
                <button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`flex items-center gap-2 px-4 py-3 text-sm font-medium 
                                        border-b-2 transition-colors whitespace-nowrap ${
                                          activeTab === tab.id
                                            ? isExportTab
                                              ? "border-purple-600 text-purple-600"
                                              : "border-indigo-600 text-indigo-600"
                                            : "border-transparent text-gray-500 hover:text-gray-700"
                                        }`}
                >
                  <Icon size={16} />
                  {tab.label}
                  {isExportTab && canProceedToPhase2 && (
                    <span className="w-2 h-2 rounded-full bg-purple-500 animate-pulse" />
                  )}
                </button>
              );
            })}
          </nav>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-8 py-6">
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
            clusteringResult={clusteringResult}
            optimizationResult={optimizationResult}
            normalizedWeights={normalizedWeights}
            weightsTotal={weightsTotal}
            // AI Persona Naming props
            useAiNaming={useAiNaming}
            setUseAiNaming={setUseAiNaming}
            isGeneratingNames={isGeneratingNames}
            personaLabels={personaLabels}
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
