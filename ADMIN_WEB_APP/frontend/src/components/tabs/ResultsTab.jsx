/**
 * Phase 2 Results Tab
 *
 * Fidelity analysis and model comparison visualization.
 *
 * ENHANCEMENTS:
 * - Column visibility toggles for the data table
 * - Advanced filtering with fidelity range slider
 * - Heatmap for Persona × Model × Prompt combinations
 * - Scatter plot for Fidelity vs Cost analysis
 * - AI vs Human click rate comparison chart
 * - Grouped bar charts for multi-dimensional comparison
 * - Sortable table columns
 */

import React, { useState, useMemo, useEffect } from "react";
import * as api from "../../services/phase2Api";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  ScatterChart,
  Scatter,
  Cell,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ComposedChart,
  Line,
  PieChart,
  Pie,
} from "recharts";
import {
  BarChart2,
  TrendingUp,
  CheckCircle,
  AlertCircle,
  Target,
  DollarSign,
  Award,
  Filter,
  Eye,
  EyeOff,
  ArrowUpDown,
  ChevronDown,
  ChevronUp,
  Grid3X3,
  Percent,
  Users,
  Cpu,
  FileText,
  SlidersHorizontal,
  X,
  Download,
  ArrowUp,
  ArrowDown,
  Minus,
  Zap,
  Layers,
} from "lucide-react";

// =============================================================================
// CONSTANTS
// =============================================================================

const FIDELITY_THRESHOLD = 0.85;
const DECISION_THRESHOLD = 0.8;

// =============================================================================
// FIDELITY CALCULATION HELPER
// =============================================================================

/**
 * Calculate fidelity based on rate type
 * Fidelity = 1 - |AI_rate - Human_rate|
 * For click rate, we use the pre-computed normalized_accuracy from backend
 * For report and ignore rates, we calculate on frontend
 */
const getFidelityForRateType = (dataPoint, rateType) => {
  if (rateType === "click") {
    // Use pre-computed normalized_accuracy for click rate (default)
    return dataPoint.normalized_accuracy || 0;
  } else if (rateType === "report") {
    const aiReport = dataPoint.ai_report_rate || 0;
    const humanReport = dataPoint.human_report_rate || 0;
    return 1 - Math.abs(aiReport - humanReport);
  } else if (rateType === "ignore") {
    // Ignore rate = 1 - click_rate - report_rate
    const aiClick = dataPoint.ai_click_rate || 0;
    const aiReport = dataPoint.ai_report_rate || 0;
    const humanClick = dataPoint.human_click_rate || 0;
    const humanReport = dataPoint.human_report_rate || 0;
    const aiIgnore = Math.max(0, 1 - aiClick - aiReport);
    const humanIgnore = Math.max(0, 1 - humanClick - humanReport);
    return 1 - Math.abs(aiIgnore - humanIgnore);
  }
  return dataPoint.normalized_accuracy || 0;
};

/**
 * Get AI rate value based on rate type
 */
const getAiRateForType = (dataPoint, rateType) => {
  if (rateType === "click") return dataPoint.ai_click_rate || 0;
  if (rateType === "report") return dataPoint.ai_report_rate || 0;
  if (rateType === "ignore") {
    const aiClick = dataPoint.ai_click_rate || 0;
    const aiReport = dataPoint.ai_report_rate || 0;
    return Math.max(0, 1 - aiClick - aiReport);
  }
  return 0;
};

/**
 * Get Human rate value based on rate type
 */
const getHumanRateForType = (dataPoint, rateType) => {
  if (rateType === "click") return dataPoint.human_click_rate || 0;
  if (rateType === "report") return dataPoint.human_report_rate || 0;
  if (rateType === "ignore") {
    const humanClick = dataPoint.human_click_rate || 0;
    const humanReport = dataPoint.human_report_rate || 0;
    return Math.max(0, 1 - humanClick - humanReport);
  }
  return 0;
};

/**
 * Get rate type label
 */
const getRateTypeLabel = (rateType) => {
  if (rateType === "click") return "Click Rate";
  if (rateType === "report") return "Report Rate";
  if (rateType === "ignore") return "Ignore Rate";
  return "Click Rate";
};

const MODEL_COLORS = {
  "gpt-4-turbo": "#22c55e",
  "gpt-4o": "#22c55e",
  "gpt-4o-mini": "#86efac",
  "claude-3-haiku": "#8b5cf6",
  "claude-3-opus": "#a78bfa",
  "claude-sonnet-4-5": "#8b5cf6",
  "llama-3-70b": "#3b82f6",
  "llama-3-8b": "#93c5fd",
  "llama-3.3-70b": "#3b82f6",
  "llama-4-maverick": "#60a5fa",
  "mixtral-8x7b": "#f97316",
  "mixtral-8x22b": "#fb923c",
  "mistral-7b": "#ef4444",
  "mistral-large": "#dc2626",
  "nova-micro": "#10b981",
  "nova-lite": "#34d399",
  "nova-pro": "#059669",
  default: "#6b7280",
};

const PROMPT_COLORS = {
  baseline: "#94a3b8",
  stats: "#3b82f6",
  cot: "#8b5cf6",
};

// Model categorization for grouped analysis
const MODEL_CATEGORIES = {
  "nova-pro": { sizeTier: "Large", openness: "Closed", architecture: "Dense" },
  "claude-3-5-sonnet": { sizeTier: "Medium", openness: "Closed", architecture: "Dense" },
  "mistral-7b": { sizeTier: "Small", openness: "Open", architecture: "Dense" },
  "claude-3-sonnet": { sizeTier: "Medium", openness: "Closed", architecture: "Dense" },
  "gpt-4o-mini": { sizeTier: "Small", openness: "Closed", architecture: "Dense" },
  "llama-4-scout": { sizeTier: "Small", openness: "Open", architecture: "MoE" },
  "llama-3.3-70b": { sizeTier: "Medium", openness: "Open", architecture: "Dense" },
  "nova-lite": { sizeTier: "Medium", openness: "Closed", architecture: "Dense" },
  "claude-3-opus": { sizeTier: "Large", openness: "Closed", architecture: "Dense" },
  "claude-3-haiku": { sizeTier: "Small", openness: "Closed", architecture: "Dense" },
  "gpt-4": { sizeTier: "Large", openness: "Closed", architecture: "Dense" },
  "nova-micro": { sizeTier: "Small", openness: "Closed", architecture: "Dense" },
  "mistral-medium": { sizeTier: "Medium", openness: "Closed", architecture: "Dense" },
  "llama-3.1-405b": { sizeTier: "Large", openness: "Open", architecture: "Dense" },
  "gpt-4o": { sizeTier: "Large", openness: "Closed", architecture: "Dense" },
  "claude-3-5-haiku": { sizeTier: "Small", openness: "Closed", architecture: "Dense" },
  "mixtral-8x7b": { sizeTier: "Medium", openness: "Open", architecture: "MoE" },
  "gpt-4-turbo": { sizeTier: "Large", openness: "Closed", architecture: "Dense" },
  "llama-4-maverick": { sizeTier: "Medium", openness: "Open", architecture: "MoE" },
  "mistral-large": { sizeTier: "Large", openness: "Closed", architecture: "Dense" },
  "llama-3.1-70b": { sizeTier: "Medium", openness: "Open", architecture: "Dense" },
  "mistral-small": { sizeTier: "Small", openness: "Closed", architecture: "Dense" },
  "mixtral-8x22b": { sizeTier: "Large", openness: "Open", architecture: "MoE" },
};

// Category display colors
const CATEGORY_COLORS = {
  sizeTier: { Small: "#22c55e", Medium: "#3b82f6", Large: "#8b5cf6" },
  openness: { Open: "#22c55e", Closed: "#ef4444" },
  architecture: { Dense: "#3b82f6", MoE: "#f97316" },
};

// Category grouping options
const CATEGORY_OPTIONS = [
  { id: "none", label: "Individual Models" },
  { id: "sizeTier", label: "Size Tier" },
  { id: "openness", label: "Open vs Closed" },
  { id: "architecture", label: "Architecture" },
];

const PERSONA_COLORS = {
  "Impulsive Risk Taker": "#ef4444",
  "Tech Savvy Skeptic": "#3b82f6",
  "Cautious Security Expert": "#22c55e",
  "Technologically Challenged": "#f59e0b",
  "Socially Influenced Clicker": "#8b5cf6",
  default: "#6b7280",
};

// =============================================================================
// SUB-TAB DEFINITIONS
// =============================================================================

const RESULT_SUB_TABS = [
  { id: "dashboard", label: "Dashboard", icon: BarChart2 },
  { id: "models", label: "Models", icon: Cpu },
  { id: "prompts", label: "Prompts", icon: FileText },
  { id: "costs", label: "Costs", icon: DollarSign },
  { id: "boundaries", label: "Boundaries", icon: AlertCircle },
];

// Visualization mode options for the unified dashboard (Bubble first, then Matrix)
const VIZ_MODES = [
  { id: "bubble", label: "Bubble", icon: Target },
  { id: "matrix", label: "Matrix", icon: Grid3X3 },
];

// Prompt label mapping
const PROMPT_LABELS = {
  cot: "Chain-of-Thought",
  stats: "Behavioral Stats",
  baseline: "Baseline",
};

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export const ResultsTab = ({
  experiments,
  currentExperiment,
  results,
  onSelectExperiment,
}) => {
  const [activeSubTab, setActiveSubTab] = useState("dashboard");
  const [selectedMetric, setSelectedMetric] = useState("normalized_accuracy");
  const [selectedRateType, setSelectedRateType] = useState("click"); // "click" | "report" | "ignore"
  const [filterPersona, setFilterPersona] = useState("all");
  const [filterModel, setFilterModel] = useState("all");
  const [filterPrompt, setFilterPrompt] = useState("all");
  const [filterStatus, setFilterStatus] = useState("all");
  const [fidelityRange, setFidelityRange] = useState([0, 100]);
  const [sortConfig, setSortConfig] = useState({
    key: "normalized_accuracy",
    direction: "desc",
  });
  const [visibleColumns, setVisibleColumns] = useState({
    persona: true,
    model: true,
    prompt: true,
    fidelity: true,
    aiClick: true,
    humanClick: true,
    deviation: true,
    status: true,
  });
  const [showColumnMenu, setShowColumnMenu] = useState(false);
  const [showAdvancedFilters, setShowAdvancedFilters] = useState(false);
  const [heatmapView, setHeatmapView] = useState("persona-model"); // persona-model, persona-prompt, model-prompt

  // Detailed combinations state (4D: LLM × Persona × Email × Prompt)
  const [detailedCombinations, setDetailedCombinations] = useState(null);
  const [loadingCombinations, setLoadingCombinations] = useState(false);
  const [combFilterEmail, setCombFilterEmail] = useState("all");
  const [combFilterPersona, setCombFilterPersona] = useState("all");
  const [combFilterModel, setCombFilterModel] = useState("all");
  const [combFilterPrompt, setCombFilterPrompt] = useState("all");
  // Email attribute filters for business analysis
  const [combFilterEmailType, setCombFilterEmailType] = useState("all"); // phishing/legitimate
  const [combFilterSenderFamiliarity, setCombFilterSenderFamiliarity] =
    useState("all"); // familiar/unfamiliar
  const [combFilterUrgency, setCombFilterUrgency] = useState("all"); // high/medium/low
  const [combFilterFraming, setCombFilterFraming] = useState("all"); // threat/reward/neutral
  const [combFilterAggression, setCombFilterAggression] = useState("all"); // very_high/high/medium/low
  const [combSortKey, setCombSortKey] = useState("fidelity");
  const [combSortDir, setCombSortDir] = useState("desc");

  // Load detailed combinations when switching to dashboard tab
  useEffect(() => {
    if (
      activeSubTab === "dashboard" &&
      currentExperiment &&
      !detailedCombinations
    ) {
      loadDetailedCombinations();
    }
  }, [activeSubTab, currentExperiment]);

  const loadDetailedCombinations = async () => {
    if (!currentExperiment?.experiment_id) return;
    setLoadingCombinations(true);
    try {
      const data = await api.getDetailedCombinations(
        currentExperiment.experiment_id,
      );
      setDetailedCombinations(data);
    } catch (error) {
      console.error("Failed to load detailed combinations:", error);
    }
    setLoadingCombinations(false);
  };

  // Extract data from results - handle both field names
  const fidelityData = results?.fidelity_results || results?.fidelity || [];
  const modelComparison = results?.model_comparison || {};
  const threshold =
    results?.thresholds?.accuracy || results?.threshold || FIDELITY_THRESHOLD;

  // Get unique values for filters
  const personas = useMemo(() => {
    const set = new Set(
      fidelityData.map((d) => d.persona_name || d.persona_id),
    );
    return ["all", ...Array.from(set)];
  }, [fidelityData]);

  const models = useMemo(() => {
    const set = new Set(fidelityData.map((d) => d.model_id));
    return ["all", ...Array.from(set)];
  }, [fidelityData]);

  const prompts = useMemo(() => {
    const set = new Set(fidelityData.map((d) => d.prompt_config || "baseline"));
    return ["all", ...Array.from(set)];
  }, [fidelityData]);

  // Filter data with all filters applied
  const filteredData = useMemo(() => {
    return fidelityData.filter((d) => {
      const personaMatch =
        filterPersona === "all" ||
        (d.persona_name || d.persona_id) === filterPersona;
      const modelMatch = filterModel === "all" || d.model_id === filterModel;
      const promptMatch =
        filterPrompt === "all" || d.prompt_config === filterPrompt;
      const fidelityValue = (d.normalized_accuracy || 0) * 100;
      const fidelityMatch =
        fidelityValue >= fidelityRange[0] && fidelityValue <= fidelityRange[1];
      const statusMatch =
        filterStatus === "all" ||
        (filterStatus === "passing" &&
          (d.meets_threshold || d.normalized_accuracy >= threshold)) ||
        (filterStatus === "failing" &&
          !(d.meets_threshold || d.normalized_accuracy >= threshold));

      return (
        personaMatch &&
        modelMatch &&
        promptMatch &&
        fidelityMatch &&
        statusMatch
      );
    });
  }, [
    fidelityData,
    filterPersona,
    filterModel,
    filterPrompt,
    fidelityRange,
    filterStatus,
    threshold,
  ]);

  // Sort data
  const sortedData = useMemo(() => {
    const sorted = [...filteredData];
    sorted.sort((a, b) => {
      let aVal, bVal;
      switch (sortConfig.key) {
        case "persona":
          aVal = a.persona_name || a.persona_id || "";
          bVal = b.persona_name || b.persona_id || "";
          break;
        case "model":
          aVal = a.model_id || "";
          bVal = b.model_id || "";
          break;
        case "prompt":
          aVal = a.prompt_config || "";
          bVal = b.prompt_config || "";
          break;
        case "normalized_accuracy":
          aVal = a.normalized_accuracy || 0;
          bVal = b.normalized_accuracy || 0;
          break;
        case "ai_click_rate":
          aVal = a.ai_click_rate || 0;
          bVal = b.ai_click_rate || 0;
          break;
        case "human_click_rate":
          aVal = a.human_click_rate || 0;
          bVal = b.human_click_rate || 0;
          break;
        case "deviation":
          aVal = (a.ai_click_rate || 0) - (a.human_click_rate || 0);
          bVal = (b.ai_click_rate || 0) - (b.human_click_rate || 0);
          break;
        default:
          aVal = a[sortConfig.key] || 0;
          bVal = b[sortConfig.key] || 0;
      }
      if (typeof aVal === "string") {
        return sortConfig.direction === "asc"
          ? aVal.localeCompare(bVal)
          : bVal.localeCompare(aVal);
      }
      return sortConfig.direction === "asc" ? aVal - bVal : bVal - aVal;
    });
    return sorted;
  }, [filteredData, sortConfig]);

  // Aggregate by model for comparison chart (includes behavioral rates)
  // Now uses selectedRateType for fidelity calculations
  const modelAggregates = useMemo(() => {
    const byModel = {};
    filteredData.forEach((d) => {
      if (!byModel[d.model_id]) {
        byModel[d.model_id] = {
          fidelityValues: [], // Store fidelity based on selected rate type
          dataPoints: [], // Store full data points for recalculation
          costs: [],
          trials: 0,
          aiClicks: [],
          humanClicks: [],
          aiReports: [],
          humanReports: [],
        };
      }
      // Calculate fidelity based on selected rate type
      const fidelity = getFidelityForRateType(d, selectedRateType);
      if (!isNaN(fidelity)) {
        byModel[d.model_id].fidelityValues.push(fidelity);
      }
      byModel[d.model_id].dataPoints.push(d);
      byModel[d.model_id].trials += d.trial_count || 1;
      // Collect behavioral rates
      if (!isNaN(d.ai_click_rate))
        byModel[d.model_id].aiClicks.push(d.ai_click_rate);
      if (!isNaN(d.human_click_rate))
        byModel[d.model_id].humanClicks.push(d.human_click_rate);
      if (!isNaN(d.ai_report_rate))
        byModel[d.model_id].aiReports.push(d.ai_report_rate);
      if (!isNaN(d.human_report_rate))
        byModel[d.model_id].humanReports.push(d.human_report_rate);
    });

    return Object.entries(byModel)
      .map(([model, data]) => {
        const meanAiClick =
          data.aiClicks.length > 0
            ? data.aiClicks.reduce((a, b) => a + b, 0) / data.aiClicks.length
            : 0;
        const meanHumanClick =
          data.humanClicks.length > 0
            ? data.humanClicks.reduce((a, b) => a + b, 0) / data.humanClicks.length
            : 0;
        const meanAiReport =
          data.aiReports.length > 0
            ? data.aiReports.reduce((a, b) => a + b, 0) / data.aiReports.length
            : 0;
        const meanHumanReport =
          data.humanReports.length > 0
            ? data.humanReports.reduce((a, b) => a + b, 0) / data.humanReports.length
            : 0;
        const meanAiIgnore = Math.max(0, 1 - meanAiClick - meanAiReport);
        const meanHumanIgnore = Math.max(0, 1 - meanHumanClick - meanHumanReport);

        return {
          model,
          mean_fidelity:
            data.fidelityValues.length > 0
              ? data.fidelityValues.reduce((a, b) => a + b, 0) / data.fidelityValues.length
              : 0,
          min_fidelity: data.fidelityValues.length > 0 ? Math.min(...data.fidelityValues) : 0,
          max_fidelity: data.fidelityValues.length > 0 ? Math.max(...data.fidelityValues) : 0,
          n_conditions: data.fidelityValues.length,
          trials: data.trials,
          passing: data.fidelityValues.filter((v) => v >= threshold).length,
          meets_threshold:
            data.fidelityValues.length > 0 && data.fidelityValues.every((v) => v >= threshold),
          // Behavioral rates per model
          mean_ai_click: meanAiClick,
          mean_human_click: meanHumanClick,
          mean_ai_report: meanAiReport,
          mean_human_report: meanHumanReport,
          mean_ai_ignore: meanAiIgnore,
          mean_human_ignore: meanHumanIgnore,
        };
      })
      .sort((a, b) => b.mean_fidelity - a.mean_fidelity);
  }, [filteredData, threshold, selectedRateType]);

  // Aggregate by prompt config - now uses selectedRateType
  const promptAggregates = useMemo(() => {
    const byPrompt = {};
    filteredData.forEach((d) => {
      const config = d.prompt_config || "baseline";
      if (!byPrompt[config]) {
        byPrompt[config] = { values: [], passing: 0, total: 0 };
      }
      const fidelity = getFidelityForRateType(d, selectedRateType);
      if (!isNaN(fidelity)) {
        byPrompt[config].values.push(fidelity);
        byPrompt[config].total++;
        if (fidelity >= threshold) {
          byPrompt[config].passing++;
        }
      }
    });

    return Object.entries(byPrompt).map(([prompt, data]) => ({
      prompt,
      mean_fidelity:
        data.values.length > 0
          ? data.values.reduce((a, b) => a + b, 0) / data.values.length
          : 0,
      n: data.values.length,
      passing: data.passing,
      total: data.total,
      passRate: data.total > 0 ? data.passing / data.total : 0,
    }));
  }, [filteredData, threshold, selectedRateType]);

  // Aggregate by persona - now uses selectedRateType
  const personaAggregates = useMemo(() => {
    const byPersona = {};
    filteredData.forEach((d) => {
      const persona = d.persona_name || d.persona_id;
      if (!byPersona[persona]) {
        byPersona[persona] = {
          fidelityValues: [],
          aiClicks: [],
          humanClicks: [],
          aiReports: [],
          humanReports: [],
        };
      }
      const fidelity = getFidelityForRateType(d, selectedRateType);
      if (!isNaN(fidelity)) {
        byPersona[persona].fidelityValues.push(fidelity);
      }
      if (!isNaN(d.ai_click_rate)) {
        byPersona[persona].aiClicks.push(d.ai_click_rate);
      }
      if (!isNaN(d.human_click_rate)) {
        byPersona[persona].humanClicks.push(d.human_click_rate);
      }
      if (!isNaN(d.ai_report_rate)) {
        byPersona[persona].aiReports.push(d.ai_report_rate);
      }
      if (!isNaN(d.human_report_rate)) {
        byPersona[persona].humanReports.push(d.human_report_rate);
      }
    });

    return Object.entries(byPersona).map(([persona, data]) => {
      const mean_ai_click =
        data.aiClicks.length > 0
          ? data.aiClicks.reduce((a, b) => a + b, 0) / data.aiClicks.length
          : 0;
      const mean_human_click =
        data.humanClicks.length > 0
          ? data.humanClicks.reduce((a, b) => a + b, 0) / data.humanClicks.length
          : 0;
      const mean_ai_report =
        data.aiReports.length > 0
          ? data.aiReports.reduce((a, b) => a + b, 0) / data.aiReports.length
          : 0;
      const mean_human_report =
        data.humanReports.length > 0
          ? data.humanReports.reduce((a, b) => a + b, 0) / data.humanReports.length
          : 0;
      const mean_ai_ignore = Math.max(0, 1 - mean_ai_click - mean_ai_report);
      const mean_human_ignore = Math.max(0, 1 - mean_human_click - mean_human_report);

      return {
        persona,
        mean_fidelity:
          data.fidelityValues.length > 0
            ? data.fidelityValues.reduce((a, b) => a + b, 0) / data.fidelityValues.length
            : 0,
        mean_ai_click,
        mean_human_click,
        mean_ai_report,
        mean_human_report,
        mean_ai_ignore,
        mean_human_ignore,
        // Get AI and Human rate based on selected rate type for deviation chart
        mean_ai_rate: selectedRateType === "click" ? mean_ai_click :
                      selectedRateType === "report" ? mean_ai_report : mean_ai_ignore,
        mean_human_rate: selectedRateType === "click" ? mean_human_click :
                         selectedRateType === "report" ? mean_human_report : mean_human_ignore,
        n: data.fidelityValues.length,
      };
    });
  }, [filteredData, selectedRateType]);

  // Heatmap data: Persona × Model - now uses selectedRateType
  const heatmapData = useMemo(() => {
    const data = [];
    const modelList = Array.from(new Set(filteredData.map((d) => d.model_id)));
    const personaList = Array.from(
      new Set(filteredData.map((d) => d.persona_name || d.persona_id)),
    );

    personaList.forEach((persona) => {
      modelList.forEach((model) => {
        const matching = filteredData.filter(
          (d) =>
            (d.persona_name || d.persona_id) === persona &&
            d.model_id === model,
        );
        if (matching.length > 0) {
          const avgFidelity =
            matching.reduce((sum, d) => sum + getFidelityForRateType(d, selectedRateType), 0) /
            matching.length;
          data.push({
            persona,
            model,
            fidelity: avgFidelity,
            count: matching.length,
          });
        }
      });
    });
    return { data, models: modelList, personas: personaList };
  }, [filteredData, selectedRateType]);

  // Heatmap data: Persona × Prompt - now uses selectedRateType
  const heatmapPersonaPrompt = useMemo(() => {
    const data = [];
    const promptList = Array.from(
      new Set(filteredData.map((d) => d.prompt_config || "baseline")),
    );
    const personaList = Array.from(
      new Set(filteredData.map((d) => d.persona_name || d.persona_id)),
    );

    personaList.forEach((persona) => {
      promptList.forEach((prompt) => {
        const matching = filteredData.filter(
          (d) =>
            (d.persona_name || d.persona_id) === persona &&
            (d.prompt_config || "baseline") === prompt,
        );
        if (matching.length > 0) {
          const avgFidelity =
            matching.reduce((sum, d) => sum + getFidelityForRateType(d, selectedRateType), 0) /
            matching.length;
          data.push({
            persona,
            prompt,
            fidelity: avgFidelity,
            count: matching.length,
          });
        }
      });
    });
    return { data, prompts: promptList, personas: personaList };
  }, [filteredData, selectedRateType]);

  // Heatmap data: Model × Prompt - now uses selectedRateType
  const heatmapModelPrompt = useMemo(() => {
    const data = [];
    const promptList = Array.from(
      new Set(filteredData.map((d) => d.prompt_config || "baseline")),
    );
    const modelList = Array.from(new Set(filteredData.map((d) => d.model_id)));

    modelList.forEach((model) => {
      promptList.forEach((prompt) => {
        const matching = filteredData.filter(
          (d) =>
            d.model_id === model && (d.prompt_config || "baseline") === prompt,
        );
        if (matching.length > 0) {
          const avgFidelity =
            matching.reduce((sum, d) => sum + getFidelityForRateType(d, selectedRateType), 0) /
            matching.length;
          data.push({
            model,
            prompt,
            fidelity: avgFidelity,
            count: matching.length,
          });
        }
      });
    });
    return { data, prompts: promptList, models: modelList };
  }, [filteredData, selectedRateType]);

  // Model × Prompt breakdown for detailed analysis - now uses selectedRateType
  const modelPromptBreakdown = useMemo(() => {
    const breakdown = {};
    filteredData.forEach((d) => {
      const key = `${d.model_id}|${d.prompt_config || "baseline"}`;
      if (!breakdown[key]) {
        breakdown[key] = {
          model: d.model_id,
          prompt: d.prompt_config || "baseline",
          fidelityValues: [],
          aiClicks: [],
          humanClicks: [],
          aiReports: [],
          humanReports: [],
        };
      }
      const fidelity = getFidelityForRateType(d, selectedRateType);
      if (!isNaN(fidelity))
        breakdown[key].fidelityValues.push(fidelity);
      if (!isNaN(d.ai_click_rate))
        breakdown[key].aiClicks.push(d.ai_click_rate);
      if (!isNaN(d.human_click_rate))
        breakdown[key].humanClicks.push(d.human_click_rate);
      if (!isNaN(d.ai_report_rate))
        breakdown[key].aiReports.push(d.ai_report_rate);
      if (!isNaN(d.human_report_rate))
        breakdown[key].humanReports.push(d.human_report_rate);
    });

    return Object.values(breakdown)
      .map((item) => {
        const avg_ai_click =
          item.aiClicks.length > 0
            ? item.aiClicks.reduce((a, b) => a + b, 0) / item.aiClicks.length
            : 0;
        const avg_human_click =
          item.humanClicks.length > 0
            ? item.humanClicks.reduce((a, b) => a + b, 0) / item.humanClicks.length
            : 0;
        const avg_ai_report =
          item.aiReports.length > 0
            ? item.aiReports.reduce((a, b) => a + b, 0) / item.aiReports.length
            : 0;
        const avg_human_report =
          item.humanReports.length > 0
            ? item.humanReports.reduce((a, b) => a + b, 0) / item.humanReports.length
            : 0;
        const avg_ai_ignore = Math.max(0, 1 - avg_ai_click - avg_ai_report);
        const avg_human_ignore = Math.max(0, 1 - avg_human_click - avg_human_report);

        // Get AI and Human rate based on selectedRateType
        const avg_ai_rate = selectedRateType === "click" ? avg_ai_click :
                           selectedRateType === "report" ? avg_ai_report : avg_ai_ignore;
        const avg_human_rate = selectedRateType === "click" ? avg_human_click :
                              selectedRateType === "report" ? avg_human_report : avg_human_ignore;

        return {
          ...item,
          avg_fidelity:
            item.fidelityValues.length > 0
              ? item.fidelityValues.reduce((a, b) => a + b, 0) / item.fidelityValues.length
              : 0,
          avg_ai_click,
          avg_human_click,
          avg_ai_report,
          avg_human_report,
          avg_ai_ignore,
          avg_human_ignore,
          avg_ai_rate,
          avg_human_rate,
          pass_rate:
            item.fidelityValues.filter((v) => v >= threshold).length /
            Math.max(item.fidelityValues.length, 1),
          n: item.fidelityValues.length,
        };
      })
      .sort((a, b) => b.avg_fidelity - a.avg_fidelity);
  }, [filteredData, threshold, selectedRateType]);

  // Calculate summary stats safely - now uses selectedRateType for fidelity
  const summaryStats = useMemo(() => {
    const validData = filteredData.filter(
      (d) =>
        d.normalized_accuracy !== undefined && !isNaN(d.normalized_accuracy),
    );

    if (validData.length === 0) {
      return {
        meanFidelity: 0,
        passingCount: 0,
        passRate: 0,
        total: 0,
        meanAiClick: 0,
        meanHumanClick: 0,
        meanAiReport: 0,
        meanHumanReport: 0,
        meanAiIgnore: 0,
        meanHumanIgnore: 0,
      };
    }

    // Calculate click rates
    const aiClickData = validData.filter((d) => !isNaN(d.ai_click_rate));
    const meanAiClick =
      aiClickData.length > 0
        ? aiClickData.reduce((a, d) => a + d.ai_click_rate, 0) /
          aiClickData.length
        : 0;

    const humanClickData = validData.filter((d) => !isNaN(d.human_click_rate));
    const meanHumanClick =
      humanClickData.length > 0
        ? humanClickData.reduce((a, d) => a + d.human_click_rate, 0) /
          humanClickData.length
        : 0;

    // Calculate report rates
    const aiReportData = validData.filter((d) => !isNaN(d.ai_report_rate));
    const meanAiReport =
      aiReportData.length > 0
        ? aiReportData.reduce((a, d) => a + d.ai_report_rate, 0) /
          aiReportData.length
        : 0;

    const humanReportData = validData.filter(
      (d) => !isNaN(d.human_report_rate),
    );
    const meanHumanReport =
      humanReportData.length > 0
        ? humanReportData.reduce((a, d) => a + d.human_report_rate, 0) /
          humanReportData.length
        : 0;

    // Calculate ignore rates (1 - click - report)
    const meanAiIgnore = Math.max(0, 1 - meanAiClick - meanAiReport);
    const meanHumanIgnore = Math.max(0, 1 - meanHumanClick - meanHumanReport);

    // Calculate fidelity based on selected rate type
    const meanFidelity =
      validData.reduce((a, d) => a + getFidelityForRateType(d, selectedRateType), 0) /
      validData.length;
    const passingCount = validData.filter(
      (d) => getFidelityForRateType(d, selectedRateType) >= threshold,
    ).length;
    const passRate = passingCount / validData.length;

    return {
      meanFidelity,
      passingCount,
      passRate,
      total: validData.length,
      meanAiClick,
      meanHumanClick,
      meanAiReport,
      meanHumanReport,
      meanAiIgnore,
      meanHumanIgnore,
    };
  }, [filteredData, threshold, selectedRateType]);

  // Cost analysis data - now uses selectedRateType for fidelity
  const costAnalysis = useMemo(() => {
    const byModel = {};
    fidelityData.forEach((d) => {
      if (!byModel[d.model_id]) {
        byModel[d.model_id] = {
          model: d.model_id,
          totalCost: 0,
          trials: 0,
          fidelityValues: [],
          latencies: [],
          conditions: 0,
        };
      }
      byModel[d.model_id].totalCost += d.cost || 0;
      byModel[d.model_id].trials += d.trial_count || 1;
      byModel[d.model_id].conditions += 1;
      const fidelity = getFidelityForRateType(d, selectedRateType);
      if (!isNaN(fidelity)) {
        byModel[d.model_id].fidelityValues.push(fidelity);
      }
      if (d.latency_ms) {
        byModel[d.model_id].latencies.push(d.latency_ms);
      }
    });

    return Object.values(byModel)
      .map((m) => {
        const avgFidelity =
          m.fidelityValues.length > 0
            ? m.fidelityValues.reduce((a, b) => a + b, 0) /
              m.fidelityValues.length
            : 0;
        const costPerTrial = m.trials > 0 ? m.totalCost / m.trials : 0;
        return {
          ...m,
          costPerTrial,
          avgFidelity,
          p50Latency:
            m.latencies.length > 0
              ? [...m.latencies].sort((a, b) => a - b)[
                  Math.floor(m.latencies.length / 2)
                ]
              : 0,
          // Value score: fidelity per unit cost (higher = better value)
          valueScore: costPerTrial > 0 ? avgFidelity / costPerTrial : 0,
        };
      })
      .sort((a, b) => b.avgFidelity - a.avgFidelity);
  }, [fidelityData, selectedRateType]);

  // Boundary conditions analysis - now uses selectedRateType
  const boundaryConditions = useMemo(() => {
    const conditions = [];
    const rateLabel = getRateTypeLabel(selectedRateType);
    const actionVerb = selectedRateType === "click" ? "Clicking" :
                       selectedRateType === "report" ? "Reporting" : "Ignoring";

    personaAggregates.forEach((p) => {
      // Use the appropriate rate based on selectedRateType
      const aiRate = p.mean_ai_rate;
      const humanRate = p.mean_human_rate;
      const rateDiff = Math.abs(aiRate - humanRate);

      if (rateDiff > 0.2) {
        conditions.push({
          type:
            aiRate > humanRate
              ? `Over ${actionVerb}`
              : `Under ${actionVerb}`,
          severity:
            rateDiff > 0.4 ? "high" : rateDiff > 0.3 ? "medium" : "low",
          persona: p.persona,
          aiRate: aiRate,
          humanRate: humanRate,
          diff: rateDiff,
          rateType: selectedRateType,
          suggestion:
            aiRate > humanRate
              ? `Adjust persona prompt to reduce AI ${rateLabel.toLowerCase()}`
              : `Adjust persona prompt to increase AI ${rateLabel.toLowerCase()}`,
        });
      }
    });
    return conditions.sort((a, b) =>
      b.severity === "high" ? 1 : a.severity === "high" ? -1 : b.diff - a.diff,
    );
  }, [personaAggregates, selectedRateType]);

  // Handle sort
  const handleSort = (key) => {
    setSortConfig((prev) => ({
      key,
      direction: prev.key === key && prev.direction === "desc" ? "asc" : "desc",
    }));
  };

  // Toggle column visibility
  const toggleColumn = (column) => {
    setVisibleColumns((prev) => ({ ...prev, [column]: !prev[column] }));
  };

  // Clear all filters
  const clearFilters = () => {
    setFilterPersona("all");
    setFilterModel("all");
    setFilterPrompt("all");
    setFilterStatus("all");
    setFidelityRange([0, 100]);
  };

  // No results state
  if (!results || fidelityData.length === 0) {
    return (
      <div className="space-y-6">
        <div className="bg-white rounded-xl border p-6">
          <h3 className="text-lg font-semibold mb-4">
            Select Completed Experiment
          </h3>
          {experiments.filter((e) => e.status === "completed").length > 0 ? (
            <div className="space-y-2">
              {experiments
                .filter((e) => e.status === "completed")
                .map((exp) => (
                  <button
                    key={exp.experiment_id}
                    onClick={() => onSelectExperiment(exp)}
                    className={`w-full p-3 rounded-lg border text-left transition ${
                      currentExperiment?.experiment_id === exp.experiment_id
                        ? "border-purple-300 bg-purple-50"
                        : "border-gray-200 hover:bg-gray-50"
                    }`}
                  >
                    <div className="font-medium">{exp.name}</div>
                    <div className="text-sm text-gray-500">
                      {exp.completed_trials?.toLocaleString() || "?"} trials
                    </div>
                  </button>
                ))}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500">
              <BarChart2 className="mx-auto mb-2 text-gray-300" size={32} />
              <p>No completed experiments</p>
              <p className="text-sm">Run an experiment to see results</p>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Sub-Tab Navigation */}
      <div className="bg-white rounded-xl border p-2 flex gap-1 overflow-x-auto">
        {RESULT_SUB_TABS.map((tab) => (
          <button
            key={tab.id}
            onClick={() => setActiveSubTab(tab.id)}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition whitespace-nowrap ${
              activeSubTab === tab.id
                ? "bg-purple-100 text-purple-700"
                : "text-gray-600 hover:bg-gray-100"
            }`}
          >
            <tab.icon size={16} />
            {tab.label}
          </button>
        ))}
      </div>

      {/* Summary Cards - Only shown on Dashboard tab */}
      {activeSubTab === "dashboard" && (
        <>
          {/* Summary Cards - Row 1: Unified Core Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-3">
            <SummaryCard
              icon={<BarChart2 className="text-purple-600" />}
              label="Total Trials"
              value={results?.total_trials?.toLocaleString() || filteredData.length}
              status="neutral"
              sublabel={`${summaryStats.total} conditions`}
            />
            <SummaryCard
              icon={<Grid3X3 className="text-blue-600" />}
              label="Combinations"
              value={summaryStats.total?.toLocaleString() || "0"}
              status="neutral"
              sublabel="Unique test cases"
            />
            <SummaryCard
              icon={<Target className="text-green-600" />}
              label="Avg Fidelity"
              value={`${(summaryStats.meanFidelity * 100).toFixed(1)}%`}
              status={
                summaryStats.meanFidelity >= threshold ? "success" : "warning"
              }
              sublabel={
                summaryStats.meanFidelity >= threshold
                  ? "Above threshold"
                  : "Below threshold"
              }
            />
            <SummaryCard
              icon={<TrendingUp className="text-blue-600" />}
              label="Pass Rate"
              value={`${(summaryStats.passRate * 100).toFixed(0)}%`}
              status={summaryStats.passRate >= 0.8 ? "success" : "warning"}
              sublabel={`${summaryStats.passingCount}/${summaryStats.total} passing`}
            />
            <SummaryCard
              icon={<CheckCircle className="text-green-600" />}
              label="Parse Success"
              value={`${((results?.parse_success_rate || 0.994) * 100).toFixed(1)}%`}
              status="success"
              sublabel="Valid responses"
            />
            <SummaryCard
              icon={<DollarSign className="text-amber-600" />}
              label="Total Cost"
              value={`$${(results?.total_cost || 0).toFixed(4)}`}
              status="neutral"
              sublabel="All trials"
            />
          </div>

          {/* Summary Cards - Row 2: Behavioral Rates Comparison with LLM Breakdown - CLICKABLE */}
          <RateTypeSwitcher
            selectedRateType={selectedRateType}
            setSelectedRateType={setSelectedRateType}
            summaryStats={summaryStats}
            modelAggregates={modelAggregates}
          />

      {/* Quick Summary - Pareto Optimal Recommendations */}
      <div className="bg-white rounded-xl border p-6">
        <h3 className="text-sm font-medium text-gray-500 mb-4">
          Quick Summary
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          {/* Best Value Model (Pareto optimal - high fidelity, low cost) */}
          <div>
            <div className="text-xs text-gray-500">Best Value Model</div>
            <div className="text-lg font-semibold text-black">
              {(() => {
                const paretoModels = costAnalysis
                  .filter(
                    (m) =>
                      !costAnalysis.some(
                        (other) =>
                          other.avgFidelity > m.avgFidelity &&
                          other.costPerTrial < m.costPerTrial,
                      ),
                  )
                  .sort((a, b) => b.valueScore - a.valueScore);
                return paretoModels[0]?.model || "N/A";
              })()}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {(() => {
                const paretoModels = costAnalysis
                  .filter(
                    (m) =>
                      !costAnalysis.some(
                        (other) =>
                          other.avgFidelity > m.avgFidelity &&
                          other.costPerTrial < m.costPerTrial,
                      ),
                  )
                  .sort((a, b) => b.valueScore - a.valueScore);
                const best = paretoModels[0];
                return best
                  ? `${(best.avgFidelity * 100).toFixed(1)}% fidelity · $${best.costPerTrial.toFixed(5)}/trial`
                  : "";
              })()}
            </div>
          </div>
          {/* Highest Fidelity Model */}
          <div>
            <div className="text-xs text-gray-500">Highest Fidelity Model</div>
            <div className="text-lg font-semibold text-black">
              {modelAggregates[0]?.model || "N/A"}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {modelAggregates[0]
                ? `${(modelAggregates[0].mean_fidelity * 100).toFixed(1)}% fidelity`
                : ""}
            </div>
          </div>
          {/* Best Prompt Config */}
          <div>
            <div className="text-xs text-gray-500">Best Prompt Config</div>
            <div className="text-lg font-semibold text-black">
              {(() => {
                const sorted = [...promptAggregates].sort(
                  (a, b) => b.mean_fidelity - a.mean_fidelity,
                );
                const prompt = sorted[0]?.prompt;
                if (prompt === "cot") return "Chain-of-Thought";
                if (prompt === "stats") return "Behavioral Stats";
                return prompt || "N/A";
              })()}
            </div>
            <div className="text-xs text-gray-500 mt-1">
              {(() => {
                const sorted = [...promptAggregates].sort(
                  (a, b) => b.mean_fidelity - a.mean_fidelity,
                );
                return sorted[0]
                  ? `${(sorted[0].mean_fidelity * 100).toFixed(1)}% avg fidelity`
                  : "";
              })()}
            </div>
          </div>
          {/* Boundary Issues */}
          <div>
            <div className="text-xs text-gray-500">Boundary Issues</div>
            <div className="text-lg font-semibold">
              {boundaryConditions.length} detected
            </div>
            <div className="text-xs text-amber-600 mt-1">
              {boundaryConditions.filter((b) => b.severity === "high").length}{" "}
              high severity
            </div>
          </div>
        </div>
      </div>
        </>
      )}

      {/* Sub-Tab Content */}
      {activeSubTab === "dashboard" && (
        <UnifiedDashboard
          detailedCombinations={detailedCombinations}
          loadingCombinations={loadingCombinations}
          onRefresh={loadDetailedCombinations}
          fidelityData={fidelityData}
          modelAggregates={modelAggregates}
          promptAggregates={promptAggregates}
          personaAggregates={personaAggregates}
          costAnalysis={costAnalysis}
          boundaryConditions={boundaryConditions}
          threshold={threshold}
          summaryStats={summaryStats}
          results={results}
          selectedRateType={selectedRateType}
          setSelectedRateType={setSelectedRateType}
        />
      )}

      {activeSubTab === "prompts" && (
        <PromptsSubTab
          promptAggregates={promptAggregates}
          modelPromptBreakdown={modelPromptBreakdown}
          selectedRateType={selectedRateType}
          setSelectedRateType={setSelectedRateType}
          summaryStats={summaryStats}
          modelAggregates={modelAggregates}
          threshold={threshold}
        />
      )}

      {activeSubTab === "models" && (
        <ModelsSubTab
          modelAggregates={modelAggregates}
          threshold={threshold}
          selectedRateType={selectedRateType}
          setSelectedRateType={setSelectedRateType}
          summaryStats={summaryStats}
        />
      )}

      {activeSubTab === "costs" && (
        <CostsSubTab
          costAnalysis={costAnalysis}
          modelAggregates={modelAggregates}
          selectedRateType={selectedRateType}
          setSelectedRateType={setSelectedRateType}
          summaryStats={summaryStats}
        />
      )}

      {activeSubTab === "boundaries" && (
        <BoundariesSubTab
          boundaryConditions={boundaryConditions}
          personaAggregates={personaAggregates}
          selectedRateType={selectedRateType}
          setSelectedRateType={setSelectedRateType}
          summaryStats={summaryStats}
          modelAggregates={modelAggregates}
        />
      )}
    </div>
  );
};

// =============================================================================
// UNIFIED DASHBOARD (Replaces Overview + Combinations + Fidelity)
// =============================================================================

const UnifiedDashboard = ({
  detailedCombinations,
  loadingCombinations,
  onRefresh,
  fidelityData,
  modelAggregates,
  promptAggregates,
  personaAggregates,
  costAnalysis,
  boundaryConditions,
  threshold,
  summaryStats,
  results,
  selectedRateType,
  setSelectedRateType,
}) => {
  // Visualization mode state (default to bubble)
  const [vizMode, setVizMode] = useState("bubble");

  // Dimension filters
  const [selectedModels, setSelectedModels] = useState([]);
  const [selectedPersonas, setSelectedPersonas] = useState([]);
  const [selectedEmails, setSelectedEmails] = useState([]);
  const [selectedPrompts, setSelectedPrompts] = useState([]);
  const [selectedEmailTypes, setSelectedEmailTypes] = useState([]);
  const [selectedUrgency, setSelectedUrgency] = useState([]);
  const [selectedFamiliarity, setSelectedFamiliarity] = useState([]);
  const [selectedFraming, setSelectedFraming] = useState([]);
  // Model category filters
  const [selectedSizeTier, setSelectedSizeTier] = useState([]);
  const [selectedOpenness, setSelectedOpenness] = useState([]);
  const [selectedArchitecture, setSelectedArchitecture] = useState([]);

  // Matrix axes configuration (removed persona from options)
  const [matrixXAxis, setMatrixXAxis] = useState("model");
  const [matrixYAxis, setMatrixYAxis] = useState("prompt");

  // Detail table state - multi-column sorting
  const [showDetailTable, setShowDetailTable] = useState(false);
  const [tableSortColumns, setTableSortColumns] = useState([
    { key: "fidelity", dir: "desc" },
  ]);

  // Get dimensions from data
  const dimensions = detailedCombinations?.dimensions || {};
  const allModels = dimensions.models || [];
  const allPersonas = dimensions.personas || [];
  const allEmails = dimensions.emails || [];
  const allPrompts = dimensions.prompts || [];
  const allEmailTypes = dimensions.email_types || [];
  const allUrgencyLevels = dimensions.urgency_levels || [];
  const allSenderFamiliarities = dimensions.sender_familiarities || [];
  const allFramingTypes = dimensions.framing_types || [];

  // Helper to get model category
  const getModelCategory = (modelName, categoryType) => {
    const category = MODEL_CATEGORIES[modelName];
    if (!category) return "Unknown";
    return category[categoryType] || "Unknown";
  };

  // Filter the detailed results based on selections
  const filteredResults = useMemo(() => {
    if (!detailedCombinations?.detailed_results) return [];

    let data = [...detailedCombinations.detailed_results];

    if (selectedModels.length > 0) {
      data = data.filter((d) => selectedModels.includes(d.model_id));
    }
    if (selectedPersonas.length > 0) {
      data = data.filter((d) => selectedPersonas.includes(d.persona_name));
    }
    if (selectedEmails.length > 0) {
      data = data.filter((d) => selectedEmails.includes(d.email_id));
    }
    if (selectedPrompts.length > 0) {
      data = data.filter((d) => selectedPrompts.includes(d.prompt_config));
    }
    if (selectedEmailTypes.length > 0) {
      data = data.filter((d) => selectedEmailTypes.includes(d.email_type));
    }
    if (selectedUrgency.length > 0) {
      data = data.filter((d) => selectedUrgency.includes(d.urgency_level));
    }
    if (selectedFamiliarity.length > 0) {
      data = data.filter((d) =>
        selectedFamiliarity.includes(d.sender_familiarity),
      );
    }
    if (selectedFraming.length > 0) {
      data = data.filter((d) => selectedFraming.includes(d.framing_type));
    }
    // Model category filters
    if (selectedSizeTier.length > 0) {
      data = data.filter((d) =>
        selectedSizeTier.includes(getModelCategory(d.model_id, "sizeTier")),
      );
    }
    if (selectedOpenness.length > 0) {
      data = data.filter((d) =>
        selectedOpenness.includes(getModelCategory(d.model_id, "openness")),
      );
    }
    if (selectedArchitecture.length > 0) {
      data = data.filter((d) =>
        selectedArchitecture.includes(getModelCategory(d.model_id, "architecture")),
      );
    }

    return data;
  }, [
    detailedCombinations,
    selectedModels,
    selectedPersonas,
    selectedEmails,
    selectedPrompts,
    selectedEmailTypes,
    selectedUrgency,
    selectedFamiliarity,
    selectedFraming,
    selectedSizeTier,
    selectedOpenness,
    selectedArchitecture,
  ]);

  // Calculate aggregated metrics for filtered data - uses selectedRateType
  const filteredMetrics = useMemo(() => {
    if (filteredResults.length === 0) {
      return {
        avgFidelity: 0,
        totalCost: 0,
        passRate: 0,
        count: 0,
        passing: 0,
      };
    }

    // Calculate fidelity based on selected rate type for each data point
    const avgFidelity =
      filteredResults.reduce((sum, d) => sum + getFidelityForRateType(d, selectedRateType), 0) /
      filteredResults.length;
    const totalCost = filteredResults.reduce(
      (sum, d) => sum + (d.cost || 0),
      0,
    );
    const passing = filteredResults.filter(
      (d) => getFidelityForRateType(d, selectedRateType) >= threshold,
    ).length;
    const passRate = passing / filteredResults.length;

    return {
      avgFidelity,
      totalCost,
      passRate,
      count: filteredResults.length,
      passing,
    };
  }, [filteredResults, threshold, selectedRateType]);

  // Generate matrix data PER PERSONA
  const matrixDataByPersona = useMemo(() => {
    if (filteredResults.length === 0) return {};

    const getAxisValues = (axis, data) => {
      switch (axis) {
        case "model":
          return [...new Set(data.map((d) => d.model_id))];
        case "prompt":
          return [...new Set(data.map((d) => d.prompt_config))];
        case "email_type":
          return [...new Set(data.map((d) => d.email_type))];
        case "urgency":
          return [...new Set(data.map((d) => d.urgency_level))];
        case "familiarity":
          return [...new Set(data.map((d) => d.sender_familiarity))];
        case "framing":
          return [...new Set(data.map((d) => d.framing_type))];
        default:
          return [];
      }
    };

    const getAxisValue = (d, axis) => {
      switch (axis) {
        case "model":
          return d.model_id;
        case "prompt":
          return d.prompt_config;
        case "email_type":
          return d.email_type;
        case "urgency":
          return d.urgency_level;
        case "familiarity":
          return d.sender_familiarity;
        case "framing":
          return d.framing_type;
        default:
          return "";
      }
    };

    const personas = [...new Set(filteredResults.map((d) => d.persona_name))];
    const result = {};

    personas.forEach((persona) => {
      const personaData = filteredResults.filter(
        (d) => d.persona_name === persona,
      );
      const rows = getAxisValues(matrixYAxis, personaData);
      const cols = getAxisValues(matrixXAxis, personaData);
      const cells = {};

      rows.forEach((row) => {
        cols.forEach((col) => {
          const matching = personaData.filter(
            (d) =>
              getAxisValue(d, matrixYAxis) === row &&
              getAxisValue(d, matrixXAxis) === col,
          );
          if (matching.length > 0) {
            const avgFidelity =
              matching.reduce((sum, d) => sum + getFidelityForRateType(d, selectedRateType), 0) /
              matching.length;
            const totalCost = matching.reduce(
              (sum, d) => sum + (d.cost || 0),
              0,
            );
            const passing = matching.filter(
              (d) => getFidelityForRateType(d, selectedRateType) >= threshold,
            ).length;
            cells[`${row}|${col}`] = {
              fidelity: avgFidelity,
              cost: totalCost,
              count: matching.length,
              passing,
              passRate: passing / matching.length,
            };
          }
        });
      });

      result[persona] = { rows, cols, cells };
    });

    return result;
  }, [filteredResults, matrixXAxis, matrixYAxis, threshold, selectedRateType]);

  // Generate bubble chart data PER PERSONA - colored by prompt config - uses selectedRateType
  const bubbleDataByPersona = useMemo(() => {
    if (filteredResults.length === 0) return {};

    const personas = [...new Set(filteredResults.map((d) => d.persona_name))];
    const result = {};

    personas.forEach((persona) => {
      const personaData = filteredResults.filter(
        (d) => d.persona_name === persona,
      );

      // Aggregate by model + prompt combination
      const byCombo = {};
      personaData.forEach((d) => {
        const key = `${d.model_id}|${d.prompt_config}`;
        if (!byCombo[key]) {
          byCombo[key] = {
            model: d.model_id,
            prompt: d.prompt_config,
            fidelities: [],
            costs: [],
            count: 0,
          };
        }
        byCombo[key].fidelities.push(getFidelityForRateType(d, selectedRateType));
        byCombo[key].costs.push(d.cost || 0);
        byCombo[key].count++;
      });

      result[persona] = Object.values(byCombo).map((data) => ({
        model: data.model,
        prompt: data.prompt,
        promptLabel: PROMPT_LABELS[data.prompt] || data.prompt,
        fidelity:
          (data.fidelities.reduce((a, b) => a + b, 0) /
            data.fidelities.length) *
          100,
        cost: data.costs.reduce((a, b) => a + b, 0), // Total cost for this combo
        count: data.count,
        passing: data.fidelities.filter((f) => f >= threshold).length,
      }));
    });

    return result;
  }, [filteredResults, threshold, selectedRateType]);

  // Calculate per-persona quick stats for best value model+prompt combination - uses selectedRateType
  const personaQuickStats = useMemo(() => {
    if (filteredResults.length === 0) return {};

    const personas = [...new Set(filteredResults.map((d) => d.persona_name))];
    const result = {};

    personas.forEach((persona) => {
      const personaData = filteredResults.filter(
        (d) => d.persona_name === persona,
      );

      // Aggregate by MODEL + PROMPT combination to find best configuration
      const byModelPrompt = {};
      personaData.forEach((d) => {
        const key = `${d.model_id}|${d.prompt_config}`;
        if (!byModelPrompt[key]) {
          byModelPrompt[key] = {
            model: d.model_id,
            prompt: d.prompt_config,
            fidelities: [],
            costs: [],
            aiClicks: [],
            humanClicks: [],
            aiReports: [],
            humanReports: [],
            count: 0,
          };
        }
        // Use selected rate type for fidelity calculation
        byModelPrompt[key].fidelities.push(getFidelityForRateType(d, selectedRateType));
        byModelPrompt[key].costs.push(d.cost || 0);
        if (!isNaN(d.ai_click_rate))
          byModelPrompt[key].aiClicks.push(d.ai_click_rate);
        if (!isNaN(d.human_click_rate))
          byModelPrompt[key].humanClicks.push(d.human_click_rate);
        if (!isNaN(d.ai_report_rate))
          byModelPrompt[key].aiReports.push(d.ai_report_rate);
        if (!isNaN(d.human_report_rate))
          byModelPrompt[key].humanReports.push(d.human_report_rate);
        byModelPrompt[key].count++;
      });

      // Calculate metrics for each model+prompt combination
      const comboStats = Object.values(byModelPrompt).map((m) => {
        const avgFidelity =
          m.fidelities.reduce((a, b) => a + b, 0) / m.fidelities.length;
        const totalCost = m.costs.reduce((a, b) => a + b, 0);
        const costPerTrial = m.count > 0 ? totalCost / m.count : 0;

        return {
          model: m.model,
          prompt: m.prompt,
          avgFidelity,
          totalCost,
          costPerTrial,
          aiClick:
            m.aiClicks.length > 0
              ? m.aiClicks.reduce((a, b) => a + b, 0) / m.aiClicks.length
              : 0,
          humanClick:
            m.humanClicks.length > 0
              ? m.humanClicks.reduce((a, b) => a + b, 0) / m.humanClicks.length
              : 0,
          aiReport:
            m.aiReports.length > 0
              ? m.aiReports.reduce((a, b) => a + b, 0) / m.aiReports.length
              : 0,
          humanReport:
            m.humanReports.length > 0
              ? m.humanReports.reduce((a, b) => a + b, 0) /
                m.humanReports.length
              : 0,
          count: m.count,
        };
      });

      // Find best value model+prompt with priority:
      // 1. Combinations that meet threshold (≥85% fidelity) - sorted by lowest cost
      // 2. If none meet threshold, pick highest fidelity combination
      const passingCombos = comboStats
        .filter((m) => m.avgFidelity >= threshold)
        .sort((a, b) => a.costPerTrial - b.costPerTrial); // Lowest cost first among passing

      const bestCombo =
        passingCombos.length > 0
          ? passingCombos[0] // Cheapest combo that meets threshold
          : comboStats.sort((a, b) => b.avgFidelity - a.avgFidelity)[0]; // Fallback to highest fidelity

      // Calculate average human rates for this persona
      const avgHumanClick =
        personaData.filter((d) => !isNaN(d.human_click_rate)).length > 0
          ? personaData
              .filter((d) => !isNaN(d.human_click_rate))
              .reduce((a, d) => a + d.human_click_rate, 0) /
            personaData.filter((d) => !isNaN(d.human_click_rate)).length
          : 0;
      const avgHumanReport =
        personaData.filter((d) => !isNaN(d.human_report_rate)).length > 0
          ? personaData
              .filter((d) => !isNaN(d.human_report_rate))
              .reduce((a, d) => a + d.human_report_rate, 0) /
            personaData.filter((d) => !isNaN(d.human_report_rate)).length
          : 0;

      result[persona] = {
        bestModel: bestCombo?.model || "N/A",
        bestPrompt: bestCombo?.prompt || "baseline",
        fidelity: bestCombo?.avgFidelity || 0,
        costPerTrial: bestCombo?.costPerTrial || 0,
        aiClick: bestCombo?.aiClick || 0,
        humanClick: avgHumanClick,
        aiReport: bestCombo?.aiReport || 0,
        humanReport: avgHumanReport,
        aiIgnore: Math.max(
          0,
          1 - (bestCombo?.aiClick || 0) - (bestCombo?.aiReport || 0),
        ),
        humanIgnore: Math.max(0, 1 - avgHumanClick - avgHumanReport),
        count: personaData.length,
        meetsThreshold: bestCombo?.avgFidelity >= threshold,
      };
    });

    return result;
  }, [filteredResults, threshold, selectedRateType]);

  // Clear all filters
  const clearAllFilters = () => {
    setSelectedModels([]);
    setSelectedPersonas([]);
    setSelectedEmails([]);
    setSelectedPrompts([]);
    setSelectedEmailTypes([]);
    setSelectedUrgency([]);
    setSelectedFamiliarity([]);
    setSelectedFraming([]);
    setSelectedSizeTier([]);
    setSelectedOpenness([]);
    setSelectedArchitecture([]);
  };

  // Toggle filter selection
  const toggleFilter = (setter, current, value) => {
    if (current.includes(value)) {
      setter(current.filter((v) => v !== value));
    } else {
      setter([...current, value]);
    }
  };

  // Multi-column sort handler
  const handleTableSort = (key) => {
    setTableSortColumns((prev) => {
      const existingIndex = prev.findIndex((s) => s.key === key);
      if (existingIndex >= 0) {
        // Toggle direction if already sorting by this column
        const updated = [...prev];
        updated[existingIndex] = {
          key,
          dir: prev[existingIndex].dir === "asc" ? "desc" : "asc",
        };
        return updated;
      } else {
        // Add to sort columns (max 3)
        const newSort = [...prev, { key, dir: "desc" }];
        return newSort.slice(-3);
      }
    });
  };

  // Multi-column sorted table data - uses selectedRateType for fidelity
  const sortedTableData = useMemo(() => {
    // Add computed fidelity based on selectedRateType
    const withComputedFidelity = filteredResults.map(d => ({
      ...d,
      fidelity: getFidelityForRateType(d, selectedRateType),
    }));
    const sorted = [...withComputedFidelity];
    sorted.sort((a, b) => {
      for (const { key, dir } of tableSortColumns) {
        const aVal = a[key] || 0;
        const bVal = b[key] || 0;
        if (aVal !== bVal) {
          return dir === "asc" ? (aVal > bVal ? 1 : -1) : aVal < bVal ? 1 : -1;
        }
      }
      return 0;
    });
    return sorted;
  }, [filteredResults, tableSortColumns, selectedRateType]);

  // Export filtered data to CSV - uses selectedRateType for fidelity
  const exportToCSV = () => {
    const rateLabel = getRateTypeLabel(selectedRateType);
    const headers = [
      "Model",
      "Persona",
      "Prompt",
      "Email Type",
      "Urgency",
      "Familiarity",
      "Framing",
      `Fidelity (${rateLabel})`,
      "Cost",
      "AI Click",
      "Human Click",
      "AI Report",
      "Human Report",
      "Status",
    ];
    const rows = filteredResults.map((row) => {
      const fidelity = getFidelityForRateType(row, selectedRateType);
      return [
        row.model_id,
        row.persona_name,
        PROMPT_LABELS[row.prompt_config] || row.prompt_config,
        row.email_type || "",
        row.urgency_level || "",
        row.sender_familiarity || "",
        row.framing_type || "",
        (fidelity * 100).toFixed(2),
        (row.cost || 0).toFixed(6),
        ((row.ai_click_rate || 0) * 100).toFixed(2),
        ((row.human_click_rate || 0) * 100).toFixed(2),
        ((row.ai_report_rate || 0) * 100).toFixed(2),
        ((row.human_report_rate || 0) * 100).toFixed(2),
        fidelity >= threshold ? "Pass" : "Fail",
      ];
    });
    const csvContent = [headers, ...rows].map((r) => r.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `dashboard_export_${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Check if we have any active filters
  const hasActiveFilters =
    selectedModels.length > 0 ||
    selectedPersonas.length > 0 ||
    selectedEmails.length > 0 ||
    selectedPrompts.length > 0 ||
    selectedEmailTypes.length > 0 ||
    selectedUrgency.length > 0 ||
    selectedFamiliarity.length > 0 ||
    selectedFraming.length > 0 ||
    selectedSizeTier.length > 0 ||
    selectedOpenness.length > 0 ||
    selectedArchitecture.length > 0;

  // Loading state
  if (loadingCombinations) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-purple-200 border-t-purple-600"></div>
        <span className="ml-4 text-gray-600">Loading dashboard data...</span>
      </div>
    );
  }

  // No data state
  if (!detailedCombinations) {
    return (
      <div className="bg-white rounded-xl border p-8 text-center">
        <BarChart2 size={48} className="mx-auto text-gray-300 mb-4" />
        <h3 className="text-lg font-semibold text-gray-900">
          No Data Available
        </h3>
        <p className="text-gray-500 mt-2">
          Run an experiment to see the unified dashboard.
        </p>
        <button
          onClick={onRefresh}
          className="mt-4 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
        >
          Load Data
        </button>
      </div>
    );
  }

  // Use allPersonas with consistent order, filtered to only those with data in current view
  const personas = allPersonas
    .filter((p) => matrixDataByPersona[p] || bubbleDataByPersona[p])
    .sort((a, b) => a.localeCompare(b)); // Alphabetical order for consistency

  return (
    <div className="space-y-4">
      {/* Top Bar: Dimension Slicer */}
      <div className="bg-white rounded-xl  p-4">
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center gap-2">
            <SlidersHorizontal size={18} className="text-purple-600" />
            <span className="font-semibold text-gray-800">
              Dimension Filters
            </span>
            {hasActiveFilters && (
              <span className="text-xs bg-purple-100 text-purple-700 px-2 py-0.5 rounded-full">
                {
                  [
                    selectedModels,
                    selectedPersonas,
                    selectedEmails,
                    selectedPrompts,
                    selectedEmailTypes,
                    selectedUrgency,
                    selectedFamiliarity,
                    selectedFraming,
                    selectedSizeTier,
                    selectedOpenness,
                    selectedArchitecture,
                  ].flat().length
                }{" "}
                active
              </span>
            )}
          </div>
          <div className="flex items-center gap-2">
            {hasActiveFilters && (
              <button
                onClick={clearAllFilters}
                className="text-sm text-gray-500 hover:text-red-600 flex items-center gap-1"
              >
                <X size={14} /> Clear All
              </button>
            )}
            <button
              onClick={onRefresh}
              className="px-3 py-1.5 bg-gray-100 rounded-lg text-sm hover:bg-gray-200 flex items-center gap-1"
            >
              <Zap size={14} /> Refresh
            </button>
            <button
              onClick={exportToCSV}
              className="px-3 py-1.5 bg-purple-600 text-white rounded-lg text-sm hover:bg-purple-700 flex items-center gap-1"
            >
              <Download size={14} /> Export
            </button>
          </div>
        </div>

        {/* Filter Chips Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          {/* Models */}
          <DimensionFilter
            label="Models"
            icon={Cpu}
            options={allModels}
            selected={selectedModels}
            onToggle={(v) => toggleFilter(setSelectedModels, selectedModels, v)}
            colorMap={MODEL_COLORS}
          />

          {/* Personas */}
          <DimensionFilter
            label="Personas"
            icon={Users}
            options={allPersonas}
            selected={selectedPersonas}
            onToggle={(v) =>
              toggleFilter(setSelectedPersonas, selectedPersonas, v)
            }
            colorMap={PERSONA_COLORS}
            truncate={true}
          />

          {/* Prompts */}
          <DimensionFilter
            label="Prompts"
            icon={FileText}
            options={allPrompts}
            selected={selectedPrompts}
            onToggle={(v) =>
              toggleFilter(setSelectedPrompts, selectedPrompts, v)
            }
            colorMap={PROMPT_COLORS}
            labelMap={PROMPT_LABELS}
          />

          {/* Email Type */}
          <DimensionFilter
            label="Email Type"
            icon={AlertCircle}
            options={allEmailTypes}
            selected={selectedEmailTypes}
            onToggle={(v) =>
              toggleFilter(setSelectedEmailTypes, selectedEmailTypes, v)
            }
            colorMap={{ phishing: "#ef4444", legitimate: "#22c55e" }}
          />
        </div>

        {/* Secondary Filters Row - Expanded by default */}
        <details open className="mt-3">
          <summary className="text-sm text-gray-500 cursor-pointer hover:text-purple-600">
            More filters (Urgency, Familiarity, Framing)
          </summary>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-3">
            <DimensionFilter
              label="Urgency"
              icon={Zap}
              options={allUrgencyLevels}
              selected={selectedUrgency}
              onToggle={(v) =>
                toggleFilter(setSelectedUrgency, selectedUrgency, v)
              }
              colorMap={{ high: "#ef4444", medium: "#f59e0b", low: "#22c55e" }}
            />
            <DimensionFilter
              label="Sender Familiarity"
              icon={Users}
              options={allSenderFamiliarities}
              selected={selectedFamiliarity}
              onToggle={(v) =>
                toggleFilter(setSelectedFamiliarity, selectedFamiliarity, v)
              }
              colorMap={{ familiar: "#3b82f6", unfamiliar: "#9ca3af" }}
            />
            <DimensionFilter
              label="Content Framing"
              icon={Target}
              options={allFramingTypes}
              selected={selectedFraming}
              onToggle={(v) =>
                toggleFilter(setSelectedFraming, selectedFraming, v)
              }
              colorMap={{
                threat: "#ef4444",
                reward: "#22c55e",
                neutral: "#9ca3af",
              }}
            />
          </div>
        </details>

        {/* Model Category Filters */}
        <details className="mt-3">
          <summary className="text-sm text-gray-500 cursor-pointer hover:text-purple-600 flex items-center gap-1">
            <Layers size={14} /> Model Categories (Size, Openness, Architecture)
          </summary>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mt-3">
            <DimensionFilter
              label="Size Tier"
              icon={Layers}
              options={["Small", "Medium", "Large"]}
              selected={selectedSizeTier}
              onToggle={(v) =>
                toggleFilter(setSelectedSizeTier, selectedSizeTier, v)
              }
              colorMap={CATEGORY_COLORS.sizeTier}
            />
            <DimensionFilter
              label="Openness"
              icon={Layers}
              options={["Open", "Closed"]}
              selected={selectedOpenness}
              onToggle={(v) =>
                toggleFilter(setSelectedOpenness, selectedOpenness, v)
              }
              colorMap={CATEGORY_COLORS.openness}
            />
            <DimensionFilter
              label="Architecture"
              icon={Layers}
              options={["Dense", "MoE"]}
              selected={selectedArchitecture}
              onToggle={(v) =>
                toggleFilter(setSelectedArchitecture, selectedArchitecture, v)
              }
              colorMap={CATEGORY_COLORS.architecture}
            />
          </div>
        </details>
      </div>

      {/* Quick Summary Stats - Filtered View */}
      <div className="bg-gradient-to-br from-purple-50 to-blue-50 rounded-xl border border-purple-200 p-4">
        <div className="flex items-center justify-between mb-3">
          <span className="text-xs font-medium text-purple-600 uppercase tracking-wide">
            Filtered Results
          </span>
          <span className="text-[10px] text-gray-400">
            Based on selected filters above
          </span>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-700">
              {filteredMetrics.count.toLocaleString()}
            </div>
            <div className="text-xs text-gray-500">Combinations</div>
          </div>
          <div className="text-center">
            <div
              className={`text-2xl font-bold ${filteredMetrics.avgFidelity >= threshold ? "text-green-600" : "text-amber-600"}`}
            >
              {(filteredMetrics.avgFidelity * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-500">Avg Fidelity</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {(filteredMetrics.passRate * 100).toFixed(0)}%
            </div>
            <div className="text-xs text-gray-500">
              Pass Rate ({filteredMetrics.passing}/{filteredMetrics.count})
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-700">
              ${filteredMetrics.totalCost.toFixed(4)}
            </div>
            <div className="text-xs text-gray-500">Total Cost</div>
          </div>
        </div>
      </div>

      {/* Visualization Mode Switcher */}
      <div className="bg-gray-200 rounded-xl border-5 border-gray-500 p-4 overflow-visible">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            {VIZ_MODES.map((mode) => (
              <button
                key={mode.id}
                onClick={() => setVizMode(mode.id)}
                className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition ${
                  vizMode === mode.id
                    ? "bg-purple-600 text-white shadow-sm"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                <mode.icon size={16} />
                {mode.label}
              </button>
            ))}
          </div>

          {/* Axis selectors for Matrix mode - persona removed */}
          {vizMode === "matrix" && (
            <div className="flex items-center gap-2 text-sm">
              <span className="text-gray-500">Y:</span>
              <select
                value={matrixYAxis}
                onChange={(e) => setMatrixYAxis(e.target.value)}
                className="border rounded px-2 py-1 text-md bg-white"
              >
                <option value="model">Model</option>
                <option value="prompt">Prompt</option>
                <option value="email_type">Email Type</option>
                <option value="urgency">Urgency</option>
                <option value="familiarity">Familiarity</option>
                <option value="framing">Framing</option>
              </select>
              <span className="text-gray-500 ml-2">X:</span>
              <select
                value={matrixXAxis}
                onChange={(e) => setMatrixXAxis(e.target.value)}
                className="border rounded px-2 py-1 text-md bg-white"
              >
                <option value="model">Model</option>
                <option value="prompt">Prompt</option>
                <option value="email_type">Email Type</option>
                <option value="urgency">Urgency</option>
                <option value="familiarity">Familiarity</option>
                <option value="framing">Framing</option>
              </select>
            </div>
          )}
        </div>

        {/* Per-Persona Quick Summary Stats - Shows best value model and rates comparison */}
        {(selectedModels.length > 0 ||
          selectedPersonas.length > 0 ||
          selectedPrompts.length > 0 ||
          selectedEmailTypes.length > 0 ||
          selectedUrgency.length > 0 ||
          selectedFamiliarity.length > 0 ||
          selectedFraming.length > 0) && (
          <div className="bg-gradient-to-br from-purple-50/50 to-blue-50/50 rounded-lg border border-purple-100 p-3 mb-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-md font-semibold text-gray-700">
                Per-Persona Best Value Analysis
              </h3>
              {/* Active Filter Tags */}
              <div className="flex flex-wrap gap-1.5">
                {selectedEmailTypes.map((t) => (
                  <span
                    key={t}
                    className={`px-2 py-0.5 rounded text-[10px] font-medium ${t === "phishing" ? "bg-red-100 text-red-700" : "bg-green-100 text-green-700"}`}
                  >
                    {t}
                  </span>
                ))}
                {selectedUrgency.map((u) => (
                  <span
                    key={u}
                    className={`px-2 py-0.5 rounded text-[10px] font-medium ${u === "high" ? "bg-red-100 text-red-700" : u === "medium" ? "bg-amber-100 text-amber-700" : "bg-green-100 text-green-700"}`}
                  >
                    {u} urgency
                  </span>
                ))}
                {selectedFamiliarity.map((f) => (
                  <span
                    key={f}
                    className={`px-2 py-0.5 rounded text-[10px] font-medium ${f === "familiar" ? "bg-blue-100 text-blue-700" : "bg-gray-100 text-gray-600"}`}
                  >
                    {f}
                  </span>
                ))}
                {selectedFraming.map((f) => (
                  <span
                    key={f}
                    className={`px-2 py-0.5 rounded text-[10px] font-medium ${f === "threat" ? "bg-red-100 text-red-700" : f === "reward" ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-600"}`}
                  >
                    {f} framing
                  </span>
                ))}
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-3">
              {personas.map((persona) => {
                const stats = personaQuickStats[persona];
                if (!stats) return null;

                return (
                  <div
                    key={persona}
                    className="bg-white rounded-lg border border-purple-100 p-2.5 shadow-sm"
                  >
                    {/* Persona Name */}
                    <div
                      className="text-md font-semibold text-black mb-1.5 truncate"
                      title={persona}
                    >
                      {persona}
                    </div>
                    <br />

                    {/* Best Value Model Row */}
                    <div className="flex items-start gap-3 mb-1.5">
                      <div className="flex-1">
                        <div className="text-[12px] text-gray-500 uppercase">
                          Best Value Model
                        </div>
                        <div className="flex items-center gap-1.5">
                          <span
                            className="text-md font-bold text-gray-700 truncate max-w-[100px]"
                            title={stats.bestModel}
                          >
                            {stats.bestModel}
                          </span>
                          <span
                            className={`px-1 py-0.5 rounded text-xs font-medium ${
                              stats.bestPrompt === "cot"
                                ? "bg-purple-100 text-purple-700"
                                : stats.bestPrompt === "stats"
                                  ? "bg-blue-100 text-blue-700"
                                  : "bg-gray-100 text-gray-600"
                            }`}
                          >
                            {PROMPT_LABELS[stats.bestPrompt] ||
                              stats.bestPrompt}
                          </span>
                        </div>
                      </div>
                      <div className="text-right">
                        <div className="text-[12px] text-gray-500">
                          Fidelity / Cost
                        </div>
                        <div className="flex items-center gap-1">
                          <span
                            className={`text-xs font-bold ${stats.fidelity >= threshold ? "text-green-600" : "text-amber-600"}`}
                          >
                            {(stats.fidelity * 100).toFixed(1)}%
                          </span>
                          <span className="text-[12px] text-gray-400">/</span>
                          <span className="text-[12px] text-gray-600 font-mono">
                            ${stats.costPerTrial.toFixed(4)}
                          </span>
                        </div>
                      </div>
                    </div>

                    {/* Rates Comparison Row */}
                    <div className="grid grid-cols-3 gap-1.5 pt-1.5 border-t border-gray-100 text-xs">
                      {/* Click Rate */}
                      <div className="text-center">
                        <div className=" text-gray-500 uppercase">Click</div>
                        <div className="flex items-center justify-center gap-0.5">
                          <span className=" font-bold text-red-500">
                            {(stats.aiClick * 100).toFixed(0)}%
                          </span>
                          <span className=" text-gray-500">vs</span>
                          <span className="font-bold text-blue-500">
                            {(stats.humanClick * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                      {/* Report Rate */}
                      <div className="text-center border-l border-r border-gray-100">
                        <div className=" text-gray-500 uppercase">Report</div>
                        <div className="flex items-center justify-center gap-0.5">
                          <span className=" font-bold text-red-500">
                            {(stats.aiReport * 100).toFixed(0)}%
                          </span>
                          <span className=" text-gray-500">vs</span>
                          <span className="font-bold text-blue-500">
                            {(stats.humanReport * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                      {/* Ignore Rate */}
                      <div className="text-center">
                        <div className=" text-gray-500 uppercase">Ignore</div>
                        <div className="flex items-center justify-center gap-0.5">
                          <span className=" font-bold text-red-500">
                            {(stats.aiIgnore * 100).toFixed(0)}%
                          </span>
                          <span className=" text-gray-500">vs</span>
                          <span className="font-bold text-blue-500">
                            {(stats.humanIgnore * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                );
              })}
            </div>
          </div>
        )}

        {/* Visualization Content - Per Persona in 2-column grid (Bubble first, then Matrix) */}
        {vizMode === "bubble" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {personas.map((persona) => (
              <div key={persona} className="border rounded-lg p-3 bg-gray-50">
                <h4
                  className="text-sm font-semibold text-gray-700 mb-2 truncate"
                  title={persona}
                >
                  {persona}
                </h4>
                <BubbleScatterCompact
                  data={bubbleDataByPersona[persona] || []}
                  threshold={threshold}
                />
              </div>
            ))}
          </div>
        )}

        {vizMode === "matrix" && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
            {personas.map((persona) => (
              <div
                key={persona}
                className="border rounded-lg p-3 bg-gray-50 overflow-visible"
                style={{ overflow: "visible" }}
              >
                <h4
                  className="text-sm font-semibold text-gray-700 mb-2 truncate"
                  title={persona}
                >
                  {persona}
                </h4>
                <MatrixHeatmapCompact
                  data={matrixDataByPersona[persona]}
                  threshold={threshold}
                  xLabel={matrixXAxis}
                  yLabel={matrixYAxis}
                />
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Bottom: Expandable Detail Table */}
      <div className="bg-white rounded-xl border">
        <button
          onClick={() => setShowDetailTable(!showDetailTable)}
          className="w-full p-4 flex items-center justify-between hover:bg-gray-50 transition"
        >
          <div className="flex items-center gap-2">
            <Grid3X3 size={18} className="text-purple-600" />
            <span className="font-medium">Detail Table</span>
            <span className="text-sm text-gray-500">
              ({filteredResults.length} combinations)
            </span>
          </div>
          {showDetailTable ? (
            <ChevronUp size={18} />
          ) : (
            <ChevronDown size={18} />
          )}
        </button>

        {showDetailTable && (
          <div className="border-t p-4">
            {/* Sort indicator */}
            {tableSortColumns.length > 0 && (
              <div className="mb-2 text-xs text-gray-500 flex items-center gap-2">
                <span>Sort by:</span>
                {tableSortColumns.map((s, i) => (
                  <span
                    key={s.key}
                    className="bg-purple-100 text-purple-700 px-2 py-0.5 rounded"
                  >
                    {s.key} ({s.dir}) {i < tableSortColumns.length - 1 && "→"}
                  </span>
                ))}
                <button
                  onClick={() =>
                    setTableSortColumns([{ key: "fidelity", dir: "desc" }])
                  }
                  className="text-red-500 hover:text-red-700"
                >
                  Reset
                </button>
              </div>
            )}
            <div className="overflow-x-auto max-h-96">
              <table className="w-full text-sm">
                <thead className="bg-gray-50 sticky top-0 z-10">
                  <tr>
                    {[
                      { key: "model_id", label: "Model" },
                      { key: "persona_name", label: "Persona" },
                      { key: "prompt_config", label: "Prompt" },
                      { key: "email_type", label: "Type" },
                      { key: "urgency_level", label: "Urgency" },
                      { key: "sender_familiarity", label: "Sender" },
                      { key: "framing_type", label: "Framing" },
                      { key: "fidelity", label: "Fidelity" },
                      { key: "cost", label: "Cost" },
                    ].map((col) => {
                      const sortIndex = tableSortColumns.findIndex(
                        (s) => s.key === col.key,
                      );
                      const sortInfo =
                        sortIndex >= 0 ? tableSortColumns[sortIndex] : null;
                      return (
                        <th
                          key={col.key}
                          className="px-3 py-2 text-left text-xs font-medium text-gray-500 uppercase cursor-pointer hover:bg-gray-100"
                          onClick={() => handleTableSort(col.key)}
                        >
                          {col.label}
                          {sortInfo && (
                            <span className="ml-1 text-purple-600">
                              {sortInfo.dir === "asc" ? "↑" : "↓"}
                              <sup className="text-[10px]">{sortIndex + 1}</sup>
                            </span>
                          )}
                        </th>
                      );
                    })}
                  </tr>
                </thead>
                <tbody className="divide-y">
                  {sortedTableData.slice(0, 100).map((row, i) => (
                    <tr key={i} className="hover:bg-gray-50">
                      <td className="px-3 py-2 font-medium text-purple-700">
                        {row.model_id}
                      </td>
                      <td
                        className="px-3 py-2 text-gray-600 truncate max-w-[120px]"
                        title={row.persona_name}
                      >
                        {row.persona_name?.slice(0, 15)}...
                      </td>
                      <td className="px-3 py-2">
                        <span
                          className={`px-2 py-0.5 rounded text-xs ${
                            row.prompt_config === "cot"
                              ? "bg-purple-100 text-purple-700"
                              : row.prompt_config === "stats"
                                ? "bg-blue-100 text-blue-700"
                                : "bg-gray-100 text-gray-700"
                          }`}
                        >
                          {PROMPT_LABELS[row.prompt_config] ||
                            row.prompt_config}
                        </span>
                      </td>
                      <td className="px-3 py-2">
                        <span
                          className={`px-2 py-0.5 rounded text-xs ${
                            row.email_type === "phishing"
                              ? "bg-red-100 text-red-700"
                              : "bg-green-100 text-green-700"
                          }`}
                        >
                          {row.email_type}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-xs text-gray-500">
                        {row.urgency_level || "-"}
                      </td>
                      <td className="px-3 py-2 text-xs text-gray-500">
                        {row.sender_familiarity || "-"}
                      </td>
                      <td className="px-3 py-2 text-xs text-gray-500">
                        {row.framing_type || "-"}
                      </td>
                      <td className="px-3 py-2">
                        <MiniBar value={row.fidelity} threshold={threshold} />
                      </td>
                      <td className="px-3 py-2 font-mono text-gray-600 text-xs">
                        ${(row.cost || 0).toFixed(5)}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
              {sortedTableData.length > 100 && (
                <div className="text-center py-2 text-sm text-gray-500">
                  Showing first 100 of {sortedTableData.length} results
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

// =============================================================================
// DIMENSION FILTER COMPONENT
// =============================================================================

const DimensionFilter = ({
  label,
  icon: Icon,
  options,
  selected,
  onToggle,
  colorMap = {},
  truncate = false,
  labelMap = {},
}) => {
  const [expanded, setExpanded] = useState(false);
  const displayOptions = expanded ? options : options.slice(0, 4);

  return (
    <div className="bg-gray-50 rounded-lg p-2">
      <div className="flex items-center gap-1 mb-2">
        <Icon size={14} className="text-gray-500" />
        <span className="text-xs font-medium text-gray-600">{label}</span>
        {selected.length > 0 && (
          <span className="text-xs bg-purple-200 text-purple-700 px-1.5 rounded-full ml-auto">
            {selected.length}
          </span>
        )}
      </div>
      <div className="flex flex-wrap gap-1">
        {displayOptions.map((opt) => {
          const isSelected = selected.includes(opt);
          const color = colorMap[opt] || colorMap.default || "#6b7280";
          const optLabel = labelMap[opt] || opt;
          const displayText =
            truncate && optLabel.length > 12
              ? optLabel.slice(0, 12) + "..."
              : optLabel;

          return (
            <button
              key={opt}
              onClick={() => onToggle(opt)}
              title={labelMap[opt] || opt}
              className={`px-2 py-1 rounded text-xs font-medium transition-all ${
                isSelected
                  ? "text-white shadow-sm"
                  : "bg-white text-gray-600 hover:bg-gray-100 border"
              }`}
              style={isSelected ? { backgroundColor: color } : {}}
            >
              {displayText}
            </button>
          );
        })}
        {options.length > 4 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="px-2 py-1 rounded text-xs text-purple-600 hover:bg-purple-50"
          >
            {expanded ? "Less" : `+${options.length - 4}`}
          </button>
        )}
      </div>
    </div>
  );
};

// =============================================================================
// COMPACT MATRIX HEATMAP (Per-Persona View)
// =============================================================================

const MatrixHeatmapCompact = ({ data, threshold, xLabel, yLabel }) => {
  if (!data) return null;
  const { rows, cols, cells } = data;

  if (rows.length === 0 || cols.length === 0) {
    return (
      <div className="flex items-center justify-center h-[150px] text-gray-400 text-xs">
        No data
      </div>
    );
  }

  const truncateLabel = (label, maxLen = 12) => {
    if (!label) return "";
    // Use PROMPT_LABELS if available
    const displayLabel = PROMPT_LABELS[label] || label;
    return displayLabel.length > maxLen
      ? displayLabel.slice(0, maxLen) + "..."
      : displayLabel;
  };

  // Calculate cell width based on number of columns (fill available space)
  const cellMinWidth = Math.max(
    60,
    Math.floor((100 - 15) / Math.max(cols.length, 1)),
  ); // 15% for row label

  return (
    <div
      className="overflow-x-auto overflow-y-visible w-full"
      style={{ overflowY: "visible" }}
    >
      <table className="w-full border-collapse table-fixed min-w-full">
        {/* Column headers */}
        <thead>
          <tr>
            <th className="w-[15%] min-w-[80px] p-1 text-[10px] font-medium text-gray-500 text-left"></th>
            {cols.map((col) => (
              <th
                key={col}
                className="p-1 text-center text-[10px] font-medium text-gray-600 truncate"
                title={PROMPT_LABELS[col] || col}
                style={{ width: `${(85 / cols.length).toFixed(1)}%` }}
              >
                {truncateLabel(col)}
              </th>
            ))}
          </tr>
        </thead>

        {/* Rows */}
        <tbody>
          {rows.map((row) => (
            <tr key={row}>
              <td
                className="w-[15%] min-w-[80px] p-1 text-[10px] font-medium text-gray-700 truncate"
                title={PROMPT_LABELS[row] || row}
              >
                {truncateLabel(row)}
              </td>
              {cols.map((col) => {
                const cell = cells[`${row}|${col}`];
                if (!cell) {
                  return (
                    <td
                      key={`${row}-${col}`}
                      className="h-12 text-center text-[10px] text-gray-300 bg-gray-50 border border-gray-100"
                    >
                      -
                    </td>
                  );
                }

                const fidelity = cell.fidelity || 0;
                const isPassing = fidelity >= threshold;

                const bgColor = isPassing
                  ? `rgba(34, 197, 94, ${0.3 + Math.min(fidelity / threshold, 1) * 0.5})`
                  : fidelity > 0.5
                    ? `rgba(234, 179, 8, ${0.3 + (fidelity / threshold) * 0.4})`
                    : `rgba(239, 68, 68, ${0.3 + (1 - fidelity) * 0.3})`;

                return (
                  <td
                    key={`${row}-${col}`}
                    className="h-12 text-center text-[10px] border border-gray-100 cursor-pointer hover:ring-1 hover:ring-purple-400 transition-all group relative"
                    style={{ backgroundColor: bgColor }}
                  >
                    <div
                      className={`font-bold ${isPassing ? "text-green-800" : "text-gray-700"}`}
                    >
                      {(fidelity * 100).toFixed(0)}%
                    </div>
                    <div className="text-[8px] text-gray-500">
                      ${(cell.cost || 0).toFixed(3)}
                    </div>

                    {/* Tooltip - fixed position with very high z-index to prevent clipping */}
                    <div
                      className="absolute top-full left-1/2 -translate-x-1/2 mt-1 hidden group-hover:block pointer-events-none"
                      style={{ zIndex: 9999 }}
                    >
                      <div className="bg-gray-900 text-white text-[10px] rounded-lg p-2.5 whitespace-nowrap shadow-2xl border border-gray-700">
                        <div className="font-semibold text-purple-300">
                          {PROMPT_LABELS[row] || row}
                        </div>
                        <div className="text-gray-400 text-[9px]">
                          {PROMPT_LABELS[col] || col}
                        </div>
                        <div className="mt-1.5 pt-1.5 border-t border-gray-700 space-y-0.5">
                          <div className="flex justify-between gap-3">
                            <span className="text-gray-400">Fidelity:</span>
                            <span
                              className={
                                isPassing
                                  ? "text-green-400 font-medium"
                                  : "text-amber-400 font-medium"
                              }
                            >
                              {(fidelity * 100).toFixed(1)}%
                            </span>
                          </div>
                          <div className="flex justify-between gap-3">
                            <span className="text-gray-400">Cost:</span>
                            <span className="font-mono">
                              ${(cell.cost || 0).toFixed(5)}
                            </span>
                          </div>
                          <div className="flex justify-between gap-3">
                            <span className="text-gray-400">Pass:</span>
                            <span>
                              {cell.passing}/{cell.count}
                            </span>
                          </div>
                        </div>
                      </div>
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

// =============================================================================
// COMPACT BUBBLE SCATTER (Per-Persona View) - Colored by Prompt
// =============================================================================

const BubbleScatterCompact = ({ data, threshold }) => {
  if (!data || data.length === 0) {
    return (
      <div className="flex items-center justify-center h-[200px] text-gray-400 text-xs">
        No data
      </div>
    );
  }

  // Prompt colors and labels
  const promptColors = {
    cot: "#8b5cf6", // purple for chain-of-thought
    stats: "#3b82f6", // blue for behavioral stats
    baseline: "#6b7280", // gray for baseline
  };

  const promptLabels = {
    cot: "Chain-of-Thought",
    stats: "Behavioral Stats",
    baseline: "Baseline",
  };

  // Get unique prompts in the data for legend
  const uniquePrompts = [...new Set(data.map((d) => d.prompt))].filter(Boolean);

  return (
    <div className="w-full">
      {/* Legend */}
      <div className="flex items-center justify-end gap-3 mb-2 pr-2">
        {uniquePrompts.map((prompt) => (
          <div key={prompt} className="flex items-center gap-1.5">
            <div
              className="w-3 h-3 rounded-full"
              style={{ backgroundColor: promptColors[prompt] || "#6b7280" }}
            />
            <span className="text-[10px] text-gray-600">
              {promptLabels[prompt] || prompt}
            </span>
          </div>
        ))}
        <div className="flex items-center gap-1.5 ml-2 pl-2 border-l border-gray-200">
          <div className="w-4 h-0 border-t-2 border-red-500 border-dashed" />
          <span className="text-[10px] text-gray-600">85% Threshold</span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={200}>
        <ScatterChart margin={{ top: 10, right: 10, bottom: 30, left: 40 }}>
          <CartesianGrid strokeDasharray="3 3" />
          <XAxis
            type="number"
            dataKey="cost"
            name="Cost"
            tickFormatter={(v) => `$${v.toFixed(3)}`}
            tick={{ fontSize: 10 }}
            label={{
              value: "Total Cost ($)",
              position: "bottom",
              offset: 15,
              fontSize: 12,
            }}
          />
          <YAxis
            type="number"
            dataKey="fidelity"
            name="Fidelity"
            domain={[0, 100]}
            tickFormatter={(v) => `${v}%`}
            tick={{ fontSize: 10 }}
            label={{
              value: "Fidelity",
              angle: -90,
              position: "insideLeft",
              fontSize: 12,
              offset: 10,
            }}
          />
          <Tooltip
            content={({ active, payload }) => {
              if (active && payload && payload.length) {
                const d = payload[0].payload;
                return (
                  <div className="bg-white border rounded shadow-lg p-2 text-xs">
                    <div className="font-bold text-purple-700">{d.model}</div>
                    <div className="text-gray-500">{d.promptLabel}</div>
                    <div className="mt-1">
                      <div>
                        Fidelity:{" "}
                        <span
                          className={
                            d.fidelity >= threshold * 100
                              ? "text-green-600"
                              : "text-amber-600"
                          }
                        >
                          {d.fidelity.toFixed(1)}%
                        </span>
                      </div>
                      <div>Cost: ${d.cost.toFixed(5)}</div>
                      <div>Combinations: {d.count}</div>
                    </div>
                  </div>
                );
              }
              return null;
            }}
          />
          <ReferenceLine
            y={threshold * 100}
            stroke="#ef4444"
            strokeDasharray="3 3"
          />
          <Scatter name="Combinations" data={data}>
            {data.map((entry, index) => (
              <Cell
                key={`cell-${index}`}
                fill={promptColors[entry.prompt] || "#6b7280"}
                r={Math.sqrt(entry.count) * 2 + 4}
              />
            ))}
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
};

// =============================================================================
// MATRIX HEATMAP VISUALIZATION (Legacy - kept for reference)
// =============================================================================

const MatrixHeatmap = ({ data, threshold, xLabel, yLabel }) => {
  const { rows, cols, cells } = data;

  if (rows.length === 0 || cols.length === 0) {
    return (
      <div className="flex items-center justify-center h-[400px] text-gray-400">
        <div className="text-center">
          <Grid3X3 size={48} className="mx-auto mb-2 opacity-50" />
          <div>No data matches the current filters</div>
        </div>
      </div>
    );
  }

  // Truncate labels helper
  const truncateLabel = (label, maxLen = 15) => {
    if (!label) return "";
    return label.length > maxLen ? label.slice(0, maxLen) + "..." : label;
  };

  return (
    <div className="overflow-auto">
      <div className="inline-block min-w-full">
        {/* Column headers */}
        <div className="flex">
          <div className="w-36 shrink-0 p-2 text-xs font-medium text-gray-500 uppercase">
            {yLabel} / {xLabel}
          </div>
          {cols.map((col) => (
            <div
              key={col}
              className="w-20 shrink-0 p-2 text-center text-xs font-medium text-gray-600 truncate"
              title={col}
            >
              {truncateLabel(col, 10)}
            </div>
          ))}
        </div>

        {/* Rows */}
        {rows.map((row) => (
          <div key={row} className="flex items-center">
            <div
              className="w-36 shrink-0 p-2 text-xs font-medium text-gray-700 truncate"
              title={row}
            >
              {truncateLabel(row)}
            </div>
            {cols.map((col) => {
              const cell = cells[`${row}|${col}`];
              if (!cell) {
                return (
                  <div
                    key={`${row}-${col}`}
                    className="w-20 h-14 shrink-0 flex items-center justify-center text-xs text-gray-300 bg-gray-50 border border-gray-100"
                  >
                    -
                  </div>
                );
              }

              const fidelity = cell.fidelity || 0;
              const isPassing = fidelity >= threshold;
              const intensity = Math.min(fidelity / threshold, 1);

              const bgColor = isPassing
                ? `rgba(34, 197, 94, ${0.2 + intensity * 0.6})`
                : fidelity > 0.5
                  ? `rgba(234, 179, 8, ${0.2 + (fidelity / threshold) * 0.5})`
                  : `rgba(239, 68, 68, ${0.2 + (1 - fidelity) * 0.4})`;

              return (
                <div
                  key={`${row}-${col}`}
                  className="w-20 h-14 shrink-0 flex flex-col items-center justify-center text-xs border border-gray-100 cursor-pointer hover:ring-2 hover:ring-purple-400 transition-all group relative"
                  style={{ backgroundColor: bgColor }}
                  title={`${row} × ${col}\nFidelity: ${(fidelity * 100).toFixed(1)}%\nCost: $${(cell.cost || 0).toFixed(5)}\nCount: ${cell.count}`}
                >
                  <div
                    className={`font-bold ${isPassing ? "text-green-800" : "text-gray-700"}`}
                  >
                    {(fidelity * 100).toFixed(0)}%
                  </div>
                  <div className="text-[10px] text-gray-500">
                    ${(cell.cost || 0).toFixed(4)}
                  </div>

                  {/* Tooltip on hover */}
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 hidden group-hover:block z-10">
                    <div className="bg-gray-900 text-white text-xs rounded-lg p-2 whitespace-nowrap shadow-lg">
                      <div className="font-medium">{row}</div>
                      <div className="text-gray-300">{col}</div>
                      <div className="mt-1 pt-1 border-t border-gray-700">
                        <div>Fidelity: {(fidelity * 100).toFixed(1)}%</div>
                        <div>Cost: ${(cell.cost || 0).toFixed(5)}</div>
                        <div>
                          Pass: {cell.passing}/{cell.count}
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              );
            })}
          </div>
        ))}

        {/* Legend */}
        <div className="flex items-center gap-4 mt-4 text-xs text-gray-500 pl-36">
          <span>Legend:</span>
          <div className="flex items-center gap-1">
            <div
              className="w-4 h-4 rounded"
              style={{ backgroundColor: "rgba(239, 68, 68, 0.4)" }}
            ></div>
            <span>&lt;50%</span>
          </div>
          <div className="flex items-center gap-1">
            <div
              className="w-4 h-4 rounded"
              style={{ backgroundColor: "rgba(234, 179, 8, 0.5)" }}
            ></div>
            <span>50-{(threshold * 100).toFixed(0)}%</span>
          </div>
          <div className="flex items-center gap-1">
            <div
              className="w-4 h-4 rounded"
              style={{ backgroundColor: "rgba(34, 197, 94, 0.7)" }}
            ></div>
            <span>≥{(threshold * 100).toFixed(0)}%</span>
          </div>
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// BUBBLE SCATTER VISUALIZATION
// =============================================================================

const BubbleScatter = ({ data, threshold }) => {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-[400px] text-gray-400">
        <div className="text-center">
          <Target size={48} className="mx-auto mb-2 opacity-50" />
          <div>No data matches the current filters</div>
        </div>
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={400}>
      <ScatterChart margin={{ top: 20, right: 30, bottom: 60, left: 60 }}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          type="number"
          dataKey="cost"
          name="Cost"
          tickFormatter={(v) => `$${(v / 1000).toFixed(3)}`}
          label={{
            value: "Cost per Trial (×1000)",
            position: "bottom",
            offset: 40,
          }}
        />
        <YAxis
          type="number"
          dataKey="fidelity"
          name="Fidelity"
          domain={[0, 100]}
          tickFormatter={(v) => `${v}%`}
          label={{
            value: "Fidelity (%)",
            angle: -90,
            position: "insideLeft",
            offset: -10,
          }}
        />
        <Tooltip
          formatter={(value, name) => {
            if (name === "Cost") return `$${(value / 1000).toFixed(4)}`;
            if (name === "Fidelity") return `${value.toFixed(1)}%`;
            return value;
          }}
          content={({ active, payload }) => {
            if (active && payload && payload.length) {
              const d = payload[0].payload;
              return (
                <div className="bg-white border rounded-lg shadow-lg p-3 text-sm">
                  <div className="font-bold text-purple-700">{d.model}</div>
                  <div className="text-gray-600 mt-1">
                    <div>
                      Fidelity:{" "}
                      <span
                        className={
                          d.fidelity >= threshold * 100
                            ? "text-green-600 font-medium"
                            : "text-amber-600"
                        }
                      >
                        {d.fidelity.toFixed(1)}%
                      </span>
                    </div>
                    <div>Cost: ${(d.cost / 1000).toFixed(5)}</div>
                    <div>Combinations: {d.count}</div>
                    <div>Passing: {d.passing}</div>
                  </div>
                </div>
              );
            }
            return null;
          }}
        />
        <ReferenceLine
          y={threshold * 100}
          stroke="#ef4444"
          strokeDasharray="5 5"
          label={{
            value: `${(threshold * 100).toFixed(0)}% threshold`,
            position: "right",
          }}
        />
        <Scatter name="Models" data={data} fill="#8b5cf6">
          {data.map((entry, index) => (
            <Cell
              key={`cell-${index}`}
              fill={entry.fidelity >= threshold * 100 ? "#22c55e" : "#f59e0b"}
              r={Math.sqrt(entry.count) * 3 + 8}
            />
          ))}
        </Scatter>
      </ScatterChart>
    </ResponsiveContainer>
  );
};

// =============================================================================
// PARALLEL COORDINATES VISUALIZATION
// =============================================================================

const ParallelCoordinates = ({
  data,
  models,
  personas,
  prompts,
  threshold,
}) => {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-[400px] text-gray-400">
        <div className="text-center">
          <SlidersHorizontal size={48} className="mx-auto mb-2 opacity-50" />
          <div>No data matches the current filters</div>
        </div>
      </div>
    );
  }

  // Calculate axis ranges
  const axes = [
    {
      key: "model",
      label: "Model",
      range: [0, models.length - 1],
      ticks: models,
    },
    {
      key: "persona",
      label: "Persona",
      range: [0, personas.length - 1],
      ticks: personas.map((p) => p.slice(0, 8) + "..."),
    },
    {
      key: "prompt",
      label: "Prompt",
      range: [0, prompts.length - 1],
      ticks: prompts,
    },
    {
      key: "email_type",
      label: "Email Type",
      range: [0, 1],
      ticks: ["Legit", "Phish"],
    },
    {
      key: "fidelity",
      label: "Fidelity %",
      range: [0, 100],
      ticks: [0, 25, 50, 75, 100],
    },
    {
      key: "cost",
      label: "Cost ×10k",
      range: [0, Math.max(...data.map((d) => d.cost), 1)],
      ticks: null,
    },
  ];

  const width = 800;
  const height = 350;
  const margin = { top: 30, right: 30, bottom: 20, left: 30 };
  const innerWidth = width - margin.left - margin.right;
  const innerHeight = height - margin.top - margin.bottom;

  const axisSpacing = innerWidth / (axes.length - 1);

  // Scale functions
  const scaleY = (value, range) => {
    const [min, max] = range;
    if (max === min) return innerHeight / 2;
    return innerHeight - ((value - min) / (max - min)) * innerHeight;
  };

  // Generate path for each data point
  const generatePath = (d) => {
    const points = axes.map((axis, i) => {
      const x = i * axisSpacing;
      const y = scaleY(d[axis.key], axis.range);
      return `${x},${y}`;
    });
    return `M${points.join("L")}`;
  };

  return (
    <div className="overflow-x-auto">
      <svg width={width} height={height} className="mx-auto">
        <g transform={`translate(${margin.left},${margin.top})`}>
          {/* Axes */}
          {axes.map((axis, i) => {
            const x = i * axisSpacing;
            return (
              <g key={axis.key} transform={`translate(${x},0)`}>
                <line
                  y1={0}
                  y2={innerHeight}
                  stroke="#e5e7eb"
                  strokeWidth={2}
                />
                <text
                  y={-10}
                  textAnchor="middle"
                  className="text-xs font-medium fill-gray-600"
                >
                  {axis.label}
                </text>
                {/* Tick marks */}
                {axis.ticks &&
                  axis.ticks.slice(0, 5).map((tick, ti) => {
                    const yPos = scaleY(ti, axis.range);
                    return (
                      <g key={ti}>
                        <line
                          x1={-5}
                          x2={0}
                          y1={yPos}
                          y2={yPos}
                          stroke="#9ca3af"
                        />
                        <text
                          x={-8}
                          y={yPos}
                          textAnchor="end"
                          alignmentBaseline="middle"
                          className="text-[10px] fill-gray-500"
                        >
                          {typeof tick === "string" ? tick : tick}
                        </text>
                      </g>
                    );
                  })}
              </g>
            );
          })}

          {/* Data lines */}
          {data.map((d, i) => (
            <path
              key={i}
              d={generatePath(d)}
              fill="none"
              stroke={d.passing ? "#22c55e" : "#ef4444"}
              strokeWidth={1.5}
              strokeOpacity={0.4}
              className="hover:stroke-opacity-100 hover:stroke-[3px] transition-all cursor-pointer"
            />
          ))}

          {/* Threshold line on fidelity axis */}
          <g
            transform={`translate(${4 * axisSpacing},${scaleY(threshold * 100, [0, 100])})`}
          >
            <line
              x1={-10}
              x2={10}
              stroke="#ef4444"
              strokeWidth={2}
              strokeDasharray="4 2"
            />
          </g>
        </g>
      </svg>

      {/* Legend */}
      <div className="flex justify-center gap-6 mt-2 text-xs">
        <div className="flex items-center gap-1">
          <div className="w-8 h-0.5 bg-green-500"></div>
          <span className="text-gray-600">
            Passing (≥{(threshold * 100).toFixed(0)}%)
          </span>
        </div>
        <div className="flex items-center gap-1">
          <div className="w-8 h-0.5 bg-red-500"></div>
          <span className="text-gray-600">Failing</span>
        </div>
        <span className="text-gray-400">
          Showing first {data.length} combinations
        </span>
      </div>
    </div>
  );
};

// =============================================================================
// OVERVIEW SUB-TAB (kept for backwards compatibility, but not used)
// =============================================================================

const OverviewSubTab = ({
  modelAggregates,
  promptAggregates,
  personaAggregates,
  threshold,
  summaryStats,
  filteredData = [],
}) => {
  // Prepare comparison data
  const comparisonData = personaAggregates.map((p) => ({
    name:
      p.persona.length > 15 ? p.persona.substring(0, 15) + "..." : p.persona,
    fullName: p.persona,
    "AI Click": p.mean_ai_click * 100,
    "Human Click": p.mean_human_click * 100,
    Fidelity: p.mean_fidelity * 100,
  }));

  // Pass/Fail pie chart data
  const passingCount = filteredData.filter(
    (d) => (d.normalized_accuracy || 0) >= threshold,
  ).length;
  const failingCount = filteredData.length - passingCount;
  const pieData = [
    { name: "Passing", value: passingCount, fill: "#22c55e" },
    { name: "Failing", value: failingCount, fill: "#ef4444" },
  ];

  // Top 5 combinations sorted by fidelity
  const topCombinations = [...filteredData]
    .sort((a, b) => (b.normalized_accuracy || 0) - (a.normalized_accuracy || 0))
    .slice(0, 5);

  // Radar chart data for persona comparison
  const radarData = personaAggregates.map((p) => ({
    persona:
      p.persona.length > 12 ? p.persona.substring(0, 12) + "..." : p.persona,
    fullName: p.persona,
    Fidelity: (p.mean_fidelity || 0) * 100,
    "AI Click": (p.mean_ai_click || 0) * 100,
    "Human Click": (p.mean_human_click || 0) * 100,
  }));

  return (
    <div className="space-y-6">
      {/* Top Row: Pass/Fail Summary + Top Combinations */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Pass/Fail Pie Chart */}
        <div className="bg-white rounded-xl border p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <CheckCircle size={20} className="text-green-600" />
            Pass/Fail Summary
          </h3>
          <div className="flex items-center justify-center">
            <ResponsiveContainer width="100%" height={180}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={50}
                  outerRadius={70}
                  paddingAngle={2}
                  dataKey="value"
                  label={({ name, percent }) =>
                    `${name} ${(percent * 100).toFixed(0)}%`
                  }
                  labelLine={false}
                >
                  {pieData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.fill} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => `${value} conditions`} />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="flex justify-center gap-6 mt-2 text-sm">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500" />
              <span>{passingCount} Passing</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500" />
              <span>{failingCount} Failing</span>
            </div>
          </div>
        </div>

        {/* Top 5 Combinations */}
        <div className="lg:col-span-2 bg-white rounded-xl border p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Award size={20} className="text-yellow-500" />
            Top 5 Performing Combinations
          </h3>
          <div className="space-y-2">
            {topCombinations.length > 0 ? (
              topCombinations.map((item, index) => (
                <TopCombinationCard
                  key={index}
                  rank={index + 1}
                  persona={item.persona_name || item.persona_id}
                  model={item.model_id}
                  prompt={item.prompt_config || "baseline"}
                  fidelity={item.normalized_accuracy || 0}
                  threshold={threshold}
                />
              ))
            ) : (
              <div className="text-center py-8 text-gray-500">
                No data available
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Radar Chart for Persona Comparison */}
      {radarData.length > 0 && (
        <div className="bg-white rounded-xl border p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Target size={20} className="text-purple-600" />
            Persona Comparison (Multi-dimensional)
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <RadarChart data={radarData}>
              <PolarGrid />
              <PolarAngleAxis dataKey="persona" tick={{ fontSize: 11 }} />
              <PolarRadiusAxis
                angle={30}
                domain={[0, 100]}
                tick={{ fontSize: 10 }}
              />
              <Radar
                name="Fidelity"
                dataKey="Fidelity"
                stroke="#8b5cf6"
                fill="#8b5cf6"
                fillOpacity={0.3}
              />
              <Radar
                name="AI Click"
                dataKey="AI Click"
                stroke="#ef4444"
                fill="#ef4444"
                fillOpacity={0.2}
              />
              <Radar
                name="Human Click"
                dataKey="Human Click"
                stroke="#3b82f6"
                fill="#3b82f6"
                fillOpacity={0.2}
              />
              <Legend />
              <Tooltip formatter={(v) => `${v.toFixed(1)}%`} />
            </RadarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Model Fidelity Comparison */}
        <div className="bg-white rounded-xl border p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Cpu size={20} className="text-purple-600" />
            Model Fidelity Comparison
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={modelAggregates} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <YAxis
                dataKey="model"
                type="category"
                width={120}
                tick={{ fontSize: 11 }}
              />
              <Tooltip
                formatter={(v) => `${(v * 100).toFixed(1)}%`}
                labelFormatter={(label) => `Model: ${label}`}
              />
              <ReferenceLine
                x={threshold}
                stroke="#ef4444"
                strokeDasharray="5 5"
                label={{
                  value: `${threshold * 100}%`,
                  position: "top",
                  fontSize: 10,
                }}
              />
              <Bar
                dataKey="mean_fidelity"
                name="Mean Fidelity"
                radius={[0, 4, 4, 0]}
              >
                {modelAggregates.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={MODEL_COLORS[entry.model] || MODEL_COLORS.default}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* AI vs Human Click Rate by Persona */}
        <div className="bg-white rounded-xl border p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Users size={20} className="text-blue-600" />
            AI vs Human Click Rate
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={comparisonData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="name"
                tick={{ fontSize: 10, angle: -20, textAnchor: "end" }}
                height={60}
              />
              <YAxis tickFormatter={(v) => `${v.toFixed(0)}%`} />
              <Tooltip
                formatter={(v) => `${v.toFixed(1)}%`}
                labelFormatter={(label, payload) =>
                  payload[0]?.payload?.fullName || label
                }
              />
              <Legend />
              <Bar dataKey="AI Click" fill="#ef4444" radius={[4, 4, 0, 0]} />
              <Bar dataKey="Human Click" fill="#3b82f6" radius={[4, 4, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Prompt Configuration Effect */}
        <div className="bg-white rounded-xl border p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <FileText size={20} className="text-green-600" />
            Prompt Configuration Effect
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={promptAggregates}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="prompt" />
              <YAxis
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} />
              <ReferenceLine
                y={threshold}
                stroke="#ef4444"
                strokeDasharray="5 5"
              />
              <Bar
                dataKey="mean_fidelity"
                name="Mean Fidelity"
                radius={[4, 4, 0, 0]}
              >
                {promptAggregates.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={PROMPT_COLORS[entry.prompt] || "#6b7280"}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
          <div className="mt-4 grid grid-cols-3 gap-2 text-center text-xs">
            {["baseline", "stats", "cot"].map((p) => (
              <div
                key={p}
                className="p-2 rounded"
                style={{ backgroundColor: PROMPT_COLORS[p] + "20" }}
              >
                <div className="font-medium uppercase">{p}</div>
                <div className="text-gray-500">
                  {p === "baseline"
                    ? "Basic persona"
                    : p === "stats"
                      ? "+ Behavioral stats"
                      : "+ Reasoning"}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Fidelity Distribution by Persona */}
        <div className="bg-white rounded-xl border p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Target size={20} className="text-amber-600" />
            Fidelity by Persona
          </h3>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={personaAggregates} layout="vertical">
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                type="number"
                domain={[0, 1]}
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
              />
              <YAxis
                dataKey="persona"
                type="category"
                width={150}
                tick={{ fontSize: 10 }}
              />
              <Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} />
              <ReferenceLine
                x={threshold}
                stroke="#ef4444"
                strokeDasharray="5 5"
              />
              <Bar
                dataKey="mean_fidelity"
                name="Mean Fidelity"
                radius={[0, 4, 4, 0]}
              >
                {personaAggregates.map((entry, index) => (
                  <Cell
                    key={`cell-${index}`}
                    fill={
                      PERSONA_COLORS[entry.persona] || PERSONA_COLORS.default
                    }
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// FIDELITY SUB-TAB
// =============================================================================

const FidelitySubTab = ({
  filteredData,
  sortedData,
  personas,
  models,
  prompts,
  filterPersona,
  setFilterPersona,
  filterModel,
  setFilterModel,
  filterPrompt,
  setFilterPrompt,
  filterStatus,
  setFilterStatus,
  fidelityRange,
  setFidelityRange,
  visibleColumns,
  toggleColumn,
  showColumnMenu,
  setShowColumnMenu,
  showAdvancedFilters,
  setShowAdvancedFilters,
  clearFilters,
  sortConfig,
  handleSort,
  threshold,
  heatmapData,
  heatmapPersonaPrompt,
  heatmapModelPrompt,
  heatmapView,
  setHeatmapView,
}) => {
  const hasActiveFilters =
    filterPersona !== "all" ||
    filterModel !== "all" ||
    filterPrompt !== "all" ||
    filterStatus !== "all" ||
    fidelityRange[0] !== 0 ||
    fidelityRange[1] !== 100;

  // Export to CSV function
  const exportToCSV = () => {
    const headers = [
      "Persona",
      "Model",
      "Prompt",
      "Fidelity",
      "AI Click",
      "Human Click",
      "Deviation",
      "Status",
    ];
    const rows = sortedData.map((row) => [
      row.persona_name || row.persona_id,
      row.model_id,
      row.prompt_config || "baseline",
      ((row.normalized_accuracy || 0) * 100).toFixed(2),
      ((row.ai_click_rate || 0) * 100).toFixed(2),
      ((row.human_click_rate || 0) * 100).toFixed(2),
      (((row.ai_click_rate || 0) - (row.human_click_rate || 0)) * 100).toFixed(
        2,
      ),
      (row.normalized_accuracy || 0) >= threshold ? "Pass" : "Fail",
    ]);
    const csvContent = [headers, ...rows].map((r) => r.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `fidelity_results_${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Get current heatmap data based on view
  const getCurrentHeatmapData = () => {
    switch (heatmapView) {
      case "persona-prompt":
        return {
          data: heatmapPersonaPrompt?.data || [],
          rows: heatmapPersonaPrompt?.personas || [],
          cols: heatmapPersonaPrompt?.prompts || [],
          rowKey: "persona",
          colKey: "prompt",
        };
      case "model-prompt":
        return {
          data: heatmapModelPrompt?.data || [],
          rows: heatmapModelPrompt?.models || [],
          cols: heatmapModelPrompt?.prompts || [],
          rowKey: "model",
          colKey: "prompt",
        };
      default: // persona-model
        return {
          data: heatmapData?.data || [],
          rows: heatmapData?.personas || [],
          cols: heatmapData?.models || [],
          rowKey: "persona",
          colKey: "model",
        };
    }
  };

  const currentHeatmap = getCurrentHeatmapData();

  return (
    <div className="space-y-6">
      {/* Quick Filter Chips */}
      <div className="bg-white rounded-xl border p-4">
        <div className="flex flex-wrap items-center gap-2">
          <span className="text-sm font-medium text-gray-500 mr-2">
            Quick Filters:
          </span>
          <FilterChip
            label="All"
            active={filterStatus === "all" && filterPersona === "all"}
            onClick={() => {
              setFilterStatus("all");
              setFilterPersona("all");
              setFilterModel("all");
              setFilterPrompt("all");
            }}
            icon={Zap}
          />
          <FilterChip
            label="Passing Only"
            active={filterStatus === "passing"}
            onClick={() =>
              setFilterStatus(filterStatus === "passing" ? "all" : "passing")
            }
            icon={CheckCircle}
          />
          <FilterChip
            label="Failing Only"
            active={filterStatus === "failing"}
            onClick={() =>
              setFilterStatus(filterStatus === "failing" ? "all" : "failing")
            }
            icon={AlertCircle}
          />
          <div className="w-px h-6 bg-gray-200 mx-2" />
          {personas
            .filter((p) => p !== "all")
            .map((persona) => (
              <FilterChip
                key={persona}
                label={
                  persona.length > 15
                    ? persona.substring(0, 15) + "..."
                    : persona
                }
                active={filterPersona === persona}
                onClick={() =>
                  setFilterPersona(filterPersona === persona ? "all" : persona)
                }
              />
            ))}
        </div>
      </div>

      {/* Fidelity Heatmap with View Switching */}
      <div className="bg-white rounded-xl border p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold flex items-center gap-2">
            <Grid3X3 size={20} className="text-purple-600" />
            Fidelity Heatmap
          </h3>
          <select
            value={heatmapView}
            onChange={(e) => setHeatmapView(e.target.value)}
            className="border rounded-lg px-3 py-1.5 text-sm bg-white focus:ring-2 focus:ring-purple-500"
          >
            <option value="persona-model">Persona × Model</option>
            <option value="persona-prompt">Persona × Prompt</option>
            <option value="model-prompt">Model × Prompt</option>
          </select>
        </div>
        <div className="overflow-x-auto">
          <div className="inline-block min-w-full">
            <div className="flex">
              <div className="w-40"></div>
              {currentHeatmap.cols.map((col) => (
                <div
                  key={col}
                  className="w-24 text-center text-xs font-medium text-gray-600 truncate px-1"
                >
                  {col}
                </div>
              ))}
            </div>
            {currentHeatmap.rows.map((row) => (
              <div key={row} className="flex items-center">
                <div className="w-40 text-xs font-medium text-gray-700 truncate pr-2">
                  {row}
                </div>
                {currentHeatmap.cols.map((col) => {
                  const cell = currentHeatmap.data.find(
                    (d) =>
                      d[currentHeatmap.rowKey] === row &&
                      d[currentHeatmap.colKey] === col,
                  );
                  const fidelity = cell?.fidelity || 0;
                  const bgColor =
                    fidelity >= threshold
                      ? `rgba(34, 197, 94, ${0.3 + fidelity * 0.7})`
                      : fidelity > 0.5
                        ? `rgba(234, 179, 8, ${0.3 + fidelity * 0.7})`
                        : `rgba(239, 68, 68, ${0.3 + (1 - fidelity) * 0.4})`;
                  return (
                    <div
                      key={`${row}-${col}`}
                      className="w-24 h-12 flex items-center justify-center text-xs font-medium border border-gray-100 cursor-pointer hover:ring-2 hover:ring-purple-400 transition-all"
                      style={{ backgroundColor: bgColor }}
                      title={`${row} × ${col}: ${(fidelity * 100).toFixed(1)}%`}
                    >
                      {cell ? `${(fidelity * 100).toFixed(0)}%` : "-"}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
        <div className="mt-4 flex items-center gap-4 text-xs text-gray-500">
          <span>Legend:</span>
          <div className="flex items-center gap-1">
            <div
              className="w-4 h-4 rounded"
              style={{ backgroundColor: "rgba(239, 68, 68, 0.5)" }}
            ></div>
            <span>&lt;50%</span>
          </div>
          <div className="flex items-center gap-1">
            <div
              className="w-4 h-4 rounded"
              style={{ backgroundColor: "rgba(234, 179, 8, 0.6)" }}
            ></div>
            <span>50-85%</span>
          </div>
          <div className="flex items-center gap-1">
            <div
              className="w-4 h-4 rounded"
              style={{ backgroundColor: "rgba(34, 197, 94, 0.8)" }}
            ></div>
            <span>≥85%</span>
          </div>
        </div>
      </div>

      {/* Filters Section */}
      <div className="bg-white rounded-xl border p-4">
        <div className="flex flex-wrap items-center gap-3">
          <Filter size={18} className="text-gray-400" />

          {/* Basic Filters */}
          <select
            value={filterPersona}
            onChange={(e) => setFilterPersona(e.target.value)}
            className="border rounded-lg px-3 py-1.5 text-sm bg-white"
          >
            {personas.map((p) => (
              <option key={p} value={p}>
                {p === "all" ? "All Personas" : p}
              </option>
            ))}
          </select>

          <select
            value={filterModel}
            onChange={(e) => setFilterModel(e.target.value)}
            className="border rounded-lg px-3 py-1.5 text-sm bg-white"
          >
            {models.map((m) => (
              <option key={m} value={m}>
                {m === "all" ? "All Models" : m}
              </option>
            ))}
          </select>

          <select
            value={filterPrompt}
            onChange={(e) => setFilterPrompt(e.target.value)}
            className="border rounded-lg px-3 py-1.5 text-sm bg-white"
          >
            {prompts.map((p) => (
              <option key={p} value={p}>
                {p === "all" ? "All Prompts" : p.toUpperCase()}
              </option>
            ))}
          </select>

          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="border rounded-lg px-3 py-1.5 text-sm bg-white"
          >
            <option value="all">All Status</option>
            <option value="passing">Passing Only</option>
            <option value="failing">Failing Only</option>
          </select>

          {/* Advanced Filters Toggle */}
          <button
            onClick={() => setShowAdvancedFilters(!showAdvancedFilters)}
            className={`flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm transition ${
              showAdvancedFilters
                ? "bg-purple-100 text-purple-700"
                : "bg-gray-100 text-gray-600 hover:bg-gray-200"
            }`}
          >
            <SlidersHorizontal size={14} />
            Advanced
            {showAdvancedFilters ? (
              <ChevronUp size={14} />
            ) : (
              <ChevronDown size={14} />
            )}
          </button>

          {/* Column Visibility Toggle */}
          <div className="relative">
            <button
              onClick={() => setShowColumnMenu(!showColumnMenu)}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm bg-gray-100 text-gray-600 hover:bg-gray-200 transition"
            >
              <Eye size={14} />
              Columns
              <ChevronDown size={14} />
            </button>
            {showColumnMenu && (
              <div className="absolute top-full left-0 mt-1 bg-white border rounded-lg shadow-lg z-10 p-2 min-w-[160px]">
                {Object.entries(visibleColumns).map(([col, visible]) => (
                  <label
                    key={col}
                    className="flex items-center gap-2 px-2 py-1 hover:bg-gray-50 rounded cursor-pointer"
                  >
                    <input
                      type="checkbox"
                      checked={visible}
                      onChange={() => toggleColumn(col)}
                      className="rounded"
                    />
                    <span className="text-sm capitalize">
                      {col.replace(/([A-Z])/g, " $1").trim()}
                    </span>
                  </label>
                ))}
              </div>
            )}
          </div>

          {/* Clear Filters */}
          {hasActiveFilters && (
            <button
              onClick={clearFilters}
              className="flex items-center gap-1 px-3 py-1.5 rounded-lg text-sm text-red-600 hover:bg-red-50 transition"
            >
              <X size={14} />
              Clear
            </button>
          )}

          <span className="ml-auto text-sm text-gray-500">
            {sortedData.length} results
          </span>

          {/* Export CSV Button */}
          <button
            onClick={exportToCSV}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium bg-green-50 text-green-700 hover:bg-green-100 transition"
          >
            <Download size={14} />
            Export CSV
          </button>
        </div>

        {/* Advanced Filters Panel */}
        {showAdvancedFilters && (
          <div className="mt-4 pt-4 border-t space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Fidelity Range: {fidelityRange[0]}% - {fidelityRange[1]}%
              </label>
              <div className="flex items-center gap-4">
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={fidelityRange[0]}
                  onChange={(e) =>
                    setFidelityRange([
                      parseInt(e.target.value),
                      fidelityRange[1],
                    ])
                  }
                  className="flex-1"
                />
                <input
                  type="range"
                  min="0"
                  max="100"
                  value={fidelityRange[1]}
                  onChange={(e) =>
                    setFidelityRange([
                      fidelityRange[0],
                      parseInt(e.target.value),
                    ])
                  }
                  className="flex-1"
                />
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Behavioral Fidelity Analysis */}
      <div className="bg-white rounded-xl border p-6">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h3 className="text-lg font-semibold flex items-center gap-2">
              <Target size={20} className="text-purple-600" />
              Behavioral Fidelity Analysis
            </h3>
            <p className="text-sm text-gray-500">
              Fidelity threshold: ≥{(threshold * 100).toFixed(0)}% to be
              considered valid AI replica
            </p>
          </div>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                {visibleColumns.persona && (
                  <th
                    className="px-4 py-3 text-left cursor-pointer hover:bg-gray-100 transition"
                    onClick={() => handleSort("persona")}
                  >
                    <div className="flex items-center gap-1">
                      Persona
                      <SortIcon
                        active={sortConfig.key === "persona"}
                        direction={sortConfig.direction}
                      />
                    </div>
                  </th>
                )}
                {visibleColumns.model && (
                  <th
                    className="px-4 py-3 text-left cursor-pointer hover:bg-gray-100 transition"
                    onClick={() => handleSort("model")}
                  >
                    <div className="flex items-center gap-1">
                      Model
                      <SortIcon
                        active={sortConfig.key === "model"}
                        direction={sortConfig.direction}
                      />
                    </div>
                  </th>
                )}
                {visibleColumns.prompt && (
                  <th
                    className="px-4 py-3 text-left cursor-pointer hover:bg-gray-100 transition"
                    onClick={() => handleSort("prompt")}
                  >
                    <div className="flex items-center gap-1">
                      Prompt
                      <SortIcon
                        active={sortConfig.key === "prompt"}
                        direction={sortConfig.direction}
                      />
                    </div>
                  </th>
                )}
                {visibleColumns.fidelity && (
                  <th
                    className="px-4 py-3 text-right cursor-pointer hover:bg-gray-100 transition"
                    onClick={() => handleSort("normalized_accuracy")}
                  >
                    <div className="flex items-center justify-end gap-1">
                      Fidelity
                      <SortIcon
                        active={sortConfig.key === "normalized_accuracy"}
                        direction={sortConfig.direction}
                      />
                    </div>
                  </th>
                )}
                {visibleColumns.aiClick && (
                  <th
                    className="px-4 py-3 text-right cursor-pointer hover:bg-gray-100 transition"
                    onClick={() => handleSort("ai_click_rate")}
                  >
                    <div className="flex items-center justify-end gap-1">
                      AI Click
                      <SortIcon
                        active={sortConfig.key === "ai_click_rate"}
                        direction={sortConfig.direction}
                      />
                    </div>
                  </th>
                )}
                {visibleColumns.humanClick && (
                  <th
                    className="px-4 py-3 text-right cursor-pointer hover:bg-gray-100 transition"
                    onClick={() => handleSort("human_click_rate")}
                  >
                    <div className="flex items-center justify-end gap-1">
                      Human Click
                      <SortIcon
                        active={sortConfig.key === "human_click_rate"}
                        direction={sortConfig.direction}
                      />
                    </div>
                  </th>
                )}
                {visibleColumns.deviation && (
                  <th
                    className="px-4 py-3 text-center cursor-pointer hover:bg-gray-100 transition"
                    onClick={() => handleSort("deviation")}
                  >
                    <div className="flex items-center justify-center gap-1">
                      Deviation
                      <SortIcon
                        active={sortConfig.key === "deviation"}
                        direction={sortConfig.direction}
                      />
                    </div>
                  </th>
                )}
                {visibleColumns.status && (
                  <th className="px-4 py-3 text-center">Status</th>
                )}
              </tr>
            </thead>
            <tbody>
              {sortedData.length > 0 ? (
                sortedData.map((row, i) => (
                  <tr key={i} className="border-t hover:bg-gray-50 transition">
                    {visibleColumns.persona && (
                      <td className="px-4 py-3 font-medium">
                        {row.persona_name || row.persona_id}
                      </td>
                    )}
                    {visibleColumns.model && (
                      <td className="px-4 py-3">{row.model_id}</td>
                    )}
                    {visibleColumns.prompt && (
                      <td className="px-4 py-3">
                        <span
                          className={`px-2 py-0.5 rounded text-xs font-medium ${
                            row.prompt_config === "cot"
                              ? "bg-purple-100 text-purple-700"
                              : row.prompt_config === "stats"
                                ? "bg-blue-100 text-blue-700"
                                : "bg-gray-100 text-gray-700"
                          }`}
                        >
                          {row.prompt_config}
                        </span>
                      </td>
                    )}
                    {visibleColumns.fidelity && (
                      <td className="px-4 py-3">
                        {isNaN(row.normalized_accuracy) ? (
                          <span className="text-gray-400">N/A</span>
                        ) : (
                          <MiniBar
                            value={row.normalized_accuracy}
                            threshold={threshold}
                          />
                        )}
                      </td>
                    )}
                    {visibleColumns.aiClick && (
                      <td className="px-4 py-3 text-right font-mono">
                        {isNaN(row.ai_click_rate)
                          ? "N/A"
                          : `${(row.ai_click_rate * 100).toFixed(1)}%`}
                      </td>
                    )}
                    {visibleColumns.humanClick && (
                      <td className="px-4 py-3 text-right font-mono">
                        {isNaN(row.human_click_rate)
                          ? "N/A"
                          : `${(row.human_click_rate * 100).toFixed(1)}%`}
                      </td>
                    )}
                    {visibleColumns.deviation && (
                      <td className="px-4 py-3 text-center">
                        {isNaN(row.ai_click_rate) ||
                        isNaN(row.human_click_rate) ? (
                          <span className="text-gray-400">N/A</span>
                        ) : (
                          <DeviationIndicator
                            value={row.ai_click_rate - row.human_click_rate}
                          />
                        )}
                      </td>
                    )}
                    {visibleColumns.status && (
                      <td className="px-4 py-3 text-center">
                        {row.meets_threshold ||
                        row.normalized_accuracy >= threshold ? (
                          <CheckCircle
                            size={18}
                            className="inline text-green-500"
                          />
                        ) : (
                          <AlertCircle
                            size={18}
                            className="inline text-red-500"
                          />
                        )}
                      </td>
                    )}
                  </tr>
                ))
              ) : (
                <tr>
                  <td
                    colSpan={
                      Object.values(visibleColumns).filter(Boolean).length
                    }
                    className="px-4 py-8 text-center text-gray-500"
                  >
                    No results match your filters
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// PROMPTS SUB-TAB
// =============================================================================

const PromptsSubTab = ({
  promptAggregates,
  modelPromptBreakdown,
  threshold,
  selectedRateType,
  setSelectedRateType,
  summaryStats,
  modelAggregates,
}) => {
  const [categoryView, setCategoryView] = useState("none");
  const [showBreakdownTable, setShowBreakdownTable] = useState(false);

  // Get category for a model
  const getModelCategory = (modelName, categoryType) => {
    const category = MODEL_CATEGORIES[modelName];
    if (!category) return "Unknown";
    return category[categoryType] || "Unknown";
  };

  // Prepare data for grouped bar chart (individual models)
  const groupedData = useMemo(() => {
    const data = [];
    const modelSet = new Set(modelPromptBreakdown.map((d) => d.model));
    const models = Array.from(modelSet);

    models.forEach((model) => {
      const row = {
        model,
        category: categoryView !== "none" ? getModelCategory(model, categoryView) : null,
        categoryColor: categoryView !== "none" ? CATEGORY_COLORS[categoryView]?.[getModelCategory(model, categoryView)] || "#6b7280" : null,
      };
      ["baseline", "stats", "cot"].forEach((prompt) => {
        const match = modelPromptBreakdown.find(
          (d) => d.model === model && d.prompt === prompt,
        );
        row[prompt] = match ? match.avg_fidelity * 100 : 0;
      });
      data.push(row);
    });

    // Sort by category if grouping is active
    if (categoryView !== "none") {
      data.sort((a, b) => {
        if (a.category !== b.category) {
          return a.category.localeCompare(b.category);
        }
        // Within category, sort by average fidelity
        const avgA = (a.baseline + a.stats + a.cot) / 3;
        const avgB = (b.baseline + b.stats + b.cot) / 3;
        return avgB - avgA;
      });
    }

    return data;
  }, [modelPromptBreakdown, categoryView]);

  // Prepare category-aggregated data
  const categoryData = useMemo(() => {
    if (categoryView === "none") return null;

    const byCategory = {};
    modelPromptBreakdown.forEach((row) => {
      const category = getModelCategory(row.model, categoryView);
      if (!byCategory[category]) {
        byCategory[category] = {
          category,
          baseline: { sum: 0, count: 0 },
          stats: { sum: 0, count: 0 },
          cot: { sum: 0, count: 0 },
        };
      }
      if (row.prompt === "baseline") {
        byCategory[category].baseline.sum += row.avg_fidelity * 100;
        byCategory[category].baseline.count++;
      } else if (row.prompt === "stats") {
        byCategory[category].stats.sum += row.avg_fidelity * 100;
        byCategory[category].stats.count++;
      } else if (row.prompt === "cot") {
        byCategory[category].cot.sum += row.avg_fidelity * 100;
        byCategory[category].cot.count++;
      }
    });

    return Object.values(byCategory).map((cat) => ({
      category: cat.category,
      baseline: cat.baseline.count > 0 ? cat.baseline.sum / cat.baseline.count : 0,
      stats: cat.stats.count > 0 ? cat.stats.sum / cat.stats.count : 0,
      cot: cat.cot.count > 0 ? cat.cot.sum / cat.cot.count : 0,
      color: CATEGORY_COLORS[categoryView]?.[cat.category] || "#6b7280",
    }));
  }, [modelPromptBreakdown, categoryView]);

  // Get category label
  const getCategoryLabel = () => {
    const option = CATEGORY_OPTIONS.find((o) => o.id === categoryView);
    return option?.label || "Category";
  };

  return (
    <div className="space-y-6">
      {/* Rate Type Selector */}
      <RateTypeSwitcher
        selectedRateType={selectedRateType}
        setSelectedRateType={setSelectedRateType}
        summaryStats={summaryStats}
        modelAggregates={modelAggregates}
        compact={true}
      />

      {/* Group By Selector */}
      <div className="bg-white rounded-xl border p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers size={18} className="text-purple-600" />
            <span className="font-medium text-gray-700">Group Models By:</span>
          </div>
          <div className="flex gap-2">
            {CATEGORY_OPTIONS.map((option) => (
              <button
                key={option.id}
                onClick={() => setCategoryView(option.id)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  categoryView === option.id
                    ? "bg-purple-600 text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Prompt Summary Cards */}
      <div className="bg-white rounded-xl border p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <FileText size={20} className="text-purple-600" />
          Prompt Configuration Comparison
        </h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {["cot", "stats", "baseline"].map((promptType) => {
            const data = promptAggregates.find((p) => p.prompt === promptType);
            return (
              <div
                key={promptType}
                className="p-4 rounded-xl border-2"
                style={{ borderColor: PROMPT_COLORS[promptType] }}
              >
                <div
                  className="text-xs font-semibold uppercase mb-2"
                  style={{ color: PROMPT_COLORS[promptType] }}
                >
                  {PROMPT_LABELS[promptType] || promptType}
                </div>
                <div className="text-3xl font-bold mb-1">
                  {data ? `${(data.mean_fidelity * 100).toFixed(1)}%` : "N/A"}
                </div>
                <div className="text-sm text-gray-500">Average Fidelity</div>
                <div className="text-xs text-gray-400 mt-2">
                  {data
                    ? `${data.passing}/${data.total} passing (${(data.passRate * 100).toFixed(0)}%)`
                    : ""}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Category × Prompt Chart - shown when category is selected */}
      {categoryView !== "none" && categoryData && (
        <div className="bg-white rounded-xl border p-6">
          <h3 className="text-lg font-semibold mb-4">
            {getCategoryLabel()} × Prompt Fidelity
          </h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={categoryData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" tick={{ fontSize: 12 }} />
              <YAxis tickFormatter={(v) => `${v.toFixed(0)}%`} domain={[0, 100]} />
              <Tooltip formatter={(v) => `${v.toFixed(1)}%`} />
              <Legend />
              <ReferenceLine
                y={threshold * 100}
                stroke="#ef4444"
                strokeDasharray="5 5"
                label="85%"
              />
              <Bar
                dataKey="baseline"
                name="Baseline"
                fill={PROMPT_COLORS.baseline}
                radius={[2, 2, 0, 0]}
              />
              <Bar
                dataKey="stats"
                name="Stats"
                fill={PROMPT_COLORS.stats}
                radius={[2, 2, 0, 0]}
              />
              <Bar
                dataKey="cot"
                name="CoT"
                fill={PROMPT_COLORS.cot}
                radius={[2, 2, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
          {/* Insight */}
          <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <div className="flex items-start gap-2">
              <Zap size={16} className="text-blue-600 mt-0.5" />
              <div className="text-sm text-blue-800">
                {categoryData.length > 0 && (
                  <>
                    <strong>Best Performing:</strong>{" "}
                    {(() => {
                      const best = categoryData.reduce((best, cat) => {
                        const avg = (cat.baseline + cat.stats + cat.cot) / 3;
                        const bestAvg = (best.baseline + best.stats + best.cot) / 3;
                        return avg > bestAvg ? cat : best;
                      }, categoryData[0]);
                      const avg = ((best.baseline + best.stats + best.cot) / 3).toFixed(1);
                      return `${best.category} models achieve ${avg}% average fidelity across all prompt types.`;
                    })()}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Model × Prompt Chart - colored by category if selected */}
      <div className="bg-white rounded-xl border p-6">
        <h3 className="text-lg font-semibold mb-4">
          {categoryView === "none"
            ? "Model × Prompt Fidelity"
            : `Model × Prompt Fidelity (by ${getCategoryLabel()})`}
        </h3>
        <ResponsiveContainer width="100%" height={400}>
          <BarChart data={groupedData}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              dataKey="model"
              tick={{ fontSize: 9, angle: -30, textAnchor: "end" }}
              height={80}
            />
            <YAxis tickFormatter={(v) => `${v.toFixed(0)}%`} domain={[0, 100]} />
            <Tooltip
              formatter={(v) => `${v.toFixed(1)}%`}
              labelFormatter={(label) => {
                const model = groupedData.find((m) => m.model === label);
                return categoryView !== "none" && model?.category
                  ? `${label} (${model.category})`
                  : label;
              }}
            />
            <Legend />
            <ReferenceLine
              y={threshold * 100}
              stroke="#ef4444"
              strokeDasharray="5 5"
              label="85%"
            />
            <Bar
              dataKey="baseline"
              name="Baseline"
              fill={PROMPT_COLORS.baseline}
              radius={[2, 2, 0, 0]}
            />
            <Bar
              dataKey="stats"
              name="Stats"
              fill={PROMPT_COLORS.stats}
              radius={[2, 2, 0, 0]}
            />
            <Bar
              dataKey="cot"
              name="CoT"
              fill={PROMPT_COLORS.cot}
              radius={[2, 2, 0, 0]}
            />
          </BarChart>
        </ResponsiveContainer>
        {/* Category legend when grouped */}
        {categoryView !== "none" && (
          <div className="mt-4 flex flex-wrap gap-4 justify-center border-t pt-4">
            {[...new Set(groupedData.map((d) => d.category))].map((cat) => (
              <div key={cat} className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: CATEGORY_COLORS[categoryView]?.[cat] || "#6b7280" }}
                />
                <span className="text-sm text-gray-600">{cat}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Detailed Breakdown Table - Collapsible */}
      <div className="bg-white rounded-xl border p-6">
        <button
          onClick={() => setShowBreakdownTable(!showBreakdownTable)}
          className="w-full flex items-center justify-between text-lg font-semibold"
        >
          <span>Model × Prompt Breakdown</span>
          {showBreakdownTable ? (
            <ChevronUp size={20} className="text-gray-500" />
          ) : (
            <ChevronDown size={20} className="text-gray-500" />
          )}
        </button>
        {showBreakdownTable && (
          <div className="overflow-x-auto mt-4">
            <table className="w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left">Model</th>
                  {categoryView !== "none" && (
                    <th className="px-4 py-3 text-left">{getCategoryLabel()}</th>
                  )}
                  <th className="px-4 py-3 text-left">Prompt</th>
                  <th className="px-4 py-3 text-right">Avg Fidelity</th>
                  <th className="px-4 py-3 text-right">AI Click</th>
                  <th className="px-4 py-3 text-right">Human Click</th>
                  <th className="px-4 py-3 text-center">Status</th>
                </tr>
              </thead>
              <tbody>
                {modelPromptBreakdown
                  .slice()
                  .sort((a, b) => {
                    if (categoryView !== "none") {
                      const catA = getModelCategory(a.model, categoryView);
                      const catB = getModelCategory(b.model, categoryView);
                      if (catA !== catB) return catA.localeCompare(catB);
                    }
                    return b.avg_fidelity - a.avg_fidelity;
                  })
                  .map((row, i) => (
                    <tr key={i} className="border-t hover:bg-gray-50">
                      <td className="px-4 py-2 font-medium">{row.model}</td>
                      {categoryView !== "none" && (
                        <td className="px-4 py-2">
                          <span
                            className="text-xs px-2 py-0.5 rounded-full text-white"
                            style={{ backgroundColor: CATEGORY_COLORS[categoryView]?.[getModelCategory(row.model, categoryView)] || "#6b7280" }}
                          >
                            {getModelCategory(row.model, categoryView)}
                          </span>
                        </td>
                      )}
                      <td className="px-4 py-2">
                        <span
                          className={`px-2 py-0.5 rounded text-xs font-medium ${
                            row.prompt === "cot"
                              ? "bg-purple-100 text-purple-700"
                              : row.prompt === "stats"
                                ? "bg-blue-100 text-blue-700"
                                : "bg-gray-100 text-gray-700"
                          }`}
                        >
                          {row.prompt}
                        </span>
                      </td>
                      <td
                        className={`px-4 py-2 text-right font-mono ${
                          row.avg_fidelity >= threshold
                            ? "text-green-600 font-bold"
                            : "text-red-600"
                        }`}
                      >
                        {(row.avg_fidelity * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-2 text-right font-mono text-gray-600">
                        {(row.avg_ai_click * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-2 text-right font-mono text-gray-600">
                        {(row.avg_human_click * 100).toFixed(1)}%
                      </td>
                      <td className="px-4 py-2 text-center">
                        {row.avg_fidelity >= threshold ? (
                          <CheckCircle size={16} className="inline text-green-500" />
                        ) : (
                          <AlertCircle size={16} className="inline text-red-500" />
                        )}
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
};

// =============================================================================
// MODELS SUB-TAB
// =============================================================================

const ModelsSubTab = ({
  modelAggregates,
  threshold,
  selectedRateType,
  setSelectedRateType,
  summaryStats,
}) => {
  const [categoryView, setCategoryView] = useState("none");
  const [showModelRanking, setShowModelRanking] = useState(true);

  // Get category for a model
  const getModelCategory = (modelName, categoryType) => {
    const category = MODEL_CATEGORIES[modelName];
    if (!category) return "Unknown";
    return category[categoryType] || "Unknown";
  };

  // Compute category aggregates
  const categoryAggregates = useMemo(() => {
    if (categoryView === "none") return null;

    const byCategory = {};
    modelAggregates.forEach((model) => {
      const category = getModelCategory(model.model, categoryView);
      if (!byCategory[category]) {
        byCategory[category] = {
          category,
          models: [],
          totalFidelity: 0,
          minFidelity: Infinity,
          maxFidelity: -Infinity,
          totalTrials: 0,
        };
      }
      byCategory[category].models.push(model);
      byCategory[category].totalFidelity += model.mean_fidelity;
      byCategory[category].minFidelity = Math.min(
        byCategory[category].minFidelity,
        model.min_fidelity
      );
      byCategory[category].maxFidelity = Math.max(
        byCategory[category].maxFidelity,
        model.max_fidelity
      );
      byCategory[category].totalTrials += model.trials;
    });

    return Object.values(byCategory)
      .map((cat) => ({
        ...cat,
        mean_fidelity: cat.totalFidelity / cat.models.length,
        min_fidelity: cat.minFidelity === Infinity ? 0 : cat.minFidelity,
        max_fidelity: cat.maxFidelity === -Infinity ? 0 : cat.maxFidelity,
        modelCount: cat.models.length,
      }))
      .sort((a, b) => b.mean_fidelity - a.mean_fidelity);
  }, [modelAggregates, categoryView]);

  // Prepare chart data grouped by category
  const groupedChartData = useMemo(() => {
    if (categoryView === "none" || !categoryAggregates) return modelAggregates;

    // Sort models within each category
    const sortedModels = [...modelAggregates].sort((a, b) => {
      const catA = getModelCategory(a.model, categoryView);
      const catB = getModelCategory(b.model, categoryView);
      if (catA !== catB) {
        // Sort by category average fidelity (best category first)
        const catAAvg = categoryAggregates.find((c) => c.category === catA)?.mean_fidelity || 0;
        const catBAvg = categoryAggregates.find((c) => c.category === catB)?.mean_fidelity || 0;
        return catBAvg - catAAvg;
      }
      return b.mean_fidelity - a.mean_fidelity;
    });

    return sortedModels.map((model) => ({
      ...model,
      category: getModelCategory(model.model, categoryView),
      categoryColor: CATEGORY_COLORS[categoryView]?.[getModelCategory(model.model, categoryView)] || "#6b7280",
    }));
  }, [modelAggregates, categoryView, categoryAggregates]);

  // Get category label for display
  const getCategoryLabel = () => {
    const option = CATEGORY_OPTIONS.find((o) => o.id === categoryView);
    return option?.label || "Category";
  };

  // Collapse model ranking when switching to category view
  const handleCategoryChange = (newView) => {
    setCategoryView(newView);
    if (newView !== "none") {
      setShowModelRanking(false);
    } else {
      setShowModelRanking(true);
    }
  };

  return (
    <div className="space-y-6">
      {/* Rate Type Selector */}
      <RateTypeSwitcher
        selectedRateType={selectedRateType}
        setSelectedRateType={setSelectedRateType}
        summaryStats={summaryStats}
        modelAggregates={modelAggregates}
        compact={true}
      />

      {/* Category View Selector */}
      <div className="bg-white rounded-xl border p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Layers size={18} className="text-purple-600" />
            <span className="font-medium text-gray-700">Group By:</span>
          </div>
          <div className="flex gap-2">
            {CATEGORY_OPTIONS.map((option) => (
              <button
                key={option.id}
                onClick={() => handleCategoryChange(option.id)}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  categoryView === option.id
                    ? "bg-purple-600 text-white"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                {option.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Category Summary Cards - shown when grouping is active */}
      {categoryView !== "none" && categoryAggregates && (
        <div className="bg-white rounded-xl border p-6">
          <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <TrendingUp size={20} className="text-purple-600" />
            {getCategoryLabel()} Performance Comparison
          </h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {categoryAggregates.map((cat, index) => (
              <div
                key={cat.category}
                className={`p-4 rounded-xl border-2 ${
                  index === 0 ? "border-yellow-400 bg-yellow-50" : "border-gray-200"
                }`}
              >
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center gap-2">
                    <div
                      className="w-3 h-3 rounded-full"
                      style={{ backgroundColor: CATEGORY_COLORS[categoryView]?.[cat.category] || "#6b7280" }}
                    />
                    <span className="font-semibold text-gray-800">{cat.category}</span>
                    {index === 0 && (
                      <span className="text-xs bg-yellow-400 text-white px-2 py-0.5 rounded-full">Best</span>
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-3">
                  <div>
                    <div
                      className={`text-2xl font-bold ${
                        cat.mean_fidelity >= threshold ? "text-green-600" : "text-amber-600"
                      }`}
                    >
                      {(cat.mean_fidelity * 100).toFixed(1)}%
                    </div>
                    <div className="text-xs text-gray-500">Avg Fidelity</div>
                  </div>
                  <div>
                    <div className="text-lg font-semibold text-gray-700">{cat.modelCount}</div>
                    <div className="text-xs text-gray-500">Models</div>
                  </div>
                </div>
                <div className="mt-3 pt-3 border-t border-gray-100">
                  <div className="flex justify-between text-xs text-gray-500">
                    <span>Min: {(cat.min_fidelity * 100).toFixed(1)}%</span>
                    <span>Max: {(cat.max_fidelity * 100).toFixed(1)}%</span>
                  </div>
                  <div className="text-xs text-gray-400 mt-1">
                    {cat.totalTrials.toLocaleString()} total trials
                  </div>
                </div>
                {/* Models in this category */}
                <div className="mt-3 pt-2 border-t border-gray-100">
                  <div className="text-xs text-gray-500 mb-1">Models:</div>
                  <div className="flex flex-wrap gap-1">
                    {cat.models
                      .sort((a, b) => b.mean_fidelity - a.mean_fidelity)
                      .slice(0, 5)
                      .map((m) => (
                        <span
                          key={m.model}
                          className="text-xs px-2 py-0.5 bg-gray-100 rounded text-gray-600"
                        >
                          {m.model}
                        </span>
                      ))}
                    {cat.models.length > 5 && (
                      <span className="text-xs text-gray-400">+{cat.models.length - 5} more</span>
                    )}
                  </div>
                </div>
              </div>
            ))}
          </div>
          {/* Insight Text */}
          <div className="mt-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
            <div className="flex items-start gap-2">
              <Zap size={16} className="text-blue-600 mt-0.5" />
              <div className="text-sm text-blue-800">
                {categoryView === "openness" && (
                  <>
                    <strong>Open vs Closed Models:</strong>{" "}
                    {categoryAggregates[0]?.category === "Open"
                      ? "Open-source models outperform closed/proprietary models on average."
                      : "Closed/proprietary models outperform open-source models on average."}
                  </>
                )}
                {categoryView === "sizeTier" && (
                  <>
                    <strong>Size Tier Analysis:</strong>{" "}
                    {categoryAggregates[0]?.category} models achieve the highest average fidelity (
                    {(categoryAggregates[0]?.mean_fidelity * 100).toFixed(1)}%).
                  </>
                )}
                {categoryView === "architecture" && (
                  <>
                    <strong>Architecture Comparison:</strong>{" "}
                    {categoryAggregates[0]?.category === "MoE"
                      ? "Mixture of Experts (MoE) architectures outperform Dense Transformers on average."
                      : "Dense Transformer architectures outperform Mixture of Experts (MoE) on average."}
                  </>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Models by Category Chart - shown when grouped, BEFORE model ranking */}
      {categoryView !== "none" && categoryAggregates && (
        <div className="bg-white rounded-xl border p-6">
          <h3 className="text-lg font-semibold mb-4">
            Models by {getCategoryLabel()} (Individual Performance)
          </h3>
          <ResponsiveContainer width="100%" height={400}>
            <ComposedChart data={groupedChartData}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="model"
                tick={{ fontSize: 9, angle: -30, textAnchor: "end" }}
                height={80}
              />
              <YAxis
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                domain={[0, 1]}
              />
              <Tooltip
                formatter={(v, name) => [`${(v * 100).toFixed(1)}%`, name]}
                labelFormatter={(label) => {
                  const model = groupedChartData.find((m) => m.model === label);
                  return `${label} (${model?.category || "Unknown"})`;
                }}
              />
              <Legend />
              <ReferenceLine
                y={threshold}
                stroke="#ef4444"
                strokeDasharray="5 5"
                label="85%"
              />
              <Bar dataKey="mean_fidelity" name="Mean Fidelity" radius={[4, 4, 0, 0]}>
                {groupedChartData.map((entry) => (
                  <Cell key={entry.model} fill={entry.categoryColor} />
                ))}
              </Bar>
            </ComposedChart>
          </ResponsiveContainer>
          <div className="mt-4 flex flex-wrap gap-4 justify-center">
            {categoryAggregates.map((cat) => (
              <div key={cat.category} className="flex items-center gap-2">
                <div
                  className="w-3 h-3 rounded-full"
                  style={{ backgroundColor: CATEGORY_COLORS[categoryView]?.[cat.category] || "#6b7280" }}
                />
                <span className="text-sm text-gray-600">
                  {cat.category} (avg: {(cat.mean_fidelity * 100).toFixed(1)}%)
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Model Ranking - Collapsible when category view is active */}
      <div className="bg-white rounded-xl border p-6">
        <button
          onClick={() => setShowModelRanking(!showModelRanking)}
          className="w-full flex items-center justify-between text-lg font-semibold"
        >
          <div className="flex items-center gap-2">
            <TrendingUp size={20} className="text-purple-600" />
            {categoryView === "none" ? "Model Ranking" : `Model Ranking (Grouped by ${getCategoryLabel()})`}
          </div>
          {showModelRanking ? (
            <ChevronUp size={20} className="text-gray-500" />
          ) : (
            <ChevronDown size={20} className="text-gray-500" />
          )}
        </button>
        {showModelRanking && (
          <div className="space-y-3 mt-4">
            {groupedChartData.map((model, index) => (
              <div
                key={model.model}
                className={`flex items-center justify-between p-4 rounded-xl border-2 ${
                  index === 0
                    ? "border-yellow-400 bg-yellow-50"
                    : "border-gray-200"
                }`}
              >
                <div className="flex items-center gap-4">
                  <div
                    className={`w-8 h-8 rounded-full flex items-center justify-center font-bold ${
                      index === 0
                        ? "bg-yellow-400 text-white"
                        : index === 1
                          ? "bg-gray-300 text-gray-700"
                          : index === 2
                            ? "bg-amber-600 text-white"
                            : "bg-gray-100 text-gray-600"
                    }`}
                  >
                    {index + 1}
                  </div>
                  <div>
                    <div className="font-semibold text-purple-700 flex items-center gap-2">
                      {model.model}
                      {categoryView !== "none" && (
                        <span
                          className="text-xs px-2 py-0.5 rounded-full text-white"
                          style={{ backgroundColor: model.categoryColor }}
                        >
                          {model.category}
                        </span>
                      )}
                    </div>
                    <div className="text-xs text-gray-500">
                      {model.trials} trials • Best:{" "}
                      {model.mean_fidelity >= threshold
                        ? "passing"
                        : model.passing > 0
                          ? `${model.passing}/${model.n_conditions} passing`
                          : "none passing"}
                    </div>
                  </div>
                </div>
                <div className="text-right">
                  <div
                    className={`text-2xl font-bold ${
                      model.mean_fidelity >= threshold
                        ? "text-green-600"
                        : "text-amber-600"
                    }`}
                  >
                    {(model.mean_fidelity * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500">Mean Fidelity</div>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Fidelity Distribution Chart - only shown for individual models view */}
      {categoryView === "none" && (
        <div className="bg-white rounded-xl border p-6">
          <h3 className="text-lg font-semibold mb-4">
            Fidelity Distribution by Model
          </h3>
          <ResponsiveContainer width="100%" height={350}>
            <ComposedChart data={modelAggregates}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="model"
                tick={{ fontSize: 10, angle: -20, textAnchor: "end" }}
                height={60}
              />
              <YAxis
                tickFormatter={(v) => `${(v * 100).toFixed(0)}%`}
                domain={[0, 1]}
              />
              <Tooltip formatter={(v) => `${(v * 100).toFixed(1)}%`} />
              <Legend />
              <ReferenceLine
                y={threshold}
                stroke="#ef4444"
                strokeDasharray="5 5"
                label="85%"
              />
              <Bar
                dataKey="mean_fidelity"
                name="Mean"
                fill="#8b5cf6"
                radius={[4, 4, 0, 0]}
              />
              <Line
                type="monotone"
                dataKey="max_fidelity"
                name="Max"
                stroke="#22c55e"
                strokeWidth={2}
                dot
              />
              <Line
                type="monotone"
                dataKey="min_fidelity"
                name="Min"
                stroke="#ef4444"
                strokeWidth={2}
                dot
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

// =============================================================================
// COSTS SUB-TAB
// =============================================================================

const CostsSubTab = ({
  costAnalysis,
  modelAggregates,
  selectedRateType,
  setSelectedRateType,
  summaryStats,
}) => {
  // Prepare scatter data for fidelity vs cost
  const scatterData = costAnalysis.map((m) => ({
    model: m.model,
    fidelity: m.avgFidelity * 100,
    costPerTrial: m.costPerTrial * 1000, // Convert to per 1000 for visibility
    valueScore: m.valueScore,
  }));

  // Find pareto optimal models
  const paretoOptimal = costAnalysis
    .filter((m) => {
      return !costAnalysis.some(
        (other) =>
          other.avgFidelity > m.avgFidelity &&
          other.costPerTrial < m.costPerTrial,
      );
    })
    .map((m) => m.model);

  return (
    <div className="space-y-6">
      {/* Rate Type Selector */}
      <RateTypeSwitcher
        selectedRateType={selectedRateType}
        setSelectedRateType={setSelectedRateType}
        summaryStats={summaryStats}
        modelAggregates={modelAggregates}
        compact={true}
      />

      {/* Cost-Fidelity Table */}
      <div className="bg-white rounded-xl border p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <DollarSign size={20} className="text-green-600" />
          Cost-Fidelity Analysis
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left">Model</th>
                <th className="px-4 py-3 text-right">Fidelity</th>
                <th className="px-4 py-3 text-right">Total Cost</th>
                <th className="px-4 py-3 text-right">Cost/Trial</th>
                <th className="px-4 py-3 text-right">Latency (p50)</th>
                <th className="px-4 py-3 text-right">Value Score</th>
              </tr>
            </thead>
            <tbody>
              {costAnalysis.map((row, i) => (
                <tr key={i} className="border-t hover:bg-gray-50">
                  <td className="px-4 py-2 font-medium">{row.model}</td>
                  <td
                    className={`px-4 py-2 text-right font-mono ${
                      row.avgFidelity >= 0.85
                        ? "text-green-600"
                        : "text-red-600"
                    }`}
                  >
                    {(row.avgFidelity * 100).toFixed(1)}%
                  </td>
                  <td className="px-4 py-2 text-right font-mono">
                    ${row.totalCost.toFixed(4)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono">
                    ${row.costPerTrial.toFixed(4)}
                  </td>
                  <td className="px-4 py-2 text-right font-mono text-gray-600">
                    {row.p50Latency ? `${row.p50Latency.toFixed(0)}ms` : "-"}
                  </td>
                  <td
                    className={`px-4 py-2 text-right font-mono font-bold ${
                      row.valueScore > 1000 ? "text-green-600" : "text-gray-600"
                    }`}
                  >
                    {row.valueScore.toFixed(1)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <p className="text-xs text-gray-500 mt-4">
          <strong>Value Score</strong> = Fidelity ÷ Cost per Trial. Higher means
          more fidelity per unit cost - Better value for research budget.
        </p>
      </div>

      {/* Fidelity vs Cost Scatter Plot */}
      <div className="bg-white rounded-xl border p-6">
        <h3 className="text-lg font-semibold mb-4">
          Fidelity vs Cost Trade-off
        </h3>
        <ResponsiveContainer width="100%" height={350}>
          <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis
              type="number"
              dataKey="costPerTrial"
              name="Cost"
              unit="$"
              tickFormatter={(v) => `$${(v / 1000).toFixed(4)}`}
              label={{
                value: "Cost per Trial ($)",
                position: "bottom",
                offset: 0,
              }}
            />
            <YAxis
              type="number"
              dataKey="fidelity"
              name="Fidelity"
              unit="%"
              domain={[0, 100]}
              label={{
                value: "Fidelity (%)",
                angle: -90,
                position: "insideLeft",
              }}
            />
            <Tooltip
              formatter={(value, name) => {
                if (name === "Cost") return `$${(value / 1000).toFixed(4)}`;
                return `${value.toFixed(1)}%`;
              }}
              labelFormatter={(label) =>
                scatterData.find((d) => d.costPerTrial === label)?.model ||
                label
              }
            />
            <ReferenceLine
              y={85}
              stroke="#ef4444"
              strokeDasharray="5 5"
              label="85% threshold"
            />
            <Scatter name="Models" data={scatterData} fill="#8b5cf6">
              {scatterData.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={
                    paretoOptimal.includes(entry.model) ? "#22c55e" : "#8b5cf6"
                  }
                />
              ))}
            </Scatter>
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Pareto Optimal Models */}
      <div className="bg-white rounded-xl border p-6">
        <h3 className="text-lg font-semibold mb-2">Pareto Optimal Models</h3>
        <p className="text-sm text-gray-500 mb-4">
          Best tradeoff between fidelity and cost
        </p>
        <div className="flex flex-wrap gap-2">
          {paretoOptimal.map((model) => (
            <span
              key={model}
              className="px-3 py-1 bg-green-100 text-green-700 rounded-full text-sm font-medium"
            >
              {model}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// BOUNDARIES SUB-TAB
// =============================================================================

const BoundariesSubTab = ({
  boundaryConditions,
  personaAggregates,
  selectedRateType,
  setSelectedRateType,
  summaryStats,
  modelAggregates,
}) => {
  const severityColors = {
    high: "border-red-300 bg-red-50",
    medium: "border-amber-300 bg-amber-50",
    low: "border-yellow-300 bg-yellow-50",
  };

  return (
    <div className="space-y-6">
      {/* Rate Type Selector */}
      <RateTypeSwitcher
        selectedRateType={selectedRateType}
        setSelectedRateType={setSelectedRateType}
        summaryStats={summaryStats}
        modelAggregates={modelAggregates}
        compact={true}
      />

      {/* Boundary Conditions */}
      <div className="bg-white rounded-xl border p-6">
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
          <AlertCircle size={20} className="text-amber-600" />
          Boundary Conditions (Where AI Fails)
        </h3>

        {boundaryConditions.length > 0 ? (
          <div className="space-y-4">
            {boundaryConditions.map((condition, i) => (
              <div
                key={i}
                className={`p-4 rounded-xl border-2 ${severityColors[condition.severity]}`}
              >
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2 mb-1">
                      <AlertCircle
                        size={16}
                        className={
                          condition.severity === "high"
                            ? "text-red-600"
                            : condition.severity === "medium"
                              ? "text-amber-600"
                              : "text-yellow-600"
                        }
                      />
                      <span className="font-semibold">{condition.type}</span>
                      <span
                        className={`text-xs px-2 py-0.5 rounded ${
                          condition.severity === "high"
                            ? "bg-red-200 text-red-800"
                            : condition.severity === "medium"
                              ? "bg-amber-200 text-amber-800"
                              : "bg-yellow-200 text-yellow-800"
                        }`}
                      >
                        {condition.severity}
                      </span>
                    </div>
                    <p className="text-sm text-gray-600">
                      AI ({(condition.aiRate * 100).toFixed(1)}%){" "}
                      {condition.type.includes("Over")
                        ? "much more"
                        : "much less"}{" "}
                      than human persona (
                      {(condition.humanRate * 100).toFixed(1)}%)
                    </p>
                    <p className="text-xs text-gray-500 mt-1">
                      Affected persona: {condition.persona}
                    </p>
                  </div>
                </div>
                <div className="mt-3 text-sm text-green-700 flex items-center gap-1">
                  <span>💡</span>
                  {condition.suggestion}
                </div>
              </div>
            ))}
          </div>
        ) : (
          <div className="text-center py-8 text-gray-500">
            <CheckCircle size={32} className="mx-auto mb-2 text-green-500" />
            <p>No significant boundary conditions detected</p>
            <p className="text-sm">
              AI behavior is within acceptable range for all personas
            </p>
          </div>
        )}
      </div>

      {/* AI vs Human Rate Comparison - uses selectedRateType */}
      <div className="bg-white rounded-xl border p-6">
        <h3 className="text-lg font-semibold mb-4">
          {getRateTypeLabel(selectedRateType)} Deviation by Persona
        </h3>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart
            data={personaAggregates.map((p) => ({
              persona:
                p.persona.length > 20
                  ? p.persona.substring(0, 20) + "..."
                  : p.persona,
              deviation: (p.mean_ai_rate - p.mean_human_rate) * 100,
              aiRate: p.mean_ai_rate * 100,
              humanRate: p.mean_human_rate * 100,
            }))}
          >
            <CartesianGrid strokeDasharray="3 3" />
            <XAxis dataKey="persona" tick={{ fontSize: 10 }} />
            <YAxis
              tickFormatter={(v) => `${v > 0 ? "+" : ""}${v.toFixed(0)}%`}
            />
            <Tooltip
              formatter={(v, name) => `${v > 0 ? "+" : ""}${v.toFixed(1)}%`}
              labelFormatter={(label) => label}
            />
            <ReferenceLine y={0} stroke="#000" />
            <ReferenceLine
              y={20}
              stroke="#ef4444"
              strokeDasharray="3 3"
              label="Alert"
            />
            <ReferenceLine y={-20} stroke="#ef4444" strokeDasharray="3 3" />
            <Bar
              dataKey="deviation"
              name="AI-Human Deviation"
              radius={[4, 4, 0, 0]}
            >
              {personaAggregates.map((entry, index) => (
                <Cell
                  key={`cell-${index}`}
                  fill={
                    Math.abs(entry.mean_ai_rate - entry.mean_human_rate) > 0.2
                      ? "#ef4444"
                      : "#22c55e"
                  }
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
        <p className="text-xs text-gray-500 mt-2 text-center">
          Positive values = AI rate higher than humans | Negative = AI rate lower
        </p>
      </div>
    </div>
  );
};

// =============================================================================
// COMBINATIONS SUB-TAB (4D: LLM × Persona × Email × Prompt)
// =============================================================================

const CombinationsSubTab = ({
  detailedCombinations,
  loadingCombinations,
  onRefresh,
  filterEmail,
  setFilterEmail,
  filterPersona,
  setFilterPersona,
  filterModel,
  setFilterModel,
  filterPrompt,
  setFilterPrompt,
  // Email attribute filters
  filterEmailType,
  setFilterEmailType,
  filterSenderFamiliarity,
  setFilterSenderFamiliarity,
  filterUrgency,
  setFilterUrgency,
  filterFraming,
  setFilterFraming,
  filterAggression,
  setFilterAggression,
  sortKey,
  setSortKey,
  sortDir,
  setSortDir,
  threshold,
}) => {
  // Get filter options from data
  const dimensions = detailedCombinations?.dimensions || {};
  const emails = ["all", ...(dimensions.emails || [])];
  const personas = ["all", ...(dimensions.personas || [])];
  const models = ["all", ...(dimensions.models || [])];
  const prompts = ["all", ...(dimensions.prompts || [])];
  // Email attribute filter options
  const emailTypes = ["all", ...(dimensions.email_types || [])];
  const senderFamiliarities = [
    "all",
    ...(dimensions.sender_familiarities || []),
  ];
  const urgencyLevels = ["all", ...(dimensions.urgency_levels || [])];
  const framingTypes = ["all", ...(dimensions.framing_types || [])];
  const aggressionLevels = ["all", ...(dimensions.aggression_levels || [])];

  // Filter and sort data
  const filteredData = useMemo(() => {
    if (!detailedCombinations?.detailed_results) return [];

    let data = [...detailedCombinations.detailed_results];

    // Apply basic filters
    if (filterEmail !== "all") {
      data = data.filter((d) => d.email_id === filterEmail);
    }
    if (filterPersona !== "all") {
      data = data.filter((d) => d.persona_name === filterPersona);
    }
    if (filterModel !== "all") {
      data = data.filter((d) => d.model_id === filterModel);
    }
    if (filterPrompt !== "all") {
      data = data.filter((d) => d.prompt_config === filterPrompt);
    }

    // Apply email attribute filters for business analysis
    if (filterEmailType !== "all") {
      data = data.filter((d) => d.email_type === filterEmailType);
    }
    if (filterSenderFamiliarity !== "all") {
      data = data.filter(
        (d) => d.sender_familiarity === filterSenderFamiliarity,
      );
    }
    if (filterUrgency !== "all") {
      data = data.filter((d) => d.urgency_level === filterUrgency);
    }
    if (filterFraming !== "all") {
      data = data.filter((d) => d.framing_type === filterFraming);
    }
    if (filterAggression !== "all") {
      data = data.filter((d) => d.aggression_level === filterAggression);
    }

    // Sort
    data.sort((a, b) => {
      let aVal = a[sortKey] || 0;
      let bVal = b[sortKey] || 0;
      if (sortDir === "asc") return aVal - bVal;
      return bVal - aVal;
    });

    return data;
  }, [
    detailedCombinations,
    filterEmail,
    filterPersona,
    filterModel,
    filterPrompt,
    filterEmailType,
    filterSenderFamiliarity,
    filterUrgency,
    filterFraming,
    filterAggression,
    sortKey,
    sortDir,
  ]);

  // Calculate summary stats for filtered data
  const summary = useMemo(() => {
    if (filteredData.length === 0)
      return { avgFidelity: 0, totalCost: 0, passing: 0 };
    const avgFidelity =
      filteredData.reduce((sum, d) => sum + d.fidelity, 0) /
      filteredData.length;
    const totalCost = filteredData.reduce((sum, d) => sum + d.cost, 0);
    const passing = filteredData.filter((d) => d.fidelity >= threshold).length;
    return { avgFidelity, totalCost, passing };
  }, [filteredData, threshold]);

  // Handle sort toggle
  const handleSort = (key) => {
    if (sortKey === key) {
      setSortDir(sortDir === "asc" ? "desc" : "asc");
    } else {
      setSortKey(key);
      setSortDir("desc");
    }
  };

  // Export to CSV
  const exportToCSV = () => {
    const headers = [
      "Persona",
      "Model",
      "Prompt",
      "Email ID",
      "Email Subject",
      "Email Type",
      "Sender Familiarity",
      "Urgency Level",
      "Framing Type",
      "Aggression Level",
      "Fidelity",
      "AI Click Rate",
      "Human Click Rate",
      "Deviation",
      "Cost",
      "Cost/Trial",
      "Trials",
      "Status",
    ];
    const rows = filteredData.map((row) => [
      row.persona_name,
      row.model_id,
      row.prompt_config,
      row.email_id,
      `"${(row.email_subject || "").replace(/"/g, '""')}"`,
      row.email_type || "",
      row.sender_familiarity || "",
      row.urgency_level || "",
      row.framing_type || "",
      row.aggression_level || "",
      (row.fidelity * 100).toFixed(2),
      (row.ai_click_rate * 100).toFixed(2),
      (row.human_click_rate * 100).toFixed(2),
      (row.click_deviation * 100).toFixed(2),
      row.cost.toFixed(6),
      row.cost_per_trial.toFixed(6),
      row.n_trials,
      row.meets_threshold ? "Pass" : "Fail",
    ]);
    const csvContent = [headers, ...rows].map((r) => r.join(",")).join("\n");
    const blob = new Blob([csvContent], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `combinations_analysis_${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  if (loadingCombinations) {
    return (
      <div className="flex items-center justify-center py-12">
        <div className="animate-spin rounded-full h-12 w-12 border-4 border-purple-200 border-t-purple-600"></div>
        <span className="ml-4 text-gray-600">
          Loading detailed combinations...
        </span>
      </div>
    );
  }

  if (!detailedCombinations) {
    return (
      <div className="bg-white rounded-xl border p-8 text-center">
        <Grid3X3 size={48} className="mx-auto text-gray-300 mb-4" />
        <h3 className="text-lg font-semibold text-gray-900">
          No Combination Data
        </h3>
        <p className="text-gray-500 mt-2">
          Run an experiment to see detailed LLM × Persona × Email × Prompt
          combinations.
        </p>
        <button
          onClick={onRefresh}
          className="mt-4 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
        >
          Load Combinations
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header with Summary */}
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-xl p-6">
        <div className="flex items-start justify-between">
          <div>
            <h3 className="text-lg font-semibold text-purple-900 flex items-center gap-2">
              <Grid3X3 size={20} />
              Business Comparison: LLM × Persona × Email × Prompt
            </h3>
            <p className="text-sm text-purple-700 mt-1">
              Compare all{" "}
              {detailedCombinations.summary?.total_combinations || 0}{" "}
              combinations to find the best configuration for your needs.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={onRefresh}
              className="px-3 py-1.5 bg-white border rounded-lg text-sm hover:bg-gray-50 flex items-center gap-1"
            >
              <Zap size={14} />
              Refresh
            </button>
            <button
              onClick={exportToCSV}
              className="px-3 py-1.5 bg-purple-600 text-white rounded-lg text-sm hover:bg-purple-700 flex items-center gap-1"
            >
              <Download size={14} />
              Export CSV
            </button>
          </div>
        </div>

        {/* Quick Stats */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
          <div className="bg-white/70 rounded-lg p-3">
            <div className="text-xs text-gray-500">Showing</div>
            <div className="text-xl font-bold text-purple-600">
              {filteredData.length}
            </div>
            <div className="text-xs text-gray-500">combinations</div>
          </div>
          <div className="bg-white/70 rounded-lg p-3">
            <div className="text-xs text-gray-500">Avg Fidelity</div>
            <div
              className={`text-xl font-bold ${summary.avgFidelity >= threshold ? "text-green-600" : "text-amber-600"}`}
            >
              {(summary.avgFidelity * 100).toFixed(1)}%
            </div>
            <div className="text-xs text-gray-500">across filtered</div>
          </div>
          <div className="bg-white/70 rounded-lg p-3">
            <div className="text-xs text-gray-500">Passing</div>
            <div className="text-xl font-bold text-green-600">
              {summary.passing}
            </div>
            <div className="text-xs text-gray-500">
              ≥{threshold * 100}% fidelity
            </div>
          </div>
          <div className="bg-white/70 rounded-lg p-3">
            <div className="text-xs text-gray-500">Total Cost</div>
            <div className="text-xl font-bold text-gray-700">
              ${summary.totalCost.toFixed(4)}
            </div>
            <div className="text-xs text-gray-500">for filtered trials</div>
          </div>
        </div>
      </div>

      {/* Filters */}
      <div className="bg-white rounded-xl border p-4">
        <div className="flex items-center gap-2 mb-3">
          <Filter size={16} className="text-gray-500" />
          <span className="font-medium text-sm">Filter Combinations</span>
        </div>

        {/* Experiment Dimensions Row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-4">
          <div>
            <label className="text-xs text-gray-500 mb-1 block">Email ID</label>
            <select
              value={filterEmail}
              onChange={(e) => setFilterEmail(e.target.value)}
              className="w-full border rounded-lg px-3 py-2 text-sm bg-white"
            >
              {emails.map((e) => (
                <option key={e} value={e}>
                  {e === "all" ? "All Emails" : e}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-500 mb-1 block">Persona</label>
            <select
              value={filterPersona}
              onChange={(e) => setFilterPersona(e.target.value)}
              className="w-full border rounded-lg px-3 py-2 text-sm bg-white"
            >
              {personas.map((p) => (
                <option key={p} value={p}>
                  {p === "all" ? "All Personas" : p}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-500 mb-1 block">
              LLM Model
            </label>
            <select
              value={filterModel}
              onChange={(e) => setFilterModel(e.target.value)}
              className="w-full border rounded-lg px-3 py-2 text-sm bg-white"
            >
              {models.map((m) => (
                <option key={m} value={m}>
                  {m === "all" ? "All Models" : m}
                </option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-500 mb-1 block">
              Prompt Config
            </label>
            <select
              value={filterPrompt}
              onChange={(e) => setFilterPrompt(e.target.value)}
              className="w-full border rounded-lg px-3 py-2 text-sm bg-white"
            >
              {prompts.map((p) => (
                <option key={p} value={p}>
                  {p === "all" ? "All Prompts" : p.toUpperCase()}
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Email Characteristics Row - Business Analysis Filters */}
        <div className="border-t pt-3">
          <div className="flex items-center gap-2 mb-2">
            <span className="text-xs font-medium text-purple-600">
              Email Characteristics
            </span>
            <span className="text-xs text-gray-400">
              (for business analysis)
            </span>
          </div>
          <div className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Type</label>
              <select
                value={filterEmailType}
                onChange={(e) => setFilterEmailType(e.target.value)}
                className="w-full border rounded-lg px-3 py-2 text-sm bg-white"
              >
                {emailTypes.map((t) => (
                  <option key={t} value={t}>
                    {t === "all"
                      ? "All Types"
                      : t === "phishing"
                        ? "⚠ Phishing"
                        : "✓ Legitimate"}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">Sender</label>
              <select
                value={filterSenderFamiliarity}
                onChange={(e) => setFilterSenderFamiliarity(e.target.value)}
                className="w-full border rounded-lg px-3 py-2 text-sm bg-white"
              >
                {senderFamiliarities.map((s) => (
                  <option key={s} value={s}>
                    {s === "all"
                      ? "All Senders"
                      : s === "familiar"
                        ? "Known Sender"
                        : "Unknown Sender"}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">
                Urgency
              </label>
              <select
                value={filterUrgency}
                onChange={(e) => setFilterUrgency(e.target.value)}
                className="w-full border rounded-lg px-3 py-2 text-sm bg-white"
              >
                {urgencyLevels.map((u) => (
                  <option key={u} value={u}>
                    {u === "all"
                      ? "All Urgency"
                      : u === "high"
                        ? "⚡ High"
                        : u === "medium"
                          ? "Medium"
                          : "Low"}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">
                Framing
              </label>
              <select
                value={filterFraming}
                onChange={(e) => setFilterFraming(e.target.value)}
                className="w-full border rounded-lg px-3 py-2 text-sm bg-white"
              >
                {framingTypes.map((f) => (
                  <option key={f} value={f}>
                    {f === "all"
                      ? "All Framing"
                      : f === "threat"
                        ? "Threat"
                        : f === "reward"
                          ? "Reward"
                          : "Neutral"}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="text-xs text-gray-500 mb-1 block">
                Aggression
              </label>
              <select
                value={filterAggression}
                onChange={(e) => setFilterAggression(e.target.value)}
                className="w-full border rounded-lg px-3 py-2 text-sm bg-white"
              >
                {aggressionLevels.map((a) => (
                  <option key={a} value={a}>
                    {a === "all"
                      ? "All Levels"
                      : a === "very_high"
                        ? "Very High"
                        : a.charAt(0).toUpperCase() + a.slice(1)}
                  </option>
                ))}
              </select>
            </div>
          </div>
        </div>
      </div>

      {/* Results Table */}
      <div className="bg-white rounded-xl border overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 border-b">
              <tr>
                <th
                  className="px-3 py-3 text-left cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort("persona_name")}
                >
                  <div className="flex items-center gap-1">
                    Persona
                    <SortIcon
                      active={sortKey === "persona_name"}
                      direction={sortDir}
                    />
                  </div>
                </th>
                <th
                  className="px-3 py-3 text-left cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort("model_id")}
                >
                  <div className="flex items-center gap-1">
                    LLM Model
                    <SortIcon
                      active={sortKey === "model_id"}
                      direction={sortDir}
                    />
                  </div>
                </th>
                <th className="px-3 py-3 text-left">Prompt</th>
                <th className="px-3 py-3 text-left">Email</th>
                <th
                  className="px-3 py-3 text-right cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort("fidelity")}
                >
                  <div className="flex items-center justify-end gap-1">
                    Fidelity
                    <SortIcon
                      active={sortKey === "fidelity"}
                      direction={sortDir}
                    />
                  </div>
                </th>
                <th className="px-3 py-3 text-right">AI Click</th>
                <th className="px-3 py-3 text-right">Human Click</th>
                <th className="px-3 py-3 text-center">Deviation</th>
                <th
                  className="px-3 py-3 text-right cursor-pointer hover:bg-gray-100"
                  onClick={() => handleSort("cost")}
                >
                  <div className="flex items-center justify-end gap-1">
                    Cost
                    <SortIcon active={sortKey === "cost"} direction={sortDir} />
                  </div>
                </th>
                <th className="px-3 py-3 text-center">Status</th>
              </tr>
            </thead>
            <tbody>
              {filteredData.slice(0, 100).map((row, i) => (
                <tr key={i} className="border-t hover:bg-gray-50">
                  <td className="px-3 py-2">
                    <div
                      className="font-medium truncate max-w-[150px]"
                      title={row.persona_name}
                    >
                      {row.persona_name}
                    </div>
                  </td>
                  <td className="px-3 py-2">
                    <span
                      className="px-2 py-0.5 rounded text-xs font-medium"
                      style={{
                        backgroundColor:
                          (MODEL_COLORS[row.model_id] || MODEL_COLORS.default) +
                          "20",
                        color:
                          MODEL_COLORS[row.model_id] || MODEL_COLORS.default,
                      }}
                    >
                      {row.model_id}
                    </span>
                  </td>
                  <td className="px-3 py-2">
                    <span
                      className="px-2 py-0.5 rounded text-xs font-medium"
                      style={{
                        backgroundColor:
                          (PROMPT_COLORS[row.prompt_config] || "#6b7280") +
                          "20",
                        color: PROMPT_COLORS[row.prompt_config] || "#6b7280",
                      }}
                    >
                      {row.prompt_config?.toUpperCase()}
                    </span>
                  </td>
                  <td className="px-3 py-2">
                    <div className="max-w-[180px]">
                      <div
                        className="text-xs font-medium truncate"
                        title={row.email_subject}
                      >
                        {row.email_id}
                      </div>
                      <div className="text-xs text-gray-500 flex flex-wrap items-center gap-1 mt-0.5">
                        {/* Email Type */}
                        {row.email_is_phishing ? (
                          <span className="px-1 py-0.5 rounded bg-red-100 text-red-600 text-[10px]">
                            Phishing
                          </span>
                        ) : (
                          <span className="px-1 py-0.5 rounded bg-green-100 text-green-600 text-[10px]">
                            Legit
                          </span>
                        )}
                        {/* Sender Familiarity */}
                        {row.sender_familiarity === "familiar" ? (
                          <span className="px-1 py-0.5 rounded bg-blue-100 text-blue-600 text-[10px]">
                            Known
                          </span>
                        ) : (
                          <span className="px-1 py-0.5 rounded bg-gray-100 text-gray-600 text-[10px]">
                            Unknown
                          </span>
                        )}
                        {/* Urgency */}
                        {row.urgency_level === "high" && (
                          <span className="px-1 py-0.5 rounded bg-orange-100 text-orange-600 text-[10px]">
                            ⚡ Urgent
                          </span>
                        )}
                        {/* Framing */}
                        {row.framing_type === "threat" && (
                          <span className="px-1 py-0.5 rounded bg-purple-100 text-purple-600 text-[10px]">
                            Threat
                          </span>
                        )}
                        {row.framing_type === "reward" && (
                          <span className="px-1 py-0.5 rounded bg-emerald-100 text-emerald-600 text-[10px]">
                            Reward
                          </span>
                        )}
                      </div>
                    </div>
                  </td>
                  <td className="px-3 py-2 text-right">
                    <MiniBar value={row.fidelity} threshold={threshold} />
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-xs">
                    {(row.ai_click_rate * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-xs">
                    {(row.human_click_rate * 100).toFixed(1)}%
                  </td>
                  <td className="px-3 py-2 text-center">
                    <DeviationIndicator value={row.click_deviation} />
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-xs text-gray-600">
                    ${row.cost.toFixed(4)}
                  </td>
                  <td className="px-3 py-2 text-center">
                    {row.meets_threshold ? (
                      <span className="px-2 py-0.5 bg-green-100 text-green-700 rounded text-xs font-medium">
                        Pass
                      </span>
                    ) : (
                      <span className="px-2 py-0.5 bg-red-100 text-red-700 rounded text-xs font-medium">
                        Fail
                      </span>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {filteredData.length > 100 && (
          <div className="p-3 bg-gray-50 border-t text-center text-sm text-gray-500">
            Showing first 100 of {filteredData.length} combinations. Export CSV
            for full data.
          </div>
        )}
      </div>

      {/* Business Insights */}
      <div className="bg-blue-50 border border-blue-200 rounded-xl p-6">
        <h4 className="font-semibold text-blue-900 flex items-center gap-2 mb-3">
          <TrendingUp size={18} />
          Business Decision Guide
        </h4>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-blue-800">
          <div>
            <strong>High Fidelity + Low Cost:</strong> Look for combinations
            with ≥85% fidelity and lower cost per trial. These offer the best
            value for deploying to employees.
          </div>
          <div>
            <strong>Phishing vs Legitimate:</strong> Use the Email Type filter
            to compare how different LLMs distinguish between phishing and
            legitimate emails with similar patterns.
          </div>
          <div>
            <strong>Known vs Unknown Sender:</strong> Filter by Sender
            Familiarity to see how LLMs handle emails from familiar vs
            unfamiliar sources - critical for social engineering defense.
          </div>
          <div>
            <strong>Urgency & Aggression:</strong> Filter by Urgency and
            Aggression to test how LLMs respond to high-pressure, aggressive
            phishing tactics vs subtle approaches.
          </div>
          <div>
            <strong>Threat vs Reward Framing:</strong> Use the Framing filter to
            compare LLM performance on threat-based emails (account suspension)
            vs reward-based (gift cards, refunds).
          </div>
          <div>
            <strong>Prompt Configuration:</strong> CoT (Chain-of-Thought) often
            provides better fidelity at slightly higher cost. Stats provides a
            balance.
          </div>
        </div>
      </div>
    </div>
  );
};

// =============================================================================
// RATE TYPE SWITCHER COMPONENT (Clickable Click/Report/Ignore Rate Selector)
// =============================================================================

const RateTypeSwitcher = ({
  selectedRateType,
  setSelectedRateType,
  summaryStats,
  modelAggregates,
  compact = false,
}) => {
  const rateTypes = [
    {
      id: "click",
      label: "Click Rate",
      aiValue: summaryStats?.meanAiClick || 0,
      humanValue: summaryStats?.meanHumanClick || 0,
      aiColor: "text-red-500",
      tooltipColor: "text-red-400",
      getModelValue: (m) => m.mean_ai_click,
    },
    {
      id: "report",
      label: "Report Rate",
      aiValue: summaryStats?.meanAiReport || 0,
      humanValue: summaryStats?.meanHumanReport || 0,
      aiColor: "text-green-500",
      tooltipColor: "text-green-400",
      getModelValue: (m) => m.mean_ai_report,
    },
    {
      id: "ignore",
      label: "Ignore Rate",
      aiValue: summaryStats?.meanAiIgnore || 0,
      humanValue: summaryStats?.meanHumanIgnore || 0,
      aiColor: "text-amber-500",
      tooltipColor: "text-amber-400",
      getModelValue: (m) => m.mean_ai_ignore,
    },
  ];

  if (compact) {
    // Compact version for subtabs - just buttons
    return (
      <div className="bg-white rounded-xl border p-4 mb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Target size={18} className="text-purple-600" />
            <span className="font-medium text-gray-700">Fidelity Analysis For:</span>
          </div>
          <div className="flex gap-2">
            {rateTypes.map((rate) => (
              <button
                key={rate.id}
                onClick={() => setSelectedRateType(rate.id)}
                className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                  selectedRateType === rate.id
                    ? "bg-purple-600 text-white shadow-md"
                    : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                }`}
              >
                {rate.label}
              </button>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Full version with stats display
  return (
    <div className="bg-white rounded-xl border p-4">
      <div className="text-xs font-medium text-gray-400 uppercase mb-3 text-center">
        Click to view fidelity analysis for each rate type
      </div>
      <div className="grid grid-cols-3 gap-4">
        {rateTypes.map((rate, index) => (
          <button
            key={rate.id}
            onClick={() => setSelectedRateType(rate.id)}
            className={`text-center p-3 rounded-xl transition-all cursor-pointer ${
              selectedRateType === rate.id
                ? "bg-purple-100 border-2 border-purple-500 shadow-md"
                : "hover:bg-gray-50 border-2 border-transparent"
            } ${index === 1 ? "border-l border-r border-gray-200" : ""}`}
          >
            <div className={`text-xs font-medium uppercase mb-2 ${
              selectedRateType === rate.id ? "text-purple-700" : "text-gray-500"
            }`}>
              {rate.label}
              {selectedRateType === rate.id && (
                <span className="ml-2 text-[10px] bg-purple-600 text-white px-2 py-0.5 rounded-full">
                  Selected
                </span>
              )}
            </div>
            <div className="flex items-center justify-center gap-4">
              <div className="relative group">
                <div className={`text-lg font-bold ${rate.aiColor} cursor-help`}>
                  {(rate.aiValue * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-400 flex items-center justify-center gap-1">
                  AI{" "}
                  <span className="text-[10px] text-purple-400">(hover)</span>
                </div>
                {/* LLM Breakdown Tooltip */}
                {modelAggregates && modelAggregates.length > 0 && (
                  <div className="absolute left-1/2 -translate-x-1/2 top-full mt-2 hidden group-hover:block z-50 pointer-events-none">
                    <div className="bg-gray-900 text-white text-[10px] rounded-lg p-3 whitespace-nowrap shadow-xl min-w-[140px]">
                      <div className="font-semibold mb-2 text-gray-300 border-b border-gray-700 pb-1">
                        Per-LLM {rate.label}
                      </div>
                      {modelAggregates.map((m) => (
                        <div
                          key={m.model}
                          className="flex justify-between gap-3 py-0.5"
                        >
                          <span
                            className="text-gray-400 truncate max-w-[80px]"
                            title={m.model}
                          >
                            {m.model}
                          </span>
                          <span className={`font-mono ${rate.tooltipColor}`}>
                            {((rate.getModelValue(m) || 0) * 100).toFixed(1)}%
                          </span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
              <div className="text-gray-300">vs</div>
              <div>
                <div className="text-lg font-bold text-blue-500">
                  {(rate.humanValue * 100).toFixed(1)}%
                </div>
                <div className="text-xs text-gray-400">Human</div>
              </div>
            </div>
          </button>
        ))}
      </div>
    </div>
  );
};

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

const SummaryCard = ({ icon, label, value, status, sublabel }) => {
  const statusColors = {
    success: "bg-green-50 border-green-200",
    warning: "bg-amber-50 border-amber-200",
    error: "bg-red-50 border-red-200",
    neutral: "bg-gray-50 border-gray-200",
  };

  return (
    <div className={`p-4 rounded-xl border ${statusColors[status]}`}>
      <div className="flex items-center gap-2 mb-1">
        {icon}
        <span className="text-sm text-gray-600">{label}</span>
      </div>
      <div className="text-2xl font-bold">{value}</div>
      <div className="text-xs text-gray-500">{sublabel}</div>
    </div>
  );
};

const SortIcon = ({ active, direction }) => {
  if (!active) {
    return <ArrowUpDown size={14} className="text-gray-300" />;
  }
  return direction === "asc" ? (
    <ChevronUp size={14} className="text-purple-600" />
  ) : (
    <ChevronDown size={14} className="text-purple-600" />
  );
};

// FilterChip - Quick filter button
const FilterChip = ({ label, active, onClick, icon: Icon }) => (
  <button
    onClick={onClick}
    className={`flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium transition-all ${
      active
        ? "bg-purple-600 text-white shadow-sm"
        : "bg-gray-100 text-gray-600 hover:bg-gray-200"
    }`}
  >
    {Icon && <Icon size={14} />}
    {label}
  </button>
);

// DeviationIndicator - Shows AI-Human click rate difference with arrow
const DeviationIndicator = ({ value }) => {
  if (isNaN(value)) return <span className="text-gray-400">-</span>;
  const absVal = Math.abs(value * 100);
  const isHigh = absVal > 20;
  const isMedium = absVal > 10;
  const color =
    value > 0.1
      ? isHigh
        ? "text-red-600"
        : "text-red-400"
      : value < -0.1
        ? isHigh
          ? "text-blue-600"
          : "text-blue-400"
        : "text-gray-400";

  return (
    <span
      className={`flex items-center justify-center gap-0.5 font-mono text-xs ${color}`}
    >
      {value > 0.02 ? (
        <ArrowUp size={12} className={isHigh ? "animate-pulse" : ""} />
      ) : value < -0.02 ? (
        <ArrowDown size={12} className={isHigh ? "animate-pulse" : ""} />
      ) : (
        <Minus size={12} />
      )}
      {absVal.toFixed(1)}%
    </span>
  );
};

// MiniBar - Inline fidelity visualization
const MiniBar = ({ value, threshold = 0.85 }) => {
  if (isNaN(value)) return <span className="text-gray-400">N/A</span>;
  const percent = Math.min(value * 100, 100);
  const isPassing = value >= threshold;

  return (
    <div className="flex items-center gap-2">
      <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
        <div
          className={`h-full rounded-full transition-all ${
            isPassing
              ? "bg-green-500"
              : percent > 50
                ? "bg-amber-400"
                : "bg-red-400"
          }`}
          style={{ width: `${percent}%` }}
        />
      </div>
      <span
        className={`text-xs font-mono ${isPassing ? "text-green-600 font-semibold" : "text-gray-600"}`}
      >
        {percent.toFixed(1)}%
      </span>
    </div>
  );
};

// TopCombinationCard - Highlight best performing combination
const TopCombinationCard = ({
  rank,
  persona,
  model,
  prompt,
  fidelity,
  threshold,
}) => {
  const isPassing = fidelity >= threshold;
  return (
    <div
      className={`flex items-center gap-3 p-3 rounded-lg border ${
        rank === 1
          ? "bg-yellow-50 border-yellow-200"
          : "bg-gray-50 border-gray-200"
      }`}
    >
      <div
        className={`w-8 h-8 rounded-full flex items-center justify-center font-bold text-sm ${
          rank === 1
            ? "bg-yellow-400 text-white"
            : rank === 2
              ? "bg-gray-300 text-gray-700"
              : rank === 3
                ? "bg-amber-600 text-white"
                : "bg-gray-200 text-gray-600"
        }`}
      >
        {rank}
      </div>
      <div className="flex-1 min-w-0">
        <div className="text-sm font-medium truncate">{persona}</div>
        <div className="text-xs text-gray-500 truncate">
          {model} / {prompt}
        </div>
      </div>
      <div
        className={`text-lg font-bold ${isPassing ? "text-green-600" : "text-amber-600"}`}
      >
        {(fidelity * 100).toFixed(1)}%
      </div>
    </div>
  );
};

export default ResultsTab;
