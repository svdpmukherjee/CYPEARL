/**
 * Calibration Tab - Prompt Validation via Held-Out Data
 *
 * This tab allows users to validate prompt configurations before full benchmarking:
 * 1. Split behavioral data into train/test sets
 * 2. Test if LLMs can predict held-out human responses
 * 3. Use LLM self-reflection to suggest prompt improvements
 */

import { useState } from "react";
import {
  Beaker,
  Play,
  RefreshCw,
  CheckCircle,
  XCircle,
  AlertTriangle,
  TrendingUp,
  Brain,
  Lightbulb,
  ChevronDown,
  ChevronRight,
  Target,
  BarChart3,
  Sparkles,
  ArrowRight,
  ToggleLeft,
  ToggleRight,
  BookOpen,
  Zap,
  Info,
  StopCircle,
} from "lucide-react";

// Card component
const Card = ({ children, className = "" }) => (
  <div
    className={`bg-white rounded-lg shadow border border-gray-200 p-4 ${className}`}
  >
    {children}
  </div>
);

// Progress bar component
const ProgressBar = ({ value, max = 100, color = "blue" }) => {
  const percentage = Math.min((value / max) * 100, 100);
  const colorClasses = {
    blue: "bg-blue-500",
    green: "bg-green-500",
    red: "bg-red-500",
    yellow: "bg-yellow-500",
    purple: "bg-purple-500",
  };
  return (
    <div className="w-full bg-gray-200 rounded-full h-2.5">
      <div
        className={`h-2.5 rounded-full transition-all duration-300 ${colorClasses[color]}`}
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
};

// Accuracy badge component
const AccuracyBadge = ({ accuracy, threshold = 0.8 }) => {
  const percentage = (accuracy * 100).toFixed(1);
  const meetsThreshold = accuracy >= threshold;

  return (
    <div
      className={`flex items-center gap-2 px-3 py-1 rounded-full ${
        meetsThreshold
          ? "bg-green-100 text-green-700"
          : "bg-red-100 text-red-700"
      }`}
    >
      {meetsThreshold ? <CheckCircle size={16} /> : <XCircle size={16} />}
      <span className="font-medium">{percentage}%</span>
    </div>
  );
};

export const CalibrationTab = ({
  personas,
  models,
  emails,
  apiBase = "/api/phase2",
}) => {
  // State
  const [selectedPersona, setSelectedPersona] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [selectedConfig, setSelectedConfig] = useState("stats");
  const [splitRatio, setSplitRatio] = useState(0.8);
  const [useICL, setUseICL] = useState(true); // ICL toggle - enabled by default
  const [testSampleSize, setTestSampleSize] = useState(null); // NEW: Quick test sample size (null = use all)
  const [useQuickTest, setUseQuickTest] = useState(false); // NEW: Toggle for quick test mode

  const [loading, setLoading] = useState(false);
  const [isCalibrationRunning, setIsCalibrationRunning] = useState(false);
  const [calibrationResult, setCalibrationResult] = useState(null);
  const [reflectionResult, setReflectionResult] = useState(null);
  const [configComparison, setConfigComparison] = useState(null);
  const [stoppedEarly, setStoppedEarly] = useState(false);
  const [autoCalibrationResult, setAutoCalibrationResult] = useState(null); // NEW: Auto-calibration result
  const [isAutoCalibrating, setIsAutoCalibrating] = useState(false); // NEW: Auto-calibration running

  const [expandedSections, setExpandedSections] = useState({
    split: true,
    calibration: true,
    reflection: false,
    suggestions: false,
  });

  const [error, setError] = useState(null);

  const toggleSection = (section) => {
    setExpandedSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  // API calls
  const splitData = async () => {
    if (!selectedPersona) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${apiBase}/calibration/split?persona_id=${selectedPersona}&split_ratio=${splitRatio}`,
        { method: "POST" },
      );
      const data = await response.json();
      if (response.ok) {
        setSplitResult(data);
      } else {
        setError(data.detail || "Split failed");
      }
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  const runCalibration = async () => {
    if (!selectedPersona || !selectedModel) return;
    setLoading(true);
    setIsCalibrationRunning(true);
    setError(null);
    setReflectionResult(null);
    setStoppedEarly(false);
    setAutoCalibrationResult(null);
    try {
      const requestBody = {
        persona_id: selectedPersona,
        model_id: selectedModel,
        prompt_config: selectedConfig,
        split_ratio: splitRatio,
        use_icl: useICL,
      };
      // NEW: Add test_sample_size if quick test mode is enabled
      if (useQuickTest && testSampleSize) {
        requestBody.test_sample_size = testSampleSize;
      }

      const response = await fetch(`${apiBase}/calibration/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestBody),
      });
      const data = await response.json();
      if (response.ok) {
        setCalibrationResult(data);
        if (data.stopped_early) {
          setStoppedEarly(true);
        }
      } else {
        setError(data.detail || "Calibration failed");
      }
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
    setIsCalibrationRunning(false);
  };

  const stopCalibration = async () => {
    try {
      const response = await fetch(`${apiBase}/calibration/stop`, {
        method: "POST",
      });
      const data = await response.json();
      if (response.ok) {
        console.log("Stop requested:", data.message);
      } else {
        setError(data.detail || "Failed to stop calibration");
      }
    } catch (err) {
      setError(err.message);
    }
  };

  const runReflection = async () => {
    if (!calibrationResult?.calibration_key) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(`${apiBase}/calibration/reflect`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          calibration_key: calibrationResult.calibration_key,
          reflection_model: selectedModel.includes("claude")
            ? selectedModel
            : "claude-3-5-sonnet",
        }),
      });
      const data = await response.json();
      if (response.ok) {
        setReflectionResult(data);
        setExpandedSections((prev) => ({
          ...prev,
          reflection: true,
          suggestions: true,
        }));
      } else {
        setError(data.detail || "Reflection failed");
      }
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  // NEW: Auto-apply suggestions and rerun calibration
  const applyAndRerun = async (maxIterations = 3) => {
    if (!calibrationResult?.calibration_key || !reflectionResult) return;
    setIsAutoCalibrating(true);
    setError(null);
    setAutoCalibrationResult(null);
    try {
      const response = await fetch(`${apiBase}/calibration/apply-suggestions`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          calibration_key: calibrationResult.calibration_key,
          suggestion_indices: [], // Apply all suggestions
          auto_rerun: true,
          max_iterations: maxIterations,
        }),
      });
      const data = await response.json();
      if (response.ok) {
        setAutoCalibrationResult(data);
        // Update main calibration result with final result if available
        if (data.final_result) {
          setCalibrationResult(data.final_result);
        }
      } else {
        setError(data.detail || "Auto-calibration failed");
      }
    } catch (err) {
      setError(err.message);
    }
    setIsAutoCalibrating(false);
  };

  const compareConfigs = async () => {
    if (!selectedPersona || !selectedModel) return;
    setLoading(true);
    setError(null);
    try {
      const response = await fetch(
        `${apiBase}/calibration/compare-configs?persona_id=${selectedPersona}&model_id=${selectedModel}&use_icl=${useICL}`,
        { method: "POST" },
      );
      const data = await response.json();
      if (response.ok) {
        setConfigComparison(data);
      } else {
        setError(data.detail || "Comparison failed");
      }
    } catch (err) {
      setError(err.message);
    }
    setLoading(false);
  };

  // No personas or emails check
  if (!personas?.length) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <Beaker size={48} className="text-gray-400 mb-4" />
        <h3 className="text-lg font-medium text-gray-900">
          No Personas Loaded
        </h3>
        <p className="text-gray-500 mt-1">
          Import personas from Phase 1 to start calibration.
        </p>
      </div>
    );
  }

  if (!emails?.length) {
    return (
      <div className="flex flex-col items-center justify-center py-12">
        <AlertTriangle size={48} className="text-amber-400 mb-4" />
        <h3 className="text-lg font-medium text-gray-900">No Emails Loaded</h3>
        <p className="text-gray-500 mt-1">
          Import email stimuli to run calibration trials.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-gradient-to-r from-purple-50 to-blue-50 border border-purple-200 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Beaker className="text-purple-600 mt-1" size={24} />
          <div>
            <h3 className="text-lg font-semibold text-purple-900">
              Prompt Calibration
            </h3>
            <p className="text-sm text-purple-700 mt-1">
              Validate prompt configurations before full benchmarking. The
              system splits behavioral data into train/test sets, tests LLM
              predictions against held-out human responses, and uses
              self-reflection to suggest improvements when accuracy is below
              threshold.
            </p>
          </div>
        </div>
      </div>

      {/* Error display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4 flex items-center gap-2">
          <XCircle className="text-red-500" size={20} />
          <span className="text-red-700">{error}</span>
          <button
            onClick={() => setError(null)}
            className="ml-auto text-red-500 hover:text-red-700"
          >
            Dismiss
          </button>
        </div>
      )}

      {/* Configuration */}
      <Card>
        <h4 className="font-medium text-gray-900 mb-4 flex items-center gap-2">
          <Target size={18} />
          Configuration
        </h4>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          {/* Persona selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Persona
            </label>
            <select
              value={selectedPersona}
              onChange={(e) => {
                setSelectedPersona(e.target.value);
                setSplitResult(null);
                setCalibrationResult(null);
                setReflectionResult(null);
              }}
              className="w-full border rounded-lg px-3 py-2 text-sm"
            >
              <option value="">Select persona...</option>
              {personas.map((p) => (
                <option key={p.persona_id} value={p.persona_id}>
                  {p.name} (
                  {(
                    (p.behavioral_statistics?.phishing_click_rate ||
                      p.behavioral_targets?.phishing_click_rate ||
                      0) * 100
                  ).toFixed(0)}
                  % click rate)
                </option>
              ))}
            </select>
          </div>

          {/* Model selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => {
                setSelectedModel(e.target.value);
                setCalibrationResult(null);
                setReflectionResult(null);
              }}
              className="w-full border rounded-lg px-3 py-2 text-sm"
            >
              <option value="">Select model...</option>
              {models.map((m) => (
                <option key={m.model_id} value={m.model_id}>
                  {m.display_name}
                </option>
              ))}
            </select>
          </div>

          {/* Prompt config */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Prompt Config
            </label>
            <select
              value={selectedConfig}
              onChange={(e) => {
                setSelectedConfig(e.target.value);
                setCalibrationResult(null);
                setReflectionResult(null);
              }}
              className="w-full border rounded-lg px-3 py-2 text-sm"
            >
              <option value="baseline">Baseline (Task-only)</option>
              <option value="stats">+ Behavioral Stats</option>
              <option value="cot">+ Chain-of-Thought</option>
            </select>
          </div>

          {/* Split ratio */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Train/Test Split: {(splitRatio * 100).toFixed(0)}% /{" "}
              {((1 - splitRatio) * 100).toFixed(0)}%
            </label>
            <input
              type="range"
              min="0.5"
              max="0.9"
              step="0.1"
              value={splitRatio}
              onChange={(e) => setSplitRatio(parseFloat(e.target.value))}
              className="w-full"
            />
          </div>
        </div>

        {/* NEW: Quick Test Mode Section */}
        <div className="mt-4 p-4 bg-gradient-to-r from-amber-50 to-orange-50 rounded-lg border border-amber-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Zap className="text-amber-600" size={20} />
              <div>
                <div className="font-medium text-gray-900 flex items-center gap-2">
                  Quick Test Mode
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full ${useQuickTest ? "bg-amber-100 text-amber-700" : "bg-gray-100 text-gray-600"}`}
                  >
                    {useQuickTest
                      ? `${testSampleSize || 50} trials`
                      : "Full test set"}
                  </span>
                </div>
                <p className="text-xs text-gray-600 mt-0.5">
                  Run fewer test trials to quickly evaluate LLM accuracy before
                  full calibration
                </p>
              </div>
            </div>
            <button
              onClick={() => {
                setUseQuickTest(!useQuickTest);
                if (!useQuickTest && !testSampleSize) {
                  setTestSampleSize(50); // Default to 50
                }
                setCalibrationResult(null);
                setReflectionResult(null);
              }}
              className={`p-2 rounded-lg transition-all ${
                useQuickTest
                  ? "bg-amber-600 text-white hover:bg-amber-700"
                  : "bg-gray-200 text-gray-600 hover:bg-gray-300"
              }`}
              title={useQuickTest ? "Use full test set" : "Enable quick test"}
            >
              {useQuickTest ? (
                <ToggleRight size={24} />
              ) : (
                <ToggleLeft size={24} />
              )}
            </button>
          </div>

          {/* Sample size slider - shown when quick test is enabled */}
          {useQuickTest && (
            <div className="mt-3 p-3 bg-white/70 rounded-lg border border-amber-100">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Test Sample Size: {testSampleSize} trials
              </label>
              <input
                type="range"
                min="10"
                max="200"
                step="10"
                value={testSampleSize || 50}
                onChange={(e) => setTestSampleSize(parseInt(e.target.value))}
                className="w-full accent-amber-600"
              />
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>10 (fastest)</span>
                <span>50</span>
                <span>100</span>
                <span>200 (more accurate)</span>
              </div>
              <div className="mt-2 text-xs text-amber-700 flex items-center gap-1">
                <Info size={12} />
                <span>
                  Quick test randomly samples from the test set. Use full set
                  for final validation.
                </span>
              </div>
            </div>
          )}
        </div>

        {/* ICL Toggle Section */}
        <div className="mt-4 p-4 bg-gradient-to-r from-indigo-50 to-purple-50 rounded-lg border border-indigo-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <BookOpen className="text-indigo-600" size={20} />
              <div>
                <div className="font-medium text-gray-900 flex items-center gap-2">
                  In-Context Learning (ICL)
                  <span
                    className={`text-xs px-2 py-0.5 rounded-full ${useICL ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-600"}`}
                  >
                    {useICL ? "Enabled" : "Disabled"}
                  </span>
                </div>
                <p className="text-xs text-gray-600 mt-0.5">
                  Include training examples in prompts to improve LLM behavioral
                  fidelity
                </p>
              </div>
            </div>
            <button
              onClick={() => {
                setUseICL(!useICL);
                setCalibrationResult(null);
                setReflectionResult(null);
                setConfigComparison(null);
              }}
              className={`p-2 rounded-lg transition-all ${
                useICL
                  ? "bg-indigo-600 text-white hover:bg-indigo-700"
                  : "bg-gray-200 text-gray-600 hover:bg-gray-300"
              }`}
              title={useICL ? "Disable ICL" : "Enable ICL"}
            >
              {useICL ? <ToggleRight size={24} /> : <ToggleLeft size={24} />}
            </button>
          </div>

          {/* ICL Info Box - shown when ICL is enabled */}
          {useICL && (
            <div className="mt-3 p-3 bg-white/70 rounded-lg border border-indigo-100">
              <div className="flex items-start gap-2">
                <Info
                  size={14}
                  className="text-indigo-500 mt-0.5 flex-shrink-0"
                />
                <div className="text-xs text-gray-600">
                  <strong className="text-gray-700">
                    Incremental prompting with ICL:
                  </strong>
                  <ul className="mt-1 ml-3 list-disc space-y-0.5">
                    <li>
                      <span className="font-medium">Baseline:</span> 29 traits +
                      6 minimal ICL examples (action only)
                    </li>
                    <li>
                      <span className="font-medium">BASELINE + Stats:</span> + 8
                      behavioral outcomes + ICL with reasoning
                    </li>
                    <li>
                      <span className="font-medium">STATS + CoT:</span> + actual
                      participant reasoning chains
                    </li>
                  </ul>
                </div>
              </div>
            </div>
          )}
        </div>

        {/* Action buttons */}
        <div className="flex gap-3 mt-4">
          {isCalibrationRunning ? (
            <button
              onClick={stopCalibration}
              className="px-4 py-2 bg-red-600 text-white rounded-lg text-sm flex items-center gap-2 hover:bg-red-700"
            >
              <StopCircle size={16} />
              Stop Calibration
            </button>
          ) : (
            <button
              onClick={runCalibration}
              disabled={!selectedPersona || !selectedModel || loading}
              className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm flex items-center gap-2 hover:bg-purple-700 disabled:opacity-50"
            >
              {loading ? (
                <RefreshCw size={16} className="animate-spin" />
              ) : (
                <Play size={16} />
              )}
              Run Calibration
            </button>
          )}

          <button
            onClick={compareConfigs}
            disabled={
              !selectedPersona ||
              !selectedModel ||
              loading ||
              isCalibrationRunning
            }
            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm flex items-center gap-2 hover:bg-blue-700 disabled:opacity-50"
          >
            <BarChart3 size={16} />
            Compare All Configs
          </button>
        </div>
      </Card>

      {/* Calibration Result */}
      {calibrationResult && (
        <Card>
          <div
            className="flex items-center justify-between cursor-pointer"
            onClick={() => toggleSection("calibration")}
          >
            <h4 className="font-medium text-gray-900 flex items-center gap-2">
              <TrendingUp size={18} />
              Calibration Results
              {stoppedEarly && (
                <span className="text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full flex items-center gap-1">
                  <StopCircle size={10} />
                  Stopped Early
                </span>
              )}
              <AccuracyBadge accuracy={calibrationResult.accuracy} />
              {/* Quick test indicator */}
              {calibrationResult.is_quick_test && (
                <span className="text-xs bg-amber-100 text-amber-700 px-2 py-0.5 rounded-full flex items-center gap-1">
                  <Zap size={10} />
                  Quick ({calibrationResult.n_trials}/
                  {calibrationResult.full_test_count})
                </span>
              )}
              {calibrationResult.use_icl !== undefined &&
                (calibrationResult.use_icl ? (
                  <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-0.5 rounded-full flex items-center gap-1">
                    <BookOpen size={10} />
                    ICL
                  </span>
                ) : (
                  <span className="text-xs bg-gray-100 text-gray-500 px-2 py-0.5 rounded-full">
                    No ICL
                  </span>
                ))}
            </h4>
            {expandedSections.calibration ? (
              <ChevronDown size={20} />
            ) : (
              <ChevronRight size={20} />
            )}
          </div>

          {expandedSections.calibration && (
            <div className="mt-4 space-y-4">
              {/* Quick test warning */}
              {calibrationResult.is_quick_test && (
                <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 flex items-center gap-2">
                  <Zap className="text-amber-600" size={16} />
                  <span className="text-sm text-amber-800">
                    Quick test mode: {calibrationResult.n_trials} of{" "}
                    {calibrationResult.full_test_count} trials. Run full test
                    set for accurate calibration results.
                  </span>
                </div>
              )}

              {/* Summary stats */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-sm text-gray-600">Accuracy</div>
                  <div className="text-2xl font-bold text-gray-900">
                    {(calibrationResult.accuracy * 100).toFixed(1)}%
                  </div>
                  <ProgressBar
                    value={calibrationResult.accuracy * 100}
                    color={calibrationResult.meets_threshold ? "green" : "red"}
                  />
                </div>

                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-sm text-gray-600">Trials</div>
                  <div className="text-2xl font-bold text-gray-900">
                    {calibrationResult.n_correct}/{calibrationResult.n_trials}
                  </div>
                  <div className="text-xs text-gray-500">
                    correct predictions
                    {calibrationResult.is_quick_test &&
                      ` (of ${calibrationResult.full_test_count} total)`}
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-sm text-gray-600">Human Click Rate</div>
                  <div className="text-2xl font-bold text-gray-900">
                    {(calibrationResult.human_click_rate * 100).toFixed(1)}%
                  </div>
                </div>

                <div className="bg-gray-50 rounded-lg p-3">
                  <div className="text-sm text-gray-600">LLM Click Rate</div>
                  <div className="text-2xl font-bold text-gray-900">
                    {(calibrationResult.llm_click_rate * 100).toFixed(1)}%
                  </div>
                  <div
                    className={`text-xs ${calibrationResult.click_rate_error > 0.1 ? "text-red-500" : "text-green-500"}`}
                  >
                    Error:{" "}
                    {(calibrationResult.click_rate_error * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Failure summary */}
              {calibrationResult.failure_summary &&
                !calibrationResult.failure_summary.no_failures && (
                  <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                    <div className="flex items-center gap-2 text-amber-800 font-medium mb-2">
                      <AlertTriangle size={18} />
                      Failure Analysis
                    </div>
                    <div className="text-sm text-amber-700 space-y-1">
                      <div>
                        Total failures:{" "}
                        {calibrationResult.failure_summary.total_failures}
                      </div>
                      <div>
                        Clicks missed:{" "}
                        {calibrationResult.failure_summary.click_as_other}{" "}
                        (human clicked, LLM didn't)
                      </div>
                      <div>
                        False clicks:{" "}
                        {calibrationResult.failure_summary.other_as_click} (LLM
                        clicked, human didn't)
                      </div>
                    </div>
                  </div>
                )}

              {/* Reflection button */}
              {!calibrationResult.meets_threshold && (
                <div className="flex items-center gap-3">
                  <button
                    onClick={runReflection}
                    disabled={loading}
                    className="px-4 py-2 bg-amber-600 text-white rounded-lg text-sm flex items-center gap-2 hover:bg-amber-700 disabled:opacity-50"
                  >
                    {loading ? (
                      <RefreshCw size={16} className="animate-spin" />
                    ) : (
                      <Brain size={16} />
                    )}
                    Analyze Failures & Get Suggestions
                  </button>
                  <span className="text-sm text-gray-500">
                    LLM will analyze failures and suggest prompt improvements
                  </span>
                </div>
              )}
            </div>
          )}
        </Card>
      )}

      {/* Reflection Results */}
      {reflectionResult && (
        <Card>
          <div
            className="flex items-center justify-between cursor-pointer"
            onClick={() => toggleSection("reflection")}
          >
            <h4 className="font-medium text-gray-900 flex items-center gap-2">
              <Brain size={18} />
              Self-Reflection Analysis
            </h4>
            {expandedSections.reflection ? (
              <ChevronDown size={20} />
            ) : (
              <ChevronRight size={20} />
            )}
          </div>

          {expandedSections.reflection && (
            <div className="mt-4 space-y-4">
              {/* Root Cause */}
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                <div className="font-medium text-blue-800 mb-2">
                  Root Cause Analysis
                </div>
                <p className="text-sm text-blue-700 whitespace-pre-wrap">
                  {reflectionResult.root_cause_analysis}
                </p>
              </div>

              {/* Suggestions */}
              {reflectionResult.suggestions?.length > 0 && (
                <div>
                  <div
                    className="flex items-center justify-between cursor-pointer mb-2"
                    onClick={() => toggleSection("suggestions")}
                  >
                    <h5 className="font-medium text-gray-800 flex items-center gap-2">
                      <Lightbulb size={16} />
                      Prompt Improvement Suggestions (
                      {reflectionResult.suggestions.length})
                    </h5>
                    {expandedSections.suggestions ? (
                      <ChevronDown size={16} />
                    ) : (
                      <ChevronRight size={16} />
                    )}
                  </div>

                  {expandedSections.suggestions && (
                    <div className="space-y-3">
                      {reflectionResult.suggestions.map((suggestion, idx) => (
                        <div key={idx} className="border rounded-lg p-3">
                          <div className="flex items-center justify-between mb-2">
                            <span
                              className={`text-xs px-2 py-0.5 rounded ${
                                suggestion.category === "behavioral_stats"
                                  ? "bg-purple-100 text-purple-700"
                                  : suggestion.category === "framing"
                                    ? "bg-blue-100 text-blue-700"
                                    : suggestion.category ===
                                        "reasoning_examples"
                                      ? "bg-green-100 text-green-700"
                                      : "bg-gray-100 text-gray-700"
                              }`}
                            >
                              {suggestion.category}
                            </span>
                            <span
                              className={`text-xs ${
                                suggestion.confidence === "high"
                                  ? "text-green-600"
                                  : suggestion.confidence === "medium"
                                    ? "text-yellow-600"
                                    : "text-gray-600"
                              }`}
                            >
                              {suggestion.confidence} confidence
                            </span>
                          </div>
                          <div className="text-sm">
                            <div className="text-red-700 mb-1">
                              <strong>Issue:</strong> {suggestion.issue}
                            </div>
                            <div className="text-green-700">
                              <strong>Change:</strong> {suggestion.change}
                            </div>
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              )}

              {/* Improved Prompt */}
              {reflectionResult.improved_prompt && (
                <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                  <div className="font-medium text-green-800 mb-2 flex items-center gap-2">
                    <Sparkles size={16} />
                    Suggested Improved Prompt Section
                  </div>
                  <pre className="text-xs bg-green-100 rounded p-3 overflow-auto max-h-48 text-green-900">
                    {reflectionResult.improved_prompt}
                  </pre>
                </div>
              )}

              {/* NEW: Auto-Apply & Rerun Button */}
              {reflectionResult.suggestions?.length > 0 && (
                <div className="mt-4 p-4 bg-gradient-to-r from-emerald-50 to-teal-50 border border-emerald-200 rounded-lg">
                  <div className="flex items-center justify-between">
                    <div>
                      <div className="font-medium text-emerald-800 flex items-center gap-2">
                        <RefreshCw size={16} />
                        Auto-Calibration
                      </div>
                      <p className="text-xs text-emerald-700 mt-1">
                        Apply suggestions and rerun calibration automatically
                        until 80% accuracy is achieved
                      </p>
                    </div>
                    <button
                      onClick={() => applyAndRerun(3)}
                      disabled={isAutoCalibrating || loading}
                      className="px-4 py-2 bg-emerald-600 text-white rounded-lg text-sm flex items-center gap-2 hover:bg-emerald-700 disabled:opacity-50"
                    >
                      {isAutoCalibrating ? (
                        <>
                          <RefreshCw size={16} className="animate-spin" />
                          Auto-Calibrating...
                        </>
                      ) : (
                        <>
                          <Sparkles size={16} />
                          Apply & Rerun (up to 3x)
                        </>
                      )}
                    </button>
                  </div>

                  {/* Auto-calibration progress/result */}
                  {autoCalibrationResult && (
                    <div className="mt-3 p-3 bg-white/80 rounded-lg">
                      <div className="flex items-center justify-between mb-2">
                        <span className="font-medium text-gray-800">
                          Auto-Calibration Result
                        </span>
                        {autoCalibrationResult.meets_threshold ? (
                          <span className="text-xs bg-green-100 text-green-700 px-2 py-1 rounded-full flex items-center gap-1">
                            <CheckCircle size={12} />
                            Success!
                          </span>
                        ) : (
                          <span className="text-xs bg-amber-100 text-amber-700 px-2 py-1 rounded-full">
                            Threshold not met
                          </span>
                        )}
                      </div>
                      <div className="grid grid-cols-3 gap-3 text-sm">
                        <div>
                          <div className="text-gray-500 text-xs">Initial</div>
                          <div className="font-medium">
                            {(
                              autoCalibrationResult.initial_accuracy * 100
                            ).toFixed(1)}
                            %
                          </div>
                        </div>
                        <div>
                          <div className="text-gray-500 text-xs">Final</div>
                          <div className="font-medium">
                            {(
                              autoCalibrationResult.final_accuracy * 100
                            ).toFixed(1)}
                            %
                          </div>
                        </div>
                        <div>
                          <div className="text-gray-500 text-xs">
                            Improvement
                          </div>
                          <div
                            className={`font-medium ${autoCalibrationResult.improvement > 0 ? "text-green-600" : "text-red-600"}`}
                          >
                            {autoCalibrationResult.improvement > 0 ? "+" : ""}
                            {(autoCalibrationResult.improvement * 100).toFixed(
                              1,
                            )}
                            %
                          </div>
                        </div>
                      </div>
                      {autoCalibrationResult.iteration_history && (
                        <div className="mt-2 text-xs text-gray-600">
                          Iterations: {autoCalibrationResult.iterations_run} of{" "}
                          {autoCalibrationResult.max_iterations}
                        </div>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          )}
        </Card>
      )}

      {/* Config Comparison Results */}
      {configComparison && (
        <Card>
          <div className="flex items-center justify-between mb-4">
            <h4 className="font-medium text-gray-900 flex items-center gap-2">
              <BarChart3 size={18} />
              Prompt Configuration Comparison
            </h4>
            <div className="flex items-center gap-2">
              {useICL ? (
                <span className="text-xs bg-indigo-100 text-indigo-700 px-2 py-1 rounded-full flex items-center gap-1">
                  <BookOpen size={12} />
                  ICL Enabled
                </span>
              ) : (
                <span className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full">
                  ICL Disabled
                </span>
              )}
            </div>
          </div>

          <div className="grid grid-cols-3 gap-4 mb-4">
            {["baseline", "stats", "cot"].map((config) => {
              const result = configComparison.results[config];
              const isBest = config === configComparison.best_config;

              return (
                <div
                  key={config}
                  className={`rounded-lg p-4 border-2 transition-all ${
                    isBest
                      ? "border-green-500 bg-green-50 shadow-md"
                      : "border-gray-200 bg-gray-50 hover:border-gray-300"
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-medium text-gray-900">
                      {config === "baseline"
                        ? "Baseline"
                        : config === "stats"
                          ? "BASELINE + Stats"
                          : "STATS + CoT"}
                    </span>
                    {isBest && (
                      <span className="text-xs bg-green-500 text-white px-2 py-0.5 rounded flex items-center gap-1">
                        <Zap size={10} />
                        BEST
                      </span>
                    )}
                  </div>

                  <div className="text-3xl font-bold mb-2">
                    {(result.accuracy * 100).toFixed(1)}%
                  </div>

                  <ProgressBar
                    value={result.accuracy * 100}
                    color={result.meets_threshold ? "green" : "red"}
                  />

                  <div className="mt-2 text-xs text-gray-600 space-y-1">
                    <div className="flex justify-between">
                      <span>Click error:</span>
                      <span
                        className={
                          result.click_rate_error > 0.15
                            ? "text-red-500"
                            : "text-green-600"
                        }
                      >
                        {(result.click_rate_error * 100).toFixed(1)}%
                      </span>
                    </div>
                    <div className="flex justify-between">
                      <span>Trials:</span>
                      <span>{result.n_trials}</span>
                    </div>
                    {result.n_correct !== undefined && (
                      <div className="flex justify-between">
                        <span>Correct:</span>
                        <span>
                          {result.n_correct}/{result.n_trials}
                        </span>
                      </div>
                    )}
                  </div>

                  {/* ICL indicator per config */}
                  {useICL && (
                    <div className="mt-3 pt-2 border-t border-gray-200">
                      <div className="text-xs text-indigo-600 flex items-center gap-1">
                        <BookOpen size={10} />
                        <span>
                          {config === "baseline"
                            ? "6 examples (minimal)"
                            : config === "stats"
                              ? "6 examples + patterns"
                              : "6 examples (full CoT)"}
                        </span>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>

          {/* Comparison Summary Table */}
          <div className="mb-4 overflow-hidden rounded-lg border border-gray-200">
            <table className="min-w-full text-sm">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-2 text-left font-medium text-gray-700">
                    Metric
                  </th>
                  <th className="px-4 py-2 text-center font-medium text-gray-700">
                    Baseline
                  </th>
                  <th className="px-4 py-2 text-center font-medium text-gray-700">
                    + Stats
                  </th>
                  <th className="px-4 py-2 text-center font-medium text-gray-700">
                    + CoT
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200">
                <tr>
                  <td className="px-4 py-2 text-gray-600">Accuracy</td>
                  {["baseline", "stats", "cot"].map((config) => {
                    const result = configComparison.results[config];
                    const isBest = config === configComparison.best_config;
                    return (
                      <td
                        key={config}
                        className={`px-4 py-2 text-center ${isBest ? "font-bold text-green-600" : ""}`}
                      >
                        {(result.accuracy * 100).toFixed(1)}%
                      </td>
                    );
                  })}
                </tr>
                <tr className="bg-gray-50">
                  <td className="px-4 py-2 text-gray-600">Meets Threshold</td>
                  {["baseline", "stats", "cot"].map((config) => {
                    const result = configComparison.results[config];
                    return (
                      <td key={config} className="px-4 py-2 text-center">
                        {result.meets_threshold ? (
                          <CheckCircle
                            size={16}
                            className="inline text-green-500"
                          />
                        ) : (
                          <XCircle size={16} className="inline text-red-500" />
                        )}
                      </td>
                    );
                  })}
                </tr>
                <tr>
                  <td className="px-4 py-2 text-gray-600">Click Rate Error</td>
                  {["baseline", "stats", "cot"].map((config) => {
                    const result = configComparison.results[config];
                    const isLow = result.click_rate_error < 0.1;
                    return (
                      <td
                        key={config}
                        className={`px-4 py-2 text-center ${isLow ? "text-green-600" : "text-red-600"}`}
                      >
                        {(result.click_rate_error * 100).toFixed(1)}%
                      </td>
                    );
                  })}
                </tr>
              </tbody>
            </table>
          </div>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
            <div className="flex items-center gap-2 text-blue-800">
              <ArrowRight size={16} />
              <span className="font-medium">Recommendation:</span>
              <span>{configComparison.recommendation}</span>
            </div>
          </div>
        </Card>
      )}

      {/* Workflow Guide */}
      <Card className="bg-gray-50">
        <h4 className="font-medium text-gray-900 mb-3">Calibration Workflow</h4>
        <div className="flex flex-wrap items-center gap-2 text-sm text-gray-600">
          <div className="flex items-center gap-1">
            <span className="w-6 h-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs">
              1
            </span>
            <span>Select persona & model</span>
          </div>
          <ArrowRight size={16} className="text-gray-400" />
          <div className="flex items-center gap-1">
            <span className="w-6 h-6 bg-indigo-500 text-white rounded-full flex items-center justify-center text-xs">
              2
            </span>
            <span>Enable/disable ICL</span>
          </div>
          <ArrowRight size={16} className="text-gray-400" />
          <div className="flex items-center gap-1">
            <span className="w-6 h-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs">
              3
            </span>
            <span>Run calibration</span>
          </div>
          <ArrowRight size={16} className="text-gray-400" />
          <div className="flex items-center gap-1">
            <span className="w-6 h-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs">
              4
            </span>
            <span>If &lt;80%, get suggestions</span>
          </div>
          <ArrowRight size={16} className="text-gray-400" />
          <div className="flex items-center gap-1">
            <span className="w-6 h-6 bg-purple-500 text-white rounded-full flex items-center justify-center text-xs">
              5
            </span>
            <span>Compare all configs</span>
          </div>
        </div>
        <div className="mt-3 pt-3 border-t border-gray-200">
          <div className="flex items-start gap-2 text-xs text-gray-500">
            <Lightbulb
              size={14}
              className="text-amber-500 mt-0.5 flex-shrink-0"
            />
            <span>
              <strong>Tip:</strong> Enable ICL (In-Context Learning) to include
              real training examples in prompts. This helps the LLM better mimic
              the persona's behavioral patterns by learning from actual
              decisions.
            </span>
          </div>
        </div>
      </Card>
    </div>
  );
};

export default CalibrationTab;
