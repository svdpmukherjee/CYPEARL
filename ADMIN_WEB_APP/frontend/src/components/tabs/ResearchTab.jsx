/**
 * CYPEARL Phase 2 - Research Tab
 *
 * Advanced research features for Phase 2:
 * 1. Prompt Ablation Study - Factorial design to find optimal prompt components
 * 2. Uncertainty Quantification - Multi-sample analysis for LLM confidence
 * 3. Behavioral Signature Matching - Beyond action matching to causal factors
 * 4. Persona Embeddings - Continuous embedding space for personas
 */

import React, { useState, useEffect } from "react";
import {
  Play,
  CheckCircle,
  XCircle,
  AlertTriangle,
  RefreshCw,
  Brain,
  Target,
  Activity,
  BarChart3,
  Layers,
  GitBranch,
  Zap,
  TrendingUp,
  ChevronDown,
  ChevronRight,
  Info,
} from "lucide-react";
import * as api from "../../services/phase2Api";

// =============================================================================
// ABLATION STUDY SECTION
// =============================================================================

const AblationStudySection = ({ personas, models }) => {
  const [selectedPersona, setSelectedPersona] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [testSampleSize, setTestSampleSize] = useState(20);
  const [includeIcl, setIncludeIcl] = useState(true);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(true);

  const runStudy = async () => {
    if (!selectedPersona || !selectedModel) {
      setError("Please select a persona and model");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await api.runAblationStudy({
        persona_id: selectedPersona,
        model_id: selectedModel,
        include_icl: includeIcl,
        test_sample_size: testSampleSize || null,
      });
      setResult(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 mb-6">
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          <Target className="w-6 h-6 text-purple-400" />
          <div>
            <h3 className="text-lg font-semibold text-white">
              Prompt Ablation Study
            </h3>
            <p className="text-sm text-gray-400">
              Test which prompt components actually matter (Traits, Stats, CoT)
            </p>
          </div>
        </div>
        {expanded ? (
          <ChevronDown className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronRight className="w-5 h-5 text-gray-400" />
        )}
      </div>

      {expanded && (
        <div className="mt-4 space-y-4">
          {/* Configuration */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Persona</label>
              <select
                value={selectedPersona}
                onChange={(e) => setSelectedPersona(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              >
                <option value="">Select...</option>
                {personas.map((p) => (
                  <option key={p.persona_id} value={p.persona_id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              >
                <option value="">Select...</option>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Test Trials (quick)
              </label>
              <input
                type="number"
                value={testSampleSize}
                onChange={(e) => setTestSampleSize(parseInt(e.target.value) || 0)}
                placeholder="All"
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              />
            </div>

            <div className="flex items-end">
              <button
                onClick={runStudy}
                disabled={loading || !selectedPersona || !selectedModel}
                className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded text-white flex items-center justify-center gap-2"
              >
                {loading ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
                Run Study
              </button>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <input
              type="checkbox"
              id="includeIcl"
              checked={includeIcl}
              onChange={(e) => setIncludeIcl(e.target.checked)}
              className="rounded"
            />
            <label htmlFor="includeIcl" className="text-sm text-gray-300">
              Include In-Context Learning examples
            </label>
          </div>

          {/* Error */}
          {error && (
            <div className="p-3 bg-red-900/30 border border-red-700 rounded text-red-300 text-sm">
              {error}
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="mt-4 space-y-4">
              {/* Summary */}
              <div className="grid grid-cols-3 gap-4">
                <div className="bg-gray-700/50 rounded p-4">
                  <div className="text-sm text-gray-400">Best Config</div>
                  <div className="text-xl font-bold text-green-400">
                    {result.best_config || "N/A"}
                  </div>
                </div>
                <div className="bg-gray-700/50 rounded p-4">
                  <div className="text-sm text-gray-400">Best Accuracy</div>
                  <div className="text-xl font-bold text-white">
                    {(result.best_accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-gray-700/50 rounded p-4">
                  <div className="text-sm text-gray-400">Recommendation</div>
                  <div className="text-lg font-medium text-blue-400">
                    {result.recommended_config || "N/A"}
                  </div>
                </div>
              </div>

              {/* Component Importance */}
              {result.component_importance && (
                <div className="bg-gray-700/30 rounded p-4">
                  <h4 className="text-sm font-medium text-gray-300 mb-3">
                    Component Importance (effect on accuracy)
                  </h4>
                  <div className="space-y-2">
                    {Object.entries(result.component_importance)
                      .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
                      .map(([comp, importance]) => (
                        <div key={comp} className="flex items-center gap-3">
                          <div className="w-20 text-sm text-gray-400 uppercase">
                            {comp}
                          </div>
                          <div className="flex-1 bg-gray-600 rounded-full h-4 overflow-hidden">
                            <div
                              className={`h-full ${importance > 0 ? "bg-green-500" : "bg-red-500"}`}
                              style={{
                                width: `${Math.min(Math.abs(importance) * 500, 100)}%`,
                              }}
                            />
                          </div>
                          <div
                            className={`w-16 text-sm text-right ${importance > 0 ? "text-green-400" : "text-red-400"}`}
                          >
                            {importance > 0 ? "+" : ""}
                            {(importance * 100).toFixed(1)}%
                          </div>
                        </div>
                      ))}
                  </div>
                </div>
              )}

              {/* Config Results Table */}
              {result.config_results && (
                <div className="bg-gray-700/30 rounded p-4">
                  <h4 className="text-sm font-medium text-gray-300 mb-3">
                    All Configurations Tested
                  </h4>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                      <thead>
                        <tr className="text-gray-400 border-b border-gray-600">
                          <th className="text-left py-2">Config</th>
                          <th className="text-right py-2">Accuracy</th>
                          <th className="text-right py-2">Click Error</th>
                          <th className="text-right py-2">Trials</th>
                        </tr>
                      </thead>
                      <tbody>
                        {Object.entries(result.config_results)
                          .sort((a, b) => b[1].accuracy - a[1].accuracy)
                          .map(([name, cfg]) => (
                            <tr
                              key={name}
                              className={`border-b border-gray-700 ${name === result.best_config ? "bg-green-900/20" : ""}`}
                            >
                              <td className="py-2 font-mono text-gray-300">
                                {name}
                                {name === result.best_config && (
                                  <span className="ml-2 text-xs text-green-400">
                                    BEST
                                  </span>
                                )}
                              </td>
                              <td className="text-right py-2 text-white">
                                {(cfg.accuracy * 100).toFixed(1)}%
                              </td>
                              <td className="text-right py-2 text-gray-400">
                                {(cfg.click_rate_error * 100).toFixed(1)}%
                              </td>
                              <td className="text-right py-2 text-gray-400">
                                {cfg.n_trials}
                              </td>
                            </tr>
                          ))}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* Recommendation */}
              {result.recommendation_reason && (
                <div className="p-3 bg-blue-900/30 border border-blue-700 rounded">
                  <div className="flex items-start gap-2">
                    <Info className="w-4 h-4 text-blue-400 mt-0.5" />
                    <p className="text-sm text-blue-300">
                      {result.recommendation_reason}
                    </p>
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// =============================================================================
// UNCERTAINTY QUANTIFICATION SECTION
// =============================================================================

const UncertaintySection = ({ personas, models }) => {
  const [selectedPersona, setSelectedPersona] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [promptConfig, setPromptConfig] = useState("cot");
  const [nSamples, setNSamples] = useState(10);
  const [testSampleSize, setTestSampleSize] = useState(10);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  const runAnalysis = async () => {
    if (!selectedPersona || !selectedModel) {
      setError("Please select a persona and model");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await api.runUncertaintyAnalysis({
        persona_id: selectedPersona,
        model_id: selectedModel,
        prompt_config: promptConfig,
        n_samples: nSamples,
        test_sample_size: testSampleSize || null,
      });
      setResult(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 mb-6">
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          <Activity className="w-6 h-6 text-blue-400" />
          <div>
            <h3 className="text-lg font-semibold text-white">
              Uncertainty Quantification
            </h3>
            <p className="text-sm text-gray-400">
              Multi-sample analysis to measure LLM prediction confidence
            </p>
          </div>
        </div>
        {expanded ? (
          <ChevronDown className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronRight className="w-5 h-5 text-gray-400" />
        )}
      </div>

      {expanded && (
        <div className="mt-4 space-y-4">
          {/* Configuration */}
          <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Persona</label>
              <select
                value={selectedPersona}
                onChange={(e) => setSelectedPersona(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              >
                <option value="">Select...</option>
                {personas.map((p) => (
                  <option key={p.persona_id} value={p.persona_id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              >
                <option value="">Select...</option>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Prompt Config
              </label>
              <select
                value={promptConfig}
                onChange={(e) => setPromptConfig(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              >
                <option value="baseline">Baseline</option>
                <option value="stats">Stats</option>
                <option value="cot">CoT</option>
              </select>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Samples/Trial
              </label>
              <input
                type="number"
                value={nSamples}
                onChange={(e) => setNSamples(parseInt(e.target.value) || 10)}
                min={3}
                max={20}
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              />
            </div>

            <div className="flex items-end">
              <button
                onClick={runAnalysis}
                disabled={loading || !selectedPersona || !selectedModel}
                className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 rounded text-white flex items-center justify-center gap-2"
              >
                {loading ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
                Analyze
              </button>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="p-3 bg-red-900/30 border border-red-700 rounded text-red-300 text-sm">
              {error}
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="mt-4 space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-700/50 rounded p-4">
                  <div className="text-sm text-gray-400">Mean Entropy</div>
                  <div className="text-xl font-bold text-white">
                    {(result.mean_normalized_entropy * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500">
                    Low = confident
                  </div>
                </div>
                <div className="bg-gray-700/50 rounded p-4">
                  <div className="text-sm text-gray-400">Majority Vote Accuracy</div>
                  <div className="text-xl font-bold text-green-400">
                    {(result.majority_vote_accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-gray-700/50 rounded p-4">
                  <div className="text-sm text-gray-400">Calibration Error</div>
                  <div className="text-xl font-bold text-yellow-400">
                    {(result.calibration_error * 100).toFixed(1)}%
                  </div>
                  <div className="text-xs text-gray-500">
                    Lower is better
                  </div>
                </div>
                <div className="bg-gray-700/50 rounded p-4">
                  <div className="text-sm text-gray-400">Overconfidence Rate</div>
                  <div className="text-xl font-bold text-red-400">
                    {(result.overconfidence_rate * 100).toFixed(1)}%
                  </div>
                </div>
              </div>

              {/* Accuracy by confidence */}
              <div className="bg-gray-700/30 rounded p-4">
                <h4 className="text-sm font-medium text-gray-300 mb-3">
                  Accuracy by Confidence Level
                </h4>
                <div className="grid grid-cols-2 gap-4">
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">High-confidence trials:</span>
                    <span className="text-green-400 font-medium">
                      {(result.high_confidence_accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-gray-400">Low-confidence trials:</span>
                    <span className="text-yellow-400 font-medium">
                      {(result.low_confidence_accuracy * 100).toFixed(1)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Uncertainty distribution */}
              {result.uncertainty_distribution && (
                <div className="bg-gray-700/30 rounded p-4">
                  <h4 className="text-sm font-medium text-gray-300 mb-3">
                    Uncertainty Distribution
                  </h4>
                  <div className="space-y-2">
                    {Object.entries(result.uncertainty_distribution).map(
                      ([level, pct]) => (
                        <div key={level} className="flex items-center gap-3">
                          <div className="w-32 text-sm text-gray-400 capitalize">
                            {level.replace("_", " ")}
                          </div>
                          <div className="flex-1 bg-gray-600 rounded-full h-3 overflow-hidden">
                            <div
                              className="h-full bg-blue-500"
                              style={{ width: `${pct * 100}%` }}
                            />
                          </div>
                          <div className="w-12 text-sm text-right text-gray-300">
                            {(pct * 100).toFixed(0)}%
                          </div>
                        </div>
                      )
                    )}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// =============================================================================
// BEHAVIORAL SIGNATURES SECTION
// =============================================================================

const SignaturesSection = ({ personas, models }) => {
  const [selectedPersona, setSelectedPersona] = useState("");
  const [selectedModel, setSelectedModel] = useState("");
  const [promptConfig, setPromptConfig] = useState("cot");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  const runAnalysis = async () => {
    if (!selectedPersona || !selectedModel) {
      setError("Please select a persona and model");
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await api.runSignatureMatching({
        persona_id: selectedPersona,
        model_id: selectedModel,
        prompt_config: promptConfig,
      });
      setResult(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 mb-6">
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          <Brain className="w-6 h-6 text-green-400" />
          <div>
            <h3 className="text-lg font-semibold text-white">
              Behavioral Signature Matching
            </h3>
            <p className="text-sm text-gray-400">
              Compare causal factors: confidence, reasoning depth, attention
              patterns
            </p>
          </div>
        </div>
        {expanded ? (
          <ChevronDown className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronRight className="w-5 h-5 text-gray-400" />
        )}
      </div>

      {expanded && (
        <div className="mt-4 space-y-4">
          <div className="p-3 bg-yellow-900/30 border border-yellow-700 rounded">
            <p className="text-sm text-yellow-300">
              Requires calibration to be run first. This analyzes the reasoning
              patterns from calibration trials.
            </p>
          </div>

          {/* Configuration */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div>
              <label className="block text-sm text-gray-400 mb-1">Persona</label>
              <select
                value={selectedPersona}
                onChange={(e) => setSelectedPersona(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              >
                <option value="">Select...</option>
                {personas.map((p) => (
                  <option key={p.persona_id} value={p.persona_id}>
                    {p.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">Model</label>
              <select
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              >
                <option value="">Select...</option>
                {models.map((m) => (
                  <option key={m.id} value={m.id}>
                    {m.name}
                  </option>
                ))}
              </select>
            </div>

            <div>
              <label className="block text-sm text-gray-400 mb-1">
                Prompt Config
              </label>
              <select
                value={promptConfig}
                onChange={(e) => setPromptConfig(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
              >
                <option value="baseline">Baseline</option>
                <option value="stats">Stats</option>
                <option value="cot">CoT</option>
              </select>
            </div>

            <div className="flex items-end">
              <button
                onClick={runAnalysis}
                disabled={loading || !selectedPersona || !selectedModel}
                className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 rounded text-white flex items-center justify-center gap-2"
              >
                {loading ? (
                  <RefreshCw className="w-4 h-4 animate-spin" />
                ) : (
                  <Play className="w-4 h-4" />
                )}
                Analyze
              </button>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="p-3 bg-red-900/30 border border-red-700 rounded text-red-300 text-sm">
              {error}
            </div>
          )}

          {/* Results */}
          {result && (
            <div className="mt-4 space-y-4">
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div className="bg-gray-700/50 rounded p-4">
                  <div className="text-sm text-gray-400">Overall Similarity</div>
                  <div className="text-xl font-bold text-white">
                    {(result.mean_overall_similarity * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-gray-700/50 rounded p-4">
                  <div className="text-sm text-gray-400">Action Accuracy</div>
                  <div className="text-xl font-bold text-green-400">
                    {(result.action_accuracy * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="bg-gray-700/50 rounded p-4">
                  <div className="text-sm text-gray-400">Urgency Correlation</div>
                  <div className="text-xl font-bold text-blue-400">
                    r={result.urgency_effect_correlation?.toFixed(2) || "N/A"}
                  </div>
                </div>
                <div className="bg-gray-700/50 rounded p-4">
                  <div className="text-sm text-gray-400">
                    Familiarity Correlation
                  </div>
                  <div className="text-xl font-bold text-purple-400">
                    r={result.familiarity_effect_correlation?.toFixed(2) || "N/A"}
                  </div>
                </div>
              </div>

              {/* Component similarities */}
              <div className="bg-gray-700/30 rounded p-4">
                <h4 className="text-sm font-medium text-gray-300 mb-3">
                  Component Similarities
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Confidence:</span>
                    <span className="ml-2 text-white">
                      {(result.mean_confidence_similarity * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Speed:</span>
                    <span className="ml-2 text-white">
                      {(result.mean_speed_similarity * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Depth:</span>
                    <span className="ml-2 text-white">
                      {(result.mean_depth_similarity * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div>
                    <span className="text-gray-400">Attention:</span>
                    <span className="ml-2 text-white">
                      {(result.mean_attention_overlap * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>

              {/* Biases */}
              {result.systematic_biases &&
                Object.keys(result.systematic_biases).length > 0 && (
                  <div className="bg-yellow-900/30 border border-yellow-700 rounded p-4">
                    <h4 className="text-sm font-medium text-yellow-300 mb-2">
                      Systematic Biases Detected
                    </h4>
                    <ul className="space-y-1">
                      {Object.entries(result.systematic_biases).map(
                        ([type, desc]) => (
                          <li key={type} className="text-sm text-yellow-200">
                            <span className="font-medium uppercase">{type}:</span>{" "}
                            {desc}
                          </li>
                        )
                      )}
                    </ul>
                  </div>
                )}

              {/* Suggestions */}
              {result.improvement_suggestions &&
                result.improvement_suggestions.length > 0 && (
                  <div className="bg-blue-900/30 border border-blue-700 rounded p-4">
                    <h4 className="text-sm font-medium text-blue-300 mb-2">
                      Improvement Suggestions
                    </h4>
                    <ul className="space-y-1 list-disc list-inside">
                      {result.improvement_suggestions.map((s, i) => (
                        <li key={i} className="text-sm text-blue-200">
                          {s}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// =============================================================================
// PERSONA EMBEDDINGS SECTION
// =============================================================================

const EmbeddingsSection = ({ personas }) => {
  const [embeddingDim, setEmbeddingDim] = useState(64);
  const [iterations, setIterations] = useState(1000);
  const [loading, setLoading] = useState(false);
  const [trained, setTrained] = useState(false);
  const [embeddingData, setEmbeddingData] = useState(null);
  const [error, setError] = useState(null);
  const [expanded, setExpanded] = useState(false);

  // Interpolation
  const [personaA, setPersonaA] = useState("");
  const [personaB, setPersonaB] = useState("");
  const [alpha, setAlpha] = useState(0.5);
  const [interpolateResult, setInterpolateResult] = useState(null);

  const trainEncoder = async () => {
    setLoading(true);
    setError(null);

    try {
      const response = await api.trainPersonaEncoder({
        embedding_dim: embeddingDim,
        n_iterations: iterations,
      });
      setTrained(true);
      setEmbeddingData(response);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const interpolate = async () => {
    if (!personaA || !personaB) {
      setError("Please select two personas to interpolate");
      return;
    }

    try {
      const response = await api.createInterpolatedPersona({
        persona_a_id: personaA,
        persona_b_id: personaB,
        alpha: alpha,
        name: `Blend (${(1 - alpha) * 100}% A + ${alpha * 100}% B)`,
      });
      setInterpolateResult(response);
    } catch (err) {
      setError(err.message);
    }
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 mb-6">
      <div
        className="flex items-center justify-between cursor-pointer"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3">
          <Layers className="w-6 h-6 text-orange-400" />
          <div>
            <h3 className="text-lg font-semibold text-white">
              Persona Embeddings
            </h3>
            <p className="text-sm text-gray-400">
              Continuous embedding space for persona interpolation
            </p>
          </div>
        </div>
        {expanded ? (
          <ChevronDown className="w-5 h-5 text-gray-400" />
        ) : (
          <ChevronRight className="w-5 h-5 text-gray-400" />
        )}
      </div>

      {expanded && (
        <div className="mt-4 space-y-4">
          {/* Training */}
          <div className="bg-gray-700/30 rounded p-4">
            <h4 className="text-sm font-medium text-gray-300 mb-3">
              Train Encoder
            </h4>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="block text-sm text-gray-400 mb-1">
                  Embedding Dimension
                </label>
                <input
                  type="number"
                  value={embeddingDim}
                  onChange={(e) => setEmbeddingDim(parseInt(e.target.value) || 64)}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-400 mb-1">
                  Iterations
                </label>
                <input
                  type="number"
                  value={iterations}
                  onChange={(e) => setIterations(parseInt(e.target.value) || 1000)}
                  className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
                />
              </div>
              <div className="flex items-end">
                <button
                  onClick={trainEncoder}
                  disabled={loading || personas.length < 2}
                  className="w-full px-4 py-2 bg-orange-600 hover:bg-orange-700 disabled:bg-gray-600 rounded text-white flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <RefreshCw className="w-4 h-4 animate-spin" />
                  ) : trained ? (
                    <CheckCircle className="w-4 h-4" />
                  ) : (
                    <Play className="w-4 h-4" />
                  )}
                  {trained ? "Retrain" : "Train"}
                </button>
              </div>
            </div>
          </div>

          {/* Error */}
          {error && (
            <div className="p-3 bg-red-900/30 border border-red-700 rounded text-red-300 text-sm">
              {error}
            </div>
          )}

          {/* Embedding Space Visualization */}
          {embeddingData && (
            <div className="bg-gray-700/30 rounded p-4">
              <h4 className="text-sm font-medium text-gray-300 mb-3">
                Embedding Space (2D projection)
              </h4>
              <div className="relative h-64 bg-gray-900 rounded overflow-hidden">
                {embeddingData.personas?.map((p) => {
                  // Normalize coordinates to 0-100%
                  const allX = embeddingData.personas.map((pp) => pp.x);
                  const allY = embeddingData.personas.map((pp) => pp.y);
                  const minX = Math.min(...allX);
                  const maxX = Math.max(...allX);
                  const minY = Math.min(...allY);
                  const maxY = Math.max(...allY);
                  const rangeX = maxX - minX || 1;
                  const rangeY = maxY - minY || 1;

                  const x = ((p.x - minX) / rangeX) * 80 + 10;
                  const y = ((p.y - minY) / rangeY) * 80 + 10;

                  return (
                    <div
                      key={p.persona_id}
                      className="absolute w-3 h-3 rounded-full bg-blue-400 transform -translate-x-1/2 -translate-y-1/2 cursor-pointer hover:scale-150 transition-transform"
                      style={{ left: `${x}%`, top: `${y}%` }}
                      title={p.name}
                    >
                      <div className="absolute left-4 top-0 whitespace-nowrap text-xs text-gray-400">
                        {p.name}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* Interpolation */}
          {trained && (
            <div className="bg-gray-700/30 rounded p-4">
              <h4 className="text-sm font-medium text-gray-300 mb-3">
                Interpolate Personas
              </h4>
              <div className="grid grid-cols-4 gap-4">
                <div>
                  <label className="block text-sm text-gray-400 mb-1">
                    Persona A
                  </label>
                  <select
                    value={personaA}
                    onChange={(e) => setPersonaA(e.target.value)}
                    className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
                  >
                    <option value="">Select...</option>
                    {personas.map((p) => (
                      <option key={p.persona_id} value={p.persona_id}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">
                    Persona B
                  </label>
                  <select
                    value={personaB}
                    onChange={(e) => setPersonaB(e.target.value)}
                    className="w-full bg-gray-700 border border-gray-600 rounded px-3 py-2 text-white"
                  >
                    <option value="">Select...</option>
                    {personas.map((p) => (
                      <option key={p.persona_id} value={p.persona_id}>
                        {p.name}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm text-gray-400 mb-1">
                    Blend (α={alpha.toFixed(2)})
                  </label>
                  <input
                    type="range"
                    min="0"
                    max="1"
                    step="0.05"
                    value={alpha}
                    onChange={(e) => setAlpha(parseFloat(e.target.value))}
                    className="w-full"
                  />
                </div>
                <div className="flex items-end">
                  <button
                    onClick={interpolate}
                    disabled={!personaA || !personaB}
                    className="w-full px-4 py-2 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 rounded text-white flex items-center justify-center gap-2"
                  >
                    <GitBranch className="w-4 h-4" />
                    Blend
                  </button>
                </div>
              </div>

              {/* Interpolation result */}
              {interpolateResult && (
                <div className="mt-4 p-4 bg-gray-800 rounded">
                  <h5 className="font-medium text-white mb-2">
                    {interpolateResult.name}
                  </h5>
                  <div className="text-sm text-gray-400 mb-2">
                    {interpolateResult.source_personas?.a.weight * 100}%{" "}
                    {interpolateResult.source_personas?.a.name} +{" "}
                    {interpolateResult.source_personas?.b.weight * 100}%{" "}
                    {interpolateResult.source_personas?.b.name}
                  </div>
                  <pre className="text-xs text-gray-300 bg-gray-900 p-2 rounded overflow-x-auto max-h-40">
                    {interpolateResult.prompt_content?.slice(0, 500)}...
                  </pre>
                </div>
              )}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// =============================================================================
// MAIN RESEARCH TAB
// =============================================================================

export const ResearchTab = ({ personas = [], models = [] }) => {
  // Transform models to expected format if needed
  const modelsList = models.map((m) =>
    typeof m === "string" ? { id: m, name: m } : m
  );

  return (
    <div className="space-y-2">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white">Research Tools</h2>
        <p className="text-gray-400 mt-1">
          Advanced analysis tools for prompt optimization, uncertainty
          quantification, and persona understanding
        </p>
      </div>

      {/* Info banner */}
      <div className="p-4 bg-blue-900/30 border border-blue-700 rounded-lg mb-6">
        <div className="flex items-start gap-3">
          <Info className="w-5 h-5 text-blue-400 mt-0.5" />
          <div className="text-sm text-blue-300">
            <p className="font-medium mb-1">Research Features Overview</p>
            <ul className="list-disc list-inside space-y-1 text-blue-200">
              <li>
                <strong>Ablation Study:</strong> Find which prompt components
                (Traits, Stats, CoT) actually improve accuracy
              </li>
              <li>
                <strong>Uncertainty:</strong> Measure how confident the LLM is
                in its predictions
              </li>
              <li>
                <strong>Signatures:</strong> Compare behavioral patterns beyond
                just actions
              </li>
              <li>
                <strong>Embeddings:</strong> Create blended personas via
                interpolation
              </li>
            </ul>
          </div>
        </div>
      </div>

      {/* Sections */}
      <AblationStudySection personas={personas} models={modelsList} />
      <UncertaintySection personas={personas} models={modelsList} />
      <SignaturesSection personas={personas} models={modelsList} />
      <EmbeddingsSection personas={personas} />
    </div>
  );
};

export default ResearchTab;
