/**
 * Clustering Tab
 *
 * Combined tab with 4 sections:
 * 1. Configure Weights - Set composite score weights
 * 2. Find Optimal K - Run K-sweep optimization
 * 3. Run Clustering - Execute with chosen config
 * 4. View Results - See generated personas
 */

import React, { useState, useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import {
  Sliders,
  Shield,
  RotateCcw,
  TrendingUp,
  CheckCircle,
  Play,
  RefreshCw,
  ChevronRight,
  Layers,
  Brain,
  Loader2,
} from "lucide-react";
import {
  Card,
  ChartCard,
  MetricCard,
  ClusterCard,
  WeightInput,
  EmptyState,
  CustomTooltip,
} from "../common";
import {
  ALGORITHM_NAMES,
  ALGORITHM_COLORS,
  DEFAULT_WEIGHTS,
  calculateCompositeScore,
} from "../../constants";

export const ClusteringTab = ({
  config,
  setConfig,
  optConfig,
  setOptConfig,
  weights,
  setWeights,
  updateWeight,
  resetToEqualWeights,
  resetToDefaultWeights,
  minClusterSize,
  setMinClusterSize,
  loading,
  onRunClustering,
  onOptimize,
  clusteringResult,
  optimizationResult,
  normalizedWeights,
  weightsTotal,
  // New props for AI persona naming
  useAiNaming,
  setUseAiNaming,
  isGeneratingNames,
  personaLabels,
}) => {
  const [activeSection, setActiveSection] = useState("config");

  // Transform optimization data for charts
  const optChartData = useMemo(() => {
    if (!optimizationResult) return [];
    const kSet = new Set();
    Object.values(optimizationResult).forEach((points) =>
      points.forEach((p) => kSet.add(p.k)),
    );
    const kValues = Array.from(kSet).sort((a, b) => a - b);

    return kValues.map((k) => {
      const row = { k };
      Object.keys(optimizationResult).forEach((algo) => {
        const point = optimizationResult[algo].find((p) => p.k === k);
        if (point) {
          row[`${algo}_composite`] = calculateCompositeScore(
            point,
            normalizedWeights,
          );
          row[`${algo}_silhouette`] = point.silhouette;
          row[`${algo}_eta`] = point.eta_squared_mean;
          row[`${algo}_balance`] = point.size_balance;
          row[`${algo}_min_size`] = point.min_cluster_size;
        }
      });
      return row;
    });
  }, [optimizationResult, normalizedWeights]);

  // Get best configuration
  const bestConfig = useMemo(() => {
    if (!optimizationResult) return null;
    let best = null;
    let bestScore = -1;

    Object.entries(optimizationResult).forEach(([algo, results]) => {
      results.forEach((r) => {
        const score = calculateCompositeScore(r, normalizedWeights);
        if (score > bestScore && r.min_cluster_size >= minClusterSize) {
          bestScore = score;
          best = { ...r, recalculated_score: score };
        }
      });
    });
    return best;
  }, [optimizationResult, normalizedWeights, minClusterSize]);

  return (
    <div className="space-y-6">
      {/* Section Navigation */}
      <div className="flex gap-2 border-b pb-4">
        <button
          onClick={() => setActiveSection("config")}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
            activeSection === "config"
              ? "bg-indigo-600 text-white"
              : "bg-gray-100 text-gray-700 hover:bg-gray-200"
          }`}
        >
          1. Configure Weights
        </button>
        <button
          onClick={() => setActiveSection("optimize")}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
            activeSection === "optimize"
              ? "bg-indigo-600 text-white"
              : "bg-gray-100 text-gray-700 hover:bg-gray-200"
          }`}
        >
          2. Find Optimal K
        </button>
        <button
          onClick={() => setActiveSection("run")}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
            activeSection === "run"
              ? "bg-indigo-600 text-white"
              : "bg-gray-100 text-gray-700 hover:bg-gray-200"
          }`}
        >
          3. Run Clustering
        </button>
      </div>

      {/* Section 1: Configure Weights */}
      {activeSection === "config" && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-md font-semibold text-gray-900 flex items-center gap-2">
                <Sliders size={18} className="text-indigo-600" />
                Composite Score Weights
              </h3>
              <div className="flex gap-2">
                <button
                  onClick={resetToEqualWeights}
                  className="px-3 py-1 text-xs bg-gray-100 hover:bg-gray-200 rounded-md flex items-center gap-1"
                >
                  <RotateCcw size={12} /> Equal
                </button>
                <button
                  onClick={resetToDefaultWeights}
                  className="px-3 py-1 text-xs bg-indigo-100 hover:bg-indigo-200 text-indigo-700 rounded-md"
                >
                  Default
                </button>
              </div>
            </div>

            <div
              className={`mb-4 p-2 rounded-lg text-sm ${
                Math.abs(weightsTotal - 1) < 0.01
                  ? "bg-green-50 text-green-700"
                  : "bg-yellow-50 text-yellow-700"
              }`}
            >
              Total: {(weightsTotal * 100).toFixed(0)}%
              {Math.abs(weightsTotal - 1) >= 0.01 &&
                " (will be normalized to 100%)"}
            </div>

            <div className="space-y-4">
              <WeightInput
                label="Behavioral (η²)"
                value={weights.behavioral}
                onChange={(v) => updateWeight("behavioral", v)}
                color="indigo"
                description="How well clusters predict phishing behavior"
                normalizedValue={normalizedWeights.behavioral}
              />
              <WeightInput
                label="Silhouette"
                value={weights.silhouette}
                onChange={(v) => updateWeight("silhouette", v)}
                color="emerald"
                description="Cluster separation quality"
                normalizedValue={normalizedWeights.silhouette}
              />
              <WeightInput
                label="Stability"
                value={weights.stability}
                onChange={(v) => updateWeight("stability", v)}
                color="amber"
                description="Cluster size balance"
                normalizedValue={normalizedWeights.stability}
              />
              <WeightInput
                label="Statistical"
                value={weights.statistical}
                onChange={(v) => updateWeight("statistical", v)}
                color="rose"
                description="Calinski-Harabasz & Davies-Bouldin"
                normalizedValue={normalizedWeights.statistical}
              />
            </div>
          </Card>

          <Card>
            <h3 className="text-md font-semibold text-gray-900 mb-4 flex items-center gap-2">
              <Shield size={18} className="text-indigo-600" />
              Constraints
            </h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Minimum Cluster Size
                </label>
                <div className="flex items-center gap-4">
                  <input
                    type="range"
                    min="10"
                    max="100"
                    value={minClusterSize}
                    onChange={(e) =>
                      setMinClusterSize(parseInt(e.target.value))
                    }
                    className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-indigo-600"
                  />
                  <input
                    type="number"
                    min="10"
                    max="100"
                    value={minClusterSize}
                    onChange={(e) =>
                      setMinClusterSize(parseInt(e.target.value) || 30)
                    }
                    className="w-20 text-center font-mono text-sm border border-gray-300 rounded-md px-2 py-1"
                  />
                </div>
                <p className="text-xs text-gray-500 mt-1">
                  Clusters smaller than this will be flagged as potentially
                  unstable
                </p>
              </div>

              <div className="pt-4 border-t border-gray-100">
                <h4 className="text-sm font-medium text-gray-700 mb-2">
                  Effect Size Reference
                </h4>
                <div className="flex gap-4 text-xs">
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded bg-yellow-400"></span>
                    <span>Small: η² &lt; 0.06</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded bg-orange-400"></span>
                    <span>Medium: 0.06 ≤ η² &lt; 0.14</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="w-3 h-3 rounded bg-green-500"></span>
                    <span>Large: η² ≥ 0.14</span>
                  </div>
                </div>
              </div>
            </div>

            <div className="mt-6 pt-4 border-t">
              <button
                onClick={() => setActiveSection("optimize")}
                className="w-full px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 flex items-center justify-center gap-2"
              >
                Continue to Find Optimal K <ChevronRight size={16} />
              </button>
            </div>
          </Card>
        </div>
      )}

      {/* Section 2: Optimization / K-Sweep */}
      {activeSection === "optimize" && (
        <div className="space-y-6">
          <Card>
            <div className="flex flex-wrap items-end gap-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Algorithm
                </label>
                <select
                  className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white"
                  value={optConfig.algorithm}
                  onChange={(e) =>
                    setOptConfig({ ...optConfig, algorithm: e.target.value })
                  }
                >
                  <option value="all">Compare All</option>
                  <option value="kmeans">K-Means</option>
                  <option value="gmm">GMM</option>
                  <option value="hierarchical">Hierarchical</option>
                  <option value="spectral">Spectral</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  K Min
                </label>
                <input
                  type="number"
                  min="2"
                  max="20"
                  className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-sm"
                  value={optConfig.k_min}
                  onChange={(e) =>
                    setOptConfig({
                      ...optConfig,
                      k_min: parseInt(e.target.value),
                    })
                  }
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  K Max
                </label>
                <input
                  type="number"
                  min="2"
                  max="25"
                  className="w-20 px-3 py-2 border border-gray-300 rounded-lg text-sm"
                  value={optConfig.k_max}
                  onChange={(e) =>
                    setOptConfig({
                      ...optConfig,
                      k_max: parseInt(e.target.value),
                    })
                  }
                />
              </div>

              <div className="flex items-center gap-2">
                <input
                  type="checkbox"
                  id="opt_use_pca"
                  checked={optConfig.use_pca}
                  onChange={(e) =>
                    setOptConfig({ ...optConfig, use_pca: e.target.checked })
                  }
                  className="w-4 h-4 accent-indigo-600"
                />
                <label htmlFor="opt_use_pca" className="text-sm text-gray-700">
                  Use PCA
                </label>
              </div>

              {optConfig.use_pca && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    PCA Variance
                  </label>
                  <select
                    className="px-3 py-2 border border-gray-300 rounded-lg text-sm bg-white"
                    value={optConfig.pca_variance}
                    onChange={(e) =>
                      setOptConfig({
                        ...optConfig,
                        pca_variance: parseFloat(e.target.value),
                      })
                    }
                  >
                    <option value={0.8}>80%</option>
                    <option value={0.85}>85%</option>
                    <option value={0.9}>90%</option>
                    <option value={0.95}>95%</option>
                    <option value={0.99}>99%</option>
                  </select>
                </div>
              )}

              <button
                onClick={onOptimize}
                disabled={loading}
                className={`px-5 py-2 rounded-lg text-sm font-medium text-white transition-colors ${
                  loading
                    ? "bg-gray-400 cursor-not-allowed"
                    : "bg-indigo-600 hover:bg-indigo-700"
                }`}
              >
                {loading ? "Running..." : "Run K-Sweep"}
              </button>
            </div>

            <div className="mt-4 pt-4 border-t border-gray-100 flex items-center gap-6 text-xs text-gray-500">
              <span className="font-medium">Current weights:</span>
              <span>
                Behavioral {(normalizedWeights.behavioral * 100).toFixed(0)}%
              </span>
              <span>
                Silhouette {(normalizedWeights.silhouette * 100).toFixed(0)}%
              </span>
              <span>
                Stability {(normalizedWeights.stability * 100).toFixed(0)}%
              </span>
              <span>
                Statistical {(normalizedWeights.statistical * 100).toFixed(0)}%
              </span>
              <span className="text-gray-400">|</span>
              <span>Min size: {minClusterSize}</span>
            </div>
          </Card>

          {optimizationResult && (
            <>
              {/* Best Configuration */}
              {bestConfig && (
                <div className="bg-indigo-50 border border-indigo-200 rounded-xl p-5">
                  <div className="flex items-start justify-between">
                    <div>
                      <h3 className="font-semibold text-indigo-900 flex items-center gap-2">
                        <CheckCircle size={18} className="text-indigo-600" />
                        Recommended Configuration
                      </h3>
                      <p className="text-sm text-indigo-700 mt-1">
                        <strong>{ALGORITHM_NAMES[bestConfig.algorithm]}</strong>{" "}
                        with <strong>K = {bestConfig.k}</strong>
                      </p>
                      <div className="flex gap-6 mt-2 text-xs text-indigo-600">
                        <span>
                          Score: {bestConfig.recalculated_score?.toFixed(4)}
                        </span>
                        <span>
                          η²: {bestConfig.eta_squared_mean?.toFixed(3)}
                        </span>
                        <span>
                          Silhouette: {bestConfig.silhouette?.toFixed(3)}
                        </span>
                        <span>Min size: {bestConfig.min_cluster_size}</span>
                      </div>
                    </div>
                    <button
                      onClick={() => {
                        setConfig((prev) => ({
                          ...prev,
                          algorithm: bestConfig.algorithm,
                          k: bestConfig.k,
                        }));
                        setActiveSection("run");
                      }}
                      className="px-4 py-2 bg-indigo-600 text-white text-sm rounded-lg hover:bg-indigo-700"
                    >
                      Use This Config →
                    </button>
                  </div>
                </div>
              )}

              {/* Charts */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <ChartCard
                  title="Composite Score"
                  subtitle="Overall quality score (higher is better)"
                >
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={optChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="k" stroke="#6b7280" fontSize={12} />
                      <YAxis stroke="#6b7280" fontSize={12} />
                      <RechartsTooltip content={<CustomTooltip />} />
                      <Legend />
                      {Object.keys(optimizationResult).map((algo) => (
                        <Line
                          key={algo}
                          type="monotone"
                          dataKey={`${algo}_composite`}
                          name={ALGORITHM_NAMES[algo]}
                          stroke={ALGORITHM_COLORS[algo]}
                          strokeWidth={2}
                          dot={{ r: 4 }}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>

                <ChartCard
                  title="Behavioral Prediction (η²)"
                  subtitle="Effect size - how well clusters predict behavior"
                >
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={optChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="k" stroke="#6b7280" fontSize={12} />
                      <YAxis stroke="#6b7280" fontSize={12} />
                      <RechartsTooltip content={<CustomTooltip />} />
                      <Legend />
                      <ReferenceLine
                        y={0.06}
                        stroke="#f59e0b"
                        strokeDasharray="5 5"
                      />
                      <ReferenceLine
                        y={0.14}
                        stroke="#22c55e"
                        strokeDasharray="5 5"
                      />
                      {Object.keys(optimizationResult).map((algo) => (
                        <Line
                          key={algo}
                          type="monotone"
                          dataKey={`${algo}_eta`}
                          name={ALGORITHM_NAMES[algo]}
                          stroke={ALGORITHM_COLORS[algo]}
                          strokeWidth={2}
                          dot={{ r: 4 }}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>

                <ChartCard
                  title="Silhouette Score"
                  subtitle="Cluster separation quality (higher is better)"
                >
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={optChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="k" stroke="#6b7280" fontSize={12} />
                      <YAxis stroke="#6b7280" fontSize={12} />
                      <RechartsTooltip content={<CustomTooltip />} />
                      <Legend />
                      {Object.keys(optimizationResult).map((algo) => (
                        <Line
                          key={algo}
                          type="monotone"
                          dataKey={`${algo}_silhouette`}
                          name={ALGORITHM_NAMES[algo]}
                          stroke={ALGORITHM_COLORS[algo]}
                          strokeWidth={2}
                          dot={{ r: 4 }}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>

                <ChartCard
                  title="Size Balance"
                  subtitle="Distribution evenness across clusters"
                >
                  <ResponsiveContainer width="100%" height={280}>
                    <LineChart data={optChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <XAxis dataKey="k" stroke="#6b7280" fontSize={12} />
                      <YAxis stroke="#6b7280" fontSize={12} />
                      <RechartsTooltip content={<CustomTooltip />} />
                      <Legend />
                      {Object.keys(optimizationResult).map((algo) => (
                        <Line
                          key={algo}
                          type="monotone"
                          dataKey={`${algo}_balance`}
                          name={ALGORITHM_NAMES[algo]}
                          stroke={ALGORITHM_COLORS[algo]}
                          strokeWidth={2}
                          dot={{ r: 4 }}
                        />
                      ))}
                    </LineChart>
                  </ResponsiveContainer>
                </ChartCard>
              </div>
            </>
          )}

          {!optimizationResult && (
            <EmptyState
              icon={<TrendingUp size={48} />}
              title="Run K-Sweep to find optimal configuration"
              description="Configure the parameters above and click 'Run K-Sweep' to compare algorithms and find the best number of clusters."
            />
          )}
        </div>
      )}

      {/* Section 3: Run Clustering */}
      {activeSection === "run" && (
        <div className="space-y-6">
          <Card>
            <h3 className="text-lg font-semibold mb-4">Run Clustering</h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4 mb-6">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Algorithm
                </label>
                <select
                  value={config.algorithm}
                  onChange={(e) =>
                    setConfig({ ...config, algorithm: e.target.value })
                  }
                  className="w-full px-3 py-2 border rounded-lg"
                >
                  {Object.entries(ALGORITHM_NAMES).map(([key, name]) => (
                    <option key={key} value={key}>
                      {name}
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Clusters (K)
                </label>
                <input
                  type="number"
                  min="2"
                  max="15"
                  value={config.k}
                  onChange={(e) =>
                    setConfig({ ...config, k: parseInt(e.target.value) })
                  }
                  className="w-full px-3 py-2 border rounded-lg"
                />
              </div>

              <div className="flex items-center gap-2 pt-6">
                <input
                  type="checkbox"
                  id="use_pca"
                  checked={config.use_pca}
                  onChange={(e) =>
                    setConfig({ ...config, use_pca: e.target.checked })
                  }
                  className="w-4 h-4 accent-indigo-600"
                />
                <label htmlFor="use_pca" className="text-sm">
                  Use PCA
                </label>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  PCA Variance
                </label>
                <select
                  value={config.pca_variance}
                  onChange={(e) =>
                    setConfig({
                      ...config,
                      pca_variance: parseFloat(e.target.value),
                    })
                  }
                  disabled={!config.use_pca}
                  className="w-full px-3 py-2 border rounded-lg disabled:bg-gray-100"
                >
                  <option value={0.8}>80%</option>
                  <option value={0.85}>85%</option>
                  <option value={0.9}>90%</option>
                  <option value={0.95}>95%</option>
                </select>
              </div>

              <div className="flex items-end">
                <button
                  onClick={onRunClustering}
                  disabled={loading || isGeneratingNames}
                  className="w-full px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 flex items-center justify-center gap-2"
                >
                  {loading ? (
                    <>
                      <RefreshCw className="animate-spin" size={16} />{" "}
                      Running...
                    </>
                  ) : isGeneratingNames ? (
                    <>
                      <Loader2 className="animate-spin" size={16} /> Naming
                      Personas...
                    </>
                  ) : (
                    <>
                      <Play size={16} /> Run Clustering
                    </>
                  )}
                </button>
              </div>
            </div>

            {/* AI Persona Naming Option */}
            <div className="mt-4 pt-4 border-t border-gray-100">
              <div className="flex items-center gap-3">
                <input
                  type="checkbox"
                  id="use_ai_naming"
                  checked={useAiNaming}
                  onChange={(e) => setUseAiNaming(e.target.checked)}
                  className="w-4 h-4 accent-purple-600"
                />
                <label
                  htmlFor="use_ai_naming"
                  className="flex items-center gap-2 text-sm cursor-pointer"
                >
                  <Brain size={16} className="text-purple-600" />
                  <span>Generate AI names for personas</span>
                </label>
                <span className="text-xs text-gray-400">
                  (uses Llama 3.3 70B Model)
                </span>
              </div>
              {useAiNaming && (
                <p className="mt-2 text-xs text-gray-500 ml-7">
                  After clustering completes, AI will generate descriptive names
                  based on psychological traits and behavioral outcomes.
                </p>
              )}
            </div>

            {bestConfig && !clusteringResult && (
              <div className="p-3 bg-indigo-50 rounded-lg text-sm text-indigo-700">
                <strong>Recommended:</strong>{" "}
                {ALGORITHM_NAMES[bestConfig.algorithm]} with K={bestConfig.k}
                (score: {bestConfig.recalculated_score?.toFixed(4)})
              </div>
            )}
          </Card>

          {/* AI Name Generation Loading Overlay */}
          {isGeneratingNames && (
            <Card>
              <div className="flex flex-col items-center justify-center py-12">
                <div className="relative mb-6">
                  <div className="w-16 h-16 border-4 border-purple-200 rounded-full animate-pulse"></div>
                  <Brain
                    className="absolute inset-0 m-auto text-purple-600 animate-pulse"
                    size={32}
                  />
                </div>
                <h3 className="text-lg font-semibold text-gray-900 mb-2">
                  Generating AI Persona Names...
                </h3>
                <p className="text-sm text-gray-500 text-center max-w-md">
                  LLM is analyzing psychological traits and behavioral patterns
                  to create descriptive names for each persona.
                </p>
                <div className="mt-4 flex items-center gap-2 text-xs text-purple-600">
                  <Loader2 className="animate-spin" size={14} />
                  <span>This usually takes 5-15 seconds</span>
                </div>
              </div>
            </Card>
          )}

          {clusteringResult && !isGeneratingNames && (
            <>
              {/* Quality Metrics with Config Info */}
              <Card>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">
                    Clustering Quality Metrics
                  </h3>
                  <div className="flex items-center gap-3 text-sm">
                    <span className="px-2 py-1 bg-indigo-100 text-indigo-700 rounded font-medium">
                      {ALGORITHM_NAMES[clusteringResult.algorithm]}
                    </span>
                    <span className="px-2 py-1 bg-gray-100 text-gray-700 rounded">
                      K = {clusteringResult.k}
                    </span>
                    {clusteringResult.pca_components && (
                      <span className="px-2 py-1 bg-purple-100 text-purple-700 rounded">
                        {clusteringResult.pca_components} PCA components
                      </span>
                    )}
                  </div>
                </div>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <MetricCard
                    label="Behavioral (η²)"
                    value={clusteringResult.metrics?.eta_squared_mean?.toFixed(
                      3,
                    )}
                    status={
                      clusteringResult.metrics?.eta_squared_mean >= 0.14
                        ? "success"
                        : clusteringResult.metrics?.eta_squared_mean >= 0.06
                          ? "warning"
                          : "danger"
                    }
                  />
                  <MetricCard
                    label="Silhouette"
                    value={clusteringResult.metrics?.silhouette?.toFixed(3)}
                    status={
                      clusteringResult.metrics?.silhouette >= 0.3
                        ? "success"
                        : "neutral"
                    }
                  />
                  <MetricCard
                    label="Min Cluster Size"
                    value={clusteringResult.metrics?.min_cluster_size}
                    status={
                      clusteringResult.metrics?.min_cluster_size >=
                      minClusterSize
                        ? "success"
                        : "danger"
                    }
                  />
                  <MetricCard
                    label="Size Balance"
                    value={clusteringResult.metrics?.size_balance?.toFixed(2)}
                    status={
                      clusteringResult.metrics?.size_balance >= 0.5
                        ? "success"
                        : "warning"
                    }
                  />
                </div>
              </Card>

              {/* Generated Personas */}
              <Card>
                <div className="flex items-center justify-between mb-4">
                  <h3 className="text-lg font-semibold">Generated Personas</h3>
                </div>
                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
                  {Object.values(clusteringResult.clusters || {}).map(
                    (cluster) => (
                      <ClusterCard
                        key={cluster.cluster_id}
                        cluster={cluster}
                        minClusterSize={minClusterSize}
                        personaLabel={personaLabels?.[cluster.cluster_id]}
                      />
                    ),
                  )}
                </div>
              </Card>
            </>
          )}

          {!clusteringResult && !isGeneratingNames && (
            <EmptyState
              icon={<Layers size={48} />}
              title="No clustering results yet"
              description="Configure parameters and run clustering to generate personas."
            />
          )}
        </div>
      )}
    </div>
  );
};

export default ClusteringTab;
