/**
 * Clustering Tab
 *
 * Combined tab with 4 sections:
 * 1. Configure Weights - Set composite score weights
 * 2. Find Optimal K - Run K-sweep optimization
 * 3. Run Clustering - Execute with chosen config
 * 4. Scientific Validation - Validate clustering quality
 */

import React, { useState, useMemo } from "react";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  ZAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
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
  XCircle,
  FlaskConical,
  AlertTriangle,
  Info,
  Target,
  BarChart3,
  ScatterChart as ScatterChartIcon,
} from "lucide-react";
import {
  Card,
  ChartCard,
  MetricCard,
  ClusterCard,
  WeightInput,
  EmptyState,
  CustomTooltip,
  SystematicCodeLegend,
} from "../common";
import {
  ALGORITHM_NAMES,
  ALGORITHM_COLORS,
  DEFAULT_WEIGHTS,
  calculateCompositeScore,
} from "../../constants";

// Cluster colors used throughout the visualization
const CLUSTER_COLORS = [
  "#6366f1",
  "#10b981",
  "#f59e0b",
  "#ef4444",
  "#8b5cf6",
  "#06b6d4",
  "#ec4899",
  "#84cc16",
  "#f97316",
  "#14b8a6",
];
const CLUSTER_BG_COLORS = [
  "bg-indigo-100 text-indigo-700",
  "bg-emerald-100 text-emerald-700",
  "bg-amber-100 text-amber-700",
  "bg-red-100 text-red-700",
  "bg-purple-100 text-purple-700",
  "bg-cyan-100 text-cyan-700",
  "bg-pink-100 text-pink-700",
  "bg-lime-100 text-lime-700",
];

/**
 * Cluster Visualization Chart Component
 *
 * Renders a 2D scatter plot with:
 * - Data points colored by cluster
 * - Centroids marked with bold X
 * - Note: Ellipse boundaries are hidden by default because in low-variance
 *   2D projections they overlap and can be misleading
 */
const ClusterVisualizationChart = ({ data }) => {
  if (!data) return null;

  const {
    method_description,
    k,
    points,
    centroids,
    axis_ranges,
    axis_features,
    decision_boundaries,
    persona_labels,
    interpretation,
  } = data;

  // Calculate total explained variance
  const totalVariance =
    (axis_features?.x_axis?.explained_variance || 0) +
    (axis_features?.y_axis?.explained_variance || 0);

  // Get unique cluster IDs from actual points data (filter out invalid values)
  const uniqueClusters = useMemo(() => {
    if (!points || points.length === 0) return [];
    const clusterSet = new Set(
      points
        .map((p) => p.cluster)
        .filter(
          (c) =>
            c !== undefined && c !== null && !isNaN(c) && typeof c === "number",
        ),
    );
    return Array.from(clusterSet).sort((a, b) => a - b);
  }, [points]);

  // Helper to get cluster name - returns only AI-generated name (no systematic code)
  const getClusterName = (clusterId) => {
    // Handle invalid cluster IDs
    if (clusterId === undefined || clusterId === null || isNaN(clusterId)) {
      return "Unknown";
    }
    // Ensure clusterId is a number
    const numericId = Number(clusterId);

    // Try direct lookup first (0-indexed)
    let label = persona_labels?.[numericId];
    if (!label) {
      // Try 1-indexed lookup (cluster 0 -> label key 1, etc.)
      label = persona_labels?.[numericId + 1];
    }
    if (!label) {
      // Try string key
      label = persona_labels?.[String(numericId)];
    }

    if (label) {
      // Return only AI-generated name (llm_name) or extract from hybrid name
      if (typeof label === "string") {
        // Label is just a string
        if (label.includes(": ")) {
          return label.split(": ").slice(1).join(": ");
        }
        return label;
      }
      if (label.llm_name) return label.llm_name;
      if (label.name && label.name.includes(": ")) {
        return label.name.split(": ").slice(1).join(": ");
      }
      return label.name || `Cluster ${numericId + 1}`;
    }
    // Fallback: display as 1-indexed for user (Cluster 1, 2, 3...)
    return `Cluster ${numericId + 1}`;
  };

  // Helper to get systematic code for a cluster
  const getSystematicCode = (clusterId) => {
    if (clusterId === undefined || clusterId === null || isNaN(clusterId))
      return null;
    const numericId = Number(clusterId);
    let label =
      persona_labels?.[numericId] ||
      persona_labels?.[numericId + 1] ||
      persona_labels?.[String(numericId)];
    if (!label) return null;

    if (typeof label === "string") {
      if (label.includes(": ")) {
        return label.split(": ")[0];
      }
      return null;
    }
    if (label?.archetype) return label.archetype;
    if (label?.name && label.name.includes(": ")) {
      return label.name.split(": ")[0];
    }
    return null;
  };

  return (
    <Card>
      <h3 className="text-md font-semibold text-gray-900 mb-4 flex items-center gap-2">
        <ScatterChartIcon size={18} className="text-cyan-600" />
        Cluster Visualization (2D Projection)
      </h3>
      <p className="text-sm text-gray-500 mb-4">
        {method_description || "PCA projection"} — Data points colored by
        cluster assignment with centroids marked.
      </p>

      <div className="h-96">
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 60, right: 20, bottom: 40, left: 40 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
            <XAxis
              type="number"
              dataKey="x"
              name="PC1"
              domain={axis_ranges?.x || ["auto", "auto"]}
              fontSize={11}
              tickFormatter={(value) => value.toFixed(2)}
              label={{
                value: axis_features?.x_axis?.name || "PC1",
                position: "bottom",
                offset: 0,
                fontSize: 12,
              }}
            />
            <YAxis
              type="number"
              dataKey="y"
              name="PC2"
              domain={axis_ranges?.y || ["auto", "auto"]}
              fontSize={11}
              tickFormatter={(value) => value.toFixed(2)}
              label={{
                value: axis_features?.y_axis?.name || "PC2",
                angle: -90,
                position: "insideLeft",
                fontSize: 12,
              }}
            />
            <ZAxis type="number" range={[40, 80]} />
            <RechartsTooltip
              cursor={{ strokeDasharray: "3 3" }}
              content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const pointData = payload[0].payload;
                  if (pointData.isCentroid) {
                    return (
                      <div className="bg-white p-2 border border-gray-200 rounded shadow-lg text-xs">
                        <p className="font-semibold text-gray-900">
                          ✕ {getClusterName(pointData.cluster)} Centroid
                        </p>
                        <p className="text-gray-600">
                          Size: {pointData.size} members
                        </p>
                      </div>
                    );
                  }
                  return (
                    <div className="bg-white p-2 border border-gray-200 rounded shadow-lg text-xs">
                      <p className="font-semibold text-gray-900">
                        {getClusterName(pointData.cluster)}
                      </p>
                      <p className="text-gray-600">
                        Participant: {pointData.participant_idx}
                      </p>
                      <p className="text-gray-600">
                        Distance to centroid:{" "}
                        {pointData.distance_to_centroid?.toFixed(2)}
                      </p>
                    </div>
                  );
                }
                return null;
              }}
            />
            <Legend
              verticalAlign="top"
              height={50}
              wrapperStyle={{ fontSize: "11px", paddingBottom: "8px" }}
              formatter={(value, entry) => {
                if (value === "centroids") return "✕ Centroids";
                // Hide any entries with "NaN", "undefined", or "Unknown"
                if (
                  value === "Unknown" ||
                  value === "Cluster NaN" ||
                  value === "Cluster undefined"
                )
                  return null;
                // Get cluster ID from the data to show systematic code
                const clusterData = entry?.payload?.[0];
                const clusterId = clusterData?.cluster;
                const systematicCode = getSystematicCode(clusterId);
                if (systematicCode) {
                  return (
                    <span>
                      {value}{" "}
                      <span style={{ color: "#6b7280", fontSize: "10px" }}>
                        ({systematicCode})
                      </span>
                    </span>
                  );
                }
                return value;
              }}
            />

            {/* Render points for each cluster */}
            {uniqueClusters.map((clusterId) => {
              // Skip invalid cluster IDs
              if (
                clusterId === undefined ||
                clusterId === null ||
                isNaN(clusterId)
              ) {
                return null;
              }
              const clusterPoints = (points || []).filter(
                (p) =>
                  p.cluster === clusterId &&
                  p.cluster !== undefined &&
                  p.cluster !== null &&
                  !isNaN(p.cluster),
              );
              if (clusterPoints.length === 0) return null;

              const color = CLUSTER_COLORS[clusterId % CLUSTER_COLORS.length];
              const clusterName = getClusterName(clusterId);

              return (
                <Scatter
                  key={`cluster-${clusterId}`}
                  name={clusterName}
                  data={clusterPoints}
                  fill={color}
                  fillOpacity={0.7}
                />
              );
            })}

            {/* Render centroids as bold black X markers */}
            {centroids && centroids.length > 0 && (
              <Scatter
                name="centroids"
                data={centroids.map((c) => ({ ...c, isCentroid: true }))}
                shape={(props) => {
                  const { cx, cy } = props;
                  if (cx == null || cy == null) return null;
                  // Bold black X marker for visibility
                  const size = 10;
                  return (
                    <g>
                      {/* White outline for contrast */}
                      <line
                        x1={cx - size}
                        y1={cy - size}
                        x2={cx + size}
                        y2={cy + size}
                        stroke="white"
                        strokeWidth={5}
                      />
                      <line
                        x1={cx + size}
                        y1={cy - size}
                        x2={cx - size}
                        y2={cy + size}
                        stroke="white"
                        strokeWidth={5}
                      />
                      {/* Black X */}
                      <line
                        x1={cx - size}
                        y1={cy - size}
                        x2={cx + size}
                        y2={cy + size}
                        stroke="#000000"
                        strokeWidth={3}
                      />
                      <line
                        x1={cx + size}
                        y1={cy - size}
                        x2={cx - size}
                        y2={cy + size}
                        stroke="#000000"
                        strokeWidth={3}
                      />
                    </g>
                  );
                }}
                legendType="cross"
              />
            )}
          </ScatterChart>
        </ResponsiveContainer>
      </div>

      {/* Centroids info */}
      {/* {centroids && centroids.length > 0 && (
        <div className="mt-4 pt-4 border-t border-gray-100">
          <h4 className="text-sm font-medium text-gray-700 mb-2">
            Cluster Centroids
          </h4>
          <div className="flex flex-wrap gap-2">
            {centroids.map((centroid, idx) => {
              const colorClass =
                CLUSTER_BG_COLORS[centroid.cluster % CLUSTER_BG_COLORS.length];
              return (
                <span
                  key={`centroid-${centroid.cluster}`}
                  className={`px-3 py-1 rounded-full text-sm ${colorClass}`}
                >
                  {getClusterName(centroid.cluster)}: {centroid.size} members
                </span>
              );
            })}
          </div>
        </div>
      )} */}

      {/* Axis features info for PCA */}
      {axis_features && (
        <div className="mt-4 pt-4 border-t border-gray-100">
          <h4 className="text-sm font-medium text-gray-700 mb-2">
            Axis Interpretation (Top Features)
          </h4>
          <div className="grid grid-cols-2 gap-4 text-xs">
            <div>
              <p className="font-medium text-gray-600 mb-1">
                {axis_features.x_axis?.name} (
                {(axis_features.x_axis?.explained_variance * 100 || 0).toFixed(
                  0,
                )}
                % variance)
              </p>
              <ul className="text-gray-500 space-y-0.5">
                {(axis_features.x_axis?.top_features || [])
                  .slice(0, 3)
                  .map((f, i) => (
                    <li key={i}>
                      • {f.feature.replace(/_/g, " ")}:{" "}
                      {f.loading > 0 ? "+" : ""}
                      {f.loading.toFixed(2)}
                    </li>
                  ))}
              </ul>
            </div>
            <div>
              <p className="font-medium text-gray-600 mb-1">
                {axis_features.y_axis?.name} (
                {(axis_features.y_axis?.explained_variance * 100 || 0).toFixed(
                  0,
                )}
                % variance)
              </p>
              <ul className="text-gray-500 space-y-0.5">
                {(axis_features.y_axis?.top_features || [])
                  .slice(0, 3)
                  .map((f, i) => (
                    <li key={i}>
                      • {f.feature.replace(/_/g, " ")}:{" "}
                      {f.loading > 0 ? "+" : ""}
                      {f.loading.toFixed(2)}
                    </li>
                  ))}
              </ul>
            </div>
          </div>
        </div>
      )}

      {/* Important note about PCA limitations */}
      <div className="mt-4 p-3 bg-amber-50 rounded-lg border border-amber-200">
        <p className="text-xs text-amber-800 flex items-start gap-2">
          <AlertTriangle size={14} className="mt-0.5 flex-shrink-0" />
          <span>
            <strong>Note:</strong> This 2D view only captures ~
            {(((axis_features?.x_axis?.explained_variance || 0) +
              (axis_features?.y_axis?.explained_variance || 0)) *
              100) |
              0}
            % of the data variance. Clusters may appear overlapped here but are
            actually well-separated in the full feature space.
          </span>
        </p>
      </div>

      {/* Interpretation */}
      {interpretation && (
        <p className="mt-3 text-sm text-gray-500">
          <Info size={14} className="inline mr-1" />
          {interpretation}
        </p>
      )}
    </Card>
  );
};

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
  onCancel,
  operationType,
  clusteringResult,
  optimizationResult,
  normalizedWeights,
  weightsTotal,
  // New props for AI persona naming
  useAiNaming,
  setUseAiNaming,
  isGeneratingNames,
  personaLabels,
  // New props for scientific validation
  validationResult,
  validationLoading,
  validationProgress,
  onRunValidation,
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
      <div className="flex gap-2 border-b pb-4 flex-wrap">
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
        <button
          onClick={() => setActiveSection("validate")}
          className={`px-4 py-2 rounded-lg text-sm font-medium transition ${
            activeSection === "validate"
              ? "bg-emerald-600 text-white"
              : clusteringResult
                ? "bg-emerald-50 text-emerald-700 hover:bg-emerald-100"
                : "bg-gray-100 text-gray-400 cursor-not-allowed"
          }`}
          disabled={!clusteringResult}
        >
          4. Validate Results
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
                    ? "bg-indigo-400 cursor-not-allowed"
                    : "bg-indigo-600 hover:bg-indigo-700"
                }`}
              >
                {loading ? (
                  <>
                    <Loader2 className="inline animate-spin mr-2" size={14} />
                    Analyzing...
                  </>
                ) : (
                  "Run K-Sweep"
                )}
              </button>
            </div>

            {/* K-Sweep Loading Overlay */}
            {loading && operationType === "optimize" && (
              <div className="mt-6 p-6 bg-gradient-to-br from-indigo-50 to-purple-50 rounded-xl border border-indigo-100">
                <div className="flex items-start gap-6">
                  {/* Animated Icon */}
                  <div className="flex-shrink-0">
                    <div className="relative w-16 h-16">
                      <div className="absolute inset-0 border-4 border-indigo-200 rounded-full"></div>
                      <div className="absolute inset-0 border-4 border-indigo-600 rounded-full border-t-transparent animate-spin"></div>
                      <TrendingUp
                        className="absolute inset-0 m-auto text-indigo-600"
                        size={24}
                      />
                    </div>
                  </div>

                  {/* Progress Info */}
                  <div className="flex-1">
                    <h3 className="text-lg font-semibold text-gray-900 mb-2">
                      Finding Optimal Configuration...
                    </h3>
                    <p className="text-sm text-gray-600 mb-4">
                      Testing{" "}
                      {optConfig.algorithm === "all"
                        ? "all algorithms"
                        : ALGORITHM_NAMES[optConfig.algorithm]}{" "}
                      across K = {optConfig.k_min} to {optConfig.k_max}
                    </p>

                    {/* Progress Steps */}
                    <div className="space-y-2">
                      <div className="flex items-center gap-2 text-sm">
                        <div className="w-5 h-5 rounded-full bg-indigo-600 flex items-center justify-center">
                          <Loader2
                            className="text-white animate-spin"
                            size={12}
                          />
                        </div>
                        <span className="text-gray-700">
                          Running clustering for each K value...
                        </span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-gray-400">
                        <div className="w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center">
                          <span className="text-xs">2</span>
                        </div>
                        <span>Calculating quality metrics</span>
                      </div>
                      <div className="flex items-center gap-2 text-sm text-gray-400">
                        <div className="w-5 h-5 rounded-full bg-gray-200 flex items-center justify-center">
                          <span className="text-xs">3</span>
                        </div>
                        <span>Computing composite scores</span>
                      </div>
                    </div>

                    {/* Estimated Time & Cancel Button */}
                    <div className="mt-4 flex items-center justify-between">
                      <div className="flex items-center gap-2 text-xs text-indigo-600">
                        <RefreshCw className="animate-spin" size={12} />
                        <span>
                          This may take 1-3 minutes on free tier servers
                        </span>
                      </div>
                      {onCancel && (
                        <button
                          onClick={onCancel}
                          className="px-3 py-1.5 text-sm bg-red-100 hover:bg-red-200 text-red-700 rounded-lg flex items-center gap-2 transition-colors"
                        >
                          <XCircle size={14} />
                          Cancel
                        </button>
                      )}
                    </div>
                  </div>
                </div>

                {/* Animated Progress Bar */}
                <div className="mt-4 h-1.5 bg-indigo-100 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-indigo-600 rounded-full animate-pulse"
                    style={{
                      width: "60%",
                      animation:
                        "pulse 1.5s ease-in-out infinite, progressIndeterminate 2s ease-in-out infinite",
                    }}
                  ></div>
                </div>

                {/* Console Tip */}
                {/* <div className="mt-3 text-xs text-gray-500 text-center">
                  Open browser console (F12) to see detailed progress logs
                </div> */}
              </div>
            )}

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
            <div className="mt-4 pt-4 border-t border-gray-100 bg-amber-100 rounded-lg p-3">
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
                  <span>
                    Do you want to generate names for personas using AI?
                  </span>
                </label>
                <span className="text-xs text-gray-800">
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
              <div className="p-3 bg-indigo-50 rounded-lg text-sm text-indigo-700 mt-2">
                <strong>Recommended:</strong>{" "}
                {ALGORITHM_NAMES[bestConfig.algorithm]} with K={bestConfig.k}
                (score: {bestConfig.recalculated_score?.toFixed(4)})
              </div>
            )}
          </Card>

          {/* Clustering Loading Overlay */}
          {loading && operationType === "cluster" && !isGeneratingNames && (
            <Card>
              <div className="flex items-start gap-6 p-4">
                {/* Animated Icon */}
                <div className="flex-shrink-0">
                  <div className="relative w-20 h-20">
                    <div className="absolute inset-0 border-4 border-indigo-100 rounded-full"></div>
                    <div className="absolute inset-0 border-4 border-indigo-600 rounded-full border-t-transparent animate-spin"></div>
                    <div className="absolute inset-2 border-4 border-purple-100 rounded-full"></div>
                    <div
                      className="absolute inset-2 border-4 border-purple-500 rounded-full border-b-transparent animate-spin"
                      style={{
                        animationDirection: "reverse",
                        animationDuration: "1.5s",
                      }}
                    ></div>
                    <Layers
                      className="absolute inset-0 m-auto text-indigo-600"
                      size={28}
                    />
                  </div>
                </div>

                {/* Progress Info */}
                <div className="flex-1">
                  <h3 className="text-lg font-semibold text-gray-900 mb-1">
                    Running {ALGORITHM_NAMES[config.algorithm]} Clustering...
                  </h3>
                  <p className="text-sm text-gray-500 mb-4">
                    Creating {config.k} clusters from your dataset
                  </p>

                  {/* Progress Steps */}
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                    <div className="flex items-center gap-3 p-3 bg-indigo-50 rounded-lg">
                      <div className="w-8 h-8 rounded-full bg-indigo-600 flex items-center justify-center">
                        <Loader2
                          className="text-white animate-spin"
                          size={16}
                        />
                      </div>
                      <div>
                        <div className="text-sm font-medium text-gray-900">
                          Preprocessing Data
                        </div>
                        <div className="text-xs text-gray-500">
                          Normalizing & applying PCA
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                      <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                        <span className="text-sm text-gray-500">2</span>
                      </div>
                      <div>
                        <div className="text-sm font-medium text-gray-400">
                          Fitting Model
                        </div>
                        <div className="text-xs text-gray-400">
                          Assigning participants to clusters
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                      <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                        <span className="text-sm text-gray-500">3</span>
                      </div>
                      <div>
                        <div className="text-sm font-medium text-gray-400">
                          Calculating Metrics
                        </div>
                        <div className="text-xs text-gray-400">
                          Silhouette, η², balance scores
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center gap-3 p-3 bg-gray-50 rounded-lg">
                      <div className="w-8 h-8 rounded-full bg-gray-200 flex items-center justify-center">
                        <span className="text-sm text-gray-500">4</span>
                      </div>
                      <div>
                        <div className="text-sm font-medium text-gray-400">
                          Characterizing Personas
                        </div>
                        <div className="text-xs text-gray-400">
                          Extracting traits & behaviors
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Progress Bar & Cancel */}
                  <div className="mt-4">
                    <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                      <div
                        className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full transition-all duration-500"
                        style={{
                          width: "30%",
                          animation:
                            "progressIndeterminate 2s ease-in-out infinite",
                        }}
                      ></div>
                    </div>
                    <div className="mt-2 flex justify-between items-center text-xs text-gray-500">
                      <span>Processing... (may take 30-60s on free tier)</span>
                      {onCancel && (
                        <button
                          onClick={onCancel}
                          className="px-3 py-1.5 text-sm bg-red-100 hover:bg-red-200 text-red-700 rounded-lg flex items-center gap-2 transition-colors"
                        >
                          <XCircle size={14} />
                          Cancel
                        </button>
                      )}
                    </div>
                  </div>

                  {/* Console Tip */}
                  <div className="mt-3 text-xs text-gray-400 text-center">
                    Open browser console (F12) to see detailed progress logs
                  </div>
                </div>
              </div>
            </Card>
          )}

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
                    reference="≥0.14"
                    status={
                      clusteringResult.metrics?.eta_squared_mean >= 0.14
                        ? "success"
                        : clusteringResult.metrics?.eta_squared_mean >= 0.06
                          ? "warning"
                          : "danger"
                    }
                    interpretation={
                      clusteringResult.metrics?.eta_squared_mean >= 0.14
                        ? "Large effect - personas predict behavior well"
                        : clusteringResult.metrics?.eta_squared_mean >= 0.06
                          ? "Medium effect - consider increasing K or behavioral weight"
                          : "Small effect - re-run with different settings"
                    }
                  />
                  <MetricCard
                    label="Silhouette"
                    value={clusteringResult.metrics?.silhouette?.toFixed(3)}
                    reference="≥0.30"
                    status={
                      clusteringResult.metrics?.silhouette >= 0.3
                        ? "success"
                        : clusteringResult.metrics?.silhouette >= 0.1
                          ? "warning"
                          : "neutral"
                    }
                    interpretation={
                      clusteringResult.metrics?.silhouette >= 0.3
                        ? "Clear geometric separation"
                        : clusteringResult.metrics?.eta_squared_mean >= 0.14
                          ? "Acceptable if η² is strong (yours is ✓)"
                          : "Clusters overlap - consider fewer clusters"
                    }
                  />
                  <MetricCard
                    label="Min Cluster Size"
                    value={clusteringResult.metrics?.min_cluster_size}
                    reference={`≥${minClusterSize}`}
                    status={
                      clusteringResult.metrics?.min_cluster_size >=
                      minClusterSize
                        ? "success"
                        : "danger"
                    }
                    interpretation={
                      clusteringResult.metrics?.min_cluster_size >=
                      minClusterSize
                        ? "All personas statistically robust"
                        : "Smallest persona too small - reduce K"
                    }
                  />
                  <MetricCard
                    label="Size Balance"
                    value={clusteringResult.metrics?.size_balance?.toFixed(2)}
                    reference="≥0.50"
                    status={
                      clusteringResult.metrics?.size_balance >= 0.5
                        ? "success"
                        : "warning"
                    }
                    interpretation={
                      clusteringResult.metrics?.size_balance >= 0.5
                        ? "Personas reasonably balanced"
                        : "Uneven distribution - one persona may be absorbing outliers"
                    }
                  />
                </div>

                {/* Decision Summary */}
                {(() => {
                  const metrics = clusteringResult.metrics || {};
                  const etaOk = metrics.eta_squared_mean >= 0.14;
                  const etaWarning = metrics.eta_squared_mean >= 0.06;
                  const silhouetteOk = metrics.silhouette >= 0.3;
                  const sizeOk = metrics.min_cluster_size >= minClusterSize;
                  const balanceOk = metrics.size_balance >= 0.5;

                  const passCount = [
                    etaOk,
                    silhouetteOk || etaOk,
                    sizeOk,
                    balanceOk,
                  ].filter(Boolean).length;
                  const isReady = etaOk && sizeOk; // η² and size are critical

                  return (
                    <div
                      className={`mt-4 p-3 rounded-lg border ${
                        isReady
                          ? "bg-green-50 border-green-200"
                          : etaWarning && sizeOk
                            ? "bg-yellow-50 border-yellow-200"
                            : "bg-red-50 border-red-200"
                      }`}
                    >
                      <div className="flex items-center gap-2">
                        <span className="text-lg">
                          {isReady ? "✓" : etaWarning && sizeOk ? "⚠" : "✗"}
                        </span>
                        <div>
                          <span
                            className={`font-medium ${
                              isReady
                                ? "text-green-700"
                                : etaWarning && sizeOk
                                  ? "text-yellow-700"
                                  : "text-red-700"
                            }`}
                          >
                            {isReady
                              ? "Ready for Phase 2"
                              : etaWarning && sizeOk
                                ? "Acceptable - Review before proceeding"
                                : "Action Required"}
                          </span>
                          <span className="text-sm text-gray-500 ml-2">
                            ({passCount}/4 metrics passed)
                          </span>
                        </div>
                      </div>
                      {!isReady && (
                        <p className="text-sm mt-1 text-gray-600">
                          {!etaWarning
                            ? "η² too low - try reducing K or increasing behavioral weight"
                            : !sizeOk
                              ? "Cluster sizes too small - reduce K for larger personas"
                              : "Review metrics above for specific recommendations"}
                        </p>
                      )}
                    </div>
                  );
                })()}
              </Card>

              {/* Generated Personas */}
              <Card>
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center gap-4">
                    <h3 className="text-lg font-semibold">
                      Generated Personas
                    </h3>
                    <SystematicCodeLegend />
                  </div>
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

      {/* Section 4: Clustering Validation */}
      {activeSection === "validate" && (
        <div className="space-y-6">
          {/* Run Validation Button */}
          <Card>
            <div className="flex items-center justify-between">
              <div>
                <h3 className="text-lg font-semibold text-gray-900 flex items-center gap-2">
                  <FlaskConical size={20} className="text-emerald-600" />
                  Clustering Validation
                </h3>
                <p className="text-sm text-gray-500 mt-1">
                  Run comprehensive validation tests to verify clustering
                  quality
                </p>
              </div>
              <button
                onClick={onRunValidation}
                disabled={validationLoading || !clusteringResult}
                className={`px-5 py-2 rounded-lg text-sm font-medium text-white transition-colors flex items-center gap-2 ${
                  validationLoading
                    ? "bg-emerald-400 cursor-not-allowed"
                    : "bg-emerald-600 hover:bg-emerald-700"
                }`}
              >
                {validationLoading ? (
                  <>
                    <Loader2 className="animate-spin" size={16} />
                    Validating...
                  </>
                ) : (
                  <>
                    <FlaskConical size={16} />
                    Run Validation
                  </>
                )}
              </button>
            </div>
          </Card>

          {/* Validation Loading State with Step-by-Step Progress */}
          {validationLoading && (
            <Card>
              <div className="p-4">
                <div className="flex items-center gap-4 mb-6">
                  <div className="relative w-14 h-14">
                    <div className="absolute inset-0 border-4 border-emerald-200 rounded-full"></div>
                    <div className="absolute inset-0 border-4 border-emerald-600 rounded-full border-t-transparent animate-spin"></div>
                    <FlaskConical
                      className="absolute inset-0 m-auto text-emerald-600"
                      size={22}
                    />
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-gray-900">
                      Running Validation...
                    </h3>
                    <p className="text-sm text-gray-500">
                      Step {validationProgress?.currentStep || 1} of{" "}
                      {validationProgress?.steps?.length || 8} •
                      {validationProgress?.steps?.filter(
                        (s) => s.status === "completed",
                      ).length || 0}{" "}
                      completed
                    </p>
                  </div>
                </div>

                {/* Step-by-Step Progress List */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                  {(validationProgress?.steps || []).map((step) => {
                    const isRunning = step.status === "running";
                    const isCompleted = step.status === "completed";
                    const isFailed = step.status === "failed";
                    const isPending = step.status === "pending";

                    return (
                      <div
                        key={step.id}
                        className={`flex items-center gap-3 p-3 rounded-lg transition-all ${
                          isRunning
                            ? "bg-emerald-50 border border-emerald-200 shadow-sm"
                            : isCompleted
                              ? "bg-green-50 border border-green-200"
                              : isFailed
                                ? "bg-red-50 border border-red-200"
                                : "bg-gray-50 border border-gray-100"
                        }`}
                      >
                        {/* Step Number/Status Icon */}
                        <div
                          className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                            isRunning
                              ? "bg-emerald-600"
                              : isCompleted
                                ? "bg-green-600"
                                : isFailed
                                  ? "bg-red-500"
                                  : "bg-gray-200"
                          }`}
                        >
                          {isRunning ? (
                            <Loader2
                              className="text-white animate-spin"
                              size={14}
                            />
                          ) : isCompleted ? (
                            <CheckCircle className="text-white" size={14} />
                          ) : isFailed ? (
                            <XCircle className="text-white" size={14} />
                          ) : (
                            <span className="text-sm text-gray-500 font-medium">
                              {step.id}
                            </span>
                          )}
                        </div>

                        {/* Step Info */}
                        <div className="flex-1 min-w-0">
                          <div
                            className={`text-sm font-medium truncate ${
                              isRunning
                                ? "text-emerald-900"
                                : isCompleted
                                  ? "text-green-900"
                                  : isFailed
                                    ? "text-red-900"
                                    : "text-gray-400"
                            }`}
                          >
                            {step.name}
                          </div>
                          <div
                            className={`text-xs truncate ${
                              isRunning
                                ? "text-emerald-600"
                                : isCompleted
                                  ? "text-green-600"
                                  : isFailed
                                    ? "text-red-500"
                                    : "text-gray-400"
                            }`}
                          >
                            {isRunning
                              ? "Running..."
                              : isCompleted
                                ? "Completed ✓"
                                : isFailed
                                  ? "Failed ✗"
                                  : "Waiting..."}
                          </div>
                        </div>
                      </div>
                    );
                  })}
                </div>

                {/* Progress Bar */}
                <div className="mt-6">
                  <div className="flex justify-between text-xs text-gray-500 mb-2">
                    <span>Progress</span>
                    <span>
                      {Math.round(
                        ((validationProgress?.steps?.filter(
                          (s) => s.status === "completed",
                        ).length || 0) /
                          (validationProgress?.steps?.length || 8)) *
                          100,
                      )}
                      %
                    </span>
                  </div>
                  <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-emerald-600 rounded-full transition-all duration-500"
                      style={{
                        width: `${((validationProgress?.steps?.filter((s) => s.status === "completed").length || 0) / (validationProgress?.steps?.length || 8)) * 100}%`,
                      }}
                    ></div>
                  </div>
                </div>

                {/* Estimated Time */}
                <div className="mt-4 text-center text-xs text-gray-400">
                  Estimated time remaining: ~
                  {Math.max(
                    0,
                    ((validationProgress?.steps?.length || 8) -
                      (validationProgress?.steps?.filter(
                        (s) => s.status === "completed",
                      ).length || 0)) *
                      5,
                  )}
                  s
                </div>
              </div>
            </Card>
          )}

          {/* Validation Results */}
          {validationResult && !validationLoading && (
            <>
              {/* LLM Readiness Banner */}
              {validationResult.llm_readiness && (
                <div
                  className={`p-4 rounded-xl border ${
                    validationResult.llm_readiness.ready_for_llm
                      ? "bg-green-50 border-green-200"
                      : "bg-amber-50 border-amber-200"
                  }`}
                >
                  <div className="flex items-center gap-3">
                    {validationResult.llm_readiness.ready_for_llm ? (
                      <CheckCircle size={24} className="text-green-600" />
                    ) : (
                      <AlertTriangle size={24} className="text-amber-600" />
                    )}
                    <div>
                      <h3
                        className={`font-semibold ${
                          validationResult.llm_readiness.ready_for_llm
                            ? "text-green-900"
                            : "text-amber-900"
                        }`}
                      >
                        {validationResult.llm_readiness.ready_for_llm
                          ? "Ready for LLM Calibration"
                          : "Review Recommendations Before Proceeding"}
                      </h3>
                      <p
                        className={`text-sm ${
                          validationResult.llm_readiness.ready_for_llm
                            ? "text-green-700"
                            : "text-amber-700"
                        }`}
                      >
                        {validationResult.llm_readiness.strengths?.length || 0}{" "}
                        strengths,{" "}
                        {validationResult.llm_readiness.concerns?.length || 0}{" "}
                        concerns,{" "}
                        {validationResult.llm_readiness.blocking_issues
                          ?.length || 0}{" "}
                        blocking issues
                      </p>
                    </div>
                  </div>
                  {(validationResult.llm_readiness.blocking_issues?.length >
                    0 ||
                    validationResult.llm_readiness.concerns?.length > 0) && (
                    <div className="mt-3 pl-9">
                      {validationResult.llm_readiness.blocking_issues?.length >
                        0 && (
                        <>
                          <p className="text-sm font-medium text-red-700 mb-1">
                            Blocking Issues:
                          </p>
                          <ul className="list-disc list-inside text-sm text-red-600 space-y-1 mb-2">
                            {validationResult.llm_readiness.blocking_issues.map(
                              (issue, i) => (
                                <li key={i}>{issue}</li>
                              ),
                            )}
                          </ul>
                        </>
                      )}
                      {validationResult.llm_readiness.concerns?.length > 0 && (
                        <>
                          <p className="text-sm font-medium text-amber-700 mb-1">
                            Concerns:
                          </p>
                          <ul className="list-disc list-inside text-sm text-amber-600 space-y-1">
                            {validationResult.llm_readiness.concerns
                              .slice(0, 3)
                              .map((concern, i) => (
                                <li key={i}>{concern}</li>
                              ))}
                          </ul>
                        </>
                      )}
                    </div>
                  )}
                </div>
              )}

              {/* Quick Metrics */}
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <MetricCard
                  label="Behavioral (η²)"
                  value={
                    validationResult.quick?.eta_squared_mean?.toFixed(3) ||
                    "N/A"
                  }
                  reference="≥0.14"
                  status={
                    validationResult.quick?.eta_squared_mean >= 0.14
                      ? "success"
                      : validationResult.quick?.eta_squared_mean >= 0.06
                        ? "warning"
                        : "danger"
                  }
                  interpretation={
                    validationResult.quick?.eta_squared_mean >= 0.14
                      ? "Large effect - personas predict behavior"
                      : validationResult.quick?.eta_squared_mean >= 0.06
                        ? "Medium effect - may need adjustment"
                        : "Small effect - re-run clustering"
                  }
                />
                <MetricCard
                  label="Silhouette Score"
                  value={
                    validationResult.quick?.silhouette?.toFixed(3) || "N/A"
                  }
                  reference="≥0.30"
                  status={
                    validationResult.quick?.silhouette >= 0.3
                      ? "success"
                      : validationResult.quick?.silhouette >= 0.1
                        ? "warning"
                        : "neutral"
                  }
                  interpretation={
                    validationResult.quick?.silhouette >= 0.3
                      ? "Clear cluster separation"
                      : "Low but acceptable if η² is strong"
                  }
                />

                <MetricCard
                  label="Cross-Val Gap"
                  value={
                    validationResult.cross_validation?.generalization?.mean_gap?.toFixed(
                      3,
                    ) || "N/A"
                  }
                  reference="<0.05"
                  status={
                    Math.abs(
                      validationResult.cross_validation?.generalization
                        ?.mean_gap || 0,
                    ) < 0.05
                      ? "success"
                      : Math.abs(
                            validationResult.cross_validation?.generalization
                              ?.mean_gap || 0,
                          ) < 0.1
                        ? "warning"
                        : "danger"
                  }
                  interpretation={
                    Math.abs(
                      validationResult.cross_validation?.generalization
                        ?.mean_gap || 0,
                    ) < 0.05
                      ? "Generalizes well to new data"
                      : "May not generalize - consider simpler model"
                  }
                />
                <MetricCard
                  label="Low Uncertainty %"
                  value={
                    validationResult.soft_assignments?.uncertainty_distribution
                      ?.uncertainty_percentages?.low_pct
                      ? `${validationResult.soft_assignments.uncertainty_distribution.uncertainty_percentages.low_pct.toFixed(0)}%`
                      : "N/A"
                  }
                  reference="≥70%"
                  status={
                    (validationResult.soft_assignments?.uncertainty_distribution
                      ?.uncertainty_percentages?.low_pct || 0) >= 70
                      ? "success"
                      : (validationResult.soft_assignments
                            ?.uncertainty_distribution?.uncertainty_percentages
                            ?.low_pct || 0) >= 50
                        ? "warning"
                        : "danger"
                  }
                  interpretation={
                    (validationResult.soft_assignments?.uncertainty_distribution
                      ?.uncertainty_percentages?.low_pct || 0) >= 70
                      ? "Confident cluster assignments"
                      : "Many borderline cases - consider fewer clusters"
                  }
                />
              </div>

              {/* Gap Statistic */}
              {validationResult.gap_statistic && (
                <Card>
                  <h3 className="text-md font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <Target size={18} className="text-indigo-600" />
                    Gap Statistic Analysis
                  </h3>
                  <div className="flex items-center gap-6">
                    <div className="flex-1">
                      <p className="text-sm text-gray-600 mb-2">
                        The Gap Statistic compares your clustering to random
                        uniform data to find the optimal K.
                      </p>
                      <div className="flex items-center gap-4">
                        <div className="px-4 py-2 bg-indigo-50 rounded-lg">
                          <span className="text-xs text-gray-500">
                            Recommended K
                          </span>
                          <p className="text-2xl font-bold text-indigo-700">
                            {validationResult.gap_statistic.optimal_k}
                          </p>
                        </div>
                        <div className="px-4 py-2 bg-gray-50 rounded-lg">
                          <span className="text-xs text-gray-500">
                            Current K
                          </span>
                          <p className="text-2xl font-bold text-gray-700">
                            {clusteringResult?.k || config.k}
                          </p>
                        </div>
                        {validationResult.gap_statistic.optimal_k !==
                          (clusteringResult?.k || config.k) && (
                          <div className="px-3 py-1 bg-amber-100 text-amber-800 text-sm rounded-lg flex items-center gap-2">
                            <AlertTriangle size={14} />
                            Consider trying K=
                            {validationResult.gap_statistic.optimal_k}
                          </div>
                        )}
                      </div>
                    </div>
                    {validationResult.gap_statistic.results_by_k && (
                      <div className="w-64 h-32">
                        <ResponsiveContainer width="100%" height="100%">
                          <LineChart
                            data={validationResult.gap_statistic.results_by_k.map(
                              (r) => ({
                                k: r.k,
                                gap: r.gap,
                              }),
                            )}
                          >
                            <CartesianGrid
                              strokeDasharray="3 3"
                              stroke="#e5e7eb"
                            />
                            <XAxis dataKey="k" fontSize={10} />
                            <YAxis fontSize={10} />
                            <RechartsTooltip
                              formatter={(value) => value.toFixed(3)}
                            />
                            <Line
                              type="monotone"
                              dataKey="gap"
                              stroke="#6366f1"
                              strokeWidth={2}
                              dot={{ r: 3 }}
                            />
                          </LineChart>
                        </ResponsiveContainer>
                      </div>
                    )}
                  </div>
                </Card>
              )}

              {/* Feature Importance */}
              {validationResult.feature_importance && (
                <Card>
                  <h3 className="text-md font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <BarChart3 size={18} className="text-purple-600" />
                    Feature Importance (Top 10)
                  </h3>
                  <p className="text-sm text-gray-500 mb-4">
                    Shows which psychological traits contribute most to cluster
                    separation.
                  </p>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart
                        data={(
                          validationResult.feature_importance
                            .feature_importance || []
                        )
                          .slice(0, 10)
                          .map((f) => ({
                            feature: f.feature_name,
                            importance: f.combined_importance,
                          }))}
                        layout="vertical"
                        margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
                      >
                        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                        <XAxis type="number" fontSize={10} domain={[0, 1]} />
                        <YAxis
                          type="category"
                          dataKey="feature"
                          fontSize={10}
                          width={110}
                        />
                        <RechartsTooltip
                          formatter={(value) => value.toFixed(3)}
                        />
                        <Bar
                          dataKey="importance"
                          fill="#8b5cf6"
                          radius={[0, 4, 4, 0]}
                        >
                          {(
                            validationResult.feature_importance
                              .feature_importance || []
                          )
                            .slice(0, 10)
                            .map((entry, index) => (
                              <Cell
                                key={`cell-${index}`}
                                fill={
                                  index < 3
                                    ? "#7c3aed"
                                    : index < 6
                                      ? "#a78bfa"
                                      : "#c4b5fd"
                                }
                              />
                            ))}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                  {validationResult.feature_importance.category_importance && (
                    <div className="mt-4 pt-4 border-t border-gray-100">
                      <h4 className="text-sm font-medium text-gray-700 mb-2">
                        Category Importance
                      </h4>
                      <div className="flex flex-wrap gap-2">
                        {Object.entries(
                          validationResult.feature_importance
                            .category_importance,
                        )
                          .sort(
                            (a, b) =>
                              (b[1]?.mean_importance || 0) -
                              (a[1]?.mean_importance || 0),
                          )
                          .map(([cat, catData]) => (
                            <span
                              key={cat}
                              className="px-3 py-1 bg-purple-50 text-purple-700 rounded-full text-sm"
                            >
                              {cat}:{" "}
                              {((catData?.mean_importance || 0) * 100).toFixed(
                                0,
                              )}
                              %
                            </span>
                          ))}
                      </div>
                    </div>
                  )}
                </Card>
              )}

              {/* Cross-Validation Details */}
              {validationResult.cross_validation && (
                <Card>
                  <h3 className="text-md font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <RefreshCw size={18} className="text-blue-600" />
                    Cross-Validation Results
                  </h3>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-3 bg-blue-50 rounded-lg">
                      <span className="text-xs text-gray-500">
                        Train Silhouette
                      </span>
                      <p className="text-lg font-semibold text-blue-700">
                        {validationResult.cross_validation.summary?.train_silhouette?.mean?.toFixed(
                          3,
                        ) || "N/A"}
                      </p>
                    </div>
                    <div className="p-3 bg-blue-50 rounded-lg">
                      <span className="text-xs text-gray-500">
                        Test Silhouette
                      </span>
                      <p className="text-lg font-semibold text-blue-700">
                        {validationResult.cross_validation.summary?.test_silhouette?.mean?.toFixed(
                          3,
                        ) || "N/A"}
                      </p>
                    </div>
                    <div className="p-3 bg-blue-50 rounded-lg">
                      <span className="text-xs text-gray-500">
                        Generalization Gap
                      </span>
                      <p
                        className={`text-lg font-semibold ${
                          Math.abs(
                            validationResult.cross_validation.generalization
                              ?.mean_gap || 0,
                          ) < 0.05
                            ? "text-green-700"
                            : "text-amber-700"
                        }`}
                      >
                        {validationResult.cross_validation.generalization?.mean_gap?.toFixed(
                          3,
                        ) || "N/A"}
                      </p>
                    </div>
                    <div className="p-3 bg-blue-50 rounded-lg">
                      <span className="text-xs text-gray-500">Std Dev</span>
                      <p className="text-lg font-semibold text-blue-700">
                        ±
                        {validationResult.cross_validation.summary?.test_silhouette?.std?.toFixed(
                          3,
                        ) || "N/A"}
                      </p>
                    </div>
                  </div>
                  <p className="mt-3 text-sm text-gray-500">
                    <Info size={14} className="inline mr-1" />A small
                    generalization gap (&lt;0.05) indicates clustering
                    generalizes well to unseen data.
                  </p>
                </Card>
              )}

              {/* Prediction Power */}
              {validationResult.prediction_error && (
                <Card>
                  <h3 className="text-md font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <TrendingUp size={18} className="text-green-600" />
                    Behavioral Prediction Power
                  </h3>
                  <div className="grid grid-cols-3 gap-4">
                    <div className="p-3 bg-green-50 rounded-lg">
                      <span className="text-xs text-gray-500">R² Score</span>
                      <p className="text-lg font-semibold text-green-700">
                        {validationResult.prediction_error.aggregated?.mean_r_squared?.toFixed(
                          3,
                        ) || "N/A"}
                      </p>
                      <span className="text-xs text-gray-400">
                        Improvement:{" "}
                        {validationResult.prediction_error.baseline_comparison?.summary?.mean_improvement_over_grand_pct?.toFixed(
                          1,
                        ) || "0"}
                        % over baseline
                      </span>
                    </div>
                    <div className="p-3 bg-green-50 rounded-lg">
                      <span className="text-xs text-gray-500">MAE</span>
                      <p className="text-lg font-semibold text-green-700">
                        {validationResult.prediction_error.aggregated?.mean_mae?.toFixed(
                          3,
                        ) || "N/A"}
                      </p>
                      <span className="text-xs text-gray-400">
                        {validationResult.prediction_error.baseline_comparison
                          ?.summary?.all_outcomes_beat_grand_mean
                          ? "✓ Beats grand mean baseline"
                          : "✗ Doesn't beat baseline"}
                      </span>
                    </div>
                    <div className="p-3 bg-green-50 rounded-lg">
                      <span className="text-xs text-gray-500">
                        Beats Baselines?
                      </span>
                      <p
                        className={`text-lg font-semibold ${
                          validationResult.prediction_error.baseline_comparison
                            ?.summary?.beats_all_baselines
                            ? "text-green-700"
                            : "text-amber-700"
                        }`}
                      >
                        {validationResult.prediction_error.baseline_comparison
                          ?.summary?.beats_all_baselines
                          ? "Yes"
                          : "No"}
                      </p>
                      <span className="text-xs text-gray-400">
                        {validationResult.prediction_error.baseline_comparison
                          ?.summary?.interpretation || ""}
                      </span>
                    </div>
                  </div>
                </Card>
              )}

              {/* Soft Assignments / Uncertainty */}
              {validationResult.soft_assignments && (
                <Card>
                  <h3 className="text-md font-semibold text-gray-900 mb-4 flex items-center gap-2">
                    <Layers size={18} className="text-amber-600" />
                    Cluster Assignment Uncertainty
                  </h3>
                  <p className="text-sm text-gray-500 mb-4">
                    Shows how confident the algorithm is about each
                    participant's cluster assignment.
                  </p>
                  <div className="flex items-center gap-4">
                    {validationResult.soft_assignments.uncertainty_distribution
                      ?.uncertainty_percentages && (
                      <>
                        <div className="flex-1 bg-gray-100 rounded-full h-6 overflow-hidden flex">
                          <div
                            className="bg-green-500 h-full flex items-center justify-center text-xs text-white font-medium"
                            style={{
                              width: `${validationResult.soft_assignments.uncertainty_distribution.uncertainty_percentages.low_pct || 0}%`,
                            }}
                          >
                            {(
                              validationResult.soft_assignments
                                .uncertainty_distribution
                                .uncertainty_percentages.low_pct || 0
                            ).toFixed(0)}
                            % Low
                          </div>
                          <div
                            className="bg-amber-500 h-full flex items-center justify-center text-xs text-white font-medium"
                            style={{
                              width: `${validationResult.soft_assignments.uncertainty_distribution.uncertainty_percentages.moderate_pct || 0}%`,
                            }}
                          >
                            {(
                              validationResult.soft_assignments
                                .uncertainty_distribution
                                .uncertainty_percentages.moderate_pct || 0
                            ).toFixed(0)}
                            % Med
                          </div>
                          <div
                            className="bg-red-500 h-full flex items-center justify-center text-xs text-white font-medium"
                            style={{
                              width: `${validationResult.soft_assignments.uncertainty_distribution.uncertainty_percentages.high_pct || 0}%`,
                            }}
                          >
                            {(
                              validationResult.soft_assignments
                                .uncertainty_distribution
                                .uncertainty_percentages.high_pct || 0
                            ).toFixed(0)}
                            % High
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                  <div className="mt-4 grid grid-cols-3 gap-4 text-center text-sm">
                    <div>
                      <span className="inline-block w-3 h-3 rounded bg-green-500 mr-1"></span>
                      <strong>Low uncertainty:</strong> Clear cluster membership
                    </div>
                    <div>
                      <span className="inline-block w-3 h-3 rounded bg-amber-500 mr-1"></span>
                      <strong>Medium:</strong> Between clusters
                    </div>
                    <div>
                      <span className="inline-block w-3 h-3 rounded bg-red-500 mr-1"></span>
                      <strong>High:</strong> Boundary cases
                    </div>
                  </div>
                </Card>
              )}

              {/* Cluster Visualization */}
              {validationResult.cluster_visualization && (
                <ClusterVisualizationChart
                  data={validationResult.cluster_visualization}
                />
              )}
            </>
          )}

          {/* Empty State */}
          {!validationResult && !validationLoading && (
            <EmptyState
              icon={<FlaskConical size={48} />}
              title="Run validation to see results"
              description="Click 'Run Validation' to analyze clustering quality with scientific methods including gap statistic, feature importance, and cross-validation."
            />
          )}
        </div>
      )}
    </div>
  );
};

export default ClusteringTab;
