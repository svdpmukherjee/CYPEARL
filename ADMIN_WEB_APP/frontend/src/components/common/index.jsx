/**
 * CYPEARL Phase 1 Dashboard - Common Reusable Components
 */

import React, { useState } from "react";
import { RefreshCw, ChevronDown, ChevronUp, AlertTriangle } from "lucide-react";
import { RISK_COLORS } from "../../constants";

// ============================================================================
// LAYOUT COMPONENTS
// ============================================================================

export const Card = ({ children, className = "" }) => (
  <div
    className={`bg-white rounded-xl border border-gray-200 p-6 shadow-sm ${className}`}
  >
    {children}
  </div>
);

export const ChartCard = ({ title, subtitle, children }) => (
  <Card>
    <div className="mb-4">
      <h3 className="text-md font-semibold text-gray-900">{title}</h3>
      {subtitle && <p className="text-sm text-gray-500">{subtitle}</p>}
    </div>
    {children}
  </Card>
);

// ============================================================================
// FEEDBACK COMPONENTS
// ============================================================================

export const EmptyState = ({ icon, title, description, action }) => (
  <div className="flex flex-col items-center justify-center py-16 text-center">
    <div className="text-gray-300 mb-4">{icon}</div>
    <h3 className="text-lg font-semibold text-gray-600 mb-2">{title}</h3>
    <p className="text-sm text-gray-500 max-w-md mb-4">{description}</p>
    {action}
  </div>
);

export const LoadingSpinner = () => (
  <div className="flex items-center justify-center py-8">
    <RefreshCw className="animate-spin text-indigo-600" size={32} />
  </div>
);

// ============================================================================
// DATA DISPLAY COMPONENTS
// ============================================================================

export const MetricCard = ({ label, value, status = "neutral" }) => {
  const statusColors = {
    success: "bg-green-50 border-green-200 text-green-700",
    warning: "bg-yellow-50 border-yellow-200 text-yellow-700",
    danger: "bg-red-50 border-red-200 text-red-700",
    neutral: "bg-gray-50 border-gray-200 text-gray-700",
  };

  return (
    <div className={`p-3 rounded-lg border ${statusColors[status]}`}>
      <div className="text-sm text-gray-500">{label}</div>
      <div className="text-xl font-bold">{value}</div>
    </div>
  );
};

export const ExpandableCard = ({
  title,
  value,
  icon,
  children,
  defaultExpanded = false,
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <div className="border rounded-lg overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full p-4 flex items-center justify-between bg-gray-50 hover:bg-gray-100 transition"
      >
        <div className="flex items-center gap-3">
          <span className="text-indigo-600">{icon}</span>
          <span className="font-medium text-gray-900">{title}</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-2xl font-bold text-indigo-600">{value}</span>
          {expanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
        </div>
      </button>
      {expanded && <div className="p-4 border-t bg-white">{children}</div>}
    </div>
  );
};

export const ClusterCard = ({ cluster, minClusterSize, personaLabel }) => {
  const riskColors = RISK_COLORS[cluster.risk_level] || RISK_COLORS.MEDIUM;
  const isSmall = cluster.n_participants < minClusterSize;

  // Use AI-generated name if available, otherwise fall back to default
  const displayName =
    personaLabel?.name || cluster.label || `Persona ${cluster.cluster_id + 1}`;

  return (
    <div
      className={`p-4 rounded-lg border ${isSmall ? "border-red-300 bg-red-50" : "border-gray-200"}`}
    >
      {/* Risk indicator at the top for better UI */}
      <div className="flex items-center justify-between mb-2">
        <span
          className="px-2 py-0.5 text-xs font-medium rounded"
          style={{
            backgroundColor: riskColors.bg,
            color: riskColors.text,
            border: `1px solid ${riskColors.border}`,
          }}
        >
          {cluster.risk_level}
        </span>
        <div className="text-xs text-gray-500">
          N=
          <span className="text-sm font-medium text-gray-800">
            {cluster.n_participants}
          </span>
        </div>
      </div>
      <div className="mb-3">
        <span className="font-semibold text-gray-900">{displayName}</span>
      </div>

      <div className="flex items-center gap-4 mb-3 text-sm">
        <div>
          <span className="text-gray-500">Click Rate: </span>
          <span className="font-medium text-red-600">
            {(cluster.phishing_click_rate * 100).toFixed(1)}%
          </span>
        </div>
        <div>
          <span className="text-gray-500">Report Rate: </span>
          <span className="font-medium text-green-600">
            {(
              (cluster.behavioral_outcomes?.report_rate?.mean || 0) * 100
            ).toFixed(1)}
            %
          </span>
        </div>
      </div>

      <p className="text-sm text-gray-600 line-clamp-2">
        {cluster.description}
      </p>

      {isSmall && (
        <div className="mt-2 text-xs text-red-600 flex items-center gap-1">
          <AlertTriangle size={12} />
          Below minimum size ({minClusterSize})
        </div>
      )}
    </div>
  );
};

// ============================================================================
// INPUT COMPONENTS
// ============================================================================

export const WeightInput = ({
  label,
  value,
  onChange,
  color,
  description,
  normalizedValue,
}) => (
  <div>
    <div className="flex justify-between items-center mb-1">
      <span className="text-sm font-medium text-gray-700">{label}</span>
      <span className="text-xs text-gray-500">
        {(value * 100).toFixed(0)}% → {(normalizedValue * 100).toFixed(0)}%
      </span>
    </div>
    <input
      type="range"
      min="0"
      max="1"
      step="0.05"
      value={value}
      onChange={(e) => onChange(e.target.value)}
      className={`w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-${color}-600`}
    />
    <p className="text-xs text-gray-500 mt-1">{description}</p>
  </div>
);

// ============================================================================
// CHART COMPONENTS
// ============================================================================

export const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload || !payload.length) return null;
  return (
    <div className="bg-white border border-gray-200 rounded-lg shadow-lg p-3 text-sm">
      <p className="font-medium text-gray-900 mb-1">{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{ color: entry.color }}>
          {entry.name}:{" "}
          {typeof entry.value === "number"
            ? entry.value.toFixed(3)
            : entry.value}
        </p>
      ))}
    </div>
  );
};

// ============================================================================
// PROCESS METRIC CARD (for Validation Tab)
// ============================================================================

export const ProcessMetricCard = ({ icon, label, description, byCluster }) => (
  <div className="p-4 bg-gray-50 rounded-lg">
    <div className="flex items-center gap-2 mb-2">
      <span className="text-indigo-600">{icon}</span>
      <span className="font-medium text-gray-900">{label}</span>
    </div>
    <p className="text-xs text-gray-500">{description}</p>
    {byCluster && (
      <div className="mt-2 text-xs text-indigo-600">Available per cluster</div>
    )}
  </div>
);
