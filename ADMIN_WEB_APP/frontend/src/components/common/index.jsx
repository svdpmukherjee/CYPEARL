/**
 * CYPEARL Phase 1 Dashboard - Common Reusable Components
 */

import React, { useState } from "react";
import {
  RefreshCw,
  ChevronDown,
  ChevronUp,
  AlertTriangle,
  Info,
} from "lucide-react";
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

export const MetricCard = ({
  label,
  value,
  status = "neutral",
  reference = null, // e.g., "≥0.14"
  interpretation = null, // e.g., "Large effect size"
}) => {
  const statusColors = {
    success: "bg-green-50 border-green-200 text-green-700",
    warning: "bg-yellow-50 border-yellow-200 text-yellow-700",
    danger: "bg-red-50 border-red-200 text-red-700",
    neutral: "bg-gray-50 border-gray-200 text-gray-700",
  };

  const statusIcons = {
    success: "✓",
    warning: "⚠",
    danger: "✗",
    neutral: "○",
  };

  return (
    <div className={`p-3 rounded-lg border ${statusColors[status]}`}>
      <div className="flex items-center justify-between">
        <div className="text-sm text-gray-500">{label}</div>
        {reference && (
          <div className="text-xs text-gray-400">target: {reference}</div>
        )}
      </div>
      <div className="flex items-baseline gap-2">
        <div className="text-xl font-bold">{value}</div>
        <span className="text-sm">{statusIcons[status]}</span>
      </div>
      {interpretation && (
        <div className="text-xs mt-1 opacity-80">{interpretation}</div>
      )}
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

  // Extract AI-generated name only (no systematic code)
  let aiName = personaLabel?.llm_name;
  if (!aiName && personaLabel?.name && personaLabel.name.includes(": ")) {
    aiName = personaLabel.name.split(": ").slice(1).join(": ");
  }
  const displayName =
    aiName ||
    personaLabel?.name ||
    cluster.label ||
    `Persona ${cluster.cluster_id + 1}`;
  const systematicCode = personaLabel?.archetype;

  // Clean description: remove "Click rate: X%, N=Y" or "Click rate: X%, Report rate: Y%, N=Z" suffix
  let description = personaLabel?.description || cluster.description || "";
  description = description
    .replace(
      /\s*Click rate:\s*[\d.]+%,?\s*(Report rate:\s*[\d.]+%,?)?\s*N=\d+\.?/gi,
      "",
    )
    .trim();

  return (
    <div
      className={`p-4 rounded-lg border ${isSmall ? "border-red-300 bg-red-50" : "border-gray-200"}`}
    >
      {/* Risk indicator and N at the top */}
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
        {systematicCode && (
          <span className="text-xs text-gray-500 ml-1">({systematicCode})</span>
        )}
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

      <p className="text-sm text-gray-600">{description}</p>

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
// SYSTEMATIC CODE LEGEND (Overlay Popover)
// ============================================================================

export const SystematicCodeLegend = () => {
  const [showPopover, setShowPopover] = useState(false);

  return (
    <div className="relative inline-block">
      <button
        onClick={() => setShowPopover(!showPopover)}
        onBlur={() => setTimeout(() => setShowPopover(false), 200)}
        className="inline-flex items-center gap-1 text-xs text-indigo-600 hover:text-indigo-800 transition-colors"
      >
        <Info size={14} />
        <span>Code Legend</span>
      </button>

      {showPopover && (
        <div className="absolute z-50 left-0 top-full mt-1 w-[600px] max-w-[90vw] p-3 bg-white rounded-lg border border-indigo-200 shadow-xl">
          <p className="text-xs text-indigo-700 mb-2">
            Systematic code (e.g.,{" "}
            <span className="font-mono bg-indigo-100 px-1 rounded">
              LO-ANL-CUR-RPT
            </span>
            ) encodes:
          </p>
          <div className="grid grid-cols-4 gap-2 text-xs">
            <div className="bg-indigo-50 p-2 rounded">
              <span className="font-semibold text-indigo-800">Risk</span>
              <ul className="text-indigo-600 mt-1">
                <li>
                  <span className="font-mono">CR</span>=Critical
                </li>
                <li>
                  <span className="font-mono">HI</span>=High
                </li>
                <li>
                  <span className="font-mono">MD</span>=Medium
                </li>
                <li>
                  <span className="font-mono">LO</span>=Low
                </li>
              </ul>
            </div>
            <div className="bg-indigo-50 p-2 rounded">
              <span className="font-semibold text-indigo-800">Cognitive</span>
              <ul className="text-indigo-600 mt-1">
                <li>
                  <span className="font-mono">INT</span>=Intuitive
                </li>
                <li>
                  <span className="font-mono">ANL</span>=Analytical
                </li>
                <li>
                  <span className="font-mono">BAL</span>=Balanced
                </li>
              </ul>
            </div>
            <div className="bg-indigo-50 p-2 rounded">
              <span className="font-semibold text-indigo-800">Trait</span>
              <ul className="text-indigo-600 mt-1">
                <li>
                  <span className="font-mono">IMP</span>=Impulsive
                </li>
                <li>
                  <span className="font-mono">TRU</span>=Trusting
                </li>
                <li>
                  <span className="font-mono">CUR</span>=Curious
                </li>
                <li>
                  <span className="font-mono">ANX</span>=Anxious
                </li>
                <li>
                  <span className="font-mono">SKP</span>=Skeptical
                </li>
                <li>
                  <span className="font-mono">CMP</span>=Compliant
                </li>
                <li>
                  <span className="font-mono">CNF</span>=Confident
                </li>
                <li>
                  <span className="font-mono">OVR</span>=Overwhelmed
                </li>
                <li>
                  <span className="font-mono">CTN</span>=Cautious
                </li>
                <li>
                  <span className="font-mono">DST</span>=Distracted
                </li>
              </ul>
            </div>
            <div className="bg-indigo-50 p-2 rounded">
              <span className="font-semibold text-indigo-800">Behavior</span>
              <ul className="text-indigo-600 mt-1">
                <li>
                  <span className="font-mono">CLK</span>=Clicker
                </li>
                <li>
                  <span className="font-mono">RPT</span>=Reporter
                </li>
                <li>
                  <span className="font-mono">IGN</span>=Ignorer
                </li>
                <li>
                  <span className="font-mono">HST</span>=Hesitant
                </li>
                <li>
                  <span className="font-mono">INS</span>=Inspector
                </li>
              </ul>
            </div>
          </div>
        </div>
      )}
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
