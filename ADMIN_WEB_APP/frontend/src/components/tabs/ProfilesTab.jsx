/**
 * Persona Profiles Tab - Redesigned for Comparison
 *
 * Focus: Help admins compare ALL personas at a glance before diving into details
 *
 * Sections:
 * 1. Persona Overview Table - Side-by-side comparison of key metrics
 * 2. Trait Profile Heatmap - Visual comparison of psychological traits
 * 3. Behavioral Outcomes Comparison - Bar chart comparison
 * 4. Selected Persona Details - Expandable detail panel (optional)
 *
 * NEW: Hierarchical Taxonomy View
 * - Toggle between flat comparison and hierarchical taxonomy
 * - Hierarchical view shows personas organized by cognitive style and risk level
 * - Helps business managers see the "big picture" and make strategic decisions
 */

import React, { useState, useMemo, useEffect } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  Users,
  ChevronDown,
  ChevronRight,
  EyeOff,
  Network,
  List,
  AlertTriangle,
  Shield,
  Brain,
  Zap,
  Scale,
  RefreshCw,
} from "lucide-react";
import {
  Card,
  EmptyState,
  CustomTooltip,
  SystematicCodeLegend,
} from "../common";
import {
  TRAIT_CATEGORIES,
  TRAIT_LABELS,
  CATEGORY_LABELS,
  OUTCOME_LABELS,
} from "../../constants";
import { getFlatTaxonomy } from "../../services/api";

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

const getZScoreColor = (z) => {
  if (z >= 1.5) return "#dc2626"; // Strong red
  if (z >= 1.0) return "#ef4444"; // Red
  if (z >= 0.5) return "#fca5a5"; // Light red
  if (z <= -1.5) return "#1d4ed8"; // Strong blue
  if (z <= -1.0) return "#3b82f6"; // Blue
  if (z <= -0.5) return "#93c5fd"; // Light blue
  return "#f3f4f6"; // Neutral gray
};

const getRiskBadge = (riskLevel) => {
  const colors = {
    CRITICAL: "bg-red-100 text-red-700 border-red-300",
    HIGH: "bg-orange-100 text-orange-700 border-orange-300",
    MEDIUM: "bg-yellow-100 text-yellow-700 border-yellow-300",
    LOW: "bg-green-100 text-green-700 border-green-300",
  };
  return colors[riskLevel] || colors.MEDIUM;
};

// ============================================================================
// SUB-COMPONENTS
// ============================================================================

/**
 * Compact Trait Heatmap - Clusters as columns, traits as rows
 */
const TraitHeatmap = ({
  clusters,
  selectedCluster,
  onSelectCluster,
  personaLabels,
}) => {
  const [expandedCategories, setExpandedCategories] = useState({});

  const toggleCategory = (cat) => {
    setExpandedCategories((prev) => ({ ...prev, [cat]: !prev[cat] }));
  };

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs">
        <thead className="sticky top-0 z-10">
          <tr className="bg-gray-50">
            <th className="sticky left-0 bg-gray-50 px-3 py-2 text-left font-medium text-gray-500 border-b min-w-[140px]">
              Trait / Cluster →
            </th>
            {clusters.map((c) => {
              const label = personaLabels?.[c.cluster_id];
              // Extract AI-generated name only (no systematic code)
              let aiName = label?.llm_name;
              if (!aiName && label?.name && label.name.includes(": ")) {
                aiName = label.name.split(": ").slice(1).join(": ");
              }
              const displayName =
                aiName || label?.name || `Persona ${c.cluster_id + 1}`;
              return (
                <th
                  key={c.cluster_id}
                  onClick={() =>
                    onSelectCluster(
                      selectedCluster === c.cluster_id ? null : c.cluster_id,
                    )
                  }
                  className={`px-2 py-2 text-center font-medium border-b cursor-pointer transition min-w-[80px] max-w-[100px] ${
                    selectedCluster === c.cluster_id
                      ? "bg-indigo-100 text-indigo-700"
                      : "hover:bg-gray-100 text-gray-700"
                  }`}
                  title={displayName}
                >
                  <div className="text-[10px] leading-tight break-words whitespace-normal">
                    {displayName}
                  </div>
                  <div
                    className={`text-[9px] font-normal px-1 py-0.5 rounded mt-0.5 ${getRiskBadge(c.risk_level)}`}
                  >
                    {c.risk_level}
                  </div>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {Object.entries(TRAIT_CATEGORIES).map(([category, traits]) => (
            <React.Fragment key={category}>
              {/* Category header - clickable to expand/collapse */}
              <tr
                className="cursor-pointer hover:bg-gray-100"
                onClick={() => toggleCategory(category)}
              >
                <td
                  colSpan={clusters.length + 1}
                  className="sticky left-0 bg-gray-100 px-3 py-1.5 text-[10px] font-bold uppercase text-gray-600 tracking-wide"
                >
                  <div className="flex items-center gap-2">
                    {expandedCategories[category] ? (
                      <ChevronDown size={12} />
                    ) : (
                      <ChevronRight size={12} />
                    )}
                    {CATEGORY_LABELS[category]} ({traits.length})
                  </div>
                </td>
              </tr>
              {/* Trait rows - only show if expanded */}
              {expandedCategories[category] &&
                traits.map((trait) => (
                  <tr key={trait} className="hover:bg-gray-50">
                    <td className="sticky left-0 bg-white px-3 py-1 text-gray-600 border-b whitespace-nowrap">
                      {TRAIT_LABELS[trait] || trait}
                    </td>
                    {clusters.map((c) => {
                      const z = c.trait_zscores?.[trait] || 0;
                      return (
                        <td
                          key={c.cluster_id}
                          className={`px-1 py-1 text-center border-b ${
                            selectedCluster === c.cluster_id
                              ? "ring-1 ring-indigo-400"
                              : ""
                          }`}
                          style={{ backgroundColor: getZScoreColor(z) }}
                          title={`${TRAIT_LABELS[trait]}: z=${z.toFixed(2)}`}
                        >
                          <span
                            className={
                              Math.abs(z) > 0.75
                                ? "text-white font-medium"
                                : "text-gray-700"
                            }
                          >
                            {z.toFixed(1)}
                          </span>
                        </td>
                      );
                    })}
                  </tr>
                ))}
            </React.Fragment>
          ))}
        </tbody>
      </table>
    </div>
  );
};

/**
 * Hierarchical Taxonomy View Component
 *
 * Displays personas in a tree structure:
 * - Level 1: Meta-types (Analytical vs Intuitive vs Balanced)
 * - Level 2: Risk profiles (Critical/High/Medium/Low)
 * - Level 3: Individual personas
 */
const HierarchicalTaxonomyView = ({
  clusters,
  personaLabels,
  onSelectPersona,
}) => {
  const [taxonomyData, setTaxonomyData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [expandedNodes, setExpandedNodes] = useState({});

  useEffect(() => {
    loadTaxonomy();
  }, [clusters]);

  const loadTaxonomy = async () => {
    if (!clusters || clusters.length === 0) return;

    setLoading(true);
    setError(null);
    try {
      const response = await getFlatTaxonomy();
      if (response.status === "success") {
        setTaxonomyData(response);
        // Expand root and meta-types by default
        const initialExpanded = {};
        response.nodes.forEach((node) => {
          if (node.level <= 1) {
            initialExpanded[node.id] = true;
          }
        });
        setExpandedNodes(initialExpanded);
      }
    } catch (e) {
      console.error("Failed to load taxonomy:", e);
      setError(e.message || "Failed to load hierarchical taxonomy");
    } finally {
      setLoading(false);
    }
  };

  const toggleNode = (nodeId) => {
    setExpandedNodes((prev) => ({
      ...prev,
      [nodeId]: !prev[nodeId],
    }));
  };

  const getCognitiveStyleIcon = (style) => {
    switch (style) {
      case "analytical":
        return <Brain size={16} className="text-blue-600" />;
      case "intuitive":
        return <Zap size={16} className="text-orange-500" />;
      case "balanced":
        return <Scale size={16} className="text-purple-500" />;
      default:
        return <Users size={16} className="text-gray-500" />;
    }
  };

  const getRiskIcon = (risk) => {
    switch (risk) {
      case "critical":
        return <AlertTriangle size={14} className="text-red-600" />;
      case "high":
        return <AlertTriangle size={14} className="text-orange-500" />;
      case "medium":
        return <Shield size={14} className="text-yellow-500" />;
      case "low":
        return <Shield size={14} className="text-green-500" />;
      default:
        return null;
    }
  };

  const renderNode = (node, allNodes) => {
    const isExpanded = expandedNodes[node.id];
    const children = allNodes.filter((n) => n.parent_id === node.id);
    const hasChildren = children.length > 0;
    const isPersona = node.level === 3;

    // Indentation based on depth
    const indent = node.depth * 24;

    return (
      <div key={node.id} className="select-none">
        <div
          className={`flex items-center gap-2 py-2 px-3 rounded-lg cursor-pointer transition-all ${
            isPersona
              ? "hover:bg-indigo-50 ml-1"
              : hasChildren
                ? "hover:bg-gray-100"
                : "hover:bg-gray-50"
          }`}
          style={{ marginLeft: `${indent}px` }}
          onClick={() => {
            if (isPersona) {
              onSelectPersona(node.cluster_id);
            } else if (hasChildren) {
              toggleNode(node.id);
            }
          }}
        >
          {/* Expand/Collapse indicator */}
          {hasChildren ? (
            <span className="w-5 h-5 flex items-center justify-center text-gray-400">
              {isExpanded ? (
                <ChevronDown size={16} />
              ) : (
                <ChevronRight size={16} />
              )}
            </span>
          ) : (
            <span className="w-5 h-5" />
          )}

          {/* Node icon based on type */}
          {node.level === 0 && <Network size={18} className="text-gray-600" />}
          {node.level === 1 && getCognitiveStyleIcon(node.cognitive_style)}
          {node.level === 2 && getRiskIcon(node.risk_level)}
          {node.level === 3 && <Users size={14} className="text-indigo-500" />}

          {/* Node name and info */}
          <div className="flex-1">
            <div
              className={`font-medium ${
                node.level === 0
                  ? "text-gray-800 text-base"
                  : node.level === 1
                    ? "text-gray-700 text-sm"
                    : node.level === 2
                      ? "text-gray-600 text-sm"
                      : "text-indigo-700 text-sm"
              }`}
            >
              {node.name}
            </div>
            {node.level <= 2 && node.description && (
              <div className="text-xs text-gray-500 mt-0.5 line-clamp-1">
                {node.description}
              </div>
            )}
          </div>

          {/* Metrics badges */}
          <div className="flex items-center gap-2">
            {node.level === 3 && node.phishing_click_rate !== undefined && (
              <span
                className={`px-2 py-0.5 rounded text-xs font-medium ${
                  node.phishing_click_rate >= 0.35
                    ? "bg-red-100 text-red-700"
                    : node.phishing_click_rate >= 0.28
                      ? "bg-orange-100 text-orange-700"
                      : node.phishing_click_rate >= 0.22
                        ? "bg-yellow-100 text-yellow-700"
                        : "bg-green-100 text-green-700"
                }`}
              >
                {(node.phishing_click_rate * 100).toFixed(0)}% click
              </span>
            )}
            {node.n_participants !== undefined && (
              <span className="text-xs text-gray-500">
                {node.n_participants} users
              </span>
            )}
            {node.metrics?.n_personas !== undefined && (
              <span className="text-xs text-gray-400">
                {node.metrics.n_personas} personas
              </span>
            )}
          </div>
        </div>

        {/* Render children if expanded */}
        {hasChildren && isExpanded && (
          <div className="border-l border-gray-200 ml-6">
            {children.map((child) => renderNode(child, allNodes))}
          </div>
        )}
      </div>
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <RefreshCw className="animate-spin text-indigo-600 mr-2" size={20} />
        <span className="text-gray-600">Loading hierarchical taxonomy...</span>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <AlertTriangle className="text-red-500 mx-auto mb-2" size={32} />
        <p className="text-red-600 text-sm">{error}</p>
        <button
          onClick={loadTaxonomy}
          className="mt-3 text-sm text-indigo-600 hover:text-indigo-700"
        >
          Try again
        </button>
      </div>
    );
  }

  if (!taxonomyData || !taxonomyData.nodes) {
    return (
      <div className="text-center py-8 text-gray-500">
        <Network className="mx-auto mb-2 opacity-50" size={32} />
        <p>No taxonomy data available. Run clustering first.</p>
      </div>
    );
  }

  // Get root node and build tree
  const rootNode = taxonomyData.nodes.find((n) => n.level === 0);

  return (
    <div className="space-y-4">
      {/* Summary Cards */}
      {taxonomyData.summary && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
          {/* Meta-type distribution */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
              Cognitive Styles
            </h4>
            <div className="space-y-1">
              {Object.entries(taxonomyData.summary.meta_types || {}).map(
                ([style, count]) => (
                  <div
                    key={style}
                    className="flex items-center justify-between text-sm"
                  >
                    <div className="flex items-center gap-2">
                      {getCognitiveStyleIcon(style)}
                      <span className="capitalize">{style}</span>
                    </div>
                    <span className="font-medium">{count}</span>
                  </div>
                ),
              )}
            </div>
          </div>

          {/* Risk distribution */}
          <div className="p-4 bg-gray-50 rounded-lg">
            <h4 className="text-xs font-medium text-gray-500 uppercase tracking-wide mb-2">
              Risk Levels
            </h4>
            <div className="space-y-1">
              {["CRITICAL", "HIGH", "MEDIUM", "LOW"].map((risk) => {
                const count =
                  taxonomyData.summary.risk_distribution?.[risk] || 0;
                return (
                  <div
                    key={risk}
                    className="flex items-center justify-between text-sm"
                  >
                    <div className="flex items-center gap-2">
                      {getRiskIcon(risk.toLowerCase())}
                      <span>{risk}</span>
                    </div>
                    <span className="font-medium">{count}</span>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Highest risk personas */}
          <div className="p-4 bg-red-50 rounded-lg col-span-2">
            <h4 className="text-xs font-medium text-red-600 uppercase tracking-wide mb-2">
              Highest Risk Personas
            </h4>
            <div className="space-y-2">
              {(taxonomyData.summary.highest_risk_personas || [])
                .slice(0, 3)
                .map((p, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between text-sm"
                  >
                    <div className="flex items-center gap-2">
                      <span className="font-medium text-red-700">{i + 1}.</span>
                      <span className="text-gray-700">{p.name}</span>
                      <span
                        className={`text-xs px-1.5 py-0.5 rounded ${
                          p.cognitive_style === "intuitive"
                            ? "bg-orange-100 text-orange-700"
                            : p.cognitive_style === "analytical"
                              ? "bg-blue-100 text-blue-700"
                              : "bg-purple-100 text-purple-700"
                        }`}
                      >
                        {p.cognitive_style}
                      </span>
                    </div>
                    <span className="font-mono text-red-600">
                      {p.click_rate}
                    </span>
                  </div>
                ))}
            </div>
          </div>
        </div>
      )}

      {/* Taxonomy Tree */}
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h4 className="text-sm font-medium text-gray-700">
              Persona Hierarchy
            </h4>
            <p className="text-xs text-gray-500">
              Click to expand/collapse. Click persona to see details.
            </p>
          </div>
          <button
            onClick={() => {
              // Expand all
              const allExpanded = {};
              taxonomyData.nodes.forEach((n) => {
                if (n.level < 3) allExpanded[n.id] = true;
              });
              setExpandedNodes(allExpanded);
            }}
            className="text-xs text-indigo-600 hover:text-indigo-700"
          >
            Expand All
          </button>
        </div>

        <div className="max-h-[500px] overflow-y-auto">
          {rootNode && renderNode(rootNode, taxonomyData.nodes)}
        </div>
      </div>

      {/* Intervention Priorities */}
      {taxonomyData.summary?.recommended_priorities?.length > 0 && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <h4 className="text-sm font-medium text-amber-800 mb-2 flex items-center gap-2">
            <AlertTriangle size={16} />
            Recommended Intervention Priorities
          </h4>
          <div className="space-y-2">
            {taxonomyData.summary.recommended_priorities.map((priority, i) => (
              <div key={i} className="flex items-start gap-3 text-sm">
                <span className="w-6 h-6 rounded-full bg-amber-200 text-amber-800 flex items-center justify-center text-xs font-bold">
                  {priority.priority}
                </span>
                <div>
                  <div className="font-medium text-amber-900">
                    {priority.group}
                  </div>
                  <div className="text-amber-700 text-xs">
                    {priority.action}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

/**
 * Behavioral Outcomes Comparison Chart
 */
const BehavioralComparisonChart = ({ clusters, personaLabels }) => {
  const chartData = useMemo(() => {
    return clusters
      .map((c) => {
        const label = personaLabels?.[c.cluster_id];
        // Extract AI-generated name only (no systematic code)
        let aiName = label?.llm_name;
        if (!aiName && label?.name && label.name.includes(": ")) {
          aiName = label.name.split(": ").slice(1).join(": ");
        }
        const displayName =
          aiName || label?.name || `Persona ${c.cluster_id + 1}`;
        return {
          name: displayName,
          fullName: displayName,
          risk: c.risk_level,
          clickRate: (c.phishing_click_rate || 0) * 100,
          reportRate: (c.behavioral_outcomes?.report_rate?.mean || 0) * 100,
          accuracy: (c.behavioral_outcomes?.overall_accuracy?.mean || 0) * 100,
        };
      })
      .sort((a, b) => b.clickRate - a.clickRate);
  }, [clusters, personaLabels]);

  return (
    <ResponsiveContainer width="100%" height={280}>
      <BarChart data={chartData} layout="vertical" margin={{ left: 20 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
        <XAxis
          type="number"
          stroke="#6b7280"
          fontSize={11}
          unit="%"
          domain={[0, 100]}
        />
        <YAxis
          type="category"
          dataKey="name"
          stroke="#6b7280"
          fontSize={10}
          width={120}
          tick={{ fill: "#374151" }}
        />
        <RechartsTooltip content={<CustomTooltip />} />
        <Legend wrapperStyle={{ fontSize: "11px" }} />
        <Bar dataKey="clickRate" name="Click Rate" fill="#ef4444" />
        <Bar dataKey="reportRate" name="Report Rate" fill="#22c55e" />
        <Bar dataKey="accuracy" name="Accuracy" fill="#3b82f6" />
      </BarChart>
    </ResponsiveContainer>
  );
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const ProfilesTab = ({
  clusteringResult,
  personaLabels,
  onSaveLabel,
}) => {
  const [selectedCluster, setSelectedCluster] = useState(null);
  const [showDetailPanel, setShowDetailPanel] = useState(false);
  const [viewMode, setViewMode] = useState("flat"); // 'flat' or 'hierarchical'

  if (!clusteringResult) {
    return (
      <EmptyState
        icon={<Users size={48} />}
        title="Run clustering first"
        description="Persona profiles require clustering results. Go to the Clustering tab to generate personas."
      />
    );
  }

  const clusters = Object.values(clusteringResult.clusters || {});
  const selectedData =
    selectedCluster !== null
      ? clusters.find((c) => c.cluster_id === selectedCluster)
      : null;

  return (
    <div className="space-y-6">
      {/* View Mode Toggle */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900">Persona Profiles</h2>
          <p className="text-sm text-gray-500">
            {viewMode === "flat"
              ? "Compare all personas side-by-side"
              : "View personas organized by cognitive style and risk level"}
          </p>
        </div>
        <div className="flex items-center gap-2 bg-gray-100 rounded-lg p-1">
          <button
            onClick={() => setViewMode("flat")}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition ${
              viewMode === "flat"
                ? "bg-white text-indigo-700 shadow-sm"
                : "text-gray-600 hover:text-gray-900"
            }`}
          >
            <List size={16} />
            Flat View
          </button>
          <button
            onClick={() => setViewMode("hierarchical")}
            className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-sm font-medium transition ${
              viewMode === "hierarchical"
                ? "bg-white text-indigo-700 shadow-sm"
                : "text-gray-600 hover:text-gray-900"
            }`}
          >
            <Network size={16} />
            Hierarchical View
          </button>
        </div>
      </div>

      {/* Hierarchical Taxonomy View */}
      {viewMode === "hierarchical" && (
        <Card>
          <div className="flex items-center gap-2 mb-4">
            <Network size={20} className="text-indigo-600" />
            <div>
              <h3 className="text-lg font-semibold text-gray-900">
                Hierarchical Persona Taxonomy
              </h3>
              <p className="text-xs text-gray-500">
                Personas organized by cognitive style
                (Analytical/Intuitive/Balanced) and risk level
              </p>
            </div>
          </div>
          <HierarchicalTaxonomyView
            clusters={clusters}
            personaLabels={personaLabels}
            onSelectPersona={(clusterId) => {
              setSelectedCluster(clusterId);
              setShowDetailPanel(true);
            }}
          />
        </Card>
      )}

      {/* Flat View - Original Content */}
      {viewMode === "flat" && (
        <>
          {/* Section 1: Persona Overview Comparison Table */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <div>
                <div className="flex items-center gap-4">
                  <h3 className="text-lg font-semibold text-gray-900">
                    Persona Overview Comparison
                  </h3>
                  <SystematicCodeLegend />
                </div>
                <p className="text-sm text-gray-500">
                  Click any row to see detailed profile
                </p>
              </div>
              <div className="flex items-center gap-4 text-xs text-gray-500">
                <span>{clusters.length} personas</span>
                <span>•</span>
                <span>
                  {clusteringResult.metrics?.eta_squared_mean?.toFixed(3)} avg
                  η²
                </span>
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-3 py-3 text-left font-medium text-gray-700">
                      Persona
                    </th>
                    <th className="px-3 py-3 text-center font-medium text-gray-700">
                      Risk
                    </th>
                    <th className="px-3 py-3 text-center font-medium text-gray-700">
                      Size
                    </th>
                    <th className="px-3 py-3 text-center font-medium text-gray-700">
                      Click Rate
                    </th>
                    <th className="px-3 py-3 text-center font-medium text-gray-700">
                      Report Rate
                    </th>
                    <th className="px-3 py-3 text-center font-medium text-gray-700">
                      Accuracy
                    </th>
                    <th className="px-3 py-3 text-left font-medium text-gray-700">
                      Key High Traits
                    </th>
                    <th className="px-3 py-3 text-left font-medium text-gray-700">
                      Key Low Traits
                    </th>
                  </tr>
                </thead>
                <tbody>
                  {clusters
                    .sort(
                      (a, b) => b.phishing_click_rate - a.phishing_click_rate,
                    )
                    .map((c) => {
                      const isSelected = selectedCluster === c.cluster_id;
                      const label = personaLabels?.[c.cluster_id];

                      return (
                        <tr
                          key={c.cluster_id}
                          onClick={() => {
                            setSelectedCluster(
                              isSelected ? null : c.cluster_id,
                            );
                            setShowDetailPanel(true);
                          }}
                          className={`border-t cursor-pointer transition ${
                            isSelected
                              ? "bg-indigo-50 border-l-4 border-l-indigo-500"
                              : "hover:bg-gray-50"
                          }`}
                        >
                          <td className="px-3 py-3">
                            <div className="font-medium text-gray-900">
                              {(() => {
                                // Extract AI-generated name only
                                let aiName = label?.llm_name;
                                if (
                                  !aiName &&
                                  label?.name &&
                                  label.name.includes(": ")
                                ) {
                                  aiName = label.name
                                    .split(": ")
                                    .slice(1)
                                    .join(": ");
                                }
                                return (
                                  aiName ||
                                  label?.name ||
                                  `Cluster ${c.cluster_id + 1}`
                                );
                              })()}
                            </div>
                            {label?.archetype && (
                              <div className="text-xs text-gray-500">
                                {label.archetype}
                              </div>
                            )}
                          </td>
                          <td className="px-3 py-3 text-center">
                            <span
                              className={`px-2 py-1 rounded text-xs font-medium ${getRiskBadge(c.risk_level)}`}
                            >
                              {c.risk_level}
                            </span>
                          </td>
                          <td className="px-3 py-3 text-center">
                            <div className="font-medium">
                              {c.n_participants}
                            </div>
                            <div className="text-xs text-gray-400">
                              {c.pct_of_population?.toFixed(1)}%
                            </div>
                          </td>
                          <td className="px-3 py-3 text-center">
                            <div className="flex items-center justify-center gap-2">
                              <div
                                className="h-2 rounded-full bg-red-200"
                                style={{ width: "60px" }}
                              >
                                <div
                                  className="h-2 rounded-full bg-red-500"
                                  style={{
                                    width: `${(c.phishing_click_rate || 0) * 100}%`,
                                  }}
                                />
                              </div>
                              <span className="font-mono text-red-600 w-12">
                                {((c.phishing_click_rate || 0) * 100).toFixed(
                                  1,
                                )}
                                %
                              </span>
                            </div>
                          </td>
                          <td className="px-3 py-3 text-center">
                            <span className="font-mono text-green-600">
                              {(
                                (c.behavioral_outcomes?.report_rate?.mean ||
                                  0) * 100
                              ).toFixed(1)}
                              %
                            </span>
                          </td>
                          <td className="px-3 py-3 text-center">
                            <span className="font-mono text-blue-600">
                              {(
                                (c.behavioral_outcomes?.overall_accuracy
                                  ?.mean || 0) * 100
                              ).toFixed(1)}
                              %
                            </span>
                          </td>
                          <td className="px-3 py-3">
                            <div className="flex flex-wrap gap-1">
                              {(c.top_high_traits || [])
                                .slice(0, 2)
                                .map(([trait, z]) => (
                                  <span
                                    key={trait}
                                    className="px-1.5 py-0.5 bg-red-100 text-red-700 rounded text-xs"
                                  >
                                    ↑
                                    {TRAIT_LABELS[trait]?.split(" ")[0] ||
                                      trait}
                                  </span>
                                ))}
                            </div>
                          </td>
                          <td className="px-3 py-3">
                            <div className="flex flex-wrap gap-1">
                              {(c.top_low_traits || [])
                                .slice(0, 2)
                                .map(([trait, z]) => (
                                  <span
                                    key={trait}
                                    className="px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded text-xs"
                                  >
                                    ↓
                                    {TRAIT_LABELS[trait]?.split(" ")[0] ||
                                      trait}
                                  </span>
                                ))}
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                </tbody>
              </table>
            </div>
          </Card>

          {/* Section 2: Visual Comparisons - Side by Side */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Behavioral Outcomes Chart */}
            <Card>
              <h3 className="text-md font-semibold text-gray-900 mb-1">
                Behavioral Outcomes Comparison
              </h3>
              <p className="text-xs text-gray-500 mb-4">
                Key metrics across all personas (sorted by click rate)
              </p>
              <BehavioralComparisonChart
                clusters={clusters}
                personaLabels={personaLabels}
              />
            </Card>

            {/* Risk Distribution Summary */}
            <Card>
              <h3 className="text-md font-semibold text-gray-900 mb-1">
                Risk Distribution Summary
              </h3>
              <p className="text-xs text-gray-500 mb-4">
                Persona count by risk level
              </p>

              <div className="space-y-4">
                {["CRITICAL", "HIGH", "MEDIUM", "LOW"].map((risk) => {
                  const count = clusters.filter(
                    (c) => c.risk_level === risk,
                  ).length;
                  const personas = clusters.filter(
                    (c) => c.risk_level === risk,
                  );
                  const totalParticipants = personas.reduce(
                    (sum, c) => sum + c.n_participants,
                    0,
                  );
                  const pct =
                    clusters.length > 0 ? (count / clusters.length) * 100 : 0;

                  return (
                    <div key={risk} className="flex items-center gap-3">
                      <span
                        className={`px-2 py-1 rounded text-xs font-medium w-20 text-center ${getRiskBadge(risk)}`}
                      >
                        {risk}
                      </span>
                      <div className="flex-1">
                        <div className="h-4 bg-gray-100 rounded-full overflow-hidden">
                          <div
                            className={`h-full rounded-full ${
                              risk === "CRITICAL"
                                ? "bg-red-500"
                                : risk === "HIGH"
                                  ? "bg-orange-500"
                                  : risk === "MEDIUM"
                                    ? "bg-yellow-500"
                                    : "bg-green-500"
                            }`}
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                      </div>
                      <div className="text-right w-32">
                        <span className="font-medium">
                          {count} persona{count !== 1 ? "s" : ""}
                        </span>
                        <span className="text-xs text-gray-500 ml-2">
                          ({totalParticipants} users)
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>

              {/* Action Summary */}
              <div className="mt-6 pt-4 border-t border-gray-100">
                <h4 className="text-sm font-medium text-gray-700 mb-2">
                  Recommended Actions
                </h4>
                <div className="space-y-2 text-xs">
                  {clusters.filter(
                    (c) =>
                      c.risk_level === "CRITICAL" || c.risk_level === "HIGH",
                  ).length > 0 && (
                    <div className="flex items-start gap-2 p-2 bg-red-50 rounded">
                      <span className="text-red-500">⚠</span>
                      <span className="text-red-700">
                        {
                          clusters.filter(
                            (c) =>
                              c.risk_level === "CRITICAL" ||
                              c.risk_level === "HIGH",
                          ).length
                        }{" "}
                        high-risk personas need targeted training interventions
                      </span>
                    </div>
                  )}
                  {clusters.filter((c) => c.risk_level === "LOW").length >
                    0 && (
                    <div className="flex items-start gap-2 p-2 bg-green-50 rounded">
                      <span className="text-green-500">✓</span>
                      <span className="text-green-700">
                        {clusters.filter((c) => c.risk_level === "LOW").length}{" "}
                        low-risk personas can serve as security champions
                      </span>
                    </div>
                  )}
                </div>
              </div>
            </Card>
          </div>

          {/* Section 3: Trait Profile Heatmap */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <div>
                <h3 className="text-md font-semibold text-gray-900">
                  Psychological Trait Heatmap
                </h3>
                <p className="text-xs text-gray-500">
                  Z-scores relative to population mean (click category to
                  expand/collapse)
                </p>
              </div>
              <div className="flex items-center gap-4 text-xs">
                <div className="flex items-center gap-1">
                  <span
                    className="w-4 h-3 rounded"
                    style={{ backgroundColor: "#3b82f6" }}
                  ></span>
                  <span>Below avg</span>
                </div>
                <div className="flex items-center gap-1">
                  <span className="w-4 h-3 rounded bg-gray-200"></span>
                  <span>Average</span>
                </div>
                <div className="flex items-center gap-1">
                  <span
                    className="w-4 h-3 rounded"
                    style={{ backgroundColor: "#ef4444" }}
                  ></span>
                  <span>Above avg</span>
                </div>
              </div>
            </div>

            <TraitHeatmap
              clusters={clusters}
              selectedCluster={selectedCluster}
              onSelectCluster={(id) => {
                setSelectedCluster(id);
                if (id !== null) setShowDetailPanel(true);
              }}
              personaLabels={personaLabels}
            />
          </Card>
        </>
      )}

      {/* Section 4: Selected Persona Detail Panel - Shown for both views */}
      {selectedData && showDetailPanel && (
        <Card>
          <div className="flex items-start justify-between mb-4">
            <div className="flex items-center gap-3">
              <h3 className="text-lg font-semibold text-gray-900">
                {(() => {
                  const label = personaLabels?.[selectedCluster];
                  let aiName = label?.llm_name;
                  if (!aiName && label?.name && label.name.includes(": ")) {
                    aiName = label.name.split(": ").slice(1).join(": ");
                  }
                  return aiName || label?.name || `Cluster ${selectedCluster}`;
                })()}{" "}
                Details
              </h3>
              <span
                className={`px-2 py-1 rounded text-xs font-medium ${getRiskBadge(selectedData.risk_level)}`}
              >
                {selectedData.risk_level}
              </span>
            </div>
            <button
              onClick={() => setShowDetailPanel(false)}
              className="text-sm text-gray-500 hover:text-gray-700 flex items-center gap-1"
            >
              <EyeOff size={14} /> Hide details
            </button>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
            {/* Overview Stats */}
            <div className="space-y-3">
              <h4 className="text-sm font-medium text-gray-700">Overview</h4>
              <div className="grid grid-cols-2 gap-2 text-sm">
                <div className="p-2 bg-gray-50 rounded text-center">
                  <div className="text-xl font-bold text-gray-900">
                    {selectedData.n_participants}
                  </div>
                  <div className="text-xs text-gray-500">Participants</div>
                </div>
                <div className="p-2 bg-red-50 rounded text-center">
                  <div className="text-xl font-bold text-red-600">
                    {((selectedData.phishing_click_rate || 0) * 100).toFixed(1)}
                    %
                  </div>
                  <div className="text-xs text-gray-500">Click Rate</div>
                </div>
              </div>
            </div>

            {/* High Traits */}
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">
                ↑ High Traits
              </h4>
              <div className="space-y-1">
                {(selectedData.top_high_traits || []).map(([trait, z]) => (
                  <div
                    key={trait}
                    className="flex items-center justify-between text-xs"
                  >
                    <span className="text-gray-600">
                      {TRAIT_LABELS[trait] || trait}
                    </span>
                    <span className="font-mono text-red-600">
                      +{z.toFixed(2)}σ
                    </span>
                  </div>
                ))}
                {!selectedData.top_high_traits?.length && (
                  <span className="text-xs text-gray-400">
                    None significant
                  </span>
                )}
              </div>
            </div>

            {/* Low Traits */}
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">
                ↓ Low Traits
              </h4>
              <div className="space-y-1">
                {(selectedData.top_low_traits || []).map(([trait, z]) => (
                  <div
                    key={trait}
                    className="flex items-center justify-between text-xs"
                  >
                    <span className="text-gray-600">
                      {TRAIT_LABELS[trait] || trait}
                    </span>
                    <span className="font-mono text-blue-600">
                      {z.toFixed(2)}σ
                    </span>
                  </div>
                ))}
                {!selectedData.top_low_traits?.length && (
                  <span className="text-xs text-gray-400">
                    None significant
                  </span>
                )}
              </div>
            </div>

            {/* Behavioral Outcomes */}
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-2">
                Behavioral Outcomes
              </h4>
              <div className="space-y-1 text-xs">
                {Object.entries(selectedData.behavioral_outcomes || {})
                  .slice(0, 5)
                  .map(([key, val]) => (
                    <div key={key} className="flex justify-between">
                      <span className="text-gray-600">
                        {OUTCOME_LABELS[key] || key}
                      </span>
                      <span className="font-mono">
                        {key.includes("rate") || key.includes("accuracy")
                          ? `${((val?.mean || 0) * 100).toFixed(1)}%`
                          : (val?.mean || 0).toFixed(1)}
                      </span>
                    </div>
                  ))}
              </div>
            </div>
          </div>

          {/* Description */}
          <div className="mt-4 pt-4 border-t border-gray-100">
            <p className="text-sm text-gray-600">
              <span className="font-medium">Description: </span>
              {personaLabels?.[selectedCluster]?.description ||
                selectedData.description}
            </p>
          </div>
        </Card>
      )}
    </div>
  );
};

export default ProfilesTab;
