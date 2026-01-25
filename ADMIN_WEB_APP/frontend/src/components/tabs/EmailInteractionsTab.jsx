/**
 * Email Interactions Tab
 *
 * Analyzes how each persona responds to different email characteristics:
 * - Urgency (High vs Low)
 * - Familiarity (Familiar vs Unfamiliar sender)
 * - Framing (Threat vs Reward)
 * - Aggressive content (Yes vs No)
 * - Email type (Phishing vs Legitimate)
 *
 * Also shows effect sizes for each interaction.
 */

import React from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  Cell,
} from "recharts";
import { Grid3X3, AlertCircle, RefreshCw } from "lucide-react";
import {
  Card,
  ChartCard,
  EmptyState,
  CustomTooltip,
  SystematicCodeLegend,
} from "../common";

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

/**
 * Get AI-generated persona name only (for charts)
 * Extracts just the AI name without systematic code
 */
const getAIName = (clusterId, personaLabels) => {
  const label = personaLabels?.[clusterId];
  if (label?.llm_name) return label.llm_name;
  if (label?.name && label.name.includes(": ")) {
    return label.name.split(": ").slice(1).join(": ");
  }
  if (label?.name) return label.name;
  return `Persona ${parseInt(clusterId) + 1}`;
};

/**
 * Get full persona name with systematic code (for tables)
 */
const getFullPersonaName = (clusterId, personaLabels) => {
  const label = personaLabels?.[clusterId];
  if (label?.name) return label.name;
  return `Persona ${parseInt(clusterId) + 1}`;
};

/**
 * Transform interaction data from API format to chart format
 * Input:  { 'high': { 0: 0.3, 1: 0.4 }, 'low': { 0: 0.2, 1: 0.25 } }
 * Output: [{ cluster: 'Persona Name', high: 30, low: 20 }, ...]
 */
const transformInteractionData = (data, personaLabels) => {
  if (!data) return [];

  const clusters = new Set();
  Object.values(data).forEach((attrData) => {
    if (attrData && typeof attrData === "object") {
      Object.keys(attrData).forEach((c) => clusters.add(c));
    }
  });

  return Array.from(clusters)
    .sort((a, b) => parseInt(a) - parseInt(b))
    .map((clusterId) => {
      const row = { cluster: getAIName(clusterId, personaLabels) };
      Object.entries(data).forEach(([attr, values]) => {
        if (values && typeof values === "object") {
          row[attr] = (values[clusterId] || 0) * 100; // Convert to percentage
        }
      });
      return row;
    });
};

/**
 * Transform effect size data for chart
 * Input:  { 0: 0.15, 1: -0.05, 2: 0.08 }
 * Output: [{ cluster: 'Persona Name', effect: 15 }, ...]
 */
const transformEffectData = (data, personaLabels) => {
  if (!data) return [];

  return Object.entries(data)
    .map(([clusterId, value]) => ({
      cluster: getAIName(clusterId, personaLabels),
      clusterId: parseInt(clusterId),
      effect: value * 100, // Convert to percentage points
    }))
    .sort((a, b) => a.clusterId - b.clusterId);
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const EmailInteractionsTab = ({
  clusteringResult,
  interactionResult,
  interactionError,
  loading,
  onAnalyzeInteractions,
  personaLabels,
}) => {
  if (!clusteringResult) {
    // return (
    //     <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-center gap-3">
    //         <AlertCircle className="text-yellow-600" size={20} />
    //         <span>Please run clustering first to analyze email interactions.</span>
    //     </div>
    // );
    return (
      <EmptyState
        icon={<Grid3X3 size={48} />}
        title="Run clustering first"
        description="Email Interaction requires clustering results."
      />
    );
  }

  return (
    <div className="space-y-6">
      <Card>
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">
              Email × Cluster Interactions
            </h3>
            <p className="text-sm text-gray-500">
              How each persona responds to different email characteristics
            </p>
          </div>
          <button
            onClick={onAnalyzeInteractions}
            disabled={loading}
            className={`px-4 py-2 rounded-lg text-sm font-medium text-white transition-colors flex items-center gap-2 ${
              loading
                ? "bg-gray-400 cursor-not-allowed"
                : "bg-indigo-600 hover:bg-indigo-700"
            }`}
          >
            {loading ? (
              <>
                <RefreshCw className="animate-spin" size={16} />
                Analyzing...
              </>
            ) : (
              "Analyze Interactions"
            )}
          </button>
        </div>
      </Card>

      {/* Error Display */}
      {interactionError && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <div className="flex items-center gap-2 text-red-700">
            <AlertCircle size={18} />
            <span className="font-medium">Analysis Error</span>
          </div>
          <p className="text-sm text-red-600 mt-1">{interactionError}</p>
        </div>
      )}

      {/* Results */}
      {interactionResult && !interactionResult.error && (
        <div className="space-y-6">
          {/* Summary */}
          {interactionResult.summary && (
            <div className="bg-gray-50 rounded-lg p-4 text-sm">
              <span className="font-medium">Data: </span>
              {interactionResult.summary.n_responses?.toLocaleString()}{" "}
              responses,{" "}
              {interactionResult.summary.n_phishing?.toLocaleString()} phishing
              emails, {interactionResult.summary.n_clusters} clusters
            </div>
          )}

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Urgency Interaction */}
            {interactionResult.by_urgency &&
              Object.keys(interactionResult.by_urgency).length > 0 && (
                <ChartCard
                  title="Urgency Susceptibility"
                  subtitle="High vs Low urgency click rates"
                >
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart
                      layout="vertical"
                      data={transformInteractionData(
                        interactionResult.by_urgency,
                        personaLabels,
                      )}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <YAxis
                        type="category"
                        dataKey="cluster"
                        stroke="#6b7280"
                        fontSize={11}
                        width={140}
                      />
                      <XAxis
                        type="number"
                        stroke="#6b7280"
                        fontSize={12}
                        unit="%"
                      />
                      <RechartsTooltip content={<CustomTooltip />} />
                      <Legend />
                      <Bar dataKey="low" name="Low Urgency" fill="#3b82f6" />
                      <Bar dataKey="high" name="High Urgency" fill="#ef4444" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartCard>
              )}

            {/* Familiarity Interaction */}
            {interactionResult.by_familiarity &&
              Object.keys(interactionResult.by_familiarity).length > 0 && (
                <ChartCard
                  title="Familiarity Trust"
                  subtitle="Familiar vs Unfamiliar sender click rates"
                >
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart
                      layout="vertical"
                      data={transformInteractionData(
                        interactionResult.by_familiarity,
                        personaLabels,
                      )}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <YAxis
                        type="category"
                        dataKey="cluster"
                        stroke="#6b7280"
                        fontSize={11}
                        width={140}
                      />
                      <XAxis
                        type="number"
                        stroke="#6b7280"
                        fontSize={12}
                        unit="%"
                      />
                      <RechartsTooltip content={<CustomTooltip />} />
                      <Legend />
                      <Bar
                        dataKey="unfamiliar"
                        name="Unfamiliar"
                        fill="#8b5cf6"
                      />
                      <Bar dataKey="familiar" name="Familiar" fill="#10b981" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartCard>
              )}

            {/* Framing Interaction (Threat vs Reward) */}
            {interactionResult.by_framing &&
              Object.keys(interactionResult.by_framing).length > 0 && (
                <ChartCard
                  title="Framing Effect"
                  subtitle="Threat vs Reward framing click rates"
                >
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart
                      layout="vertical"
                      data={transformInteractionData(
                        interactionResult.by_framing,
                        personaLabels,
                      )}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <YAxis
                        type="category"
                        dataKey="cluster"
                        stroke="#6b7280"
                        fontSize={11}
                        width={140}
                      />
                      <XAxis
                        type="number"
                        stroke="#6b7280"
                        fontSize={12}
                        unit="%"
                      />
                      <RechartsTooltip content={<CustomTooltip />} />
                      <Legend />
                      <Bar dataKey="reward" name="Reward" fill="#f59e0b" />
                      <Bar dataKey="threat" name="Threat" fill="#dc2626" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartCard>
              )}

            {/* Aggressive Content */}
            {interactionResult.by_aggressive &&
              Object.keys(interactionResult.by_aggressive).length > 0 && (
                <ChartCard
                  title="Aggressive Content Effect"
                  subtitle="With vs Without aggressive content"
                >
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart
                      layout="vertical"
                      data={transformInteractionData(
                        interactionResult.by_aggressive,
                        personaLabels,
                      )}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <YAxis
                        type="category"
                        dataKey="cluster"
                        stroke="#6b7280"
                        fontSize={11}
                        width={140}
                      />
                      <XAxis
                        type="number"
                        stroke="#6b7280"
                        fontSize={12}
                        unit="%"
                      />
                      <RechartsTooltip content={<CustomTooltip />} />
                      <Legend />
                      <Bar
                        dataKey="False"
                        name="Non-Aggressive"
                        fill="#6b7280"
                      />
                      <Bar dataKey="True" name="Aggressive" fill="#dc2626" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartCard>
              )}

            {/* Email Type (Phishing vs Legitimate) */}
            {interactionResult.by_email_type &&
              Object.keys(interactionResult.by_email_type).length > 0 && (
                <ChartCard
                  title="Email Type Response"
                  subtitle="Phishing vs Legitimate email click rates"
                >
                  <ResponsiveContainer width="100%" height={320}>
                    <BarChart
                      layout="vertical"
                      data={transformInteractionData(
                        interactionResult.by_email_type,
                        personaLabels,
                      )}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                      <YAxis
                        type="category"
                        dataKey="cluster"
                        stroke="#6b7280"
                        fontSize={11}
                        width={140}
                      />
                      <XAxis
                        type="number"
                        stroke="#6b7280"
                        fontSize={12}
                        unit="%"
                      />
                      <RechartsTooltip content={<CustomTooltip />} />
                      <Legend />
                      <Bar
                        dataKey="legitimate"
                        name="Legitimate"
                        fill="#10b981"
                      />
                      <Bar dataKey="phishing" name="Phishing" fill="#ef4444" />
                    </BarChart>
                  </ResponsiveContainer>
                </ChartCard>
              )}

            {/* Urgency Effect Size */}
            {interactionResult.interaction_effects?.urgency_effect && (
              <ChartCard
                title="Urgency Effect Size"
                subtitle="High - Low urgency (positive = more susceptible)"
              >
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart
                    layout="vertical"
                    data={transformEffectData(
                      interactionResult.interaction_effects.urgency_effect,
                      personaLabels,
                    )}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <YAxis
                      type="category"
                      dataKey="cluster"
                      stroke="#6b7280"
                      fontSize={11}
                      width={140}
                    />
                    <XAxis
                      type="number"
                      stroke="#6b7280"
                      fontSize={12}
                      unit="%"
                    />
                    <RechartsTooltip content={<CustomTooltip />} />
                    <ReferenceLine x={0} stroke="#000" />
                    <Bar dataKey="effect" name="Effect Size">
                      {transformEffectData(
                        interactionResult.interaction_effects.urgency_effect,
                        personaLabels,
                      ).map((entry, index) => (
                        <Cell
                          key={index}
                          fill={entry.effect > 0 ? "#ef4444" : "#3b82f6"}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            )}

            {/* Familiarity Effect Size */}
            {interactionResult.interaction_effects?.familiarity_effect && (
              <ChartCard
                title="Familiarity Effect Size"
                subtitle="Familiar - Unfamiliar (positive = trusts familiar more)"
              >
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart
                    layout="vertical"
                    data={transformEffectData(
                      interactionResult.interaction_effects.familiarity_effect,
                      personaLabels,
                    )}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <YAxis
                      type="category"
                      dataKey="cluster"
                      stroke="#6b7280"
                      fontSize={11}
                      width={140}
                    />
                    <XAxis
                      type="number"
                      stroke="#6b7280"
                      fontSize={12}
                      unit="%"
                    />
                    <RechartsTooltip content={<CustomTooltip />} />
                    <ReferenceLine x={0} stroke="#000" />
                    <Bar dataKey="effect" name="Effect Size">
                      {transformEffectData(
                        interactionResult.interaction_effects
                          .familiarity_effect,
                        personaLabels,
                      ).map((entry, index) => (
                        <Cell
                          key={index}
                          fill={entry.effect > 0 ? "#10b981" : "#8b5cf6"}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            )}

            {/* Framing Effect Size */}
            {interactionResult.interaction_effects?.framing_effect && (
              <ChartCard
                title="Framing Effect Size"
                subtitle="Threat - Reward (positive = more susceptible to threat)"
              >
                <ResponsiveContainer width="100%" height={320}>
                  <BarChart
                    layout="vertical"
                    data={transformEffectData(
                      interactionResult.interaction_effects.framing_effect,
                      personaLabels,
                    )}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                    <YAxis
                      type="category"
                      dataKey="cluster"
                      stroke="#6b7280"
                      fontSize={11}
                      width={140}
                    />
                    <XAxis
                      type="number"
                      stroke="#6b7280"
                      fontSize={12}
                      unit="%"
                    />
                    <RechartsTooltip content={<CustomTooltip />} />
                    <ReferenceLine x={0} stroke="#000" />
                    <Bar dataKey="effect" name="Effect Size">
                      {transformEffectData(
                        interactionResult.interaction_effects.framing_effect,
                        personaLabels,
                      ).map((entry, index) => (
                        <Cell
                          key={index}
                          fill={entry.effect > 0 ? "#dc2626" : "#f59e0b"}
                        />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </ChartCard>
            )}
          </div>

          {/* Vulnerability Summary */}
          {interactionResult.interaction_effects && (
            <Card>
              <div className="flex items-center gap-4 mb-4">
                <h3 className="text-lg font-semibold">
                  Vulnerability Summary by Persona
                </h3>
                <SystematicCodeLegend />
              </div>
              <div className="overflow-x-auto">
                <table className="w-full text-sm">
                  <thead className="bg-gray-50">
                    <tr>
                      <th className="px-4 py-3 text-left font-medium">
                        Persona
                      </th>
                      <th className="px-4 py-3 text-center font-medium">
                        Risk Level
                      </th>
                      <th className="px-4 py-3 text-center font-medium">
                        Urgency Effect
                      </th>
                      <th className="px-4 py-3 text-center font-medium">
                        Familiarity Effect
                      </th>
                      <th className="px-4 py-3 text-center font-medium">
                        Framing Effect
                      </th>
                      <th className="px-4 py-3 text-left font-medium">
                        Primary Vulnerability
                      </th>
                    </tr>
                  </thead>
                  <tbody>
                    {Object.keys(
                      interactionResult.interaction_effects.urgency_effect ||
                        {},
                    ).map((clusterId) => {
                      const urgency =
                        (interactionResult.interaction_effects.urgency_effect?.[
                          clusterId
                        ] || 0) * 100;
                      const familiarity =
                        (interactionResult.interaction_effects
                          .familiarity_effect?.[clusterId] || 0) * 100;
                      const framing =
                        (interactionResult.interaction_effects.framing_effect?.[
                          clusterId
                        ] || 0) * 100;

                      // Get risk level from clustering result
                      const cluster = clusteringResult?.clusters?.[clusterId];
                      const riskLevel = cluster?.risk_level || "MEDIUM";
                      const riskColors = {
                        CRITICAL: "bg-red-100 text-red-700",
                        HIGH: "bg-orange-100 text-orange-700",
                        MEDIUM: "bg-yellow-100 text-yellow-700",
                        LOW: "bg-green-100 text-green-700",
                      };

                      // Determine primary vulnerability
                      const effects = [
                        { name: "Urgency", value: urgency },
                        { name: "Familiarity", value: familiarity },
                        { name: "Threat Framing", value: framing },
                      ];
                      const maxEffect = effects.reduce(
                        (max, e) => (e.value > max.value ? e : max),
                        effects[0],
                      );

                      return (
                        <tr
                          key={clusterId}
                          className="border-t hover:bg-gray-50"
                        >
                          <td className="px-4 py-3 font-medium">
                            {getFullPersonaName(clusterId, personaLabels)}
                          </td>
                          <td className="px-4 py-3 text-center">
                            <span
                              className={`px-2 py-1 rounded text-xs font-medium ${riskColors[riskLevel]}`}
                            >
                              {riskLevel}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-center">
                            <span
                              className={`font-mono ${urgency > 5 ? "text-red-600" : urgency < -5 ? "text-blue-600" : "text-gray-500"}`}
                            >
                              {urgency > 0 ? "+" : ""}
                              {urgency.toFixed(1)}%
                            </span>
                          </td>
                          <td className="px-4 py-3 text-center">
                            <span
                              className={`font-mono ${familiarity > 5 ? "text-green-600" : familiarity < -5 ? "text-purple-600" : "text-gray-500"}`}
                            >
                              {familiarity > 0 ? "+" : ""}
                              {familiarity.toFixed(1)}%
                            </span>
                          </td>
                          <td className="px-4 py-3 text-center">
                            <span
                              className={`font-mono ${framing > 5 ? "text-red-600" : framing < -5 ? "text-amber-600" : "text-gray-500"}`}
                            >
                              {framing > 0 ? "+" : ""}
                              {framing.toFixed(1)}%
                            </span>
                          </td>
                          <td className="px-4 py-3">
                            {maxEffect.value > 5 ? (
                              <span className="px-2 py-1 bg-red-100 text-red-700 rounded text-xs">
                                {maxEffect.name}
                              </span>
                            ) : (
                              <span className="text-gray-400 text-xs">
                                No strong vulnerability
                              </span>
                            )}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Card>
          )}
        </div>
      )}

      {/* Empty State */}
      {!interactionResult && !interactionError && (
        <EmptyState
          icon={<Grid3X3 size={48} />}
          title="No interaction analysis yet"
          description="Click 'Analyze Interactions' to see how clusters respond to different email types."
        />
      )}
    </div>
  );
};

export default EmailInteractionsTab;
