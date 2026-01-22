/**
 * AI Export Tab - Phase 2 Preparation
 *
 * This tab prepares persona definitions for AI simulation in Phase 2.
 * Supports three prompt configurations:
 * (a) task-only baseline
 * (b) augmented with behavioral statistics
 * (c) augmented with chain-of-thought reasoning
 */

import { useState } from "react";
import {
  Sparkles,
  Download,
  Info,
  CheckCircle,
  XCircle,
} from "lucide-react";
import { Card, EmptyState } from "../common";
import { RISK_COLORS } from "../../constants";

// What each format includes (for display)
const FORMAT_FEATURES = {
  baseline: {
    name: "Baseline (Task-Only)",
    description: "Core persona data for minimal prompts",
    includes: [
      "persona_id, name, archetype",
      "description",
      "risk_level",
      "trait_zscores (all 29 traits)",
      "behavioral_statistics (6 metrics)",
      "cognitive_style",
    ],
    excludes: [
      "psychological_profile (derived)",
      "vulnerability_profile (derived)",
      "reasoning_examples",
      "boundary_conditions",
    ],
  },
  stats: {
    name: "+ Behavioral Statistics",
    description: "Adds derived psychological profiles",
    includes: [
      "Everything from Baseline",
      "psychological_profile",
      "  - distinguishing_traits (formatted)",
      "  - trust_level, impulsivity",
      "vulnerability_profile",
      "  - urgency/authority susceptible",
      "  - familiarity_trusting",
    ],
    excludes: ["reasoning_examples", "boundary_conditions"],
  },
  full: {
    name: "+ Chain-of-Thought",
    description: "Full export with reasoning examples",
    includes: [
      "Everything from Stats",
      "reasoning_examples (scenario-based)",
      "boundary_conditions (AI limitations)",
    ],
    excludes: [],
  },
};

// Helper functions
const determineCognitiveStyle = (cluster) => {
  const crt = cluster.trait_zscores?.crt_score || 0;
  const nfc = cluster.trait_zscores?.need_for_cognition || 0;
  const impulsivity = cluster.trait_zscores?.impulsivity_total || 0;

  if (crt > 0.5 && nfc > 0.5) return "analytical";
  if (impulsivity > 0.5) return "impulsive";
  // Note: 'intuitive' is not a valid backend enum value, map to 'balanced'
  return "balanced";
};

const generateReasoningExamples = (cluster) => {
  const examples = [];

  if (cluster.risk_level === "CRITICAL" || cluster.risk_level === "HIGH") {
    examples.push({
      scenario: "high_urgency_phishing",
      email_cues:
        "Urgent language, threat of account suspension, deadline pressure",
      reasoning: `As someone with high urgency susceptibility, I notice the "Act Now!" 
            message triggers my anxiety. Without pausing to verify, I feel compelled to click 
            to avoid the threatened consequence.`,
      action: "click",
      confidence: "low",
    });
  }

  if (cluster.trait_zscores?.crt_score > 0.5) {
    examples.push({
      scenario: "sophisticated_phishing",
      email_cues:
        "Professional appearance, IT department sender, but mismatched domain",
      reasoning: `Let me analyze this carefully. The sender claims to be from IT, but 
            the email domain doesn't match our company. The link URL preview shows a 
            suspicious domain. I should report this.`,
      action: "report",
      confidence: "high",
    });
  }

  if (cluster.trait_zscores?.authority_susceptibility > 0.5) {
    examples.push({
      scenario: "authority_phishing",
      email_cues:
        "CEO name in sender, executive request, immediate action needed",
      reasoning: `This email claims to be from the CEO. I feel pressure to comply 
            immediately with requests from authority figures. The urgency makes me want 
            to act quickly without questioning.`,
      action: "click",
      confidence: "medium",
    });
  }

  if (cluster.trait_zscores?.trust_propensity < -0.3) {
    examples.push({
      scenario: "unfamiliar_sender",
      email_cues: "Unknown sender, external domain, unsolicited contact",
      reasoning: `I don't recognize this sender. My natural skepticism makes me 
            question why they're contacting me. I'll hover over the link to check the 
            actual URL before doing anything.`,
      action: "ignore",
      confidence: "high",
    });
  }

  return examples;
};

const generatePersonaDefinition = (cluster, format) => {
  const topTraits = [
    ...(cluster.top_high_traits || []).map(
      ([t, z]) => `high ${t.replace(/_/g, " ")} (+${z.toFixed(1)}Ïƒ)`,
    ),
    ...(cluster.top_low_traits || []).map(
      ([t, z]) => `low ${t.replace(/_/g, " ")} (${z.toFixed(1)}Ïƒ)`,
    ),
  ].slice(0, 5);

  // Convert cluster_id to 1-indexed for display
  const displayClusterId = cluster.cluster_id + 1;

  const base = {
    persona_id: `PERSONA_${displayClusterId}`,
    cluster_id: displayClusterId,
    name: cluster.label || `Persona ${displayClusterId}`,
    archetype:
      cluster.archetype || cluster.label || `Persona ${displayClusterId}`,
    risk_level: cluster.risk_level || "MEDIUM",
    n_participants: cluster.n_participants || cluster.size || 0,
    pct_of_population: cluster.pct_of_population || 0,
    description: cluster.description || "",
    target_accuracy: 0.85,
    acceptance_range: [0.8, 0.9],
    // Required fields with defaults for baseline format
    trait_zscores: cluster.trait_zscores || {},
    distinguishing_high_traits: (cluster.top_high_traits || []).map(([t]) => t),
    distinguishing_low_traits: (cluster.top_low_traits || []).map(([t]) => t),
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
      urgency_effect: cluster.email_interaction_effects?.urgency_effect || 0,
      familiarity_effect:
        cluster.email_interaction_effects?.familiarity_effect || 0,
      framing_effect: cluster.email_interaction_effects?.framing_effect || 0,
    },
    boundary_conditions: [],
    reasoning_examples: [],
  };

  if (format === "baseline") {
    return base;
  }

  const withStats = {
    ...base,
    cluster_id: cluster.cluster_id,
    n_participants: cluster.n_participants || cluster.size || 0,
    pct_of_population: cluster.pct_of_population || 0,
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
      urgency_effect: cluster.email_interaction_effects?.urgency_effect || 0,
      familiarity_effect:
        cluster.email_interaction_effects?.familiarity_effect || 0,
      framing_effect: cluster.email_interaction_effects?.framing_effect || 0,
    },
    trait_zscores: cluster.trait_zscores || {},
    distinguishing_high_traits: (cluster.top_high_traits || []).map(([t]) => t),
    distinguishing_low_traits: (cluster.top_low_traits || []).map(([t]) => t),
    cognitive_style: determineCognitiveStyle(cluster),
    psychological_profile: {
      distinguishing_traits: topTraits,
      cognitive_style: determineCognitiveStyle(cluster),
      trust_level:
        cluster.trait_zscores?.trust_propensity > 0.5
          ? "high"
          : cluster.trait_zscores?.trust_propensity < -0.5
            ? "low"
            : "moderate",
      impulsivity:
        cluster.trait_zscores?.impulsivity_total > 0.5
          ? "high"
          : cluster.trait_zscores?.impulsivity_total < -0.5
            ? "low"
            : "moderate",
    },
    vulnerability_profile: {
      urgency_susceptible: cluster.trait_zscores?.urgency_susceptibility > 0.5,
      authority_susceptible:
        cluster.trait_zscores?.authority_susceptibility > 0.5,
      familiarity_trusting: cluster.trait_zscores?.trust_propensity > 0.3,
    },
  };

  if (format === "stats") {
    return withStats;
  }

  // Full format with chain-of-thought
  return {
    ...withStats,
    reasoning_examples: generateReasoningExamples(cluster),
    boundary_conditions: [
      cluster.trait_zscores?.impulsivity_total > 0.5
        ? {
            type: "fast_heuristic_decisions",
            description:
              "May make fast, heuristic decisions that AI struggles to replicate",
            severity: "high",
          }
        : null,
      cluster.trait_zscores?.urgency_susceptibility > 0.5
        ? {
            type: "emotional_manipulation",
            description:
              "Emotional manipulation triggers may not affect AI similarly",
            severity: "medium",
          }
        : null,
      cluster.trait_zscores?.crt_score < -0.5
        ? {
            type: "intuitive_processing",
            description:
              "Intuitive processing may lead to different decision patterns than AI deliberation",
            severity: "medium",
          }
        : null,
    ].filter(Boolean),
  };
};

export const AIExportTab = ({ clusteringResult, personaLabels }) => {
  const [selectedFormat, setSelectedFormat] = useState("full");

  if (!clusteringResult) {
    return (
      <EmptyState
        icon={<Sparkles size={48} />}
        title="Run clustering first"
        description="Export requires finalized clustering results."
      />
    );
  }

  const clusters = Object.values(clusteringResult.clusters || {});

  // Helper to get persona name (AI-generated or default)
  const getPersonaName = (cluster) => {
    const clusterId = cluster.cluster_id;
    if (personaLabels?.[clusterId]?.name) {
      return personaLabels[clusterId].name;
    }
    return cluster.label || `Persona ${clusterId + 1}`;
  };

  // Helper to get persona archetype (AI-generated or default)
  const getPersonaArchetype = (cluster) => {
    const clusterId = cluster.cluster_id;
    if (personaLabels?.[clusterId]?.archetype) {
      return personaLabels[clusterId].archetype;
    }
    return cluster.archetype || cluster.label || `Persona ${clusterId + 1}`;
  };

  // Modified generatePersonaDefinition to use AI-generated names
  const generatePersonaDefinitionWithNames = (cluster, format) => {
    const def = generatePersonaDefinition(cluster, format);
    // Override with AI-generated names if available
    def.name = getPersonaName(cluster);
    def.archetype = getPersonaArchetype(cluster);
    return def;
  };

  const allPersonaDefinitions = clusters.map((c) =>
    generatePersonaDefinitionWithNames(c, selectedFormat),
  );

  const handleExport = () => {
    const exportData = {
      format: selectedFormat,
      n_personas: allPersonaDefinitions.length,
      personas: allPersonaDefinitions,
      export_timestamp: new Date().toISOString(),
      clustering_config: {
        algorithm: clusteringResult.algorithm,
        k: clusteringResult.k,
      },
    };

    // Download as JSON - FIXED: Only download, don't proceed to Phase 2
    const blob = new Blob([JSON.stringify(exportData, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `cypearl_personas_${selectedFormat}_${new Date().toISOString().split("T")[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);

    // Show confirmation message
    alert(
      `Personas exported successfully! ${allPersonaDefinitions.length} personas saved to file.\n\nTo proceed to Phase 2, click the "Proceed to Phase 2" button in the header.`,
    );

    // NOTE: onExport callback REMOVED - Export button should only download JSON,
    // not proceed to Phase 2. Use "Proceed to Phase 2" button instead.
  };

  return (
    <div className="space-y-6">
      {/* Export Purpose */}
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Sparkles className="text-purple-600 mt-1" size={20} />
          <div>
            <h4 className="font-semibold text-purple-900">
              Phase 2 Preparation
            </h4>
            <p className="text-sm text-purple-700">
              Preview persona definitions that will be used in Phase 2 AI
              simulation. Each persona can be configured with three different
              prompt formats: baseline (task-only), behavioral statistics
              augmented, or chain-of-thought reasoning augmented. Review how
              each format structures the persona data before proceeding to Phase
              2.
            </p>
          </div>
        </div>
      </div>

      {/* Important Clarification */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex items-start gap-3">
          <Info className="text-blue-600 mt-0.5 shrink-0" size={18} />
          <div className="text-sm text-blue-800">
            <strong>Note:</strong> The core persona data (29 traits, 6
            behavioral metrics) is the <strong>same</strong> across all formats.
            The format selection only controls which{" "}
            <strong>additional derived fields</strong> are included in the
            export. In Phase 2, the prompt configuration (baseline/stats/CoT)
            determines how this data is <em>presented</em> to the LLM.
          </div>
        </div>
      </div>


      {/* Export Configuration */}
      <Card>
        <h3 className="text-lg font-semibold mb-4">Export Format Selection</h3>
        <div className="grid grid-cols-3 gap-4 mb-6">
          {["baseline", "stats", "full"].map((format) => {
            const features = FORMAT_FEATURES[format];
            const isSelected = selectedFormat === format;
            return (
              <div
                key={format}
                className={`p-4 border rounded-lg cursor-pointer transition ${isSelected ? "border-indigo-500 bg-indigo-50" : "hover:bg-gray-50"}`}
                onClick={() => setSelectedFormat(format)}
              >
                <label className="flex items-center gap-2 cursor-pointer">
                  <input
                    type="radio"
                    name="format"
                    checked={isSelected}
                    onChange={() => setSelectedFormat(format)}
                  />
                  <div>
                    <div className="font-medium">{features.name}</div>
                    <div className="text-xs text-gray-500">
                      {features.description}
                    </div>
                  </div>
                </label>
              </div>
            );
          })}
        </div>

        {/* Format Details */}
        <div className="bg-gray-50 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 mb-3">
            What's included in "{FORMAT_FEATURES[selectedFormat].name}"
          </h4>
          <div className="grid md:grid-cols-2 gap-4">
            <div>
              <div className="text-xs font-medium text-green-700 mb-2 flex items-center gap-1">
                <CheckCircle size={12} /> Includes:
              </div>
              <ul className="text-xs text-gray-600 space-y-1">
                {FORMAT_FEATURES[selectedFormat].includes.map((item, i) => (
                  <li
                    key={i}
                    className={
                      item.startsWith("  ") ? "ml-3 text-gray-500" : ""
                    }
                  >
                    {item.startsWith("  ") ? item : `• ${item}`}
                  </li>
                ))}
              </ul>
            </div>
            {FORMAT_FEATURES[selectedFormat].excludes.length > 0 && (
              <div>
                <div className="text-xs font-medium text-red-700 mb-2 flex items-center gap-1">
                  <XCircle size={12} /> Excludes:
                </div>
                <ul className="text-xs text-gray-600 space-y-1">
                  {FORMAT_FEATURES[selectedFormat].excludes.map((item, i) => (
                    <li key={i}>• {item}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </Card>

      {/* Preview Persona Definitions */}
      <Card>
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold">Example Persona Definition</h3>
          <button
            onClick={handleExport}
            className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm flex items-center gap-2 hover:bg-indigo-700"
          >
            <Download size={16} />
            Export JSON ({clusters.length} personas)
          </button>
        </div>

        <div className="space-y-4">
          {clusters.slice(0, 1).map((cluster) => {
            const def = generatePersonaDefinitionWithNames(cluster, selectedFormat);
            const riskColors =
              RISK_COLORS[def.risk_level] || RISK_COLORS.MEDIUM;

            return (
              <div
                key={cluster.cluster_id}
                className="border rounded-lg overflow-hidden"
              >
                <div className="bg-gray-50 px-4 py-2 flex items-center justify-between">
                  <span className="font-mono text-sm">{def.name}</span>
                  <span
                    className="px-2 py-0.5 rounded text-xs"
                    style={{
                      backgroundColor: riskColors.bg,
                      color: riskColors.text,
                    }}
                  >
                    {def.risk_level}
                  </span>
                </div>
                <pre className="p-4 text-xs overflow-auto max-h-64 bg-gray-900 text-green-400">
                  {JSON.stringify(def, null, 2)}
                </pre>
              </div>
            );
          })}
          {/* {clusters.length > 1 && (
                        <p className="text-sm text-gray-500 text-center">
                            ... and {clusters.length - 1} more personas
                        </p>
                    )} */}
        </div>
      </Card>

      {/* AI Model Target Matrix */}
      {/* <Card>
                <h3 className="text-lg font-semibold mb-4">Phase 2 Model Assignment Preview</h3>
                <p className="text-sm text-gray-500 mb-4">
                    Each persona will be tested across 5 AWS Bedrock models Ã— 3 prompt configurations = 15 combinations
                </p>
                <table className="w-full text-sm">
                    <thead className="bg-gray-50">
                        <tr>
                            <th className="px-4 py-2 text-left">Model</th>
                            <th className="px-4 py-2 text-left">Tier</th>
                            <th className="px-4 py-2 text-left">Est. Cost/1K calls</th>
                            <th className="px-4 py-2 text-left">Trials Needed</th>
                        </tr>
                    </thead>
                    <tbody>
                        {[
                            { model: 'Claude Sonnet 4.5', tier: 'Frontier', cost: '$13', trials: 'â‰¥30/email' },
                            { model: 'Amazon Nova Pro', tier: 'Mid', cost: '$2', trials: 'â‰¥30/email' },
                            { model: 'Llama 4 Maverick', tier: 'Budget', cost: '$1', trials: 'â‰¥30/email' },
                            { model: 'Mistral Large', tier: 'Mid', cost: '$8', trials: 'â‰¥30/email' },
                            { model: 'Cohere Command R+', tier: 'Mid', cost: '$11', trials: 'â‰¥30/email' },
                        ].map(m => (
                            <tr key={m.model} className="border-t">
                                <td className="px-4 py-2">{m.model}</td>
                                <td className="px-4 py-2">{m.tier}</td>
                                <td className="px-4 py-2">{m.cost}</td>
                                <td className="px-4 py-2">{m.trials}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </Card> */}

      {/* Boundary Conditions to Watch */}
      {/* <Card>
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <AlertTriangle className="text-amber-600" size={20} />
                    Predicted Boundary Conditions
                </h3>
                <p className="text-sm text-gray-500 mb-4">
                    Based on persona characteristics, these are likely AI failure cases to monitor in Phase 2:
                </p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                    <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                        <h4 className="font-medium text-red-900">Fast, Heuristic Clicks</h4>
                        <p className="text-sm text-red-700 mt-1">
                            High-impulsivity personas may click before LLMs can simulate deliberation
                        </p>
                        <div className="mt-2 text-xs text-red-600">
                            Affected: High impulsivity clusters
                        </div>
                    </div>
                    <div className="p-4 bg-orange-50 border border-orange-200 rounded-lg">
                        <h4 className="font-medium text-orange-900">Empathy/Urgency Responses</h4>
                        <p className="text-sm text-orange-700 mt-1">
                            Emotional manipulation may not trigger same response in LLMs
                        </p>
                        <div className="mt-2 text-xs text-orange-600">
                            Affected: High urgency susceptibility clusters
                        </div>
                    </div>
                    <div className="p-4 bg-yellow-50 border border-yellow-200 rounded-lg">
                        <h4 className="font-medium text-yellow-900">Over-Deliberation</h4>
                        <p className="text-sm text-yellow-700 mt-1">
                            LLMs may over-analyze where humans act on gut feeling
                        </p>
                        <div className="mt-2 text-xs text-yellow-600">
                            Affected: Low CRT score clusters
                        </div>
                    </div>
                </div>
            </Card>  */}
    </div>
  );
};

export default AIExportTab;
