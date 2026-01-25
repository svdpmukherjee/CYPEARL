/**
 * StepGuide Component - Provides contextual guidance for users
 *
 * This component displays step-by-step instructions, explanations,
 * and tips for each phase/tab of the CYPEARL pipeline.
 */

import React, { useState, useEffect } from "react";
import {
  Info,
  ChevronDown,
  ChevronUp,
  Lightbulb,
  AlertTriangle,
  CheckCircle,
  ArrowRight,
  HelpCircle,
  BookOpen,
  Target,
  Zap,
  Shield,
  Clock,
  TrendingUp,
} from "lucide-react";

// ============================================================================
// GUIDANCE CONTENT - Phase 1
// ============================================================================

export const PHASE1_GUIDANCE = {
  dataset: {
    title: "Step 1: Review Dataset",
    subtitle: "Check data quality before clustering",
    icon: BookOpen,
    color: "gray",
    description:
      "Verify your dataset has sufficient participants and quality for reliable persona discovery.",
    whatToDo: [
      "Confirm 100+ participants for reliable clustering",
      "Ensure missing data is under 10%",
    ],
    whyItMatters:
      "Data quality directly affects persona validity. Poor data leads to unreliable behavioral segments.",
    tips: [
      "More participants = more robust personas",
      "Industry diversity improves cross-domain use",
    ],
    nextStep: "Proceed to Clustering to discover behavioral personas.",
    canSkip: false,
    skipReason: null,
  },
  clustering: {
    title: "Step 2: Run Clustering",
    subtitle: "Discover distinct behavioral groups",
    icon: Target,
    color: "gray",
    description:
      "Use K-Sweep to find the optimal number of personas, then run clustering to segment participants.",
    whatToDo: [
      "Run K-Sweep optimization (typical range: 4-8 clusters)",
      "Set minimum cluster size 30+ for statistical validity",
      "Enable AI naming for automatic persona descriptions",
    ],
    whyItMatters:
      "Good clusters predict phishing behavior (η² > 0.10) with clear separation between personas.",
    tips: [
      "Higher behavioral weight ensures clusters predict phishing susceptibility",
      "Too similar? Try fewer clusters. Too varied? Try more.",
    ],
    nextStep:
      "Review the Personas tab to understand each group's characteristics.",
    canSkip: false,
    skipReason: null,
  },
  profiles: {
    title: "Step 3: Review Personas",
    subtitle: "Understand each persona's profile",
    icon: Target,
    color: "gray",
    description:
      "Examine distinguishing traits, risk levels, and behavioral patterns for each discovered persona.",
    whatToDo: [
      "Review distinguishing traits (z-scores > ±0.5 are meaningful)",
      "Check risk levels: Critical (>35% click), Low (<22% click)",
      "Customize persona names if needed",
    ],
    whyItMatters:
      "Understanding personas is essential for interpreting AI simulation results in Phase 2.",
    tips: [
      "Very small personas (<5%) may be outliers",
      "Note the cognitive style (analytical vs impulsive)",
    ],
    nextStep:
      "Export personas to Phase 2, or explore optional validation tabs.",
    canSkip: false,
    skipReason: null,
  },
  validation: {
    title: "Validation (Optional)",
    subtitle: "Verify behavioral prediction strength",
    icon: CheckCircle,
    color: "gray",
    description:
      "Check η² statistics showing how well personas predict phishing behavior.",
    whatToDo: [
      "Target η² > 0.10 for each behavioral outcome",
      "Identify outcomes with weak differentiation",
    ],
    whyItMatters:
      "High η² confirms personas are behaviorally distinct, not just psychologically different.",
    tips: [
      "η² > 0.20 is excellent for behavioral research",
      "Low η²? Try re-clustering with higher behavioral weight",
    ],
    nextStep: "Explore Interactions or proceed to Export.",
    canSkip: true,
    skipReason: "Skip if η² from Clustering was satisfactory (> 0.10)",
  },
  interactions: {
    title: "Interactions (Optional)",
    subtitle: "Email response patterns by persona",
    icon: Zap,
    color: "gray",
    description:
      "Analyze how personas respond differently to urgency, sender familiarity, and framing.",
    whatToDo: [
      "Click 'Analyze Interactions' to generate results",
      "Identify susceptibility patterns per persona",
    ],
    whyItMatters:
      "These patterns become calibration targets for LLM simulation in Phase 2.",
    tips: [
      "Large urgency effects = impulsive personas",
      "Large familiarity effects = trust-based decision making",
    ],
    nextStep: "Check Cross-Industry or proceed to Export.",
    canSkip: true,
    skipReason: "Skip if you only need overall behavioral profiles",
  },
  industry: {
    title: "Cross-Industry (Optional)",
    subtitle: "Check persona generalizability",
    icon: Shield,
    color: "gray",
    description:
      "Test if personas work across different industries or are sector-specific.",
    whatToDo: [
      "Run cross-industry analysis",
      "Target Cramer's V < 0.2 for good transferability",
    ],
    whyItMatters:
      "Industry-independent personas can be deployed in any organization without recalibration.",
    tips: ["V > 0.3 suggests industry-specific variants may be needed"],
    nextStep: "Try Expert Validation or proceed to Export.",
    canSkip: true,
    skipReason: "Skip for single-industry deployments",
  },
  expert: {
    title: "Expert Review (Optional)",
    subtitle: "Refine personas with expert feedback",
    icon: HelpCircle,
    color: "gray",
    description:
      "Optional validation stage for refining personas with expert input before Phase 2.",
    whatToDo: [
      "Have experts rate personas on Realism and Actionability",
      "Refine persona names and descriptions based on feedback",
    ],
    whyItMatters:
      "Expert input validates that personas are realistic and actionable for security interventions.",
    tips: [
      "Useful for research publications requiring external validation",
      "Skip for rapid prototyping",
    ],
    nextStep: "Proceed to Export to prepare personas for Phase 2.",
    canSkip: true,
    skipReason:
      "Skip for rapid prototyping. Recommended for publication-quality research.",
  },
  export: {
    title: "Step 4: Export to Phase 2",
    subtitle: "Prepare personas for AI calibration",
    icon: ArrowRight,
    color: "gray",
    description:
      "Review and export your personas to Phase 2 for LLM calibration experiments.",
    whatToDo: [
      "Review persona definitions",
      "Select export format (Full recommended)",
      "Click 'Proceed to Phase 2'",
    ],
    whyItMatters:
      "Full export provides LLMs with complete behavioral context for accurate persona simulation.",
    tips: ["You can return to refine personas anytime"],
    nextStep: "Continue to Phase 2 for LLM setup and calibration.",
    canSkip: false,
    skipReason: null,
  },
};

// ============================================================================
// GUIDANCE CONTENT - Phase 2
// ============================================================================

export const PHASE2_GUIDANCE = {
  overview: {
    title: "Phase 2: Overview",
    subtitle: "Review imported data before experiments",
    icon: BookOpen,
    color: "gray",
    description:
      "Verify personas from Phase 1 and email stimuli are loaded correctly.",
    whatToDo: [
      "Confirm personas imported from Phase 1",
      "Check email stimuli are loaded (16 factorial design)",
    ],
    whyItMatters:
      "Experiments require both personas and emails. This confirms your data is ready.",
    tips: [
      "Factorial email design enables effect preservation testing",
      "Upload additional emails in JSON/CSV if needed",
    ],
    nextStep: "Configure LLM providers in the Provider Setup tab.",
    canSkip: false,
    skipReason: null,
  },
  providers: {
    title: "Step 1: Configure Providers",
    subtitle: "Set up API access for LLMs",
    icon: Shield,
    color: "gray",
    description:
      "Enter your OpenRouter API key to access Claude, GPT, Mistral, and other models.",
    whatToDo: [
      "Enter OpenRouter API key (or use .env auto-populate)",
      "Select models across tiers: Frontier, Mid-tier, Budget",
    ],
    whyItMatters:
      "Different LLMs mimic personas differently. Testing multiple models identifies the best fit.",
    tips: [
      "Frontier models (Claude 3.5, GPT-4o) have highest fidelity",
      "Budget models are 10-100x cheaper but less accurate",
    ],
    nextStep: "Configure prompts in the Prompts tab.",
    canSkip: false,
    skipReason: null,
  },
  prompts: {
    title: "Step 2: Configure Prompts",
    subtitle: "Set up prompt templates",
    icon: Lightbulb,
    color: "gray",
    description:
      "Review the three prompt levels: Baseline, Stats, and Chain-of-Thought (CoT).",
    whatToDo: [
      "Understand: Baseline (traits), Stats (+outcomes), CoT (+reasoning)",
      "Preview rendered prompts for each persona",
    ],
    whyItMatters:
      "CoT typically produces highest fidelity but costs more tokens.",
    tips: [
      "Start with all three configs to compare effectiveness",
      "CoT includes actual participant reasoning examples",
    ],
    nextStep: "Build your experiment in the Experiment Builder tab.",
    canSkip: false,
    skipReason: null,
  },
  experiment: {
    title: "Step 3: Build Experiment",
    subtitle: "Configure the test matrix",
    icon: Target,
    color: "gray",
    description: "Define which personas × LLMs × prompts × emails to test.",
    whatToDo: [
      "Select personas, models, prompt configs, and emails",
      "Set trials per condition (30 recommended)",
    ],
    whyItMatters:
      "Full factorial: 5 personas × 10 models × 3 configs × 16 emails × 30 trials = 72,000 calls.",
    tips: [
      "Start small: 2-3 models, 1-2 configs for initial testing",
      "Cost estimate shown before running",
    ],
    nextStep: "Execute the experiment in the Execution tab.",
    canSkip: false,
    skipReason: null,
  },
  execution: {
    title: "Step 4: Run Experiment",
    subtitle: "Execute and monitor progress",
    icon: Zap,
    color: "gray",
    description: "Start execution, monitor progress and costs in real-time.",
    whatToDo: [
      "Click 'Start Experiment' to begin",
      "Monitor progress and watch for errors",
    ],
    whyItMatters:
      "System handles rate limiting, retries, and checkpointing automatically.",
    tips: [
      "Auto-checkpoints every 100 trials - safe to pause",
      "Can stop early and analyze partial results",
    ],
    nextStep: "View results in the Results tab once complete.",
    canSkip: false,
    skipReason: null,
  },
  results: {
    title: "Step 5: Analyze Results",
    subtitle: "Review fidelity metrics",
    icon: TrendingUp,
    color: "gray",
    description:
      "Examine fidelity matrix showing LLM × Persona accuracy vs human data.",
    whatToDo: [
      "Check accuracy ≥85%, decision agreement ≥80%, effect preservation r≥0.80",
      "Identify best model-prompt combinations per persona",
    ],
    whyItMatters:
      "Only high-fidelity configs should be deployed for real predictions.",
    tips: [
      "Effect preservation ensures urgency/familiarity effects are maintained",
      "Export best configurations for CISO deployment",
    ],
    nextStep: "Review boundary conditions or proceed to Publish.",
    canSkip: false,
    skipReason: null,
  },
  boundaries: {
    title: "Boundaries (Optional)",
    subtitle: "Where AI simulation fails",
    icon: AlertTriangle,
    color: "gray",
    description: "Identify where LLM personas deviate from human behavior.",
    whatToDo: [
      "Review high-severity boundary conditions",
      "Note which personas/emails cause most errors",
    ],
    whyItMatters:
      "Boundary conditions tell you when to trust AI vs require human testing.",
    tips: [
      "Impulsive personas often fail - LLMs over-rationalize",
      "High-urgency + unfamiliar sender is hardest to simulate",
    ],
    nextStep: "Fine-tune in Calibration or proceed to Publish.",
    canSkip: true,
    skipReason: "Skip if all fidelity metrics meet thresholds (≥85%)",
  },
  calibration: {
    title: "Calibration (Optional)",
    subtitle: "Validate on held-out data",
    icon: Target,
    color: "gray",
    description:
      "Test LLM predictions against held-out human data before full experiments.",
    whatToDo: [
      "Run calibration trials (uses 20% held-out test data)",
      "Target calibration accuracy ≥80%",
    ],
    whyItMatters:
      "Validates prompt configuration works before expensive full experiments.",
    tips: [
      "Focus on high-value personas first",
      "Small prompt tweaks can yield significant gains",
    ],
    nextStep: "If ≥80%, proceed to Publish validated configs.",
    canSkip: true,
    skipReason:
      "Skip to run full experiment directly and analyze fidelity afterwards",
  },
  publish: {
    title: "Step 6: Publish Configs",
    subtitle: "Export for deployment",
    icon: ArrowRight,
    color: "gray",
    description:
      "Export validated persona-model-prompt configurations for CISO use.",
    whatToDo: [
      "Select configurations that met fidelity thresholds",
      "Generate deployment package (JSON export)",
    ],
    whyItMatters:
      "CISOs can use published configs without running new experiments.",
    tips: [
      "Only publish configs with ≥85% fidelity",
      "Include boundary condition documentation",
    ],
    nextStep: "Use published configs in CISO dashboard for predictions.",
    canSkip: false,
    skipReason: null,
  },
};

// ============================================================================
// STEP GUIDE COMPONENT
// ============================================================================

export const StepGuide = ({
  phase,
  tab,
  collapsed = false,
  showNextStep = true,
  className = "",
}) => {
  const [isExpanded, setIsExpanded] = useState(!collapsed);

  // Reset expansion state when tab changes - always start collapsed on new tab
  useEffect(() => {
    setIsExpanded(!collapsed);
  }, [tab, collapsed]);

  // Get the appropriate guidance
  const guidance = phase === 1 ? PHASE1_GUIDANCE[tab] : PHASE2_GUIDANCE[tab];

  if (!guidance) {
    return null;
  }

  const Icon = guidance.icon;
  // Consistent gray color scheme for all step guides
  const colorClasses = {
    gray: {
      bg: "bg-gray-200",
      border: "border-gray-200",
      icon: "text-gray-600",
      title: "text-gray-900",
    },
    // Legacy colors kept for backward compatibility but all now default to gray
    blue: {
      bg: "bg-gray-50",
      border: "border-gray-200",
      icon: "text-gray-600",
      title: "text-gray-900",
    },
    indigo: {
      bg: "bg-gray-50",
      border: "border-gray-200",
      icon: "text-gray-600",
      title: "text-gray-900",
    },
    purple: {
      bg: "bg-gray-50",
      border: "border-gray-200",
      icon: "text-gray-600",
      title: "text-gray-900",
    },
    green: {
      bg: "bg-gray-50",
      border: "border-gray-200",
      icon: "text-gray-600",
      title: "text-gray-900",
    },
    yellow: {
      bg: "bg-gray-50",
      border: "border-gray-200",
      icon: "text-gray-600",
      title: "text-gray-900",
    },
    orange: {
      bg: "bg-gray-50",
      border: "border-gray-200",
      icon: "text-gray-600",
      title: "text-gray-900",
    },
    cyan: {
      bg: "bg-gray-50",
      border: "border-gray-200",
      icon: "text-gray-600",
      title: "text-gray-900",
    },
    amber: {
      bg: "bg-gray-50",
      border: "border-gray-200",
      icon: "text-gray-600",
      title: "text-gray-900",
    },
  };

  const colors = colorClasses[guidance.color] || colorClasses.blue;

  return (
    <div
      className={`rounded-xl border ${colors.border} ${colors.bg} overflow-hidden ${className}`}
    >
      {/* Header - Always visible */}
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-4 py-3 flex items-center justify-between hover:bg-white/50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <div className={`p-2 rounded-lg bg-white shadow-sm`}>
            <Icon className={colors.icon} size={20} />
          </div>
          <div className="text-left">
            <h3 className={`font-semibold ${colors.title}`}>
              {guidance.title}
            </h3>
            <p className="text-sm text-gray-600">{guidance.subtitle}</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          {guidance.canSkip && (
            <span className="text-xs px-2 py-1 rounded-full bg-white text-gray-500 border border-gray-200">
              Optional
            </span>
          )}
          {isExpanded ? (
            <ChevronUp className="text-gray-400" size={20} />
          ) : (
            <ChevronDown className="text-gray-400" size={20} />
          )}
        </div>
      </button>

      {/* Expanded Content */}
      {isExpanded && (
        <div className="px-4 pb-4 space-y-4">
          {/* Description */}
          <p className="text-gray-700 text-sm leading-relaxed">
            {guidance.description}
          </p>

          {/* What To Do */}
          <div className="bg-gray-100 rounded-lg p-3 border border-gray-100">
            <h4 className="font-medium text-gray-900 mb-2 flex items-center gap-2">
              <CheckCircle size={16} className="text-green-500" />
              What to do:
            </h4>
            <ul className="space-y-1">
              {guidance.whatToDo.map((item, index) => (
                <li
                  key={index}
                  className="text-sm text-gray-600 flex items-start gap-2"
                >
                  <span className="text-gray-400 mt-1">•</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Why It Matters */}
          <div className="bg-gray-100 rounded-lg p-3 border border-gray-100">
            <h4 className="font-medium text-gray-900 mb-2 flex items-center gap-2">
              <Info size={16} className="text-blue-500" />
              Why it matters:
            </h4>
            <p className="text-sm text-gray-600">{guidance.whyItMatters}</p>
          </div>

          {/* Tips */}
          <div className="bg-gray-100 rounded-lg p-3 border border-gray-100">
            <h4 className="font-medium text-gray-900 mb-2 flex items-center gap-2">
              <Lightbulb size={16} className="text-amber-500" />
              Tips:
            </h4>
            <ul className="space-y-1">
              {guidance.tips.map((tip, index) => (
                <li
                  key={index}
                  className="text-sm text-gray-600 flex items-start gap-2"
                >
                  <span className="text-amber-400 mt-1">*</span>
                  <span>{tip}</span>
                </li>
              ))}
            </ul>
          </div>

          {/* Skip Info */}
          {guidance.canSkip && guidance.skipReason && (
            <div className="bg-amber-50 rounded-lg p-3 border border-amber-200">
              <h4 className="font-medium text-amber-900 mb-1 flex items-center gap-2">
                <AlertTriangle size={16} className="text-amber-500" />
                Can I skip this step?
              </h4>
              <p className="text-sm text-amber-700">{guidance.skipReason}</p>
            </div>
          )}

          {/* Next Step */}
          {showNextStep && guidance.nextStep && (
            <div className="flex items-center gap-2 pt-2 border-t border-gray-200">
              <ArrowRight size={16} className="text-gray-400" />
              <span className="text-sm text-gray-600">
                <strong>Next:</strong> {guidance.nextStep}
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// QUICK TIP COMPONENT (Smaller inline hints)
// ============================================================================

export const QuickTip = ({ children, type = "info", className = "" }) => {
  const typeConfig = {
    info: {
      icon: Info,
      bg: "bg-blue-50",
      border: "border-blue-200",
      text: "text-blue-700",
      iconColor: "text-blue-500",
    },
    tip: {
      icon: Lightbulb,
      bg: "bg-amber-50",
      border: "border-amber-200",
      text: "text-amber-700",
      iconColor: "text-amber-500",
    },
    warning: {
      icon: AlertTriangle,
      bg: "bg-red-50",
      border: "border-red-200",
      text: "text-red-700",
      iconColor: "text-red-500",
    },
    success: {
      icon: CheckCircle,
      bg: "bg-green-50",
      border: "border-green-200",
      text: "text-green-700",
      iconColor: "text-green-500",
    },
  };

  const config = typeConfig[type];
  const Icon = config.icon;

  return (
    <div
      className={`flex items-start gap-2 p-3 rounded-lg border ${config.bg} ${config.border} ${className}`}
    >
      <Icon size={16} className={`${config.iconColor} mt-0.5 flex-shrink-0`} />
      <span className={`text-sm ${config.text}`}>{children}</span>
    </div>
  );
};

// ============================================================================
// PIPELINE OVERVIEW COMPONENT
// ============================================================================

export const PipelineOverview = ({
  currentPhase,
  currentTab,
  completedSteps = [],
}) => {
  const phase1Steps = Object.keys(PHASE1_GUIDANCE);
  const phase2Steps = Object.keys(PHASE2_GUIDANCE);

  const getStepStatus = (phase, step) => {
    const stepKey = `phase${phase}-${step}`;
    if (completedSteps.includes(stepKey)) return "completed";
    if (currentPhase === phase && currentTab === step) return "current";
    return "pending";
  };

  return (
    <div className="bg-white rounded-xl border border-gray-200 p-4">
      <h3 className="font-semibold text-gray-900 mb-3">Pipeline Progress</h3>

      {/* Phase 1 */}
      <div className="mb-4">
        <h4 className="text-sm font-medium text-gray-700 mb-2">
          Phase 1: Persona Discovery
        </h4>
        <div className="flex flex-wrap gap-1">
          {phase1Steps.map((step) => {
            const status = getStepStatus(1, step);
            const guidance = PHASE1_GUIDANCE[step];
            return (
              <span
                key={step}
                className={`px-2 py-1 rounded text-xs font-medium ${
                  status === "completed"
                    ? "bg-green-100 text-green-700"
                    : status === "current"
                      ? "bg-indigo-100 text-indigo-700 ring-2 ring-indigo-300"
                      : "bg-gray-100 text-gray-500"
                }`}
                title={guidance.title}
              >
                {step.charAt(0).toUpperCase() + step.slice(1)}
              </span>
            );
          })}
        </div>
      </div>

      {/* Phase 2 */}
      <div>
        <h4 className="text-sm font-medium text-gray-700 mb-2">
          Phase 2: AI Calibration
        </h4>
        <div className="flex flex-wrap gap-1">
          {phase2Steps.map((step) => {
            const status = getStepStatus(2, step);
            const guidance = PHASE2_GUIDANCE[step];
            return (
              <span
                key={step}
                className={`px-2 py-1 rounded text-xs font-medium ${
                  status === "completed"
                    ? "bg-green-100 text-green-700"
                    : status === "current"
                      ? "bg-purple-100 text-purple-700 ring-2 ring-purple-300"
                      : "bg-gray-100 text-gray-500"
                }`}
                title={guidance.title}
              >
                {step.charAt(0).toUpperCase() + step.slice(1)}
              </span>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default StepGuide;
