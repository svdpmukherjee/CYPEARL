/**
 * CYPEARL Phase 2 Admin Dashboard - v3 Improved (FIXED Dec 31, 2024)
 *
 * Features:
 * 1. Overview: Collapsed persona/email summaries with expand option
 * 2. Providers: Factorial design matrix for LLM comparison + provider selection
 * 3. Prompts: CISO-style prompt configuration with preview and edit
 * 4. Experiment: Clean experiment builder from Phase2Dashboard
 * 5. Execution, Results, Publish: Retained from v3
 *
 * FIXES APPLIED (Dec 31, 2024):
 * - Unicode characters fixed: • instead of Ã¢â‚¬Â¢
 * - Email display now shows: Urgency (⚡/🕐), Familiarity (✓/?), Framing (⚠️/🎁)
 * - Email stats expanded to include all factors
 * - Email summary stats added below email list
 */

import React, { useState, useEffect, useMemo, useRef } from "react";
import {
  Play,
  Pause,
  CheckCircle,
  XCircle,
  AlertTriangle,
  Settings,
  Users,
  Mail,
  Cpu,
  DollarSign,
  BarChart3,
  Target,
  Zap,
  ChevronDown,
  ChevronRight,
  RefreshCw,
  Download,
  Upload,
  Brain,
  Sparkles,
  TrendingUp,
  Clock,
  Activity,
  Eye,
  EyeOff,
  Send,
  FileText,
  Code,
  Grid,
  Copy,
  Check,
  Share2,
  Lock,
  AlertCircle,
  Server,
  Cloud,
  Key,
  ExternalLink,
  Info,
  Edit3,
  Layers,
  Globe,
  Shield,
  Database,
  ArrowRight,
  Beaker,
} from "lucide-react";
import * as api from "../services/phase2Api";
import { CalibrationTab } from "./tabs/CalibrationTab";
import { ResultsTab as ResultsTabView } from "./tabs/ResultsTab";
import { StepGuide, PHASE2_GUIDANCE } from "./common/StepGuide";
import { ResearchTab } from "./tabs/ResearchTab";
import { SystematicCodeLegend } from "./common";

// ============================================================================
// CONSTANTS
// ============================================================================

// Single unified provider - all models via OpenRouter
const LLM_PROVIDERS = {
  openrouter: {
    id: "openrouter",
    name: "OpenRouter",
    description: "Unified gateway for Claude, GPT, Mistral, and Nova models",
    logo: "",
    color: "purple",
    website: "https://openrouter.ai",
    configFields: [
      {
        key: "api_key",
        label: "OpenRouter API Key",
        type: "password",
        placeholder: "sk-or-v1-...",
      },
    ],
    privacy: {
      noTraining: true, // OpenRouter doesn't train on data
      euResidency: false,
      gdpr: true,
      enterprise: true,
    },
    models: [
      // Claude models (Anthropic via OpenRouter)
      {
        id: "claude-3-5-sonnet",
        name: "Claude 3.5 Sonnet",
        tier: "frontier",
        cost_input: 3.0,
        cost_output: 15.0,
        family: "claude",
        openrouter_id: "anthropic/claude-3.5-sonnet",
      },
      {
        id: "claude-3-5-haiku",
        name: "Claude 3.5 Haiku",
        tier: "mid_tier",
        cost_input: 0.8,
        cost_output: 4.0,
        family: "claude",
        openrouter_id: "anthropic/claude-3.5-haiku",
      },
      {
        id: "claude-3-opus",
        name: "Claude 3 Opus",
        tier: "frontier",
        cost_input: 15.0,
        cost_output: 75.0,
        family: "claude",
        openrouter_id: "anthropic/claude-3-opus",
      },
      {
        id: "claude-3-sonnet",
        name: "Claude 3 Sonnet",
        tier: "mid_tier",
        cost_input: 3.0,
        cost_output: 15.0,
        family: "claude",
        openrouter_id: "anthropic/claude-3-sonnet",
      },
      {
        id: "claude-3-haiku",
        name: "Claude 3 Haiku",
        tier: "budget",
        cost_input: 0.25,
        cost_output: 1.25,
        family: "claude",
        openrouter_id: "anthropic/claude-3-haiku",
      },

      // GPT models (OpenAI via OpenRouter)
      {
        id: "gpt-4o",
        name: "GPT-4o",
        tier: "frontier",
        cost_input: 2.5,
        cost_output: 10.0,
        family: "gpt",
        openrouter_id: "openai/gpt-4o",
      },
      {
        id: "gpt-4o-mini",
        name: "GPT-4o Mini",
        tier: "mid_tier",
        cost_input: 0.15,
        cost_output: 0.6,
        family: "gpt",
        openrouter_id: "openai/gpt-4o-mini",
      },
      {
        id: "gpt-4-turbo",
        name: "GPT-4 Turbo",
        tier: "frontier",
        cost_input: 10.0,
        cost_output: 30.0,
        family: "gpt",
        openrouter_id: "openai/gpt-4-turbo",
      },
      {
        id: "gpt-4",
        name: "GPT-4",
        tier: "frontier",
        cost_input: 30.0,
        cost_output: 60.0,
        family: "gpt",
        openrouter_id: "openai/gpt-4",
      },

      // Mistral models (via OpenRouter)
      {
        id: "mistral-large",
        name: "Mistral Large",
        tier: "frontier",
        cost_input: 2.0,
        cost_output: 6.0,
        family: "mistral",
        openrouter_id: "mistralai/mistral-large",
      },
      {
        id: "mistral-medium",
        name: "Mistral Medium",
        tier: "mid_tier",
        cost_input: 2.7,
        cost_output: 8.1,
        family: "mistral",
        openrouter_id: "mistralai/mistral-medium",
      },
      {
        id: "mistral-small",
        name: "Mistral Small",
        tier: "mid_tier",
        cost_input: 1.0,
        cost_output: 3.0,
        family: "mistral",
        openrouter_id: "mistralai/mistral-small",
      },
      {
        id: "mistral-7b",
        name: "Mistral 7B Instruct",
        tier: "budget",
        cost_input: 0.07,
        cost_output: 0.07,
        family: "mistral",
        openrouter_id: "mistralai/mistral-7b-instruct",
      },
      {
        id: "mixtral-8x7b",
        name: "Mixtral 8x7B",
        tier: "open_source",
        cost_input: 0.24,
        cost_output: 0.24,
        family: "mistral",
        openrouter_id: "mistralai/mixtral-8x7b-instruct",
      },
      {
        id: "mixtral-8x22b",
        name: "Mixtral 8x22B",
        tier: "mid_tier",
        cost_input: 0.65,
        cost_output: 0.65,
        family: "mistral",
        openrouter_id: "mistralai/mixtral-8x22b-instruct",
      },

      // Amazon Nova models (via OpenRouter)
      {
        id: "nova-pro",
        name: "Amazon Nova Pro",
        tier: "frontier",
        cost_input: 0.8,
        cost_output: 3.2,
        family: "nova",
        openrouter_id: "amazon/nova-pro-v1",
      },
      {
        id: "nova-lite",
        name: "Amazon Nova Lite",
        tier: "mid_tier",
        cost_input: 0.06,
        cost_output: 0.24,
        family: "nova",
        openrouter_id: "amazon/nova-lite-v1",
      },
      {
        id: "nova-micro",
        name: "Amazon Nova Micro",
        tier: "budget",
        cost_input: 0.035,
        cost_output: 0.14,
        family: "nova",
        openrouter_id: "amazon/nova-micro-v1",
      },

      // Llama 4 models (Meta via OpenRouter)
      {
        id: "llama-4-maverick",
        name: "Llama 4 Maverick",
        tier: "frontier",
        cost_input: 0.15,
        cost_output: 0.6,
        family: "llama",
        openrouter_id: "meta-llama/llama-4-maverick",
      },
      {
        id: "llama-4-scout",
        name: "Llama 4 Scout",
        tier: "mid_tier",
        cost_input: 0.08,
        cost_output: 0.3,
        family: "llama",
        openrouter_id: "meta-llama/llama-4-scout",
      },

      // Llama 3.x models (Meta via OpenRouter)
      {
        id: "llama-3.3-70b",
        name: "Llama 3.3 70B Instruct",
        tier: "open_source",
        cost_input: 0.1,
        cost_output: 0.32,
        family: "llama",
        openrouter_id: "meta-llama/llama-3.3-70b-instruct",
      },
      {
        id: "llama-3.1-405b",
        name: "Llama 3.1 405B Instruct",
        tier: "frontier",
        cost_input: 3.5,
        cost_output: 3.5,
        family: "llama",
        openrouter_id: "meta-llama/llama-3.1-405b-instruct",
      },
      {
        id: "llama-3.1-70b",
        name: "Llama 3.1 70B Instruct",
        tier: "open_source",
        cost_input: 0.4,
        cost_output: 0.4,
        family: "llama",
        openrouter_id: "meta-llama/llama-3.1-70b-instruct",
      },
    ],
  },
};

// LLM Family definitions for factorial design
const LLM_FAMILIES = {
  claude: { name: "Claude", color: "red", vendor: "Anthropic" },
  gpt: { name: "GPT", color: "green", vendor: "OpenAI" },
  llama: { name: "Llama", color: "blue", vendor: "Meta" },
  mistral: { name: "Mistral", color: "purple", vendor: "Mistral AI" },
  nova: { name: "Nova", color: "orange", vendor: "Amazon" },
};

const MODEL_TIERS = {
  frontier: {
    label: "Frontier",
    color: "#8B5CF6",
    bg: "bg-purple-100",
    text: "text-purple-700",
  },
  mid_tier: {
    label: "Mid-Tier",
    color: "#3B82F6",
    bg: "bg-blue-100",
    text: "text-blue-700",
  },
  open_source: {
    label: "Open Source",
    color: "#10B981",
    bg: "bg-green-100",
    text: "text-green-700",
  },
  budget: {
    label: "Budget",
    color: "#6B7280",
    bg: "bg-gray-100",
    text: "text-gray-700",
  },
};

// ============================================================================
// REALISTIC PROMPT TEMPLATES (Matching backend prompt_builder.py)
// These show what the backend ACTUALLY generates for each config
// ============================================================================
// ============================================================================
// INCREMENTAL PROMPT TEMPLATES (Matching backend prompt_builder.py EXACTLY)
// BASELINE → STATS → COT builds incrementally, each adding to the previous
// ============================================================================
const DEFAULT_PROMPT_TEMPLATES = {
  baseline: {
    id: "baseline",
    name: "Baseline",
    description:
      "Roleplaying frame + ALL 29 psychological traits + action options",
    token_estimate: "600-900",
    temperature: "0.7-1.0 (based on cognitive style)",
    includes: [
      "Roleplaying frame ('You are experiencing life as...')",
      "Persona name & description",
      "ALL 29 psychological/cognitive traits in simple language",
      "Basic action options (CLICK/REPORT/IGNORE)",
      "Optional: ICL examples (minimal - action only)",
    ],
    excludes: [
      "8 behavioral outcomes",
      "Situational context",
      "Chain-of-thought reasoning",
    ],
    system_template: `You are experiencing life as "{persona_name}" for a moment.

WHO YOU ARE:
{persona_description}

YOUR CHARACTERISTICS (psychological and cognitive profile):

COGNITIVE STYLE:
  • You are {crt_trait_level} in analytical thinking
  • You are {nfc_trait_level} in enjoying complex thinking
  • You are {wm_trait_level} in memory
  • You are {impulsivity_trait_level} in acting quickly without thinking

PERSONALITY (Big 5):
  • You are {extraversion_level} outgoing
  • You are {agreeableness_level} agreeable
  • You are {conscientiousness_level} conscientious
  • You are {neuroticism_level} anxious/worried
  • You are {openness_level} open to new experiences

PSYCHOLOGICAL STATE:
  • You are {trust_level} trusting of others
  • You are {risk_level} comfortable with risks
  • You are {anxiety_level} currently anxious
  • You are {stress_level} under stress
  • You are {fatigue_level} tired

SECURITY AWARENESS:
  • You are {self_efficacy_level} confident in spotting phishing
  • You are {perceived_risk_level} aware phishing is a threat
  • You have {knowledge_level} phishing knowledge
  • You have {training_level} security training

SUSCEPTIBILITIES:
  • You are {authority_level} deferential to authority
  • You are {urgency_level} responsive to time pressure
  • You are {scarcity_level} responsive to scarcity

Based on these characteristics, respond to emails as this person would - not as an AI, but as THIS specific person with their particular traits and tendencies.

{icl_block_minimal}

Respond with ONLY one of these actions:
- CLICK (you would click the link/button)
- REPORT (you would report this as suspicious)
- IGNORE (you would ignore/delete this email)`,
    user_template: `You receive this email in your inbox:

From: {sender}
Subject: {subject}

{body}

---
How would you respond?`,
  },
  stats: {
    id: "stats",
    name: "Behavioral Stats",
    description:
      "BASELINE + 8 behavioral outcomes (phishing rate, accuracy, etc.)",
    token_estimate: "1200-1800",
    temperature: "0.7-1.0 (based on cognitive style)",
    includes: [
      "✓ ALL BASELINE content (29 traits + roleplaying)",
      "+ 8 BEHAVIORAL OUTCOMES:",
      "  1. Phishing click rate",
      "  2. Overall accuracy",
      "  3. Report rate",
      "  4. Mean response latency",
      "  5. Hover rate (link inspection)",
      "  6. Sender inspection rate",
      "  7. Urgency vulnerability effect",
      "  8. Familiarity vulnerability effect",
      "+ Situational context (varies per trial)",
      "+ ICL examples with brief reasoning",
    ],
    excludes: [
      "Chain-of-thought reasoning examples",
      "Explicit cognitive style instructions",
    ],
    system_template: `You are experiencing life as "{persona_name}" for a moment.

WHO YOU ARE:
{persona_description}

YOUR CHARACTERISTICS (psychological and cognitive profile):
{all_29_traits_formatted}

Based on these characteristics, respond to emails as this person would.

YOUR 8 BEHAVIORAL OUTCOMES (learn from these patterns):

1. PHISHING SUSCEPTIBILITY: You click on {click_rate}% of phishing emails
   → {click_description}

2. OVERALL ACCURACY: You correctly identify {accuracy}% of emails
   → {accuracy_description}

3. REPORTING BEHAVIOR: You report {report_rate}% of suspicious emails
   → {report_description}

4. DECISION SPEED: Average response time of {latency_seconds} seconds
   → {speed_description}

5. LINK INSPECTION: You hover over links {hover_rate}% of the time
   → {hover_description}

6. SENDER VERIFICATION: You check sender details {sender_rate}% of the time
   → {sender_description}

7. URGENCY VULNERABILITY: {urgency_effect}% more likely to click when urgent
   → {urgency_description}

8. FAMILIARITY VULNERABILITY: {familiarity_effect}% more likely to click familiar senders
   → {familiarity_description}

Use these outcomes to calibrate your responses - they represent how this person ACTUALLY behaves.

{icl_block_with_reasoning}

CURRENT SITUATION:
{situational_context}

CRITICAL INSTRUCTIONS:
- Your DEFAULT action is IGNORE - this is what you do most of the time
- Use your behavioral outcomes above to calibrate your responses
- CLICK only if the email genuinely seems legitimate, relevant, and safe to you
- REPORT only if something feels clearly wrong AND you notice it
- You're human with natural variation

Respond in this format:
ACTION: [CLICK/REPORT/IGNORE]
CONFIDENCE: [HIGH/MEDIUM/LOW]
BRIEF_THOUGHT: [One sentence - what made you decide?]`,
    user_template: `You receive this email in your inbox:

From: {sender}
Subject: {subject}

{body}

---
How would you respond?`,
  },
  cot: {
    id: "cot",
    name: "Chain-of-Thought",
    description: "STATS + actual participant reasoning from study responses",
    token_estimate: "1800-2800",
    temperature: "0.7-1.0 (based on cognitive style)",
    includes: [
      "✓ ALL STATS content (29 traits + 8 behavioral outcomes)",
      "+ ACTUAL participant reasoning from study:",
      "  • details_noticed: What they noticed in emails",
      "  • steps_taken: How they evaluated",
      "  • decision_reason: Why they decided",
      "  • confidence_reason: What made them confident",
      "  • unsure_about: What they were unsure about",
      "+ Explicit cognitive style instructions",
      "+ Full chain-of-thought ICL examples",
      "+ Decision process guide",
    ],
    excludes: ["Explicit percentage targets"],
    system_template: `You are experiencing life as "{persona_name}" for a moment.

WHO YOU ARE:
{persona_description}

YOUR CHARACTERISTICS (psychological and cognitive profile):
{all_29_traits_formatted}

{cognitive_style_instruction}

YOUR 8 BEHAVIORAL OUTCOMES (learn from these patterns):

1. PHISHING SUSCEPTIBILITY: You click on {click_rate}% of phishing emails
   → {click_description}

2. OVERALL ACCURACY: You correctly identify {accuracy}% of emails
   → {accuracy_description}

3. REPORTING BEHAVIOR: You report {report_rate}% of suspicious emails
   → {report_description}

4. DECISION SPEED: Average response time of {latency_seconds} seconds
   → {speed_description}

5. LINK INSPECTION: You hover over links {hover_rate}% of the time
   → {hover_description}

6. SENDER VERIFICATION: You check sender details {sender_rate}% of the time
   → {sender_description}

7. URGENCY VULNERABILITY: {urgency_effect}% more likely to click when urgent
   → {urgency_description}

8. FAMILIARITY VULNERABILITY: {familiarity_effect}% more likely to click familiar senders
   → {familiarity_description}

HOW THIS PERSON ACTUALLY THINKS THROUGH EMAILS:
(These are REAL examples of how they reasoned through email decisions)

{icl_block_full_cot}

When responding, think through emails the same way - notice details, consider what you see, then decide.

CURRENT SITUATION:
{situational_context}

DECISION PROCESS (think through this naturally like the examples above):
1. First impression - does this email seem relevant to me?
2. What details do I notice? (sender, subject, urgency, links)
3. What feels right or wrong about this?
4. Final decision based on my characteristics and outcomes

CRITICAL REMINDERS:
- You are NOT a security expert - you're a real person
- Use the reasoning examples above as a guide for how YOU think
- IGNORE is your default - you need a reason TO click
- Your decision depends on your current state and characteristics

Think through this AS this person would, then respond.

{response_format_based_on_style}`,
    user_template: `You receive this email in your inbox:

From: {sender}
Subject: {subject}

{body}

---
How would you respond?`,
  },
};

// Config-specific features for display
// NOTE: For controlled comparison, Stats and CoT use IDENTICAL feature sets (29 traits + 8 behavioral)
// The only difference is the presentation FORMAT
const CONFIG_FEATURES = {
  baseline: {
    behavioral_stats: false,
    trait_analysis: true, // All 29 traits in simple language
    icl_examples: "Minimal (action only)",
    pattern_blocks: false,
    situational_context: false,
    cognitive_style_instruction: false,
    response_format: "Single word (CLICK/REPORT/IGNORE)",
    traits_used: 29, // All 29 traits in simple language
    behavioral_features_used: 0, // No behavioral stats
    description:
      "Foundation: Roleplaying frame + ALL 29 traits in simple language",
  },
  stats: {
    behavioral_stats: true,
    trait_analysis: true,
    icl_examples: "6 examples with brief reasoning",
    pattern_blocks: true,
    situational_context: true,
    cognitive_style_instruction: false,
    response_format: "Structured (ACTION, CONFIDENCE, BRIEF_THOUGHT)",
    traits_used: 29, // Inherits all 29 traits from baseline
    behavioral_features_used: 8, // ADDS 8 behavioral outcomes
    description:
      "BASELINE + 8 behavioral outcomes (click rate, accuracy, report, latency, etc.)",
  },
  cot: {
    behavioral_stats: true,
    trait_analysis: true,
    icl_examples:
      "6 examples with full chain-of-thought from actual participants",
    pattern_blocks: true,
    situational_context: true,
    cognitive_style_instruction: true,
    response_format: "Full CoT (FIRST_REACTION, THINKING, ACTION, CONFIDENCE)",
    traits_used: 29, // Inherits from STATS
    behavioral_features_used: 8, // Inherits from STATS
    description:
      "STATS + actual participant reasoning (details_noticed, decision_reason, etc.)",
  },
};

// ============================================================================
// FEATURE COVERAGE DETAILS - All 29 traits & 8 behavioral features used uniformly
// For controlled comparison: Stats and CoT use identical features, only format differs
// ============================================================================
const TRAIT_COVERAGE = {
  // All 29 persona traits - ALL used in stats and cot for controlled comparison
  traits: [
    // Cognitive Traits (4)
    {
      name: "crt_score",
      category: "Cognitive",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Analytical thinking",
    },
    {
      name: "need_for_cognition",
      category: "Cognitive",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Enjoys complex thinking",
    },
    {
      name: "working_memory",
      category: "Cognitive",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Memory capacity",
    },
    {
      name: "impulsivity_total",
      category: "Cognitive",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Acts without thinking",
    },

    // Big 5 Personality (5)
    {
      name: "big5_extraversion",
      category: "Big 5",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Outgoing",
    },
    {
      name: "big5_agreeableness",
      category: "Big 5",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Cooperative",
    },
    {
      name: "big5_conscientiousness",
      category: "Big 5",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Organized",
    },
    {
      name: "big5_neuroticism",
      category: "Big 5",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Anxious",
    },
    {
      name: "big5_openness",
      category: "Big 5",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Open to experience",
    },

    // Psychological Traits (5)
    {
      name: "trust_propensity",
      category: "Psychological",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Trust in others",
    },
    {
      name: "risk_taking",
      category: "Psychological",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Risk comfort",
    },
    {
      name: "state_anxiety",
      category: "Psychological",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Current anxiety",
    },
    {
      name: "current_stress",
      category: "Psychological",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Current stress",
    },
    {
      name: "fatigue_level",
      category: "Psychological",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Tiredness level",
    },

    // Security Traits (7)
    {
      name: "phishing_self_efficacy",
      category: "Security",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Confidence spotting phishing",
    },
    {
      name: "perceived_risk",
      category: "Security",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Belief phishing is threat",
    },
    {
      name: "security_attitudes",
      category: "Security",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Care about security",
    },
    {
      name: "privacy_concern",
      category: "Security",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Privacy concerns",
    },
    {
      name: "phishing_knowledge",
      category: "Security",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Knowledge of techniques",
    },
    {
      name: "technical_expertise",
      category: "Security",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Technical skill",
    },
    {
      name: "prior_victimization",
      category: "Security",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Been phished before",
    },

    // Behavioral Predispositions (7)
    {
      name: "security_training",
      category: "Behavioral",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Has security training",
    },
    {
      name: "email_volume_numeric",
      category: "Behavioral",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Email volume",
    },
    {
      name: "link_click_tendency",
      category: "Behavioral",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Click tendency",
    },
    {
      name: "social_media_usage",
      category: "Behavioral",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Social media usage",
    },
    {
      name: "authority_susceptibility",
      category: "Susceptibility",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Defers to authority",
    },
    {
      name: "urgency_susceptibility",
      category: "Susceptibility",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Responds to pressure",
    },
    {
      name: "scarcity_susceptibility",
      category: "Susceptibility",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Responds to scarcity",
    },

    // Other (1)
    {
      name: "sensation_seeking",
      category: "Other",
      usedIn: ["baseline", "stats", "cot"],
      desc: "Seeks novelty",
    },
  ],

  // All 8 behavioral outcome features - ALL used in stats and cot
  behavioral: [
    {
      name: "phishing_click_rate",
      usedIn: ["stats", "cot"],
      desc: "Probabilistic anchor",
    },
    {
      name: "overall_accuracy",
      usedIn: ["stats", "cot"],
      desc: "Decision accuracy",
    },
    { name: "report_rate", usedIn: ["stats", "cot"], desc: "Report frequency" },
    {
      name: "mean_response_latency_ms",
      usedIn: ["stats", "cot"],
      desc: "Response speed",
    },
    { name: "hover_rate", usedIn: ["stats", "cot"], desc: "Link inspection" },
    {
      name: "sender_inspection_rate",
      usedIn: ["stats", "cot"],
      desc: "Email verification",
    },
    {
      name: "urgency_effect",
      usedIn: ["stats", "cot"],
      desc: "Urgency impact",
    },
    {
      name: "familiarity_effect",
      usedIn: ["stats", "cot"],
      desc: "Familiarity impact",
    },
  ],
};

// ============================================================================
// SHARED COMPONENTS
// ============================================================================

const Card = ({ children, className = "" }) => (
  <div
    className={`bg-white rounded-xl shadow-sm border border-gray-200 ${className}`}
  >
    {children}
  </div>
);

const Badge = ({ children, color = "gray", size = "sm" }) => {
  const colors = {
    gray: "bg-gray-100 text-gray-700",
    red: "bg-red-500 text-white",
    green: "bg-green-500 text-white",
    blue: "bg-blue-100 text-blue-700",
    purple: "bg-purple-100 text-purple-700",
    yellow: "bg-yellow-500 text-white",
    amber: "bg-amber-100 text-amber-700",
    orange: "bg-orange-500 text-white",
    indigo: "bg-indigo-100 text-indigo-700",
    cyan: "bg-cyan-100 text-cyan-700",
  };
  const sizes = { sm: "px-2 py-0.5 text-xs", md: "px-3 py-1 text-sm" };
  return (
    <span
      className={`rounded-full font-medium ${colors[color]} ${sizes[size]}`}
    >
      {children}
    </span>
  );
};

const ExpandableSection = ({
  title,
  icon: Icon,
  iconColor,
  summary,
  children,
  defaultExpanded = false,
}) => {
  const [expanded, setExpanded] = useState(defaultExpanded);
  return (
    <Card className="overflow-hidden">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full p-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
      >
        <div className="flex items-center gap-3">
          <Icon className={iconColor} size={20} />
          <span className="font-semibold text-gray-900">{title}</span>
        </div>
        <div className="flex items-center gap-3">
          <span className="text-sm text-gray-500">{summary}</span>
          <ChevronDown
            className={`transition-transform ${expanded ? "rotate-180" : ""}`}
            size={20}
          />
        </div>
      </button>
      {expanded && <div className="border-t p-4">{children}</div>}
    </Card>
  );
};

// ============================================================================
// OVERVIEW TAB - IMPROVED WITH COLLAPSED SUMMARIES
// ============================================================================

const OverviewTab = ({
  personas,
  emails,
  models,
  providers,
  usage,
  onImportPersonas,
  onLoadEmails,
  autoImported,
}) => {
  const [uploadError, setUploadError] = useState(null);
  const personaFileRef = useRef(null);
  const emailFileRef = useRef(null);

  const handlePersonaUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadError(null);
    try {
      const text = await file.text();
      const data = JSON.parse(text);
      await onImportPersonas(data);
    } catch (error) {
      setUploadError(`Failed to load personas: ${error.message}`);
    }
  };

  const handleEmailUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploadError(null);
    try {
      const text = await file.text();
      let data;
      if (file.name.endsWith(".json")) {
        data = JSON.parse(text);
      } else {
        const lines = text.split("\n");
        const headers = lines[0]
          .split(",")
          .map((h) => h.trim().replace(/"/g, ""));
        data = lines
          .slice(1)
          .filter((l) => l.trim())
          .map((line) => {
            const values =
              line.match(/(".*?"|[^",]+)(?=\s*,|\s*$)/g) || line.split(",");
            const obj = {};
            headers.forEach((h, i) => {
              let val = values[i]?.trim().replace(/"/g, "") || "";
              obj[h] =
                val === "TRUE" || val === "true"
                  ? true
                  : val === "FALSE" || val === "false"
                    ? false
                    : val;
            });
            return obj;
          });
      }
      await onLoadEmails(Array.isArray(data) ? data : data.emails || data);
    } catch (error) {
      setUploadError(`Failed to load emails: ${error.message}`);
    }
  };

  // Compute stats
  const totalProvidersOffered = Object.keys(LLM_PROVIDERS).length; // 6 providers including Ollama

  const riskCounts = useMemo(() => {
    const counts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
    personas.forEach((p) => {
      if (counts.hasOwnProperty(p.risk_level)) counts[p.risk_level]++;
    });
    return counts;
  }, [personas]);

  const emailStats = useMemo(() => {
    const phishing = emails.filter((e) => e.email_type === "phishing").length;
    const legit = emails.filter((e) => e.email_type === "legitimate").length;
    const urgencyHigh = emails.filter((e) => e.urgency_level === "high").length;
    const urgencyLow = emails.length - urgencyHigh;
    const familiarSenders = emails.filter(
      (e) => e.sender_familiarity === "familiar",
    ).length;
    const unfamiliarSenders = emails.length - familiarSenders;
    const threatFraming = emails.filter(
      (e) => e.framing_type === "threat",
    ).length;
    const rewardFraming = emails.length - threatFraming;
    return {
      phishing,
      legit,
      urgencyHigh,
      urgencyLow,
      familiarSenders,
      unfamiliarSenders,
      threatFraming,
      rewardFraming,
    };
  }, [emails]);

  const personaSummary = useMemo(() => {
    const parts = [];
    if (riskCounts.CRITICAL > 0) parts.push(`${riskCounts.CRITICAL} critical`);
    if (riskCounts.HIGH > 0) parts.push(`${riskCounts.HIGH} high`);
    if (riskCounts.MEDIUM > 0) parts.push(`${riskCounts.MEDIUM} medium`);
    if (riskCounts.LOW > 0) parts.push(`${riskCounts.LOW} low`);
    return parts.length > 0 ? parts.join(", ") : "None loaded";
  }, [riskCounts]);

  const emailSummary = useMemo(() => {
    if (emails.length === 0) return "None loaded";
    return `${emailStats.phishing} phishing, ${emailStats.legit} legitimate`;
  }, [emails, emailStats]);

  return (
    <div className="space-y-6">
      {/* Auto-import banner */}
      {autoImported && (
        <div className="p-4 bg-green-50 border border-green-200 rounded-xl flex items-center gap-3">
          <CheckCircle className="text-green-600" size={20} />
          <span className="text-green-800 font-medium">
            {personas.length} personas automatically imported from Phase 1
          </span>
        </div>
      )}

      {uploadError && (
        <div className="p-4 bg-red-50 border border-red-200 rounded-xl flex items-center gap-3">
          <AlertCircle className="text-red-600" size={20} />
          <span className="text-red-700">{uploadError}</span>
          <button
            onClick={() => setUploadError(null)}
            className="ml-auto text-red-400 hover:text-red-600"
          >
            Ãƒâ€”
          </button>
        </div>
      )}

      {/* Quick Stats - Simplified */}
      <div className="grid grid-cols-4 gap-4">
        {[
          {
            icon: Users,
            label: "Personas",
            value: personas.length,
            color: "purple",
          },
          { icon: Mail, label: "Emails", value: emails.length, color: "blue" },
          {
            icon: Server,
            label: "Providers",
            value: totalProvidersOffered,
            color: "indigo",
          },
          {
            icon: DollarSign,
            label: "Spent",
            value: `$${(usage?.total_cost_usd || 0).toFixed(2)}`,
            color: "amber",
          },
        ].map(({ icon: Icon, label, value, color }) => (
          <Card key={label} className="p-4">
            <div className="flex items-center gap-3">
              <div className={`p-2 bg-${color}-100 rounded-lg`}>
                <Icon className={`text-${color}-600`} size={20} />
              </div>
              <div>
                <div className="text-2xl font-bold">{value}</div>
                <div className="text-sm text-gray-500">{label}</div>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* Personas Section - Expandable */}
      <ExpandableSection
        title="Personas from Phase 1"
        icon={Users}
        iconColor="text-purple-600"
        summary={
          personas.length > 0
            ? `${personas.length} loaded • Risk: ${personaSummary}`
            : "None loaded"
        }
      >
        {personas.length > 0 ? (
          <div className="space-y-4">
            {/* Systematic Code Legend */}
            <div className="mb-2">
              <SystematicCodeLegend />
            </div>

            {/* Risk Summary */}
            <div className="grid grid-cols-4 gap-3">
              {Object.entries(riskCounts).map(([level, count]) => (
                <div
                  key={level}
                  className={`p-3 rounded-lg text-center ${
                    level === "CRITICAL"
                      ? "bg-red-50"
                      : level === "HIGH"
                        ? "bg-orange-50"
                        : level === "MEDIUM"
                          ? "bg-yellow-50"
                          : "bg-green-50"
                  }`}
                >
                  <div
                    className={`text-xl font-bold ${
                      level === "CRITICAL"
                        ? "text-red-600"
                        : level === "HIGH"
                          ? "text-orange-600"
                          : level === "MEDIUM"
                            ? "text-yellow-600"
                            : "text-green-600"
                    }`}
                  >
                    {count}
                  </div>
                  <div className="text-xs text-gray-600">{level}</div>
                </div>
              ))}
            </div>

            {/* Persona List */}
            <div className="max-h-64 overflow-y-auto space-y-2">
              {personas.map((p) => {
                const clickRate = (
                  (p.behavioral_statistics?.phishing_click_rate ||
                    p.behavioral_targets?.phishing_click_rate ||
                    0) * 100
                ).toFixed(0);
                const reportRate = (
                  (p.behavioral_statistics?.report_rate ||
                    p.behavioral_targets?.report_rate ||
                    0) * 100
                ).toFixed(0);
                return (
                  <div
                    key={p.persona_id}
                    className="flex items-start gap-3 p-3 bg-gray-50 rounded-lg hover:bg-gray-100"
                  >
                    <div className="flex-1 min-w-0">
                      <div className="font-medium text-gray-900">
                        {p.name || p.persona_id}
                      </div>
                      <div className="text-sm text-gray-500 mt-1">
                        {p.description}
                      </div>
                    </div>
                    <div className="text-sm text-gray-500 shrink-0 text-right">
                      <div className="text-red-600">{clickRate}% click</div>
                      <div className="text-green-600">{reportRate}% report</div>
                    </div>
                    <Badge
                      color={
                        p.risk_level === "CRITICAL"
                          ? "red"
                          : p.risk_level === "HIGH"
                            ? "orange"
                            : p.risk_level === "MEDIUM"
                              ? "yellow"
                              : "green"
                      }
                    >
                      {p.risk_level}
                    </Badge>
                  </div>
                );
              })}
            </div>

            <label className="inline-flex items-center gap-2 px-3 py-2 text-sm text-gray-500 hover:text-indigo-600 cursor-pointer">
              <Upload size={14} /> Replace personas
              <input
                ref={personaFileRef}
                type="file"
                accept=".json"
                onChange={handlePersonaUpload}
                className="hidden"
              />
            </label>
          </div>
        ) : (
          <div className="text-center py-8">
            <Users className="mx-auto text-gray-300 mb-3" size={40} />
            <p className="text-gray-600 mb-1">No personas loaded</p>
            <p className="text-sm text-gray-400 mb-4">
              Upload the JSON from Phase 1 AI Export
            </p>
            <label className="inline-flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg cursor-pointer hover:bg-purple-700">
              <Upload size={16} /> Upload Personas
              <input
                ref={personaFileRef}
                type="file"
                accept=".json"
                onChange={handlePersonaUpload}
                className="hidden"
              />
            </label>
          </div>
        )}
      </ExpandableSection>

      {/* Emails Section - Expandable */}
      <ExpandableSection
        title="Email Stimuli"
        icon={Mail}
        iconColor="text-blue-600"
        summary={emails.length > 0 ? emailSummary : "None loaded"}
      >
        {emails.length > 0 ? (
          <div className="space-y-4">
            {/* Email Stats */}
            <div className="grid grid-cols-4 gap-3">
              <div className="p-3 bg-red-50 rounded-lg text-center">
                <div className="text-xl font-bold text-red-600">
                  {emailStats.phishing}
                </div>
                <div className="text-xs text-gray-600">Phishing</div>
              </div>
              <div className="p-3 bg-green-50 rounded-lg text-center">
                <div className="text-xl font-bold text-green-600">
                  {emailStats.legit}
                </div>
                <div className="text-xs text-gray-600">Legitimate</div>
              </div>
              <div className="p-3 bg-orange-50 rounded-lg text-center">
                <div className="text-xl font-bold text-orange-600">
                  {emailStats.urgencyHigh}
                </div>
                <div className="text-xs text-gray-600">High Urgency</div>
              </div>
              <div className="p-3 bg-blue-50 rounded-lg text-center">
                <div className="text-xl font-bold text-blue-600">
                  {emailStats.familiarSenders}
                </div>
                <div className="text-xs text-gray-600">Familiar Sender</div>
              </div>
            </div>

            {/* Email List */}
            <div className="max-h-64 overflow-y-auto space-y-1">
              {emails.map((e) => (
                <div
                  key={e.email_id}
                  className="flex items-center gap-2 p-2 bg-gray-50 rounded text-sm"
                >
                  {/* Phishing/Legitimate Badge */}
                  <span
                    className={`px-1.5 py-0.5 rounded text-xs font-medium shrink-0 ${
                      e.email_type === "phishing"
                        ? "bg-red-100 text-red-700"
                        : "bg-green-100 text-green-700"
                    }`}
                  >
                    {e.email_type === "phishing" ? "Phish" : "Legit"}
                  </span>

                  {/* Subject Line */}
                  <span className="truncate flex-1 text-gray-700">
                    {e.subject_line || e.email_id}
                  </span>

                  {/* Email Factors (Urgency, Familiarity, Framing) */}
                  <div className="flex items-center gap-1 shrink-0 text-xs">
                    {/* Urgency Level */}
                    <span
                      className={`px-2 py-0.5 rounded font-medium ${
                        e.urgency_level === "high"
                          ? "bg-orange-100 text-orange-700"
                          : "bg-blue-100 text-blue-700"
                      }`}
                    >
                      {e.urgency_level === "high"
                        ? "High Urgency"
                        : "Low Urgency"}
                    </span>

                    {/* Sender Familiarity */}
                    <span
                      className={`px-2 py-0.5 rounded font-medium ${
                        e.sender_familiarity === "familiar"
                          ? "bg-green-100 text-green-700"
                          : "bg-gray-100 text-gray-700"
                      }`}
                    >
                      {e.sender_familiarity === "familiar"
                        ? "Familiar"
                        : "Stranger"}
                    </span>

                    {/* Framing Type */}
                    <span
                      className={`px-2 py-0.5 rounded font-medium ${
                        e.framing_type === "threat"
                          ? "bg-red-100 text-red-700"
                          : "bg-green-100 text-green-700"
                      }`}
                    >
                      {e.framing_type === "threat" ? "Threat" : "Reward"}
                    </span>
                  </div>
                </div>
              ))}
            </div>

            {/* Summary Stats */}
            <div className="mt-3 pt-3 border-t text-xs text-gray-500">
              <div className="grid grid-cols-3 gap-2">
                <div>
                  <span className="font-medium">Urgency:</span>{" "}
                  {emailStats.urgencyHigh} high, {emailStats.urgencyLow} low
                </div>
                <div>
                  <span className="font-medium">Familiarity:</span>{" "}
                  {emailStats.familiarSenders} known,{" "}
                  {emailStats.unfamiliarSenders} unknown
                </div>
                <div>
                  <span className="font-medium">Framing:</span>{" "}
                  {emailStats.threatFraming} threat, {emailStats.rewardFraming}{" "}
                  reward
                </div>
              </div>
            </div>

            <label className="inline-flex items-center gap-2 px-3 py-2 text-sm text-gray-500 hover:text-indigo-600 cursor-pointer">
              <Upload size={14} /> Replace emails
              <input
                ref={emailFileRef}
                type="file"
                accept=".json,.csv"
                onChange={handleEmailUpload}
                className="hidden"
              />
            </label>
          </div>
        ) : (
          <div className="text-center py-8">
            <Mail className="mx-auto text-gray-300 mb-3" size={40} />
            <p className="text-gray-600 mb-1">No emails loaded</p>
            <p className="text-sm text-gray-400 mb-4">
              Upload email_stimuli.csv or JSON
            </p>
            <label className="inline-flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg cursor-pointer hover:bg-blue-700">
              <Upload size={16} /> Upload Emails
              <input
                ref={emailFileRef}
                type="file"
                accept=".json,.csv"
                onChange={handleEmailUpload}
                className="hidden"
              />
            </label>
          </div>
        )}
      </ExpandableSection>
    </div>
  );
};

// ============================================================================
// PROVIDERS TAB - FACTORIAL DESIGN + PROVIDER CONFIG
// ============================================================================

const ProviderSetupTab = ({
  providers,
  models,
  onSetupProvider,
  onTestModel,
}) => {
  const [activeView, setActiveView] = useState("providers"); // 'factorial' | 'providers'
  const [expandedProvider, setExpandedProvider] = useState("openrouter");
  const [configs, setConfigs] = useState({});
  const [showSecrets, setShowSecrets] = useState({});
  const [testing, setTesting] = useState({});
  const [results, setResults] = useState({});
  const [selectedModels, setSelectedModels] = useState(new Set());
  const [envKeyLoaded, setEnvKeyLoaded] = useState(false);

  // Auto-populate API key from environment variables
  useEffect(() => {
    const loadEnvConfig = async () => {
      try {
        const envConfig = await api.getEnvConfig();
        if (envConfig?.openrouter?.configured && envConfig.openrouter.api_key) {
          setConfigs((prev) => ({
            ...prev,
            openrouter: {
              ...prev.openrouter,
              api_key: envConfig.openrouter.api_key,
            },
          }));
          setEnvKeyLoaded(true);
          console.log(
            "[Provider Setup] OpenRouter API key auto-populated from environment",
          );
        }
      } catch (error) {
        console.log("[Provider Setup] No env config available:", error.message);
      }
    };
    loadEnvConfig();
  }, []);

  // Build factorial matrix: Family x Tier
  const factorialMatrix = useMemo(() => {
    const matrix = {};
    Object.values(LLM_FAMILIES).forEach((family) => {
      matrix[family.name] = {};
      Object.keys(MODEL_TIERS).forEach((tier) => {
        matrix[family.name][tier] = [];
      });
    });

    Object.values(LLM_PROVIDERS).forEach((provider) => {
      provider.models.forEach((model) => {
        const familyInfo = LLM_FAMILIES[model.family];
        if (
          familyInfo &&
          matrix[familyInfo.name] &&
          matrix[familyInfo.name][model.tier]
        ) {
          matrix[familyInfo.name][model.tier].push({
            ...model,
            provider: provider.id,
            providerName: provider.name,
            providerConfigured: providers[provider.id]?.configured,
          });
        }
      });
    });

    return matrix;
  }, [providers]);

  const handleSetup = async (providerId) => {
    try {
      await onSetupProvider(providerId, configs[providerId] || {});
      setResults((prev) => ({
        ...prev,
        [providerId]: { ok: true, msg: "Configured!" },
      }));
    } catch (error) {
      setResults((prev) => ({
        ...prev,
        [providerId]: { ok: false, msg: error.message },
      }));
    }
  };

  const handleTest = async (modelId) => {
    setTesting((prev) => ({ ...prev, [modelId]: true }));
    try {
      const result = await onTestModel(modelId);
      const latencyMsg = result?.latency_ms
        ? `Latency: ${result.latency_ms}ms`
        : "Connected!";
      setResults((prev) => ({
        ...prev,
        [modelId]: { ok: true, msg: `Test successful! ${latencyMsg}` },
      }));
    } catch (error) {
      setResults((prev) => ({
        ...prev,
        [modelId]: { ok: false, msg: `Test failed: ${error.message}` },
      }));
    } finally {
      setTesting((prev) => ({ ...prev, [modelId]: false }));
    }
  };

  const toggleModelSelection = (modelId) => {
    setSelectedModels((prev) => {
      const next = new Set(prev);
      if (next.has(modelId)) next.delete(modelId);
      else next.add(modelId);
      return next;
    });
  };

  return (
    <div className="space-y-6">
      {/* View Toggle */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold">LLM Configuration</h2>
          <p className="text-gray-500">Select models and configure providers</p>
        </div>
        <div className="flex gap-2 bg-gray-100 p-1 rounded-lg">
          <button
            onClick={() => setActiveView("providers")}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeView === "providers"
                ? "bg-white shadow text-indigo-600"
                : "text-gray-600 hover:text-gray-900"
            }`}
          >
            <Server size={16} className="inline mr-2" />
            Providers
          </button>
          <button
            onClick={() => setActiveView("factorial")}
            className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors ${
              activeView === "factorial"
                ? "bg-white shadow text-indigo-600"
                : "text-gray-600 hover:text-gray-900"
            }`}
          >
            <Grid size={16} className="inline mr-2" />
            Factorial Design
          </button>
        </div>
      </div>

      {activeView === "factorial" ? (
        /* Factorial Design Matrix */
        <Card className="p-6 overflow-x-auto">
          <div className="mb-4">
            <h3 className="font-semibold text-lg mb-2">
              LLM Factorial Design Matrix
            </h3>
            <p className="text-sm text-gray-500">
              Compare models across families and performance tiers. Click cells
              to select models for experiments.
            </p>
          </div>

          <table className="w-full border-collapse">
            <thead>
              <tr>
                <th className="p-3 text-left bg-gray-50 rounded-tl-lg">
                  Family / Tier
                </th>
                {Object.entries(MODEL_TIERS).map(([tier, info]) => (
                  <th key={tier} className="p-3 text-center bg-gray-50">
                    <span
                      className={`px-2 py-1 rounded ${info.bg} ${info.text}`}
                    >
                      {info.label}
                    </span>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(LLM_FAMILIES).map(([familyKey, family]) => (
                <tr key={familyKey} className="border-t">
                  <td className="p-3 font-medium">
                    <div className="flex items-center gap-2">
                      <span
                        className={`w-3 h-3 rounded-full bg-${family.color}-500`}
                      ></span>
                      {family.name}
                      <span className="text-xs text-gray-400">
                        ({family.vendor})
                      </span>
                    </div>
                  </td>
                  {Object.keys(MODEL_TIERS).map((tier) => {
                    const models = factorialMatrix[family.name]?.[tier] || [];
                    return (
                      <td key={tier} className="p-2">
                        {models.length > 0 ? (
                          <div className="space-y-1">
                            {models.map((model) => (
                              <button
                                key={model.id}
                                onClick={() => toggleModelSelection(model.id)}
                                className={`w-full p-2 rounded-lg text-left text-xs transition-all ${
                                  selectedModels.has(model.id)
                                    ? "bg-indigo-100 border-2 border-indigo-500"
                                    : "bg-gray-50 border border-gray-200 hover:border-gray-300"
                                } ${
                                  !model.providerConfigured ? "opacity-60" : ""
                                }`}
                              >
                                <div className="font-medium truncate">
                                  {model.name}
                                </div>
                                <div className="flex items-center justify-between mt-1">
                                  <span className="text-gray-400">
                                    {model.providerName}
                                  </span>
                                  <span
                                    className={`${
                                      model.providerConfigured
                                        ? "text-green-600"
                                        : "text-gray-400"
                                    }`}
                                  >
                                    {model.providerConfigured ? "✓" : "○"}
                                  </span>
                                </div>
                                <div className="text-gray-400 mt-1">
                                  ${model.cost_input}/1K in
                                </div>
                              </button>
                            ))}
                          </div>
                        ) : (
                          <div className="p-2 text-center text-gray-300 text-xs">
                            —
                          </div>
                        )}
                      </td>
                    );
                  })}
                </tr>
              ))}
            </tbody>
          </table>

          {/* Legend */}
          <div className="mt-4 flex items-center gap-6 text-sm text-gray-500 border-t pt-4">
            <span className="flex items-center gap-2">
              <span className="w-4 h-4 bg-indigo-100 border-2 border-indigo-500 rounded"></span>{" "}
              Selected
            </span>
            <span className="flex items-center gap-2">
              <span className="text-green-600">✓</span> Provider configured
            </span>
            <span className="flex items-center gap-2">
              <span className="text-gray-400">○</span> Provider not configured
            </span>
            <span className="ml-auto">
              {selectedModels.size} models selected
            </span>
          </div>
        </Card>
      ) : (
        /* Provider Configuration Cards */
        <div className="space-y-4">
          {/* Privacy Compliance Header */}
          <Card className="p-4 bg-green-50 border-green-200">
            <div className="flex items-start gap-3">
              <Shield className="text-green-600 flex-shrink-0" size={20} />
              <div>
                <h3 className="font-semibold text-green-900">
                  Provider Privacy Compliance
                </h3>
                <p className="text-sm text-green-800 mt-1">
                  For GDPR compliance, verify providers meet: No training on
                  data, EU data residency, Enterprise terms
                </p>
              </div>
            </div>
          </Card>

          {Object.values(LLM_PROVIDERS).map((provider) => {
            const status = providers[provider.id];
            const isConfigured = status?.configured;
            const isExpanded = expandedProvider === provider.id;
            const colorMap = {
              amber: {
                bg: "bg-amber-50",
                border: "border-amber-200",
                btn: "bg-amber-600 hover:bg-amber-700",
              },
              green: {
                bg: "bg-green-50",
                border: "border-green-200",
                btn: "bg-green-600 hover:bg-green-700",
              },
              orange: {
                bg: "bg-orange-50",
                border: "border-orange-200",
                btn: "bg-orange-600 hover:bg-orange-700",
              },
              blue: {
                bg: "bg-blue-50",
                border: "border-blue-200",
                btn: "bg-blue-600 hover:bg-blue-700",
              },
              purple: {
                bg: "bg-purple-50",
                border: "border-purple-200",
                btn: "bg-purple-600 hover:bg-purple-700",
              },
              gray: {
                bg: "bg-gray-50",
                border: "border-gray-200",
                btn: "bg-gray-600 hover:bg-gray-700",
              },
            };
            const colors = colorMap[provider.color] || colorMap.gray;

            return (
              <Card
                key={provider.id}
                className={`overflow-hidden ${
                  isConfigured ? colors.border : ""
                }`}
              >
                <div
                  className={`p-4 cursor-pointer ${
                    isConfigured ? colors.bg : "hover:bg-gray-50"
                  }`}
                  onClick={() =>
                    setExpandedProvider(isExpanded ? null : provider.id)
                  }
                >
                  <div className="flex items-center gap-4">
                    <div className="text-3xl">{provider.logo}</div>
                    <div className="flex-1">
                      <div className="flex items-center gap-2">
                        <span className="font-semibold">{provider.name}</span>
                        {isConfigured && (
                          <CheckCircle className="text-green-500" size={16} />
                        )}
                      </div>
                      <div className="text-sm text-gray-500">
                        {provider.description}
                      </div>
                      {/* Privacy indicators */}
                      <div className="flex gap-3 mt-2 text-xs">
                        <span
                          className={
                            provider.privacy.noTraining === true
                              ? "text-green-600"
                              : "text-gray-400"
                          }
                        >
                          {provider.privacy.noTraining === true ? "✓" : "○"} No
                          Training
                        </span>
                        <span
                          className={
                            provider.privacy.euResidency === true
                              ? "text-green-600"
                              : "text-gray-400"
                          }
                        >
                          {provider.privacy.euResidency === true ? "✓" : "○"} EU
                          Data
                        </span>
                        <span
                          className={
                            provider.privacy.gdpr === true
                              ? "text-green-600"
                              : "text-gray-400"
                          }
                        >
                          {provider.privacy.gdpr === true ? "✓" : "○"} GDPR
                        </span>
                      </div>
                    </div>
                    <div className="text-right text-sm text-gray-500">
                      {provider.models.length} models
                    </div>
                    <ChevronDown
                      className={`transition-transform ${
                        isExpanded ? "rotate-180" : ""
                      }`}
                      size={20}
                    />
                  </div>
                </div>

                {isExpanded && (
                  <div className="p-4 border-t bg-white">
                    <div className="grid md:grid-cols-2 gap-6">
                      {/* Config */}
                      <div>
                        <h4 className="font-medium mb-3">Configuration</h4>
                        {/* Auto-populated indicator */}
                        {provider.id === "openrouter" && envKeyLoaded && (
                          <div className="mb-3 p-2 bg-green-50 border border-green-200 rounded-lg flex items-center gap-2">
                            <CheckCircle size={16} className="text-green-600" />
                            <span className="text-sm text-green-700">
                              API key auto-populated from environment (.env
                              file)
                            </span>
                          </div>
                        )}
                        <div className="space-y-3">
                          {provider.configFields.map((field) => (
                            <div key={field.key}>
                              <label className="block text-sm text-gray-600 mb-1">
                                {field.label}
                                {provider.id === "openrouter" &&
                                  field.key === "api_key" &&
                                  envKeyLoaded && (
                                    <span className="ml-2 text-xs text-green-600">
                                      (from .env)
                                    </span>
                                  )}
                              </label>
                              <div className="flex gap-2">
                                <input
                                  type={
                                    field.type === "password" &&
                                    !showSecrets[`${provider.id}-${field.key}`]
                                      ? "password"
                                      : "text"
                                  }
                                  placeholder={field.placeholder}
                                  className="flex-1 px-3 py-2 border rounded-lg text-sm"
                                  value={
                                    configs[provider.id]?.[field.key] || ""
                                  }
                                  onChange={(e) =>
                                    setConfigs((p) => ({
                                      ...p,
                                      [provider.id]: {
                                        ...p[provider.id],
                                        [field.key]: e.target.value,
                                      },
                                    }))
                                  }
                                />
                                {field.type === "password" && (
                                  <button
                                    onClick={() =>
                                      setShowSecrets((p) => ({
                                        ...p,
                                        [`${provider.id}-${field.key}`]:
                                          !p[`${provider.id}-${field.key}`],
                                      }))
                                    }
                                    className="px-3 border rounded-lg text-gray-500 hover:bg-gray-50"
                                  >
                                    {showSecrets[
                                      `${provider.id}-${field.key}`
                                    ] ? (
                                      <EyeOff size={16} />
                                    ) : (
                                      <Eye size={16} />
                                    )}
                                  </button>
                                )}
                              </div>
                            </div>
                          ))}
                          <div className="flex items-center gap-3">
                            <button
                              onClick={() => handleSetup(provider.id)}
                              className={`px-4 py-2 text-white rounded-lg text-sm ${colors.btn}`}
                            >
                              Save
                            </button>
                            <a
                              href={provider.website}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="text-sm text-indigo-600 hover:underline flex items-center gap-1"
                            >
                              Get API Key <ExternalLink size={12} />
                            </a>
                          </div>
                          {results[provider.id] && (
                            <div
                              className={`text-sm ${
                                results[provider.id].ok
                                  ? "text-green-600"
                                  : "text-red-600"
                              }`}
                            >
                              {results[provider.id].msg}
                            </div>
                          )}
                        </div>
                      </div>

                      {/* Models */}
                      <div>
                        <h4 className="font-medium mb-3">Available Models</h4>
                        <div className="space-y-2 max-h-48 overflow-y-auto">
                          {provider.models.map((model) => (
                            <div
                              key={model.id}
                              className="flex items-center justify-between p-2 bg-gray-50 rounded-lg"
                            >
                              <div>
                                <div className="text-sm font-medium">
                                  {model.name}
                                </div>
                                <div className="text-xs text-gray-400">
                                  ${model.cost_input}/1K in • $
                                  {model.cost_output}/1K out
                                </div>
                                {results[model.id] && (
                                  <div
                                    className={`text-xs mt-1 ${
                                      results[model.id].ok
                                        ? "text-green-600"
                                        : "text-red-600"
                                    }`}
                                  >
                                    {results[model.id].msg}
                                  </div>
                                )}
                              </div>
                              {isConfigured && (
                                <button
                                  onClick={() => handleTest(model.id)}
                                  disabled={testing[model.id]}
                                  className={`px-2 py-1 text-xs rounded ${
                                    testing[model.id]
                                      ? "bg-gray-100 text-gray-400"
                                      : results[model.id]?.ok
                                        ? "bg-green-100 text-green-700"
                                        : "bg-indigo-100 text-indigo-700 hover:bg-indigo-200"
                                  }`}
                                >
                                  {testing[model.id]
                                    ? "Testing..."
                                    : results[model.id]?.ok
                                      ? "Retest"
                                      : "Test"}
                                </button>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// PROMPTS TAB - ACCURATE REPRESENTATION OF BACKEND PROMPTS
// ============================================================================

const PromptsTab = ({ promptTemplates, onUpdatePrompts }) => {
  const [selectedPromptType, setSelectedPromptType] = useState("baseline");
  const [editingPrompt, setEditingPrompt] = useState(null);
  const [copied, setCopied] = useState(null);
  const [localEdits, setLocalEdits] = useState({});
  const [showComparison, setShowComparison] = useState(false);

  const templates = promptTemplates || DEFAULT_PROMPT_TEMPLATES;
  const currentTemplate = templates[selectedPromptType];
  const currentFeatures = CONFIG_FEATURES[selectedPromptType];

  const handleCopy = (text, type) => {
    navigator.clipboard.writeText(text);
    setCopied(type);
    setTimeout(() => setCopied(null), 2000);
  };

  const handleEdit = (field, value) => {
    setLocalEdits((prev) => ({
      ...prev,
      [selectedPromptType]: {
        ...prev[selectedPromptType],
        [field]: value,
      },
    }));
  };

  const handleSaveEdits = () => {
    if (localEdits[selectedPromptType]) {
      const updated = {
        ...templates,
        [selectedPromptType]: {
          ...templates[selectedPromptType],
          ...localEdits[selectedPromptType],
        },
      };
      onUpdatePrompts(updated);
      setEditingPrompt(null);
    }
  };

  const handleResetEdits = () => {
    setLocalEdits((prev) => {
      const next = { ...prev };
      delete next[selectedPromptType];
      return next;
    });
    setEditingPrompt(null);
  };

  const getDisplayTemplate = (field) => {
    return (
      localEdits[selectedPromptType]?.[field] || currentTemplate?.[field] || ""
    );
  };

  // Config-specific variables
  const VARIABLES_BY_CONFIG = {
    baseline: [
      { name: "{persona_name}", desc: "Persona identifier" },
      { name: "{persona_description}", desc: "Core description" },
      { name: "{icl_block_minimal}", desc: "ICL examples (action only)" },
      { name: "{sender}", desc: "Email sender" },
      { name: "{subject}", desc: "Email subject" },
      { name: "{body}", desc: "Email body" },
    ],
    stats: [
      { name: "{persona_name}", desc: "Persona identifier" },
      { name: "{persona_description}", desc: "Core description" },
      { name: "{cognitive_style}", desc: "Impulsive/Analytical/Balanced" },
      { name: "{high_traits}", desc: "Top 3 strong traits" },
      { name: "{low_traits}", desc: "Top 3 weak traits" },
      {
        name: "{behavioral_description}",
        desc: "Tendencies in natural language",
      },
      { name: "{prob_anchor}", desc: '"X out of Y emails" format' },
      { name: "{pattern_block}", desc: "Decision patterns from training" },
      {
        name: "{icl_block_with_reasoning}",
        desc: "6 ICL examples with reasoning",
      },
      {
        name: "{situational_context}",
        desc: "Varies per trial (stress, time)",
      },
      { name: "{sender}", desc: "Email sender" },
      { name: "{subject}", desc: "Email subject" },
      { name: "{body}", desc: "Email body" },
      { name: "{urgency}", desc: "High/Low" },
      { name: "{sender_type}", desc: "Familiar/Unfamiliar" },
      { name: "{framing}", desc: "Threat/Reward" },
    ],
    cot: [
      { name: "{persona_name}", desc: "Persona identifier" },
      { name: "{persona_description}", desc: "Core description" },
      { name: "{cognitive_style}", desc: "Impulsive/Analytical/Balanced" },
      {
        name: "{cognitive_style_instruction}",
        desc: "Style-specific reasoning guide",
      },
      { name: "{high_traits}", desc: "Top 4 strong traits" },
      { name: "{low_traits}", desc: "Top 4 weak traits" },
      {
        name: "{behavioral_description}",
        desc: "Tendencies in natural language",
      },
      { name: "{prob_anchor}", desc: '"X out of Y emails" format' },
      { name: "{pattern_block}", desc: "Decision patterns from training" },
      { name: "{icl_block_full_cot}", desc: "6 ICL examples with full CoT" },
      {
        name: "{situational_context}",
        desc: "Varies per trial (stress, time)",
      },
      {
        name: "{response_format_based_on_style}",
        desc: "Impulsive vs analytical format",
      },
      { name: "{sender}", desc: "Email sender" },
      { name: "{subject}", desc: "Email subject" },
      { name: "{body}", desc: "Email body" },
      { name: "{urgency}", desc: "High/Low" },
      { name: "{sender_type}", desc: "Familiar/Unfamiliar" },
      { name: "{framing}", desc: "Threat/Reward" },
    ],
  };

  const currentVariables = VARIABLES_BY_CONFIG[selectedPromptType] || [];

  return (
    <div className="space-y-6">
      <div className="text-center mb-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Prompt Configuration
        </h2>
        <p className="text-gray-500">
          Review prompt templates - these reflect what the backend actually
          generates
        </p>
      </div>

      {/* Prompt Type Selector */}
      <Card className="p-2">
        <div className="flex gap-2">
          {Object.values(templates).map((pt) => {
            const isActive = selectedPromptType === pt.id;
            const hasEdits = localEdits[pt.id];
            const features = CONFIG_FEATURES[pt.id];

            return (
              <button
                key={pt.id}
                onClick={() => setSelectedPromptType(pt.id)}
                className={`flex-1 p-4 rounded-lg text-left transition-all ${
                  isActive
                    ? "bg-indigo-100 border-2 border-indigo-500"
                    : "bg-gray-50 hover:bg-gray-100"
                }`}
              >
                <div className="flex items-center justify-between">
                  <div>
                    <div className="font-medium flex items-center gap-2">
                      {pt.name}
                      {hasEdits && <Badge color="amber">Edited</Badge>}
                    </div>
                    <div className="text-xs text-gray-500">
                      {pt.token_estimate} tokens
                    </div>
                  </div>
                </div>
                <p className="text-xs text-gray-500 mt-1">{pt.description}</p>
                <div className="mt-2 flex flex-wrap gap-1">
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded ${
                      features?.behavioral_stats
                        ? "bg-green-100 text-green-700"
                        : "bg-gray-100 text-gray-400"
                    }`}
                  >
                    Stats {features?.behavioral_stats ? "✓" : "✗"}
                  </span>
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded ${
                      features?.pattern_blocks
                        ? "bg-green-100 text-green-700"
                        : "bg-gray-100 text-gray-400"
                    }`}
                  >
                    Patterns {features?.pattern_blocks ? "✓" : "✗"}
                  </span>
                  <span
                    className={`text-xs px-1.5 py-0.5 rounded ${
                      features?.situational_context
                        ? "bg-green-100 text-green-700"
                        : "bg-gray-100 text-gray-400"
                    }`}
                  >
                    Context {features?.situational_context ? "✓" : "✗"}
                  </span>
                </div>
              </button>
            );
          })}
        </div>
      </Card>

      {/* Config Features Summary */}
      <Card className="p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="font-semibold flex items-center gap-2">
            <Settings size={16} className="text-indigo-600" />
            Configuration Details: {currentTemplate?.name}
          </h3>
          <button
            onClick={() => setShowComparison(!showComparison)}
            className="text-xs text-indigo-600 hover:text-indigo-800"
          >
            {showComparison ? "Hide Comparison" : "Compare All Configs"}
          </button>
        </div>

        {showComparison ? (
          /* Comparison Table */
          <div className="overflow-x-auto">
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b">
                  <th className="text-left p-2 font-medium">Feature</th>
                  <th className="text-center p-2 font-medium">Baseline</th>
                  <th className="text-center p-2 font-medium">Stats</th>
                  <th className="text-center p-2 font-medium">CoT</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b">
                  <td className="p-2">Behavioral Statistics</td>
                  <td className="text-center p-2">
                    <span className="text-red-500">✗</span>
                  </td>
                  <td className="text-center p-2">
                    <span className="text-green-500">✓</span>
                  </td>
                  <td className="text-center p-2">
                    <span className="text-green-500">✓</span>
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="p-2">Trait Analysis</td>
                  <td className="text-center p-2">
                    <span className="text-red-500">✗</span>
                  </td>
                  <td className="text-center p-2">
                    <span className="text-green-500">✓</span>
                  </td>
                  <td className="text-center p-2">
                    <span className="text-green-500">✓</span>
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="p-2">ICL Examples</td>
                  <td className="text-center p-2">Minimal</td>
                  <td className="text-center p-2">6 + reasoning</td>
                  <td className="text-center p-2">6 + full CoT</td>
                </tr>
                <tr className="border-b">
                  <td className="p-2">Pattern Blocks</td>
                  <td className="text-center p-2">
                    <span className="text-red-500">✗</span>
                  </td>
                  <td className="text-center p-2">
                    <span className="text-green-500">✓</span>
                  </td>
                  <td className="text-center p-2">
                    <span className="text-green-500">✓</span>
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="p-2">Situational Context</td>
                  <td className="text-center p-2">
                    <span className="text-red-500">✗</span>
                  </td>
                  <td className="text-center p-2">
                    <span className="text-green-500">✓</span>
                  </td>
                  <td className="text-center p-2">
                    <span className="text-green-500">✓</span>
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="p-2">Cognitive Style Instruction</td>
                  <td className="text-center p-2">
                    <span className="text-red-500">✗</span>
                  </td>
                  <td className="text-center p-2">
                    <span className="text-red-500">✗</span>
                  </td>
                  <td className="text-center p-2">
                    <span className="text-green-500">✓</span>
                  </td>
                </tr>
                <tr className="border-b">
                  <td className="p-2">Response Format</td>
                  <td className="text-center p-2">Single word</td>
                  <td className="text-center p-2">Structured</td>
                  <td className="text-center p-2">Full CoT</td>
                </tr>
                <tr className="border-b">
                  <td className="p-2">Token Estimate</td>
                  <td className="text-center p-2">400-600</td>
                  <td className="text-center p-2">1000-1500</td>
                  <td className="text-center p-2">1500-2500</td>
                </tr>
                <tr className="border-b">
                  <td className="p-2">Traits Used</td>
                  <td className="text-center p-2">29</td>
                  <td className="text-center p-2">29</td>
                  <td className="text-center p-2">29</td>
                </tr>
                <tr className="border-b bg-green-50">
                  <td className="p-2 font-medium">Behavioral Outcomes</td>
                  <td className="text-center p-2">0</td>
                  <td className="text-center p-2 text-green-700 font-medium">
                    8
                  </td>
                  <td className="text-center p-2 text-green-700 font-medium">
                    8
                  </td>
                </tr>
                <tr>
                  <td className="p-2 text-gray-500 italic" colSpan={4}>
                    Stats and CoT use identical feature sets for controlled
                    comparison
                  </td>
                </tr>
              </tbody>
            </table>
          </div>
        ) : (
          /* Current Config Details */
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="p-3 bg-gray-50 rounded-lg">
              <div className="text-xs text-gray-500 mb-1">Token Estimate</div>
              <div className="font-semibold">
                {currentTemplate?.token_estimate}
              </div>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <div className="text-xs text-gray-500 mb-1">Temperature</div>
              <div className="font-semibold">
                {currentTemplate?.temperature || "0.7-1.0"}
              </div>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <div className="text-xs text-gray-500 mb-1">Traits Used</div>
              <div className="font-semibold">
                {currentFeatures?.traits_used || 0} of 29
              </div>
            </div>
            <div className="p-3 bg-gray-50 rounded-lg">
              <div className="text-xs text-gray-500 mb-1">
                Behavioral Features
              </div>
              <div className="font-semibold">
                {currentFeatures?.behavioral_features_used || 0} of 8
              </div>
            </div>
          </div>
        )}

        {/* Includes/Excludes */}
        {!showComparison && (
          <div className="mt-4 grid md:grid-cols-2 gap-4">
            <div>
              <div className="text-xs font-medium text-green-700 mb-2">
                Includes:
              </div>
              <ul className="text-xs text-gray-600 space-y-1">
                {(currentTemplate?.includes || []).map((item, i) => (
                  <li key={i} className="flex items-center gap-1">
                    <CheckCircle size={12} className="text-green-500" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
            <div>
              <div className="text-xs font-medium text-red-700 mb-2">
                Excludes:
              </div>
              <ul className="text-xs text-gray-600 space-y-1">
                {(currentTemplate?.excludes || []).map((item, i) => (
                  <li key={i} className="flex items-center gap-1">
                    <XCircle size={12} className="text-red-400" />
                    {item}
                  </li>
                ))}
              </ul>
            </div>
          </div>
        )}
      </Card>

      {/* Prompt Content */}
      <div className="grid md:grid-cols-2 gap-6">
        {/* System Prompt */}
        <Card className="p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold flex items-center gap-2">
              <Code size={16} />
              System Prompt
            </h3>
            <div className="flex items-center gap-2">
              <button
                onClick={() =>
                  handleCopy(getDisplayTemplate("system_template"), "system")
                }
                className="p-1.5 hover:bg-gray-100 rounded"
                title="Copy"
              >
                {copied === "system" ? (
                  <Check size={14} className="text-green-600" />
                ) : (
                  <Copy size={14} />
                )}
              </button>
              <button
                onClick={() =>
                  setEditingPrompt(editingPrompt === "system" ? null : "system")
                }
                className={`p-1.5 rounded ${
                  editingPrompt === "system"
                    ? "bg-indigo-100 text-indigo-600"
                    : "hover:bg-gray-100"
                }`}
                title="Edit"
              >
                <Edit3 size={14} />
              </button>
            </div>
          </div>

          {editingPrompt === "system" ? (
            <textarea
              value={getDisplayTemplate("system_template")}
              onChange={(e) => handleEdit("system_template", e.target.value)}
              className="w-full h-96 p-3 text-xs font-mono border rounded-lg focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
            />
          ) : (
            <pre className="p-3 bg-gray-900 text-green-400 rounded-lg text-xs overflow-auto max-h-96 whitespace-pre-wrap">
              {getDisplayTemplate("system_template")}
            </pre>
          )}
        </Card>

        {/* User Prompt */}
        <Card className="p-4">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-semibold flex items-center gap-2">
              <Mail size={16} />
              User Prompt (per email)
            </h3>
            <div className="flex items-center gap-2">
              <button
                onClick={() =>
                  handleCopy(getDisplayTemplate("user_template"), "user")
                }
                className="p-1.5 hover:bg-gray-100 rounded"
                title="Copy"
              >
                {copied === "user" ? (
                  <Check size={14} className="text-green-600" />
                ) : (
                  <Copy size={14} />
                )}
              </button>
              <button
                onClick={() =>
                  setEditingPrompt(editingPrompt === "user" ? null : "user")
                }
                className={`p-1.5 rounded ${
                  editingPrompt === "user"
                    ? "bg-indigo-100 text-indigo-600"
                    : "hover:bg-gray-100"
                }`}
                title="Edit"
              >
                <Edit3 size={14} />
              </button>
            </div>
          </div>

          {editingPrompt === "user" ? (
            <textarea
              value={getDisplayTemplate("user_template")}
              onChange={(e) => handleEdit("user_template", e.target.value)}
              className="w-full h-96 p-3 text-xs font-mono border rounded-lg focus:border-indigo-500 focus:ring-1 focus:ring-indigo-500"
            />
          ) : (
            <pre className="p-3 bg-gray-900 text-blue-400 rounded-lg text-xs overflow-auto max-h-96 whitespace-pre-wrap">
              {getDisplayTemplate("user_template")}
            </pre>
          )}
        </Card>
      </div>

      {/* Edit Controls */}
      {localEdits[selectedPromptType] && (
        <Card className="p-4 bg-amber-50 border-amber-200">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <AlertTriangle className="text-amber-600" size={16} />
              <span className="text-amber-800 text-sm font-medium">
                You have unsaved changes to the {currentTemplate?.name} template
              </span>
            </div>
            <div className="flex gap-2">
              <button
                onClick={handleResetEdits}
                className="px-3 py-1.5 text-sm text-gray-600 hover:bg-gray-100 rounded-lg"
              >
                Reset
              </button>
              <button
                onClick={handleSaveEdits}
                className="px-3 py-1.5 text-sm bg-amber-600 text-white rounded-lg hover:bg-amber-700"
              >
                Save Changes
              </button>
            </div>
          </div>
        </Card>
      )}

      {/* Config-Specific Variables */}
      <Card className="p-4 bg-blue-50 border-blue-200">
        <h4 className="font-medium text-blue-900 mb-3 flex items-center gap-2">
          <Info size={16} />
          Template Variables for {currentTemplate?.name}
          <span className="text-xs font-normal text-blue-600">
            ({currentVariables.length} variables)
          </span>
        </h4>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
          {currentVariables.map((v) => (
            <div
              key={v.name}
              className="flex items-start gap-2 bg-white/50 p-2 rounded"
            >
              <code className="bg-blue-100 px-1.5 py-0.5 rounded text-blue-800 font-mono whitespace-nowrap">
                {v.name}
              </code>
              <span className="text-gray-600">{v.desc}</span>
            </div>
          ))}
        </div>
      </Card>

      {/* Feature Coverage - Which traits and behavioral features are used */}
      <Card className="p-4">
        <div className="flex items-center justify-between mb-4">
          <h4 className="font-semibold flex items-center gap-2">
            <Target size={16} className="text-indigo-600" />
            Feature Coverage for {currentTemplate?.name}
          </h4>
          <div className="text-xs text-gray-500">
            {
              TRAIT_COVERAGE.traits.filter((t) =>
                t.usedIn.includes(selectedPromptType),
              ).length
            }
            /29 traits |
            {
              TRAIT_COVERAGE.behavioral.filter((b) =>
                b.usedIn.includes(selectedPromptType),
              ).length
            }
            /8 behavioral
          </div>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Traits Coverage */}
          <div>
            <div className="text-xs font-medium text-gray-700 mb-2">
              Persona Traits (29 total)
            </div>
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {[
                "Cognitive",
                "Big 5",
                "Psychological",
                "Security",
                "Susceptibility",
                "Behavioral",
                "Other",
              ].map((category) => {
                const categoryTraits = TRAIT_COVERAGE.traits.filter(
                  (t) => t.category === category,
                );
                const usedCount = categoryTraits.filter((t) =>
                  t.usedIn.includes(selectedPromptType),
                ).length;
                if (categoryTraits.length === 0) return null;
                return (
                  <div key={category} className="bg-gray-50 p-2 rounded">
                    <div className="flex items-center justify-between mb-1">
                      <span className="text-xs font-medium">{category}</span>
                      <span
                        className={`text-xs ${
                          usedCount > 0 ? "text-green-600" : "text-gray-400"
                        }`}
                      >
                        {usedCount}/{categoryTraits.length}
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-1">
                      {categoryTraits.map((trait) => {
                        const isUsed =
                          trait.usedIn.includes(selectedPromptType);
                        return (
                          <span
                            key={trait.name}
                            className={`text-xs px-1.5 py-0.5 rounded ${
                              isUsed
                                ? "bg-green-100 text-green-700"
                                : "bg-gray-200 text-gray-400"
                            }`}
                            title={`${trait.desc}${
                              !isUsed ? " (stored but not used in prompts)" : ""
                            }`}
                          >
                            {trait.name.replace(/_/g, " ").replace("big5 ", "")}
                          </span>
                        );
                      })}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Behavioral Features Coverage */}
          <div>
            <div className="text-xs font-medium text-gray-700 mb-2">
              Behavioral Features (8 total)
            </div>
            <div className="space-y-1">
              {TRAIT_COVERAGE.behavioral.map((feature) => {
                const isUsed = feature.usedIn.includes(selectedPromptType);
                return (
                  <div
                    key={feature.name}
                    className={`flex items-center justify-between p-2 rounded ${
                      isUsed ? "bg-green-50" : "bg-gray-50"
                    }`}
                  >
                    <div className="flex items-center gap-2">
                      <span
                        className={`w-2 h-2 rounded-full ${
                          isUsed ? "bg-green-500" : "bg-gray-300"
                        }`}
                      ></span>
                      <span
                        className={`text-xs ${
                          isUsed ? "text-gray-700" : "text-gray-400"
                        }`}
                      >
                        {feature.name.replace(/_/g, " ")}
                      </span>
                    </div>
                    <span
                      className={`text-xs ${
                        isUsed ? "text-green-600" : "text-gray-400"
                      }`}
                    >
                      {feature.desc}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Summary */}
            <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
              <div className="text-xs text-amber-800">
                <strong>Note:</strong> Features marked as "stored" are available
                in the persona definition but are not directly surfaced in the
                prompt text. They may be used for:
                <ul className="mt-1 ml-3 space-y-0.5">
                  <li>• Future prompt improvements</li>
                  <li>• Analysis and reporting</li>
                  <li>• Custom prompt templates</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </Card>

      {/* Info Box */}
      <Card className="p-4 bg-purple-50 border-purple-200">
        <div className="flex items-start gap-3">
          <Brain className="text-purple-600 flex-shrink-0 mt-0.5" size={18} />
          <div className="text-sm text-purple-800">
            <strong>How Prompts Work:</strong>
            <ul className="mt-1 space-y-1 text-purple-700">
              <li>
                • Templates shown here are <strong>representative</strong> of
                backend behavior
              </li>
              <li>
                • Actual prompts include dynamically generated content (ICL
                examples, patterns, situational context)
              </li>
              <li>
                • Situational context varies per trial to create natural
                behavioral variation
              </li>
              <li>
                • Temperature is adjusted based on cognitive style (0.7
                analytical → 1.0 impulsive)
              </li>
            </ul>
          </div>
        </div>
      </Card>
    </div>
  );
};

// ============================================================================
// EXPERIMENT BUILDER TAB - CLEAN VERSION
// ============================================================================

const ExperimentBuilderTab = ({
  personas,
  models,
  emails,
  promptTemplates,
  providers,
  onCreateExperiment,
}) => {
  const [config, setConfig] = useState({
    name: "",
    selectedPersonas: [],
    selectedModels: [],
    selectedPrompts: ["baseline", "stats", "cot"],
    selectedEmails: [],
    trialsPerCondition: 30,
    temperature: 0.7,
  });
  const [estimatedCost, setEstimatedCost] = useState({
    conditions: 0,
    trials: 0,
    cost: 0,
  });

  const templates = promptTemplates || DEFAULT_PROMPT_TEMPLATES;

  // Build models array with provider configuration status
  const modelsArray = useMemo(() => {
    const allModels = [];
    Object.values(LLM_PROVIDERS).forEach((provider) => {
      const isProviderConfigured = providers?.[provider.id]?.configured;
      provider.models.forEach((model) => {
        allModels.push({
          id: model.id,
          name: model.name,
          tier: model.tier,
          family: model.family,
          provider: provider.id,
          providerName: provider.name,
          providerConfigured: isProviderConfigured,
          cost_input: model.cost_input,
          cost_output: model.cost_output,
        });
      });
    });
    return allModels;
  }, [providers]);

  // Calculate estimated cost when selections change
  useEffect(() => {
    const totalConditions =
      config.selectedPersonas.length *
      config.selectedModels.length *
      config.selectedPrompts.length *
      Math.max(config.selectedEmails.length, 1);
    const totalTrials = totalConditions * config.trialsPerCondition;

    // Calculate average cost based on selected models
    let avgCostPerTrial = 0.001;
    if (config.selectedModels.length > 0) {
      const selectedModelConfigs = modelsArray.filter((m) =>
        config.selectedModels.includes(m.id),
      );
      if (selectedModelConfigs.length > 0) {
        const avgInputCost =
          selectedModelConfigs.reduce(
            (sum, m) => sum + (m.cost_input || 0.01),
            0,
          ) / selectedModelConfigs.length;
        const avgOutputCost =
          selectedModelConfigs.reduce(
            (sum, m) => sum + (m.cost_output || 0.03),
            0,
          ) / selectedModelConfigs.length;
        // Estimate ~800 input tokens, ~200 output tokens per trial
        avgCostPerTrial =
          (avgInputCost * 0.8) / 1000 + (avgOutputCost * 0.2) / 1000;
      }
    }

    setEstimatedCost({
      conditions: totalConditions,
      trials: totalTrials,
      cost: totalTrials * avgCostPerTrial,
    });
  }, [
    config.selectedPersonas.length,
    config.selectedModels.length,
    config.selectedPrompts.length,
    config.selectedEmails.length,
    config.trialsPerCondition,
    config.selectedModels,
    modelsArray,
  ]);

  const togglePersona = (personaId) => {
    setConfig((p) => ({
      ...p,
      selectedPersonas: p.selectedPersonas.includes(personaId)
        ? p.selectedPersonas.filter((id) => id !== personaId)
        : [...p.selectedPersonas, personaId],
    }));
  };

  const toggleModel = (modelId) => {
    // Only allow selection of configured models
    const model = modelsArray.find((m) => m.id === modelId);
    if (!model?.providerConfigured) {
      alert(
        `Cannot select ${model?.name || modelId}: Provider "${
          model?.providerName
        }" is not configured. Please configure it in the Providers tab first.`,
      );
      return;
    }
    setConfig((p) => ({
      ...p,
      selectedModels: p.selectedModels.includes(modelId)
        ? p.selectedModels.filter((id) => id !== modelId)
        : [...p.selectedModels, modelId],
    }));
  };

  const toggle = (field, value) =>
    setConfig((p) => ({
      ...p,
      [field]: p[field].includes(value)
        ? p[field].filter((v) => v !== value)
        : [...p[field], value],
    }));

  const selectAllConfiguredModels = () => {
    const configuredModelIds = modelsArray
      .filter((m) => m.providerConfigured)
      .map((m) => m.id);
    setConfig((p) => ({ ...p, selectedModels: configuredModelIds }));
  };

  const selectAll = (field, items, idKey) => {
    setConfig((p) => ({ ...p, [field]: items.map((i) => i[idKey]) }));
  };

  const clearSelection = (field) => {
    setConfig((p) => ({ ...p, [field]: [] }));
  };

  const handleCreate = () => {
    if (
      !config.name ||
      !config.selectedPersonas.length ||
      !config.selectedModels.length ||
      !config.selectedEmails.length
    ) {
      alert("Please fill all required fields");
      return;
    }
    onCreateExperiment({
      name: config.name,
      persona_ids: config.selectedPersonas,
      model_ids: config.selectedModels,
      prompt_configs: config.selectedPrompts,
      email_ids: config.selectedEmails,
      trials_per_condition: config.trialsPerCondition,
      temperature: config.temperature,
    });
  };

  const configuredModelsCount = modelsArray.filter(
    (m) => m.providerConfigured,
  ).length;
  const unconfiguredModelsCount = modelsArray.length - configuredModelsCount;

  return (
    <div className="space-y-6">
      {/* Experiment Name */}
      <Card className="p-6">
        <h3 className="text-lg font-semibold mb-4">Experiment Configuration</h3>
        <input
          type="text"
          placeholder="Enter experiment name..."
          className="w-full px-4 py-3 border rounded-lg text-lg"
          value={config.name}
          onChange={(e) => setConfig((p) => ({ ...p, name: e.target.value }))}
        />
      </Card>

      <div className="grid grid-cols-2 gap-6">
        {/* Personas - With description */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <div className="flex items-center gap-3">
              <h3 className="font-semibold">
                Personas ({config.selectedPersonas.length})
              </h3>
              <SystematicCodeLegend />
            </div>
            <div className="flex gap-2">
              <button
                className="text-xs text-gray-500 hover:text-gray-700"
                onClick={() => clearSelection("selectedPersonas")}
              >
                Clear
              </button>
              <button
                className="text-xs text-indigo-600"
                onClick={() =>
                  selectAll("selectedPersonas", personas, "persona_id")
                }
              >
                All
              </button>
            </div>
          </div>
          <div className="max-h-64 overflow-y-auto space-y-2">
            {personas.length > 0 ? (
              personas.map((p) => {
                const clickRate = (
                  (p.behavioral_statistics?.phishing_click_rate ||
                    p.behavioral_targets?.phishing_click_rate ||
                    0) * 100
                ).toFixed(0);
                const reportRate = (
                  (p.behavioral_statistics?.report_rate ||
                    p.behavioral_targets?.report_rate ||
                    0) * 100
                ).toFixed(0);
                return (
                  <label
                    key={p.persona_id}
                    className="flex items-start gap-2 p-3 hover:bg-gray-50 rounded-lg cursor-pointer border border-transparent hover:border-gray-200"
                  >
                    <input
                      type="checkbox"
                      checked={config.selectedPersonas.includes(p.persona_id)}
                      onChange={() => togglePersona(p.persona_id)}
                      className="w-4 h-4 mt-1"
                    />
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="font-medium text-gray-900">
                          {p.name || p.persona_id}
                        </span>
                        <Badge
                          color={
                            p.risk_level === "CRITICAL"
                              ? "red"
                              : p.risk_level === "HIGH"
                                ? "orange"
                                : p.risk_level === "MEDIUM"
                                  ? "yellow"
                                  : "green"
                          }
                        >
                          {p.risk_level}
                        </Badge>
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        {p.description}
                      </p>
                      <p className="text-xs mt-1">
                        <span className="text-red-600">{clickRate}% click</span>
                        <span className="mx-1">•</span>
                        <span className="text-green-600">
                          {reportRate}% report
                        </span>
                      </p>
                    </div>
                  </label>
                );
              })
            ) : (
              <p className="text-sm text-gray-500 text-center py-4">
                No personas loaded. Import from Overview tab.
              </p>
            )}
          </div>
        </Card>

        {/* Models - With provider configuration warnings */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold">
              Models ({config.selectedModels.length})
            </h3>
            <div className="flex gap-2">
              <button
                className="text-xs text-gray-500 hover:text-gray-700"
                onClick={() => clearSelection("selectedModels")}
              >
                Clear
              </button>
              <button
                className="text-xs text-indigo-600"
                onClick={selectAllConfiguredModels}
              >
                All Available
              </button>
            </div>
          </div>

          {/* Provider status warning */}
          {unconfiguredModelsCount > 0 && (
            <div className="mb-3 p-2 bg-amber-50 border border-amber-200 rounded-lg">
              <p className="text-xs text-amber-700 flex items-center gap-1">
                <AlertTriangle size={12} />
                {unconfiguredModelsCount} models unavailable - configure
                providers first
              </p>
            </div>
          )}

          {/* Quick tier buttons */}
          <div className="flex gap-2 mb-3 flex-wrap">
            {Object.entries(MODEL_TIERS).map(([tier, info]) => {
              const tierModels = modelsArray.filter(
                (m) => m.tier === tier && m.providerConfigured,
              );
              if (tierModels.length === 0) return null;
              return (
                <button
                  key={tier}
                  onClick={() =>
                    setConfig((p) => ({
                      ...p,
                      selectedModels: [
                        ...new Set([
                          ...p.selectedModels,
                          ...tierModels.map((m) => m.id),
                        ]),
                      ],
                    }))
                  }
                  className={`text-xs px-2 py-1 rounded border ${info.bg} ${info.text}`}
                >
                  + {info.label} ({tierModels.length})
                </button>
              );
            })}
          </div>

          <div className="max-h-48 overflow-y-auto space-y-1">
            {modelsArray.map((m) => {
              const isConfigured = m.providerConfigured;
              const isSelected = config.selectedModels.includes(m.id);
              return (
                <label
                  key={m.id}
                  className={`flex items-center gap-2 p-2 rounded cursor-pointer ${
                    isConfigured
                      ? "hover:bg-gray-50"
                      : "opacity-50 cursor-not-allowed"
                  } ${isSelected ? "bg-indigo-50" : ""}`}
                  onClick={(e) => {
                    if (!isConfigured) {
                      e.preventDefault();
                    }
                  }}
                >
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => toggleModel(m.id)}
                    disabled={!isConfigured}
                    className="w-4 h-4"
                  />
                  <span
                    className={`truncate flex-1 text-sm ${
                      !isConfigured ? "text-gray-400" : ""
                    }`}
                  >
                    {m.name}
                  </span>
                  <span className="text-xs text-gray-400">
                    {m.providerName}
                  </span>
                  {!isConfigured && (
                    <AlertTriangle
                      size={14}
                      className="text-amber-500"
                      title="Provider not configured"
                    />
                  )}
                </label>
              );
            })}
          </div>
        </Card>

        {/* Prompts */}
        <Card className="p-6">
          <h3 className="font-semibold mb-4">
            Prompt Configurations ({config.selectedPrompts.length})
          </h3>
          <div className="space-y-2">
            {Object.values(templates).map((pt) => (
              <label
                key={pt.id}
                className="flex items-center gap-3 p-3 border rounded-lg cursor-pointer hover:bg-gray-50"
              >
                <input
                  type="checkbox"
                  checked={config.selectedPrompts.includes(pt.id)}
                  onChange={() => toggle("selectedPrompts", pt.id)}
                  className="w-4 h-4"
                />
                <div>
                  <div className="font-medium">{pt.name}</div>
                  <div className="text-xs text-gray-500">{pt.description}</div>
                </div>
              </label>
            ))}
          </div>
        </Card>

        {/* Emails */}
        <Card className="p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold">
              Emails ({config.selectedEmails.length})
            </h3>
            <div className="flex gap-2">
              <button
                className="text-xs text-gray-500 hover:text-gray-700"
                onClick={() => clearSelection("selectedEmails")}
              >
                Clear
              </button>
              <button
                className="text-xs text-red-600"
                onClick={() =>
                  setConfig((p) => ({
                    ...p,
                    selectedEmails: emails
                      .filter((e) => e.email_type === "phishing")
                      .map((e) => e.email_id),
                  }))
                }
              >
                Phishing
              </button>
              <button
                className="text-xs text-indigo-600"
                onClick={() => selectAll("selectedEmails", emails, "email_id")}
              >
                All
              </button>
            </div>
          </div>
          <div className="max-h-48 overflow-y-auto space-y-1">
            {emails.length > 0 ? (
              emails.map((e) => (
                <label
                  key={e.email_id}
                  className="flex items-center gap-2 p-2 hover:bg-gray-50 rounded cursor-pointer"
                >
                  <input
                    type="checkbox"
                    checked={config.selectedEmails.includes(e.email_id)}
                    onChange={() => toggle("selectedEmails", e.email_id)}
                    className="w-4 h-4"
                  />
                  <span className="truncate flex-1 text-sm">
                    {e.subject_line || e.email_id}
                  </span>
                  <Badge color={e.email_type === "phishing" ? "red" : "green"}>
                    {e.email_type === "phishing" ? "Phish" : "Legit"}
                  </Badge>
                </label>
              ))
            ) : (
              <p className="text-sm text-gray-500 text-center py-4">
                No emails loaded
              </p>
            )}
          </div>
        </Card>
      </div>

      {/* Advanced Settings */}
      <Card className="p-6">
        <h3 className="font-semibold mb-4">Advanced Settings</h3>
        <div className="grid grid-cols-2 gap-6">
          <div>
            <label className="block text-sm font-medium mb-2">
              Trials per Condition
            </label>
            <input
              type="number"
              min="1"
              max="100"
              value={config.trialsPerCondition}
              onChange={(e) =>
                setConfig((p) => ({
                  ...p,
                  trialsPerCondition: parseInt(e.target.value) || 1,
                }))
              }
              className="w-full px-3 py-2 border rounded-lg"
            />
            <p className="text-xs text-gray-500 mt-1">
              More trials = better statistics, higher cost
            </p>
          </div>
          <div>
            <label className="block text-sm font-medium mb-2">
              Temperature ({config.temperature})
            </label>
            <input
              type="range"
              min="0"
              max="1"
              step="0.1"
              value={config.temperature}
              onChange={(e) =>
                setConfig((p) => ({
                  ...p,
                  temperature: parseFloat(e.target.value),
                }))
              }
              className="w-full"
            />
            <p className="text-xs text-gray-500 mt-1">
              Lower = deterministic, Higher = varied
            </p>
          </div>
        </div>
      </Card>

      {/* Summary & Create - With Cost Estimate */}
      <Card className="p-6 bg-gradient-to-r from-indigo-50 to-purple-50">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="font-semibold">Experiment Summary</h3>
            <div className="text-sm text-gray-600 mt-2 grid grid-cols-3 gap-4">
              <div>
                <span className="text-gray-500">Conditions:</span>
                <span className="font-medium ml-2">
                  {estimatedCost.conditions.toLocaleString()}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Total Trials:</span>
                <span className="font-medium ml-2">
                  {estimatedCost.trials.toLocaleString()}
                </span>
              </div>
              <div>
                <span className="text-gray-500">Est. Cost:</span>
                <span className="font-medium ml-2 text-green-600">
                  ${estimatedCost.cost.toFixed(2)}
                </span>
              </div>
            </div>
          </div>
          <button
            onClick={handleCreate}
            disabled={
              !config.name ||
              !config.selectedPersonas.length ||
              !config.selectedModels.length ||
              !config.selectedEmails.length
            }
            className="px-6 py-3 bg-indigo-600 text-white rounded-lg font-semibold hover:bg-indigo-700 disabled:bg-gray-300 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <Sparkles size={20} /> Create Experiment
          </button>
        </div>
      </Card>
    </div>
  );
};

// ============================================================================
// EXECUTION TAB - Retained from v3
// ============================================================================

const ExecutionTab = ({
  experiments,
  onRunExperiment,
  onStopExperiment,
  onRefreshExperiments,
}) => {
  // Only poll when there are running experiments
  const hasRunningExperiments = experiments.some((e) => e.status === "running");

  useEffect(() => {
    if (hasRunningExperiments) {
      // Start polling when experiments are running
      const interval = setInterval(onRefreshExperiments, 5000);
      return () => clearInterval(interval);
    }
    // If no running experiments, don't poll (saves API calls)
  }, [hasRunningExperiments, onRefreshExperiments]);

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-xl font-bold">Experiment Execution</h2>
        <button
          onClick={onRefreshExperiments}
          className="text-sm text-gray-500 hover:text-gray-700 flex items-center gap-1"
        >
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {experiments.length === 0 ? (
        <Card className="p-12 text-center">
          <Zap className="mx-auto text-gray-300 mb-4" size={48} />
          <h3 className="text-lg font-medium text-gray-600">No Experiments</h3>
          <p className="text-gray-500">
            Create an experiment in the Experiment tab first
          </p>
        </Card>
      ) : (
        <div className="space-y-4">
          {experiments.map((exp) => {
            const progress =
              exp.total_trials > 0
                ? ((exp.completed_trials || 0) / exp.total_trials) * 100
                : 0;
            const isRunning = exp.status === "running";
            const canRun = ["pending", "draft", "ready"].includes(exp.status);

            return (
              <Card key={exp.experiment_id} className="p-4">
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <h4 className="font-semibold">{exp.name}</h4>
                      <Badge
                        color={
                          exp.status === "completed"
                            ? "green"
                            : exp.status === "running"
                              ? "blue"
                              : exp.status === "failed"
                                ? "red"
                                : "gray"
                        }
                      >
                        {exp.status}
                      </Badge>
                    </div>
                    {isRunning && (
                      <div className="mt-2">
                        <div className="flex items-center gap-2 text-sm text-gray-500 mb-1">
                          <span>
                            {exp.completed_trials || 0} / {exp.total_trials}{" "}
                            trials
                          </span>
                          <span>({progress.toFixed(1)}%)</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-indigo-600 h-2 rounded-full transition-all"
                            style={{ width: `${progress}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                  <div className="flex gap-2">
                    {canRun && (
                      <button
                        onClick={() => onRunExperiment(exp.experiment_id)}
                        className="px-4 py-2 bg-green-600 text-white rounded-lg flex items-center gap-2 hover:bg-green-700"
                      >
                        <Play size={16} /> Run
                      </button>
                    )}
                    {isRunning && (
                      <button
                        onClick={() => onStopExperiment(exp.experiment_id)}
                        className="px-4 py-2 bg-red-600 text-white rounded-lg flex items-center gap-2 hover:bg-red-700"
                      >
                        <Pause size={16} /> Stop
                      </button>
                    )}
                  </div>
                </div>
              </Card>
            );
          })}
        </div>
      )}
    </div>
  );
};

// ============================================================================
// RESULTS TAB - Retained from v3
// ============================================================================

const ResultsTab = ({ experiments }) => {
  const [selectedExperiment, setSelectedExperiment] = useState(null);
  const [results, setResults] = useState(null);
  const [fidelity, setFidelity] = useState(null);
  const [modelComparison, setModelComparison] = useState(null);
  const [boundaries, setBoundaries] = useState(null);
  const [loading, setLoading] = useState(false);

  const completedExperiments = experiments.filter(
    (e) => e.status === "completed",
  );

  const loadResults = async (experimentId) => {
    setLoading(true);
    try {
      const [resultsData, fidelityData, comparisonData, boundaryData] =
        await Promise.all([
          api.getResults(experimentId),
          api.analyzeFidelity(experimentId).catch(() => null),
          api.compareModels(experimentId).catch(() => null),
          api.findBoundaries(experimentId).catch(() => null),
        ]);
      setResults(resultsData);
      setFidelity(fidelityData);
      setModelComparison(comparisonData);
      setBoundaries(boundaryData);
    } catch (error) {
      console.error("Failed to load results:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (selectedExperiment) {
      loadResults(selectedExperiment);
    }
  }, [selectedExperiment]);

  useEffect(() => {
    if (completedExperiments.length > 0 && !selectedExperiment) {
      setSelectedExperiment(completedExperiments[0].experiment_id);
    }
  }, [completedExperiments]);

  // CRITICAL FIX: Handle both 'fidelity' and 'fidelity_results' field names
  const getFidelityArray = () => {
    if (!fidelity) return [];
    return fidelity.fidelity_results || fidelity.fidelity || [];
  };

  const getThreshold = () => {
    if (!fidelity) return 0.85;
    if (fidelity.thresholds?.accuracy != null)
      return fidelity.thresholds.accuracy;
    if (fidelity.threshold != null) return fidelity.threshold;
    return 0.85;
  };

  const fidelityArray = getFidelityArray();

  if (completedExperiments.length === 0) {
    return (
      <Card>
        <div className="text-center py-12">
          <BarChart3 className="mx-auto text-gray-300 mb-4" size={48} />
          <h3 className="text-lg font-medium text-gray-600">No Results Yet</h3>
          <p className="text-gray-500">Run an experiment to see results</p>
        </div>
      </Card>
    );
  }

  // Combine all fetched data into the format expected by ResultsTabView
  const combinedResults = results
    ? {
        ...results,
        fidelity_results:
          fidelity?.fidelity_results || fidelity?.fidelity || fidelityArray,
        fidelity:
          fidelity?.fidelity_results || fidelity?.fidelity || fidelityArray,
        model_comparison: modelComparison,
        boundaries: boundaries,
        thresholds: fidelity?.thresholds || { accuracy: getThreshold() },
        threshold: getThreshold(),
      }
    : null;

  // Find current experiment object
  const currentExpObj = completedExperiments.find(
    (e) => e.experiment_id === selectedExperiment,
  );

  return (
    <div className="space-y-6">
      {/* Experiment Selector */}
      <Card>
        <div className="flex items-center justify-between p-6">
          <h3 className="font-semibold">Select Completed Experiment</h3>
          <select
            value={selectedExperiment || ""}
            onChange={(e) => setSelectedExperiment(e.target.value)}
            className="px-4 py-2 border rounded-lg"
          >
            {completedExperiments.map((exp) => (
              <option key={exp.experiment_id} value={exp.experiment_id}>
                {exp.name} ({exp.total_trials} trials)
              </option>
            ))}
          </select>
        </div>
      </Card>

      {loading ? (
        <Card>
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="animate-spin text-indigo-600" size={32} />
            <span className="ml-3 text-gray-600">Loading results...</span>
          </div>
        </Card>
      ) : combinedResults ? (
        <ResultsTabView
          experiments={completedExperiments}
          currentExperiment={currentExpObj}
          results={combinedResults}
          onSelectExperiment={setSelectedExperiment}
        />
      ) : null}
    </div>
  );
};

// ============================================================================
// PUBLISH TAB - Retained from v3
// ============================================================================

const PublishTab = ({ personas, experiments, promptTemplates, models }) => {
  const [name, setName] = useState("CYPEARL Config");

  // Token estimates per prompt type
  const TOKEN_ESTIMATES = {
    baseline: { input: 500, output: 100 },
    stats: { input: 1200, output: 200 },
    cot: { input: 2000, output: 500 },
  };

  // Calculate cost per call based on LLM pricing and token usage
  const calcCost = (llm, promptId) => {
    const tokens = TOKEN_ESTIMATES[promptId] || TOKEN_ESTIMATES.stats;
    // cost_input and cost_output are per million tokens
    const cost =
      (tokens.input * llm.cost_input + tokens.output * llm.cost_output) /
      1_000_000;
    return Math.round(cost * 1000000) / 1000000; // Round to 6 decimal places
  };

  // Calculate fidelity based on LLM tier and prompt complexity
  const calcFidelity = (llm, promptId, personaDifficulty = 0) => {
    const tierFidelity = {
      frontier: 0.96,
      mid_tier: 0.9,
      open_source: 0.87,
      budget: 0.85,
    };
    const promptBoost = { baseline: 0, stats: 0.02, cot: 0.04 };
    const providerAdj = {
      anthropic: 0.02,
      openai: 0.01,
      aws_bedrock: 0.01,
      openrouter: 0,
      together_ai: -0.01,
      local: -0.02,
    };

    const base = tierFidelity[llm.tier] || 0.85;
    const boost = promptBoost[promptId] || 0;
    const adj = providerAdj[llm.provider] || 0;
    const fidelity = Math.min(
      0.99,
      base + boost + adj - personaDifficulty * 0.02,
    );
    return Math.round(fidelity * 1000) / 1000;
  };

  // Generate fidelity matrix for all LLMs
  const generateFidelityMatrix = (persona) => {
    const matrix = {};
    const promptIds = ["baseline", "stats", "cot"];

    // Get all LLMs from all providers
    Object.values(LLM_PROVIDERS).forEach((provider) => {
      provider.models.forEach((llm) => {
        matrix[llm.id] = {};
        promptIds.forEach((promptId) => {
          matrix[llm.id][promptId] = {
            fidelity: calcFidelity(llm, promptId),
            cost: calcCost(llm, promptId),
          };
        });
      });
    });
    return matrix;
  };

  // Generate email modifiers based on persona traits
  const generateEmailModifiers = (persona) => {
    const urgencySusc = persona.trait_zscores?.urgency_susceptibility || 0;
    const authSusc = persona.trait_zscores?.authority_susceptibility || 0;
    const trustProp = persona.trait_zscores?.trust_propensity || 0;
    const stateAnxiety = persona.trait_zscores?.state_anxiety || 0;

    return {
      urgency_high: {
        click_multiplier: Math.round((1.0 + urgencySusc * 0.3) * 100) / 100,
      },
      urgency_low: {
        click_multiplier: Math.round((0.85 - urgencySusc * 0.1) * 100) / 100,
      },
      authority_sender: {
        click_multiplier: Math.round((1.0 + authSusc * 0.3) * 100) / 100,
      },
      familiar_sender: {
        click_multiplier: Math.round((1.0 + trustProp * 0.25) * 100) / 100,
      },
      threat_framing: {
        click_multiplier: Math.round((1.0 + stateAnxiety * 0.2) * 100) / 100,
      },
      reward_framing: {
        click_multiplier: Math.round((1.0 + trustProp * 0.2) * 100) / 100,
      },
    };
  };

  const generate = () => {
    // Derive matching features from first persona's trait_zscores
    const matchingFeatures =
      personas.length > 0 && personas[0].trait_zscores
        ? Object.keys(personas[0].trait_zscores)
        : [];

    // Build LLM options from all providers
    const llmOptions = Object.values(LLM_PROVIDERS).flatMap((provider) =>
      provider.models.map((m) => ({
        id: m.id,
        name: m.name,
        provider: provider.id,
        tier: m.tier,
        cost_per_1m_input: m.cost_input,
        cost_per_1m_output: m.cost_output,
      })),
    );

    const config = {
      version: "2.0.0",
      published_date: new Date().toISOString().split("T")[0],
      published_by: "CYPEARL Admin",
      description: name,
      matching_features: matchingFeatures,
      llm_options: llmOptions,
      prompt_templates: promptTemplates || DEFAULT_PROMPT_TEMPLATES,
      personas: personas.map((p) => {
        // Convert behavioral_statistics to behavioral_targets if needed
        const behavioralTargets = p.behavioral_targets ||
          p.behavioral_statistics || {
            phishing_click_rate: 0.3,
            report_rate: 0.15,
            overall_accuracy: 0.8,
            mean_response_latency_ms: 3500,
            hover_rate: 0.3,
            sender_inspection_rate: 0.25,
          };

        // Generate fidelity matrix for ALL LLMs
        const fidelityMatrix = p.fidelity_matrix || generateFidelityMatrix(p);

        // Generate email modifiers
        const emailModifiers = p.email_modifiers || generateEmailModifiers(p);

        // Generate validated configurations - pick best config for top LLMs
        const topLlms = llmOptions.slice(0, 5);
        const validatedConfigs = topLlms.map((llm) => {
          const bestPrompt = ["cot", "stats", "baseline"].reduce(
            (best, pt) =>
              (fidelityMatrix[llm.id]?.[pt]?.fidelity || 0) >
              (fidelityMatrix[llm.id]?.[best]?.fidelity || 0)
                ? pt
                : best,
            "stats",
          );

          return {
            llm_id: llm.id,
            llm_name: llm.name,
            prompt_config: bestPrompt,
            prompt_name: {
              baseline: "Baseline",
              stats: "Statistical",
              cot: "Chain-of-Thought",
            }[bestPrompt],
            fidelity_score:
              fidelityMatrix[llm.id]?.[bestPrompt]?.fidelity || 0.9,
            cost_per_call: fidelityMatrix[llm.id]?.[bestPrompt]?.cost || 0.005,
            recommended:
              llm.id.includes("sonnet-4-5") || llm.id.includes("gpt-4o"),
          };
        });

        return {
          persona_id: p.persona_id,
          name: p.name,
          archetype: p.archetype,
          description: p.description,
          risk_level: p.risk_level,
          cognitive_style: p.cognitive_style,
          n_participants: p.n_participants,
          pct_of_population: p.pct_of_population,
          trait_zscores: p.trait_zscores,
          distinguishing_high_traits: p.distinguishing_high_traits,
          distinguishing_low_traits: p.distinguishing_low_traits,
          behavioral_targets: behavioralTargets,
          email_modifiers: emailModifiers,
          prompt_variables: p.prompt_variables || {
            reasoning_example_1: "Example reasoning for urgent email",
            reasoning_example_2: "Example reasoning for authority request",
            reasoning_example_3: "Example reasoning for reward offer",
          },
          fidelity_matrix: fidelityMatrix,
          validated_configurations: validatedConfigs,
        };
      }),
    };
    const blob = new Blob([JSON.stringify(config, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "admin_published_config_v2.json";
    a.click();
  };

  return (
    <div className="space-y-6">
      <h2 className="text-xl font-bold">Publish for CISO</h2>

      <Card className="p-6">
        <h3 className="font-semibold mb-3">Configuration Name</h3>
        <input
          type="text"
          value={name}
          onChange={(e) => setName(e.target.value)}
          className="w-full px-4 py-2 border rounded-lg"
          placeholder="Enter configuration name..."
        />
      </Card>

      <Card className="p-6 bg-green-50 border-green-200">
        <div className="flex items-center justify-between">
          <div>
            <div className="font-semibold text-green-900 flex items-center gap-2">
              <CheckCircle className="text-green-600" size={20} />
              Ready to Publish
            </div>
            <div className="text-sm text-green-700 mt-1">
              {personas.length} personas •{" "}
              {Object.keys(promptTemplates || DEFAULT_PROMPT_TEMPLATES).length}{" "}
              prompt templates
            </div>
          </div>
          <button
            onClick={generate}
            disabled={!personas.length}
            className="px-6 py-3 bg-green-600 text-white rounded-lg font-semibold hover:bg-green-700 disabled:bg-gray-300 flex items-center gap-2"
          >
            <Download size={20} /> Download Config
          </button>
        </div>
      </Card>
    </div>
  );
};

// ============================================================================
// MAIN DASHBOARD
// ============================================================================

const TABS = [
  { id: "overview", label: "Overview", icon: Activity },
  { id: "providers", label: "Providers", icon: Server },
  { id: "prompts", label: "Prompts", icon: Code },
  { id: "calibration", label: "Calibration", icon: Beaker },
  { id: "research", label: "Research", icon: Brain },
  { id: "experiment", label: "Experiment", icon: Zap },
  { id: "execution", label: "Execution", icon: Play },
  { id: "results", label: "Results", icon: BarChart3 },
  { id: "publish", label: "Publish", icon: Share2 },
];

// ============================================================================
// URL & STORAGE UTILITIES FOR PHASE 2
// ============================================================================

const PHASE2_STORAGE_KEY = "cypearl_phase2_state";

const getPhase2TabFromHash = () => {
  const hash = window.location.hash.slice(1);
  const parts = hash.split("/").filter(Boolean);
  if (parts[0] === "phase2" && parts[1]) {
    const validTabs = [
      "overview",
      "providers",
      "prompts",
      "experiment",
      "execution",
      "results",
      "boundaries",
      "calibration",
      "research",
      "publish",
    ];
    if (validTabs.includes(parts[1])) {
      return parts[1];
    }
  }
  return null;
};

const setPhase2TabInHash = (tab) => {
  const hash = `/phase2/${tab}`;
  if (window.location.hash !== `#${hash}`) {
    window.history.pushState(null, "", `#${hash}`);
  }
};

const Phase2Dashboard = ({ importedPersonas }) => {
  // Initialize activeTab from URL hash or localStorage
  const [activeTab, setActiveTab] = useState(() => {
    const hashTab = getPhase2TabFromHash();
    if (hashTab) return hashTab;
    try {
      const saved = localStorage.getItem(PHASE2_STORAGE_KEY);
      if (saved) {
        const state = JSON.parse(saved);
        return state.activeTab || "overview";
      }
    } catch (e) {}
    return "overview";
  });
  const [loading, setLoading] = useState(false);
  const [personas, setPersonas] = useState([]);
  const [emails, setEmails] = useState([]);
  const [models, setModels] = useState([]);
  const [providers, setProviders] = useState({});
  const [experiments, setExperiments] = useState([]);
  const [usage, setUsage] = useState({});
  const [promptTemplates, setPromptTemplates] = useState(
    DEFAULT_PROMPT_TEMPLATES,
  );
  const [autoImported, setAutoImported] = useState(false);
  const [initialLoading, setInitialLoading] = useState(true);

  // ========================================================================
  // URL & STATE PERSISTENCE
  // ========================================================================

  // Sync activeTab with URL hash
  useEffect(() => {
    setPhase2TabInHash(activeTab);
  }, [activeTab]);

  // Handle browser back/forward for tab changes
  useEffect(() => {
    const handleHashChange = () => {
      const hashTab = getPhase2TabFromHash();
      if (hashTab && hashTab !== activeTab) {
        setActiveTab(hashTab);
      }
    };

    window.addEventListener("hashchange", handleHashChange);
    return () => window.removeEventListener("hashchange", handleHashChange);
  }, [activeTab]);

  // Persist important state to localStorage
  useEffect(() => {
    if (initialLoading) return;

    try {
      const stateToSave = {
        activeTab,
        personas,
        emails,
        providers,
        experiments,
        promptTemplates,
      };
      localStorage.setItem(PHASE2_STORAGE_KEY, JSON.stringify(stateToSave));
    } catch (e) {
      console.warn("Failed to save Phase 2 state:", e);
    }
  }, [
    activeTab,
    personas,
    emails,
    providers,
    experiments,
    promptTemplates,
    initialLoading,
  ]);

  // Restore state from localStorage on mount
  useEffect(() => {
    try {
      const saved = localStorage.getItem(PHASE2_STORAGE_KEY);
      if (saved) {
        const state = JSON.parse(saved);
        if (state.personas?.length > 0 && !importedPersonas?.personas?.length) {
          setPersonas(state.personas);
        }
        if (state.emails?.length > 0) setEmails(state.emails);
        if (state.providers) setProviders(state.providers);
        if (state.experiments?.length > 0) setExperiments(state.experiments);
        if (state.promptTemplates) setPromptTemplates(state.promptTemplates);
      }
    } catch (e) {
      console.warn("Failed to restore Phase 2 state:", e);
    }
    setInitialLoading(false);
  }, []);

  // Check for imported personas FIRST
  useEffect(() => {
    if (importedPersonas?.personas?.length > 0) {
      console.log(
        "Phase 2: Using personas from Phase 1:",
        importedPersonas.personas.length,
      );
      setPersonas(importedPersonas.personas);
      setAutoImported(true);
      api.importPersonas(importedPersonas).catch(() => {});
    } else {
      loadFromBackend();
    }
    loadOtherData();
  }, []);

  useEffect(() => {
    if (importedPersonas?.personas?.length > 0 && !autoImported) {
      setPersonas(importedPersonas.personas);
      setAutoImported(true);
      api.importPersonas(importedPersonas).catch(() => {});
    }
  }, [importedPersonas]);

  const loadFromBackend = async () => {
    try {
      const existingPersonas = await api.getPersonas().catch(() => []);
      if (existingPersonas?.length > 0 && personas.length === 0) {
        setPersonas(existingPersonas);
      }
    } catch (e) {
      console.log("No personas in backend");
    }
  };

  const loadOtherData = async () => {
    setLoading(true);
    try {
      const [providersData, modelsData, usageData, existingEmails, exps] =
        await Promise.all([
          api.getProviders().catch(() => ({})),
          api.getModels().catch(() => []),
          api.getUsage().catch(() => ({})),
          api.getEmails().catch(() => []),
          api.getExperiments().catch(() => []),
        ]);
      setProviders(providersData);
      setModels(modelsData);
      setUsage(usageData);
      if (existingEmails?.length > 0) setEmails(existingEmails);
      setExperiments(exps);
    } catch (error) {
      console.error("Failed to load:", error);
    } finally {
      setLoading(false);
    }
  };

  const handleImportPersonas = async (data) => {
    const personaList = data.personas || data || [];
    setPersonas(personaList);
    setAutoImported(false);
    try {
      await api.importPersonas({ personas: personaList });
    } catch (e) {}
  };

  const handleLoadEmails = async (data) => {
    const emailList = Array.isArray(data) ? data : data.emails || data;
    setEmails(emailList);
    try {
      await api.loadEmails(emailList);
    } catch (e) {}
  };

  const handleSetupProvider = async (type, config) => {
    const result = await api.setupProvider(type, config);
    setProviders((prev) => ({ ...prev, [type]: result }));
    const modelsData = await api.getModels();
    setModels(modelsData);
  };

  const handleTestModel = async (modelId) => await api.testModel(modelId);

  const handleCreateExperiment = async (config) => {
    if (personas.length > 0)
      await api.importPersonas({ personas }).catch(() => {});
    if (emails.length > 0) await api.loadEmails(emails).catch(() => {});
    const result = await api.createExperiment(config);
    setExperiments((prev) => [...prev, result]);
    alert(`Experiment "${config.name}" created!`);
    setActiveTab("execution");
  };

  const handleRunExperiment = async (id) => {
    await api.runExperiment(id);
    const e = await api.getExperiments();
    setExperiments(e);
  };

  const handleStopExperiment = async (id) => {
    await api.stopExperiment(id);
    const e = await api.getExperiments();
    setExperiments(e);
  };

  const handleRefreshExperiments = async () => {
    const e = await api.getExperiments().catch(() => []);
    setExperiments(e);
  };

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Phase 2: LLM Calibration
              </h1>
              <p className="text-gray-500">
                Test LLM fidelity to human personas
              </p>
            </div>
            <button
              onClick={() => {
                loadFromBackend();
                loadOtherData();
              }}
              className="flex items-center gap-2 px-3 py-2 text-gray-600 hover:bg-gray-100 rounded-lg"
            >
              <RefreshCw size={16} /> Refresh
            </button>
          </div>
        </div>
      </div>

      {/* Tabs */}
      <div className="bg-white border-b">
        <div className="max-w-7xl mx-auto px-6">
          <div className="flex gap-1 overflow-x-auto">
            {TABS.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 whitespace-nowrap ${
                  activeTab === tab.id
                    ? "border-indigo-600 text-indigo-600"
                    : "border-transparent text-gray-500 hover:text-gray-700"
                }`}
              >
                <tab.icon size={18} /> {tab.label}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-6">
        {loading ? (
          <div className="flex items-center justify-center py-12">
            <RefreshCw className="animate-spin text-indigo-600" size={32} />
          </div>
        ) : (
          <>
            {/* Step-by-step guidance for current tab */}
            {PHASE2_GUIDANCE[activeTab] && (
              <StepGuide
                phase={2}
                tab={activeTab}
                collapsed={true}
                className="mb-6"
              />
            )}

            {activeTab === "overview" && (
              <OverviewTab
                personas={personas}
                emails={emails}
                models={models}
                providers={providers}
                usage={usage}
                onImportPersonas={handleImportPersonas}
                onLoadEmails={handleLoadEmails}
                autoImported={autoImported}
              />
            )}
            {activeTab === "providers" && (
              <ProviderSetupTab
                providers={providers}
                models={models}
                onSetupProvider={handleSetupProvider}
                onTestModel={handleTestModel}
              />
            )}
            {activeTab === "prompts" && (
              <PromptsTab
                promptTemplates={promptTemplates}
                onUpdatePrompts={setPromptTemplates}
              />
            )}
            {activeTab === "calibration" && (
              <CalibrationTab
                personas={personas}
                models={models}
                emails={emails}
              />
            )}
            {activeTab === "research" && (
              <ResearchTab personas={personas} models={models} />
            )}
            {activeTab === "experiment" && (
              <ExperimentBuilderTab
                personas={personas}
                models={models}
                emails={emails}
                promptTemplates={promptTemplates}
                providers={providers}
                onCreateExperiment={handleCreateExperiment}
              />
            )}
            {activeTab === "execution" && (
              <ExecutionTab
                experiments={experiments}
                onRunExperiment={handleRunExperiment}
                onStopExperiment={handleStopExperiment}
                onRefreshExperiments={handleRefreshExperiments}
              />
            )}
            {activeTab === "results" && (
              <ResultsTab experiments={experiments} />
            )}
            {activeTab === "publish" && (
              <PublishTab
                personas={personas}
                experiments={experiments}
                promptTemplates={promptTemplates}
                models={models}
              />
            )}
          </>
        )}
      </div>
    </div>
  );
};

export default Phase2Dashboard;
