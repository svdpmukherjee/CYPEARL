/**
 * Phase 2 Provider Setup Tab - Enhanced with Privacy-by-Design
 * 
 * Incorporates:
 * 1. Factorial model selection (Capability × Openness × Safety × Architecture)
 * 2. Privacy compliance indicators
 * 3. Data residency and retention policies
 * 4. Cost estimation per tier
 */

import React, { useState } from 'react';
import {
    Settings, Check, X, AlertTriangle, RefreshCw, Eye, EyeOff,
    Shield, Globe, Server, Cpu, DollarSign, Lock, Database,
    Info, CheckCircle, XCircle
} from 'lucide-react';

// ============================================================================
// CONSTANTS - FACTORIAL MODEL CLASSIFICATION
// ============================================================================

const MODEL_FACTORS = {
    capability: {
        label: 'Capability Tier',
        description: 'Model reasoning and task performance level',
        levels: {
            frontier: { label: 'Frontier', color: 'purple', description: 'Max reasoning, safety, long context' },
            mid_tier: { label: 'Mid-Tier', color: 'blue', description: 'Strong but cost-effective workhorses' },
            budget: { label: 'Budget', color: 'green', description: 'Minimal capability, very cheap' },
            open_source: { label: 'Open Source', color: 'orange', description: 'Self-hostable, customizable' }
        }
    },
    openness: {
        label: 'Deployment Model',
        description: 'How the model is accessed and controlled',
        levels: {
            closed_api: { label: 'Closed API', icon: Globe, description: 'Vendor-hosted, managed' },
            open_hosted: { label: 'Open (Hosted)', icon: Server, description: 'Open weights, cloud-hosted' },
            open_self: { label: 'Open (Self-Hosted)', icon: Database, description: 'Full control, on-premise' }
        }
    },
    safety: {
        label: 'Safety Regime',
        description: 'Alignment and guardrail strength',
        levels: {
            strong: { label: 'Strong Safety', color: 'green', description: 'Heavy guardrails, red-teamed' },
            standard: { label: 'Standard', color: 'blue', description: 'General instruction-tuned' },
            minimal: { label: 'Reasoning-First', color: 'amber', description: 'Weak guardrails, may over-rationalize' }
        }
    },
    architecture: {
        label: 'Architecture',
        description: 'Model structure and specialization',
        levels: {
            large_dense: { label: 'Large Dense', description: '70B+ parameters' },
            medium_dense: { label: 'Medium Dense', description: '7-30B parameters' },
            moe: { label: 'MoE/Specialized', description: 'Mixture-of-Experts or reasoning-focused' }
        }
    }
};

// Privacy compliance indicators - OpenRouter unified provider
const PROVIDER_PRIVACY = {
    openrouter: {
        noTraining: true,
        dataRetention: 'None (pass-through)',
        euDataResidency: false,
        enterpriseTerms: true,
        systemCard: 'Varies by model',
        gdprCompliant: true
    }
};

// Enhanced model catalog with factorial classification - ALL via OpenRouter
const MODEL_CATALOG = {
    // =========================================================================
    // FRONTIER TIER - Highest capability models
    // =========================================================================
    'claude-3-5-sonnet': {
        name: 'Claude 3.5 Sonnet',
        provider: 'openrouter',
        openrouterId: 'anthropic/claude-3.5-sonnet',
        factors: { capability: 'frontier', openness: 'closed_api', safety: 'strong', architecture: 'large_dense' },
        costPer1kTokens: { input: 3.0, output: 15.0 },
        contextWindow: 200000,
        recommended: true,
        notes: 'Best balance of cost vs performance for persona work'
    },
    'claude-3-opus': {
        name: 'Claude 3 Opus',
        provider: 'openrouter',
        openrouterId: 'anthropic/claude-3-opus',
        factors: { capability: 'frontier', openness: 'closed_api', safety: 'strong', architecture: 'large_dense' },
        costPer1kTokens: { input: 15.0, output: 75.0 },
        contextWindow: 200000,
        notes: 'Highest-end Claude for complex reasoning; upper bound on performance'
    },
    'gpt-4o': {
        name: 'GPT-4o',
        provider: 'openrouter',
        openrouterId: 'openai/gpt-4o',
        factors: { capability: 'frontier', openness: 'closed_api', safety: 'strong', architecture: 'large_dense' },
        costPer1kTokens: { input: 2.5, output: 10.0 },
        contextWindow: 128000,
        recommended: true,
        notes: 'Strong general reasoning, multimodal capabilities'
    },
    'gpt-4-turbo': {
        name: 'GPT-4 Turbo',
        provider: 'openrouter',
        openrouterId: 'openai/gpt-4-turbo',
        factors: { capability: 'frontier', openness: 'closed_api', safety: 'strong', architecture: 'large_dense' },
        costPer1kTokens: { input: 10.0, output: 30.0 },
        contextWindow: 128000,
        notes: 'Strong general reasoning, good safety tooling'
    },
    'gpt-4': {
        name: 'GPT-4',
        provider: 'openrouter',
        openrouterId: 'openai/gpt-4',
        factors: { capability: 'frontier', openness: 'closed_api', safety: 'strong', architecture: 'large_dense' },
        costPer1kTokens: { input: 30.0, output: 60.0 },
        contextWindow: 8192,
        notes: 'Original GPT-4, reliable baseline'
    },
    'mistral-large': {
        name: 'Mistral Large',
        provider: 'openrouter',
        openrouterId: 'mistralai/mistral-large',
        factors: { capability: 'frontier', openness: 'closed_api', safety: 'standard', architecture: 'large_dense' },
        costPer1kTokens: { input: 2.0, output: 6.0 },
        contextWindow: 128000,
        notes: 'EU provider, strong multilingual, competitive pricing'
    },
    'nova-pro': {
        name: 'Amazon Nova Pro',
        provider: 'openrouter',
        openrouterId: 'amazon/nova-pro-v1',
        factors: { capability: 'frontier', openness: 'closed_api', safety: 'standard', architecture: 'medium_dense' },
        costPer1kTokens: { input: 0.8, output: 3.2 },
        contextWindow: 300000,
        recommended: true,
        notes: 'Very cost-effective frontier model'
    },

    // =========================================================================
    // MID-TIER - Strong but cost-effective workhorses
    // =========================================================================
    'claude-3-5-haiku': {
        name: 'Claude 3.5 Haiku',
        provider: 'openrouter',
        openrouterId: 'anthropic/claude-3.5-haiku',
        factors: { capability: 'mid_tier', openness: 'closed_api', safety: 'strong', architecture: 'medium_dense' },
        costPer1kTokens: { input: 0.8, output: 4.0 },
        contextWindow: 200000,
        recommended: true,
        notes: 'Fast, efficient, maintains Anthropic safety'
    },
    'claude-3-sonnet': {
        name: 'Claude 3 Sonnet',
        provider: 'openrouter',
        openrouterId: 'anthropic/claude-3-sonnet',
        factors: { capability: 'mid_tier', openness: 'closed_api', safety: 'strong', architecture: 'large_dense' },
        costPer1kTokens: { input: 3.0, output: 15.0 },
        contextWindow: 200000,
        notes: 'Previous generation, still very capable'
    },
    'gpt-4o-mini': {
        name: 'GPT-4o Mini',
        provider: 'openrouter',
        openrouterId: 'openai/gpt-4o-mini',
        factors: { capability: 'mid_tier', openness: 'closed_api', safety: 'strong', architecture: 'medium_dense' },
        costPer1kTokens: { input: 0.15, output: 0.6 },
        contextWindow: 128000,
        recommended: true,
        notes: 'Excellent cost-performance ratio'
    },
    'mistral-medium': {
        name: 'Mistral Medium',
        provider: 'openrouter',
        openrouterId: 'mistralai/mistral-medium',
        factors: { capability: 'mid_tier', openness: 'closed_api', safety: 'standard', architecture: 'medium_dense' },
        costPer1kTokens: { input: 2.7, output: 8.1 },
        contextWindow: 32000,
        notes: 'High efficiency, near-frontier performance'
    },
    'mistral-small': {
        name: 'Mistral Small',
        provider: 'openrouter',
        openrouterId: 'mistralai/mistral-small',
        factors: { capability: 'mid_tier', openness: 'closed_api', safety: 'standard', architecture: 'medium_dense' },
        costPer1kTokens: { input: 1.0, output: 3.0 },
        contextWindow: 32000,
        notes: 'Good balance of speed and capability'
    },
    'mixtral-8x22b': {
        name: 'Mixtral 8x22B',
        provider: 'openrouter',
        openrouterId: 'mistralai/mixtral-8x22b-instruct',
        factors: { capability: 'mid_tier', openness: 'open_hosted', safety: 'standard', architecture: 'moe' },
        costPer1kTokens: { input: 0.65, output: 0.65 },
        contextWindow: 65000,
        notes: 'Large MoE architecture, open weights'
    },
    'nova-lite': {
        name: 'Amazon Nova Lite',
        provider: 'openrouter',
        openrouterId: 'amazon/nova-lite-v1',
        factors: { capability: 'mid_tier', openness: 'closed_api', safety: 'standard', architecture: 'medium_dense' },
        costPer1kTokens: { input: 0.06, output: 0.24 },
        contextWindow: 300000,
        notes: 'Very cheap, long context'
    },

    // =========================================================================
    // OPEN SOURCE - Self-hostable, customizable
    // =========================================================================
    'mixtral-8x7b': {
        name: 'Mixtral 8x7B',
        provider: 'openrouter',
        openrouterId: 'mistralai/mixtral-8x7b-instruct',
        factors: { capability: 'open_source', openness: 'open_hosted', safety: 'standard', architecture: 'moe' },
        costPer1kTokens: { input: 0.24, output: 0.24 },
        contextWindow: 32000,
        recommended: true,
        notes: 'MoE architecture, efficient inference, open weights'
    },

    // =========================================================================
    // BUDGET TIER - Minimal capability, very cheap
    // =========================================================================
    'claude-3-haiku': {
        name: 'Claude 3 Haiku',
        provider: 'openrouter',
        openrouterId: 'anthropic/claude-3-haiku',
        factors: { capability: 'budget', openness: 'closed_api', safety: 'strong', architecture: 'medium_dense' },
        costPer1kTokens: { input: 0.25, output: 1.25 },
        contextWindow: 200000,
        recommended: true,
        notes: 'Fast, cheap, maintains Anthropic safety'
    },
    'mistral-7b': {
        name: 'Mistral 7B Instruct',
        provider: 'openrouter',
        openrouterId: 'mistralai/mistral-7b-instruct',
        factors: { capability: 'budget', openness: 'open_self', safety: 'minimal', architecture: 'medium_dense' },
        costPer1kTokens: { input: 0.07, output: 0.07 },
        contextWindow: 32000,
        notes: 'Strong performance per FLOP, open weights'
    },
    'nova-micro': {
        name: 'Amazon Nova Micro',
        provider: 'openrouter',
        openrouterId: 'amazon/nova-micro-v1',
        factors: { capability: 'budget', openness: 'closed_api', safety: 'standard', architecture: 'medium_dense' },
        costPer1kTokens: { input: 0.035, output: 0.14 },
        contextWindow: 128000,
        notes: 'Extremely cheap, text-only'
    },

    // =========================================================================
    // LLAMA 4 MODELS (Meta via OpenRouter)
    // =========================================================================
    'llama-4-maverick': {
        name: 'Llama 4 Maverick',
        provider: 'openrouter',
        openrouterId: 'meta-llama/llama-4-maverick',
        factors: { capability: 'frontier', openness: 'open_hosted', safety: 'standard', architecture: 'moe' },
        costPer1kTokens: { input: 0.15, output: 0.6 },
        contextWindow: 1050000,
        recommended: true,
        notes: 'Llama 4 flagship, 400B params MoE, open weights'
    },
    'llama-4-scout': {
        name: 'Llama 4 Scout',
        provider: 'openrouter',
        openrouterId: 'meta-llama/llama-4-scout',
        factors: { capability: 'mid_tier', openness: 'open_hosted', safety: 'standard', architecture: 'moe' },
        costPer1kTokens: { input: 0.08, output: 0.3 },
        contextWindow: 328000,
        notes: 'Llama 4 efficient, 109B params MoE, 17B active'
    },

    // =========================================================================
    // LLAMA 3.x MODELS (Meta via OpenRouter)
    // =========================================================================
    'llama-3.3-70b': {
        name: 'Llama 3.3 70B Instruct',
        provider: 'openrouter',
        openrouterId: 'meta-llama/llama-3.3-70b-instruct',
        factors: { capability: 'open_source', openness: 'open_hosted', safety: 'standard', architecture: 'large_dense' },
        costPer1kTokens: { input: 0.1, output: 0.32 },
        contextWindow: 128000,
        recommended: true,
        notes: 'Latest Llama 3, strong multilingual, open weights'
    },
    'llama-3.1-405b': {
        name: 'Llama 3.1 405B Instruct',
        provider: 'openrouter',
        openrouterId: 'meta-llama/llama-3.1-405b-instruct',
        factors: { capability: 'frontier', openness: 'open_hosted', safety: 'standard', architecture: 'large_dense' },
        costPer1kTokens: { input: 3.5, output: 3.5 },
        contextWindow: 128000,
        notes: 'Largest open model, 405B params, frontier performance'
    },
    'llama-3.1-70b': {
        name: 'Llama 3.1 70B Instruct',
        provider: 'openrouter',
        openrouterId: 'meta-llama/llama-3.1-70b-instruct',
        factors: { capability: 'open_source', openness: 'open_hosted', safety: 'standard', architecture: 'large_dense' },
        costPer1kTokens: { input: 0.4, output: 0.4 },
        contextWindow: 128000,
        notes: 'Canonical open model, strong ecosystem'
    }
};

// ============================================================================
// COMPONENTS
// ============================================================================

const Card = ({ children, className = '' }) => (
    <div className={`bg-white rounded-xl shadow-sm border border-gray-200 p-6 ${className}`}>
        {children}
    </div>
);

const Badge = ({ children, color = 'gray' }) => {
    const colors = {
        gray: 'bg-gray-100 text-gray-700',
        red: 'bg-red-100 text-red-700',
        green: 'bg-green-100 text-green-700',
        blue: 'bg-blue-100 text-blue-700',
        purple: 'bg-purple-100 text-purple-700',
        yellow: 'bg-yellow-100 text-yellow-700',
        amber: 'bg-amber-100 text-amber-700',
        orange: 'bg-orange-100 text-orange-700'
    };
    return (
        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colors[color] || colors.gray}`}>
            {children}
        </span>
    );
};

const PrivacyIndicator = ({ label, value, good = true }) => (
    <div className="flex items-center gap-2 text-sm">
        {value === true || value === 'Yes' ? (
            <CheckCircle size={14} className="text-green-500" />
        ) : value === false || value === 'No' ? (
            <XCircle size={14} className="text-red-500" />
        ) : (
            <AlertTriangle size={14} className="text-amber-500" />
        )}
        <span className="text-gray-600">{label}:</span>
        <span className={`font-medium ${value === true ? 'text-green-700' :
            value === false ? 'text-red-700' :
                'text-amber-700'
            }`}>
            {typeof value === 'boolean' ? (value ? 'Yes' : 'No') : value}
        </span>
    </div>
);

// Privacy Compliance Panel
const PrivacyCompliancePanel = ({ provider }) => {
    const privacy = PROVIDER_PRIVACY[provider];
    if (!privacy) return null;

    return (
        <div className="mt-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
            <h4 className="text-sm font-semibold text-gray-900 mb-3 flex items-center gap-2">
                <Shield size={16} className="text-green-600" />
                Privacy & Compliance
            </h4>
            <div className="grid grid-cols-2 gap-2">
                <PrivacyIndicator label="No Training on Data" value={privacy.noTraining} />
                <PrivacyIndicator label="EU Data Residency" value={privacy.euDataResidency} />
                <PrivacyIndicator label="GDPR Compliant" value={privacy.gdprCompliant} />
                <PrivacyIndicator label="Enterprise Terms" value={privacy.enterpriseTerms} />
                <PrivacyIndicator label="System Card Published" value={privacy.systemCard} />
                <div className="flex items-center gap-2 text-sm">
                    <Info size={14} className="text-blue-500" />
                    <span className="text-gray-600">Retention:</span>
                    <span className="font-medium text-gray-900">{privacy.dataRetention}</span>
                </div>
            </div>
        </div>
    );
};

// Model Card with Factorial Classification
const ModelCard = ({ modelId, model, selected, onToggle, showPrivacy = false }) => {
    const factors = model.factors;
    const capabilityColors = {
        frontier: 'purple',
        mid_tier: 'blue',
        budget: 'green',
        open_source: 'orange'
    };

    return (
        <div
            className={`p-4 rounded-lg border-2 transition-all cursor-pointer ${selected
                ? 'border-indigo-500 bg-indigo-50'
                : 'border-gray-200 hover:border-gray-300'
                }`}
            onClick={() => onToggle(modelId)}
        >
            <div className="flex items-start justify-between mb-2">
                <div>
                    <h4 className="font-semibold text-gray-900">{model.name}</h4>
                    <p className="text-xs text-gray-500">{model.provider}</p>
                </div>
                {model.recommended && (
                    <Badge color="green">Recommended</Badge>
                )}
            </div>

            {/* Factor Tags */}
            <div className="flex flex-wrap gap-1 mb-2">
                <Badge color={capabilityColors[factors.capability]}>
                    {MODEL_FACTORS.capability.levels[factors.capability].label}
                </Badge>
                <Badge color="gray">
                    {MODEL_FACTORS.openness.levels[factors.openness].label}
                </Badge>
                <Badge color={factors.safety === 'strong' ? 'green' : factors.safety === 'minimal' ? 'amber' : 'blue'}>
                    {MODEL_FACTORS.safety.levels[factors.safety].label}
                </Badge>
            </div>

            {/* Cost */}
            <div className="text-xs text-gray-500 mb-2">
                <DollarSign size={12} className="inline" />
                ${model.costPer1kTokens.input}/${model.costPer1kTokens.output} per 1K tokens (in/out)
            </div>

            {/* Notes */}
            <p className="text-xs text-gray-600">{model.notes}</p>

            {/* Selection indicator */}
            <div className="mt-2 flex items-center gap-2">
                <input
                    type="checkbox"
                    checked={selected}
                    onChange={() => onToggle(modelId)}
                    className="rounded border-gray-300 text-indigo-600"
                />
                <span className="text-sm text-gray-600">
                    {selected ? 'Selected for experiments' : 'Click to select'}
                </span>
            </div>
        </div>
    );
};

// Factorial Selection Guide
const FactorialSelectionGuide = () => (
    <Card className="mb-6">
        <div className="flex items-start gap-3">
            <Info className="text-blue-600 mt-1 flex-shrink-0" size={20} />
            <div>
                <h3 className="font-semibold text-gray-900 mb-2">
                    Factorial Model Selection for Persona Benchmarking
                </h3>
                <p className="text-sm text-gray-600 mb-3">
                    For rigorous AI-persona research, select models that span multiple dimensions:
                </p>
                <div className="grid grid-cols-2 gap-4 text-sm">
                    <div>
                        <strong className="text-purple-700">Capability Tier</strong>
                        <p className="text-gray-600">Frontier → Mid → Budget to study scaling effects on behavioral fidelity</p>
                    </div>
                    <div>
                        <strong className="text-blue-700">Openness</strong>
                        <p className="text-gray-600">Closed API vs Open Weights for transparency and governance analysis</p>
                    </div>
                    <div>
                        <strong className="text-green-700">Safety Regime</strong>
                        <p className="text-gray-600">Map where models over-protect or under-protect in phishing scenarios</p>
                    </div>
                    <div>
                        <strong className="text-amber-700">Architecture</strong>
                        <p className="text-gray-600">Dense vs MoE to study structural effects on persona behavior</p>
                    </div>
                </div>
                <p className="text-xs text-gray-500 mt-3 italic">
                    Recommendation: Select 15-20 models covering all factor combinations for publishable benchmarking results.
                </p>
            </div>
        </div>
    </Card>
);

// Privacy Warning Banner
const PrivacyWarningBanner = () => (
    <div className="mb-6 p-4 bg-amber-50 border border-amber-200 rounded-lg">
        <div className="flex items-start gap-3">
            <Shield className="text-amber-600 mt-0.5 flex-shrink-0" size={20} />
            <div>
                <h4 className="font-semibold text-amber-900 mb-1">Privacy-by-Design Reminder</h4>
                <ul className="text-sm text-amber-800 space-y-1">
                    <li>• <strong>Persona data only</strong>: LLMs should only receive aggregate persona templates, never raw employee records</li>
                    <li>• <strong>Minimum cluster size</strong>: Personas should represent ≥50 participants to prevent re-identification</li>
                    <li>• <strong>Enterprise terms</strong>: Use providers with no-training, low-retention policies for sensitive persona data</li>
                    <li>• <strong>Organizational profiling</strong>: Treat persona specs as confidential security information</li>
                </ul>
            </div>
        </div>
    </div>
);

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const ProviderSetupTab = ({
    providers,
    models,
    onSetupProvider,
    onTestModel,
    selectedModels = [],
    onModelSelectionChange
}) => {
    const [activeProvider, setActiveProvider] = useState('anthropic');
    const [showPrivacyDetails, setShowPrivacyDetails] = useState(true);
    const [filterTier, setFilterTier] = useState('all');

    // Filter models by tier
    const filteredModels = Object.entries(MODEL_CATALOG).filter(([id, model]) => {
        if (filterTier === 'all') return true;
        return model.factors.capability === filterTier;
    });

    // Group models by tier for display
    const modelsByTier = {
        frontier: filteredModels.filter(([_, m]) => m.factors.capability === 'frontier'),
        mid_tier: filteredModels.filter(([_, m]) => m.factors.capability === 'mid_tier'),
        open_source: filteredModels.filter(([_, m]) => m.factors.capability === 'open_source'),
        budget: filteredModels.filter(([_, m]) => m.factors.capability === 'budget')
    };

    const handleToggleModel = (modelId) => {
        if (onModelSelectionChange) {
            const newSelection = selectedModels.includes(modelId)
                ? selectedModels.filter(id => id !== modelId)
                : [...selectedModels, modelId];
            onModelSelectionChange(newSelection);
        }
    };

    // Calculate selection coverage
    const selectionCoverage = {
        frontier: selectedModels.filter(id => MODEL_CATALOG[id]?.factors.capability === 'frontier').length,
        mid_tier: selectedModels.filter(id => MODEL_CATALOG[id]?.factors.capability === 'mid_tier').length,
        open_source: selectedModels.filter(id => MODEL_CATALOG[id]?.factors.capability === 'open_source').length,
        budget: selectedModels.filter(id => MODEL_CATALOG[id]?.factors.capability === 'budget').length
    };

    return (
        <div className="space-y-6">
            {/* Privacy Warning */}
            <PrivacyWarningBanner />

            {/* Factorial Selection Guide */}
            <FactorialSelectionGuide />

            {/* Selection Summary */}
            <Card>
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Cpu className="text-indigo-600" size={20} />
                    Model Selection Summary
                </h3>
                <div className="grid grid-cols-4 gap-4 mb-4">
                    <div className="p-3 bg-purple-50 rounded-lg text-center">
                        <div className="text-2xl font-bold text-purple-700">{selectionCoverage.frontier}</div>
                        <div className="text-xs text-purple-600">Frontier</div>
                    </div>
                    <div className="p-3 bg-blue-50 rounded-lg text-center">
                        <div className="text-2xl font-bold text-blue-700">{selectionCoverage.mid_tier}</div>
                        <div className="text-xs text-blue-600">Mid-Tier</div>
                    </div>
                    <div className="p-3 bg-orange-50 rounded-lg text-center">
                        <div className="text-2xl font-bold text-orange-700">{selectionCoverage.open_source}</div>
                        <div className="text-xs text-orange-600">Open Source</div>
                    </div>
                    <div className="p-3 bg-green-50 rounded-lg text-center">
                        <div className="text-2xl font-bold text-green-700">{selectionCoverage.budget}</div>
                        <div className="text-xs text-green-600">Budget</div>
                    </div>
                </div>
                <div className="text-sm text-gray-600">
                    <strong>{selectedModels.length}</strong> models selected for experiments
                    {selectedModels.length < 15 && (
                        <span className="text-amber-600 ml-2">
                            (Recommend 15-20 for comprehensive benchmarking)
                        </span>
                    )}
                </div>
            </Card>

            {/* Filter Controls */}
            <div className="flex items-center gap-4">
                <span className="text-sm text-gray-600">Filter by tier:</span>
                {['all', 'frontier', 'mid_tier', 'open_source', 'budget'].map(tier => (
                    <button
                        key={tier}
                        onClick={() => setFilterTier(tier)}
                        className={`px-3 py-1 rounded-full text-sm transition-colors ${filterTier === tier
                            ? 'bg-indigo-600 text-white'
                            : 'bg-gray-100 text-gray-600 hover:bg-gray-200'
                            }`}
                    >
                        {tier === 'all' ? 'All' : MODEL_FACTORS.capability.levels[tier]?.label || tier}
                    </button>
                ))}
                <button
                    onClick={() => setShowPrivacyDetails(!showPrivacyDetails)}
                    className="ml-auto flex items-center gap-1 text-sm text-gray-600 hover:text-gray-900"
                >
                    {showPrivacyDetails ? <EyeOff size={14} /> : <Eye size={14} />}
                    {showPrivacyDetails ? 'Hide' : 'Show'} Privacy Details
                </button>
            </div>

            {/* Model Grid by Tier */}
            {Object.entries(modelsByTier).map(([tier, tierModels]) => {
                if (tierModels.length === 0) return null;
                const tierInfo = MODEL_FACTORS.capability.levels[tier];

                return (
                    <Card key={tier}>
                        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                            <Badge color={tierInfo.color}>{tierInfo.label}</Badge>
                            <span className="text-gray-600 text-sm font-normal">
                                {tierInfo.description}
                            </span>
                        </h3>
                        <div className="grid grid-cols-2 gap-4">
                            {tierModels.map(([modelId, model]) => (
                                <ModelCard
                                    key={modelId}
                                    modelId={modelId}
                                    model={model}
                                    selected={selectedModels.includes(modelId)}
                                    onToggle={handleToggleModel}
                                    showPrivacy={showPrivacyDetails}
                                />
                            ))}
                        </div>

                        {/* Provider Privacy Details */}
                        {showPrivacyDetails && (
                            <div className="mt-4 grid grid-cols-2 gap-4">
                                {[...new Set(tierModels.map(([_, m]) => m.provider))].map(provider => (
                                    <PrivacyCompliancePanel key={provider} provider={provider} />
                                ))}
                            </div>
                        )}
                    </Card>
                );
            })}
        </div>
    );
};

export default ProviderSetupTab;