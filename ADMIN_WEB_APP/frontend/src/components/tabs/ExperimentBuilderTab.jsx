/**
 * Phase 2 Experiment Builder Tab
 * 
 * Create and configure experiments with persona/model/prompt selections.
 */

import React, { useState, useMemo, useEffect } from 'react';
import {
    Beaker, Users, Cpu, Mail, Settings, DollarSign,
    CheckCircle, AlertCircle, Play, Calculator, Eye
} from 'lucide-react';

// Prompt configuration options
const PROMPT_CONFIGS = [
    {
        id: 'baseline',
        name: 'Baseline (Task-only)',
        description: 'Minimal persona description, task instructions only',
        icon: '📝',
    },
    {
        id: 'stats',
        name: 'Behavioral Statistics',
        description: 'Adds click rates, response times, psychological profile',
        icon: '📊',
    },
    {
        id: 'cot',
        name: 'Chain-of-Thought',
        description: 'Full context with reasoning examples and boundary conditions',
        icon: '🧠',
    },
];

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export const ExperimentBuilderTab = ({
    personas,
    emails,
    models,
    experiments,
    isReady,
    onCreateExperiment,
    onEstimateCost,
    loading
}) => {
    // Form state
    const [experimentName, setExperimentName] = useState('');
    const [experimentDesc, setExperimentDesc] = useState('');
    const [selectedPersonas, setSelectedPersonas] = useState([]);
    const [selectedModels, setSelectedModels] = useState([]);
    const [selectedPromptConfigs, setSelectedPromptConfigs] = useState(['cot']);
    const [selectedEmails, setSelectedEmails] = useState([]);
    const [trialsPerCondition, setTrialsPerCondition] = useState(30);
    const [temperature, setTemperature] = useState(0.3);

    // Cost estimate state
    const [costEstimate, setCostEstimate] = useState(null);
    const [estimating, setEstimating] = useState(false);

    // Preview state
    const [showPreview, setShowPreview] = useState(false);

    // Flatten models
    const allModels = useMemo(() => {
        return Object.values(models).flat().filter(m => m.is_available);
    }, [models]);

    // Calculate experiment size
    const experimentSize = useMemo(() => {
        const nPersonas = selectedPersonas.length;
        const nModels = selectedModels.length;
        const nPrompts = selectedPromptConfigs.length;
        const nEmails = selectedEmails.length;
        const totalConditions = nPersonas * nModels * nPrompts * nEmails;
        const totalTrials = totalConditions * trialsPerCondition;

        return {
            nPersonas,
            nModels,
            nPrompts,
            nEmails,
            totalConditions,
            totalTrials,
        };
    }, [selectedPersonas, selectedModels, selectedPromptConfigs, selectedEmails, trialsPerCondition]);

    // Estimate cost when selection changes
    useEffect(() => {
        const estimateCost = async () => {
            if (experimentSize.totalTrials === 0) {
                setCostEstimate(null);
                return;
            }

            setEstimating(true);
            try {
                const estimate = await onEstimateCost({
                    personas: selectedPersonas,
                    models: selectedModels,
                    promptConfigs: selectedPromptConfigs,
                    emails: selectedEmails,
                    trialsPerCondition,
                });
                setCostEstimate(estimate);
            } catch (error) {
                console.error('Cost estimation failed:', error);
            } finally {
                setEstimating(false);
            }
        };

        const debounce = setTimeout(estimateCost, 500);
        return () => clearTimeout(debounce);
    }, [selectedPersonas, selectedModels, selectedPromptConfigs, selectedEmails, trialsPerCondition]);

    // Toggle selection helpers
    const togglePersona = (personaId) => {
        setSelectedPersonas(prev =>
            prev.includes(personaId)
                ? prev.filter(p => p !== personaId)
                : [...prev, personaId]
        );
    };

    const toggleModel = (modelId) => {
        setSelectedModels(prev =>
            prev.includes(modelId)
                ? prev.filter(m => m !== modelId)
                : [...prev, modelId]
        );
    };

    const togglePromptConfig = (configId) => {
        setSelectedPromptConfigs(prev =>
            prev.includes(configId)
                ? prev.filter(p => p !== configId)
                : [...prev, configId]
        );
    };

    const toggleEmail = (emailId) => {
        setSelectedEmails(prev =>
            prev.includes(emailId)
                ? prev.filter(e => e !== emailId)
                : [...prev, emailId]
        );
    };

    // Select all helpers
    const selectAllPersonas = () => setSelectedPersonas(personas.map(p => p.persona_id));
    const selectAllModels = () => setSelectedModels(allModels.map(m => m.model_id));
    const selectAllEmails = () => setSelectedEmails(emails.map(e => e.email_id));
    const selectPhishingEmails = () => setSelectedEmails(
        emails.filter(e => e.email_type === 'phishing').map(e => e.email_id)
    );

    // Create experiment
    const handleCreateExperiment = async () => {
        if (!experimentName.trim()) {
            alert('Please enter an experiment name');
            return;
        }

        if (experimentSize.totalTrials === 0) {
            alert('Please select at least one option from each category');
            return;
        }

        const result = await onCreateExperiment({
            name: experimentName,
            description: experimentDesc,
            persona_ids: selectedPersonas,
            model_ids: selectedModels,
            prompt_configs: selectedPromptConfigs,
            email_ids: selectedEmails,
            trials_per_condition: trialsPerCondition,
            temperature,
        });

        if (result.success) {
            // Reset form
            setExperimentName('');
            setExperimentDesc('');
            setSelectedPersonas([]);
            setSelectedModels([]);
            setSelectedPromptConfigs(['cot']);
            setSelectedEmails([]);
            alert('Experiment created successfully! Go to Execution tab to run it.');
        }
    };

    // Validation
    const isValid = experimentName.trim() &&
        selectedPersonas.length > 0 &&
        selectedModels.length > 0 &&
        selectedPromptConfigs.length > 0 &&
        selectedEmails.length > 0;

    return (
        <div className="space-y-6">
            {/* Readiness Check */}
            {!isReady && (
                <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 flex items-start gap-3">
                    <AlertCircle className="text-yellow-600 mt-0.5" size={20} />
                    <div>
                        <h4 className="font-semibold text-yellow-900">Setup Required</h4>
                        <p className="text-sm text-yellow-700">
                            Complete the following before creating experiments:
                            {personas.length === 0 && ' Import personas from Phase 1.'}
                            {emails.length === 0 && ' Load email stimuli.'}
                            {allModels.length === 0 && ' Configure at least one provider.'}
                        </p>
                    </div>
                </div>
            )}

            {/* Experiment Name */}
            <div className="bg-white rounded-xl border p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Beaker className="text-purple-600" size={20} />
                    New Experiment
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            Experiment Name *
                        </label>
                        <input
                            type="text"
                            value={experimentName}
                            onChange={(e) => setExperimentName(e.target.value)}
                            placeholder="e.g., Full Model Comparison - All Personas"
                            className="w-full px-3 py-2 border rounded-lg"
                        />
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            Description (optional)
                        </label>
                        <input
                            type="text"
                            value={experimentDesc}
                            onChange={(e) => setExperimentDesc(e.target.value)}
                            placeholder="Brief description of experiment goals"
                            className="w-full px-3 py-2 border rounded-lg"
                        />
                    </div>
                </div>
            </div>

            {/* Selection Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Personas Selection */}
                <div className="bg-white rounded-xl border p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h4 className="font-semibold flex items-center gap-2">
                            <Users className="text-purple-600" size={18} />
                            Personas ({selectedPersonas.length}/{personas.length})
                        </h4>
                        <button
                            onClick={selectAllPersonas}
                            className="text-xs text-purple-600 hover:text-purple-800"
                        >
                            Select All
                        </button>
                    </div>
                    <div className="max-h-64 overflow-y-auto space-y-2">
                        {personas.map(persona => (
                            <label
                                key={persona.persona_id}
                                className={`flex items-center gap-3 p-2 rounded cursor-pointer transition ${selectedPersonas.includes(persona.persona_id)
                                    ? 'bg-purple-50 border border-purple-200'
                                    : 'hover:bg-gray-50 border border-transparent'
                                    }`}
                            >
                                <input
                                    type="checkbox"
                                    checked={selectedPersonas.includes(persona.persona_id)}
                                    onChange={() => togglePersona(persona.persona_id)}
                                    className="w-4 h-4 accent-purple-600"
                                />
                                <div className="flex-1">
                                    <div className="font-medium text-sm">{persona.name}</div>
                                    <div className="text-xs text-gray-500">
                                        {persona.risk_level} • {(persona.behavioral_statistics?.phishing_click_rate * 100 || 0).toFixed(0)}% click
                                    </div>
                                </div>
                            </label>
                        ))}
                        {personas.length === 0 && (
                            <p className="text-sm text-gray-500 text-center py-4">
                                No personas loaded
                            </p>
                        )}
                    </div>
                </div>

                {/* Models Selection */}
                <div className="bg-white rounded-xl border p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h4 className="font-semibold flex items-center gap-2">
                            <Cpu className="text-blue-600" size={18} />
                            Models ({selectedModels.length}/{allModels.length})
                        </h4>
                        <button
                            onClick={selectAllModels}
                            className="text-xs text-blue-600 hover:text-blue-800"
                        >
                            Select All
                        </button>
                    </div>
                    <div className="max-h-64 overflow-y-auto space-y-2">
                        {allModels.map(model => (
                            <label
                                key={model.model_id}
                                className={`flex items-center gap-3 p-2 rounded cursor-pointer transition ${selectedModels.includes(model.model_id)
                                    ? 'bg-blue-50 border border-blue-200'
                                    : 'hover:bg-gray-50 border border-transparent'
                                    }`}
                            >
                                <input
                                    type="checkbox"
                                    checked={selectedModels.includes(model.model_id)}
                                    onChange={() => toggleModel(model.model_id)}
                                    className="w-4 h-4 accent-blue-600"
                                />
                                <div className="flex-1">
                                    <div className="font-medium text-sm">{model.display_name}</div>
                                    <div className="text-xs text-gray-500">
                                        {model.tier} • ${model.cost_per_1k_input}/1K tokens
                                    </div>
                                </div>
                            </label>
                        ))}
                        {allModels.length === 0 && (
                            <p className="text-sm text-gray-500 text-center py-4">
                                No models available. Configure providers first.
                            </p>
                        )}
                    </div>
                </div>

                {/* Prompt Configurations */}
                <div className="bg-white rounded-xl border p-6">
                    <h4 className="font-semibold flex items-center gap-2 mb-4">
                        <Settings className="text-green-600" size={18} />
                        Prompt Configurations ({selectedPromptConfigs.length}/3)
                    </h4>
                    <div className="space-y-2">
                        {PROMPT_CONFIGS.map(config => (
                            <label
                                key={config.id}
                                className={`flex items-start gap-3 p-3 rounded cursor-pointer transition ${selectedPromptConfigs.includes(config.id)
                                    ? 'bg-green-50 border border-green-200'
                                    : 'hover:bg-gray-50 border border-transparent'
                                    }`}
                            >
                                <input
                                    type="checkbox"
                                    checked={selectedPromptConfigs.includes(config.id)}
                                    onChange={() => togglePromptConfig(config.id)}
                                    className="w-4 h-4 accent-green-600 mt-1"
                                />
                                <div>
                                    <div className="font-medium text-sm flex items-center gap-2">
                                        <span>{config.icon}</span>
                                        {config.name}
                                    </div>
                                    <div className="text-xs text-gray-500">{config.description}</div>
                                </div>
                            </label>
                        ))}
                    </div>
                </div>

                {/* Emails Selection */}
                <div className="bg-white rounded-xl border p-6">
                    <div className="flex items-center justify-between mb-4">
                        <h4 className="font-semibold flex items-center gap-2">
                            <Mail className="text-amber-600" size={18} />
                            Emails ({selectedEmails.length}/{emails.length})
                        </h4>
                        <div className="flex gap-2">
                            <button
                                onClick={selectPhishingEmails}
                                className="text-xs text-amber-600 hover:text-amber-800"
                            >
                                Phishing Only
                            </button>
                            <button
                                onClick={selectAllEmails}
                                className="text-xs text-amber-600 hover:text-amber-800"
                            >
                                Select All
                            </button>
                        </div>
                    </div>
                    <div className="max-h-64 overflow-y-auto space-y-2">
                        {emails.map(email => (
                            <label
                                key={email.email_id}
                                className={`flex items-center gap-3 p-2 rounded cursor-pointer transition ${selectedEmails.includes(email.email_id)
                                    ? 'bg-amber-50 border border-amber-200'
                                    : 'hover:bg-gray-50 border border-transparent'
                                    }`}
                            >
                                <input
                                    type="checkbox"
                                    checked={selectedEmails.includes(email.email_id)}
                                    onChange={() => toggleEmail(email.email_id)}
                                    className="w-4 h-4 accent-amber-600"
                                />
                                <div className="flex-1">
                                    <div className="font-medium text-sm">{email.email_id}</div>
                                    <div className="text-xs text-gray-500">
                                        {email.email_type} • {email.urgency_level} urgency • {email.sender_familiarity}
                                    </div>
                                </div>
                            </label>
                        ))}
                        {emails.length === 0 && (
                            <p className="text-sm text-gray-500 text-center py-4">
                                No emails loaded
                            </p>
                        )}
                    </div>
                </div>
            </div>

            {/* Advanced Settings */}
            <div className="bg-white rounded-xl border p-6">
                <h4 className="font-semibold mb-4">Advanced Settings</h4>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            Trials Per Condition
                        </label>
                        <div className="flex items-center gap-3">
                            <input
                                type="range"
                                min="10"
                                max="100"
                                step="10"
                                value={trialsPerCondition}
                                onChange={(e) => setTrialsPerCondition(parseInt(e.target.value))}
                                className="flex-1"
                            />
                            <input
                                type="number"
                                min="10"
                                max="100"
                                value={trialsPerCondition}
                                onChange={(e) => setTrialsPerCondition(parseInt(e.target.value) || 30)}
                                className="w-20 px-2 py-1 border rounded text-center"
                            />
                        </div>
                        <p className="text-xs text-gray-500 mt-1">
                            Minimum 30 recommended for statistical significance
                        </p>
                    </div>
                    <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">
                            Temperature
                        </label>
                        <div className="flex items-center gap-3">
                            <input
                                type="range"
                                min="0"
                                max="1"
                                step="0.1"
                                value={temperature}
                                onChange={(e) => setTemperature(parseFloat(e.target.value))}
                                className="flex-1"
                            />
                            <input
                                type="number"
                                min="0"
                                max="1"
                                step="0.1"
                                value={temperature}
                                onChange={(e) => setTemperature(parseFloat(e.target.value) || 0.3)}
                                className="w-20 px-2 py-1 border rounded text-center"
                            />
                        </div>
                        <p className="text-xs text-gray-500 mt-1">
                            Lower = more deterministic, Higher = more varied responses
                        </p>
                    </div>
                </div>
            </div>

            {/* Summary & Cost */}
            <div className="bg-white rounded-xl border p-6">
                <h4 className="font-semibold mb-4 flex items-center gap-2">
                    <Calculator className="text-green-600" size={18} />
                    Experiment Summary
                </h4>
                <div className="grid grid-cols-2 md:grid-cols-6 gap-4 mb-4">
                    <div className="p-3 bg-gray-50 rounded-lg text-center">
                        <div className="text-2xl font-bold text-purple-600">{experimentSize.nPersonas}</div>
                        <div className="text-xs text-gray-500">Personas</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg text-center">
                        <div className="text-2xl font-bold text-blue-600">{experimentSize.nModels}</div>
                        <div className="text-xs text-gray-500">Models</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg text-center">
                        <div className="text-2xl font-bold text-green-600">{experimentSize.nPrompts}</div>
                        <div className="text-xs text-gray-500">Prompts</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg text-center">
                        <div className="text-2xl font-bold text-amber-600">{experimentSize.nEmails}</div>
                        <div className="text-xs text-gray-500">Emails</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg text-center">
                        <div className="text-2xl font-bold">{experimentSize.totalConditions}</div>
                        <div className="text-xs text-gray-500">Conditions</div>
                    </div>
                    <div className="p-3 bg-gray-50 rounded-lg text-center">
                        <div className="text-2xl font-bold">{experimentSize.totalTrials.toLocaleString()}</div>
                        <div className="text-xs text-gray-500">Total Trials</div>
                    </div>
                </div>

                {/* Cost Estimate */}
                {costEstimate && (
                    <div className="p-4 bg-green-50 border border-green-200 rounded-lg mb-4">
                        <div className="flex items-center justify-between">
                            <div>
                                <span className="text-sm text-gray-600">Estimated Cost: </span>
                                <span className="text-2xl font-bold text-green-700">
                                    ${costEstimate.total_cost?.toFixed(2) || '0.00'}
                                </span>
                            </div>
                            <div className="text-right text-sm text-gray-500">
                                <div>~{costEstimate.estimated_time || 'N/A'} runtime</div>
                                <div>{costEstimate.total_tokens?.toLocaleString() || 0} tokens</div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Create Button */}
                <div className="flex items-center gap-3">
                    <button
                        onClick={handleCreateExperiment}
                        disabled={!isValid || loading}
                        className="px-6 py-3 bg-purple-600 text-white rounded-lg font-medium hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                    >
                        {loading ? (
                            <span className="animate-spin">⏳</span>
                        ) : (
                            <CheckCircle size={18} />
                        )}
                        Create Experiment
                    </button>
                    <button
                        onClick={() => setShowPreview(true)}
                        disabled={experimentSize.totalTrials === 0}
                        className="px-4 py-3 border rounded-lg text-gray-700 hover:bg-gray-50 disabled:opacity-50 flex items-center gap-2"
                    >
                        <Eye size={18} />
                        Preview Prompt
                    </button>
                </div>

                {!isValid && (
                    <p className="text-sm text-amber-600 mt-2">
                        Select at least one option from each category
                    </p>
                )}
            </div>
        </div>
    );
};

export default ExperimentBuilderTab;