/**
 * Phase 2 Overview Tab - Enhanced with Privacy-by-Design
 * 
 * Key improvements based on Perplexity conversation:
 * 1. Two-layer architecture visualization (your backend vs LLM layer)
 * 2. Minimum cluster size validation (n ≥ 50 recommended)
 * 3. Organizational profiling risk warnings
 * 4. Privacy mode options for persona data sent to LLMs
 * 5. Clear data flow visualization
 */

import React, { useState, useRef, useMemo } from 'react';
import {
    Users, Mail, Cpu, Database, Upload, RefreshCw,
    CheckCircle, AlertCircle, TrendingUp, DollarSign,
    Shield, AlertTriangle, Eye, EyeOff, Lock, Info,
    Layers, Server, Globe, ArrowRight, XCircle
} from 'lucide-react';

// ============================================================================
// CONSTANTS
// ============================================================================

const MINIMUM_CLUSTER_SIZE = 50; // GDPR-recommended minimum for anonymity

const PRIVACY_MODES = {
    full: {
        label: 'Full Persona Data',
        description: 'Send complete persona specs including all z-scores and behavioral statistics',
        risk: 'high',
        recommended: false
    },
    standard: {
        label: 'Standard (Recommended)',
        description: 'Send essential traits, behavioral targets, and reasoning examples only',
        risk: 'medium',
        recommended: true
    },
    minimal: {
        label: 'Minimal',
        description: 'Send only natural language descriptions and key distinguishing traits',
        risk: 'low',
        recommended: false
    }
};

// ============================================================================
// HELPER COMPONENTS
// ============================================================================

const Card = ({ children, className = '' }) => (
    <div className={`bg-white rounded-xl shadow-sm border border-gray-200 p-6 ${className}`}>
        {children}
    </div>
);

const StatCard = ({ icon, label, value, sublabel, status }) => {
    const statusColors = {
        success: 'border-green-200 bg-green-50',
        warning: 'border-yellow-200 bg-yellow-50',
        error: 'border-red-200 bg-red-50',
        neutral: 'border-gray-200 bg-gray-50'
    };

    return (
        <div className={`p-4 rounded-xl border ${statusColors[status] || statusColors.neutral}`}>
            <div className="flex items-center gap-3">
                {icon}
                <div>
                    <div className="text-2xl font-bold">{value}</div>
                    <div className="text-sm text-gray-600">{label}</div>
                    <div className="text-xs text-gray-500">{sublabel}</div>
                </div>
            </div>
        </div>
    );
};

const Badge = ({ children, color = 'gray' }) => {
    const colors = {
        gray: 'bg-gray-100 text-gray-700',
        red: 'bg-red-100 text-red-700',
        green: 'bg-green-100 text-green-700',
        blue: 'bg-blue-100 text-blue-700',
        purple: 'bg-purple-100 text-purple-700',
        yellow: 'bg-yellow-100 text-yellow-700',
        amber: 'bg-amber-100 text-amber-700'
    };
    return (
        <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${colors[color] || colors.gray}`}>
            {children}
        </span>
    );
};

// ============================================================================
// DATA FLOW VISUALIZATION
// ============================================================================

const DataFlowVisualization = () => (
    <Card>
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Layers className="text-indigo-600" size={20} />
            Privacy-by-Design Data Flow
        </h3>
        <div className="relative">
            {/* Flow diagram */}
            <div className="flex items-center justify-between gap-4 p-4 bg-gray-50 rounded-lg">
                {/* Layer 1: Client Side */}
                <div className="flex-1 p-4 bg-blue-50 border-2 border-blue-200 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                        <Database className="text-blue-600" size={20} />
                        <span className="font-semibold text-blue-900">Layer 1: Client Side</span>
                    </div>
                    <ul className="text-xs text-blue-800 space-y-1">
                        <li>• Raw employee psycho-cog data</li>
                        <li>• Pseudonymization</li>
                        <li>• Employee → Persona mapping</li>
                    </ul>
                    <div className="mt-2 px-2 py-1 bg-blue-100 rounded text-xs text-blue-700">
                        <Lock size={12} className="inline mr-1" />
                        Data stays with CISO
                    </div>
                </div>

                <ArrowRight className="text-gray-400 flex-shrink-0" size={24} />

                {/* Layer 2: Your Platform */}
                <div className="flex-1 p-4 bg-purple-50 border-2 border-purple-200 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                        <Server className="text-purple-600" size={20} />
                        <span className="font-semibold text-purple-900">Layer 2: CYPEARL Platform</span>
                    </div>
                    <ul className="text-xs text-purple-800 space-y-1">
                        <li>• Aggregate persona templates</li>
                        <li>• Cluster-level statistics</li>
                        <li>• Experiment configuration</li>
                    </ul>
                    <div className="mt-2 px-2 py-1 bg-purple-100 rounded text-xs text-purple-700">
                        <Shield size={12} className="inline mr-1" />
                        GDPR processor role
                    </div>
                </div>

                <ArrowRight className="text-gray-400 flex-shrink-0" size={24} />

                {/* Layer 3: LLM Providers */}
                <div className="flex-1 p-4 bg-green-50 border-2 border-green-200 rounded-lg">
                    <div className="flex items-center gap-2 mb-2">
                        <Globe className="text-green-600" size={20} />
                        <span className="font-semibold text-green-900">Layer 3: LLM Providers</span>
                    </div>
                    <ul className="text-xs text-green-800 space-y-1">
                        <li>• Persona descriptions only</li>
                        <li>• No individual data</li>
                        <li>• No-training agreements</li>
                    </ul>
                    <div className="mt-2 px-2 py-1 bg-green-100 rounded text-xs text-green-700">
                        <CheckCircle size={12} className="inline mr-1" />
                        Sub-processor, minimal data
                    </div>
                </div>
            </div>

            {/* Key principle */}
            <div className="mt-4 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                <div className="flex items-start gap-2">
                    <Info className="text-amber-600 mt-0.5 flex-shrink-0" size={16} />
                    <p className="text-sm text-amber-800">
                        <strong>Key Principle:</strong> Raw employee data never leaves Layer 1.
                        LLMs only receive aggregate persona templates that cannot identify individuals.
                    </p>
                </div>
            </div>
        </div>
    </Card>
);

// ============================================================================
// PERSONA PRIVACY VALIDATION
// ============================================================================

const PersonaPrivacyValidation = ({ personas }) => {
    const validation = useMemo(() => {
        if (!personas || personas.length === 0) return null;

        const smallClusters = personas.filter(p =>
            (p.n_participants || 0) < MINIMUM_CLUSTER_SIZE
        );
        const totalParticipants = personas.reduce((sum, p) =>
            sum + (p.n_participants || 0), 0
        );
        const avgClusterSize = totalParticipants / personas.length;

        return {
            smallClusters,
            totalParticipants,
            avgClusterSize,
            hasPrivacyRisk: smallClusters.length > 0,
            riskLevel: smallClusters.length === 0 ? 'low' :
                smallClusters.length <= 2 ? 'medium' : 'high'
        };
    }, [personas]);

    if (!validation) return null;

    return (
        <Card>
            <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                <Shield className="text-green-600" size={20} />
                Persona Privacy Validation
            </h3>

            {/* Summary Stats */}
            <div className="grid grid-cols-3 gap-4 mb-4">
                <div className="p-3 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-gray-900">{personas.length}</div>
                    <div className="text-xs text-gray-600">Total Personas</div>
                </div>
                <div className="p-3 bg-gray-50 rounded-lg">
                    <div className="text-2xl font-bold text-gray-900">
                        {validation.avgClusterSize.toFixed(0)}
                    </div>
                    <div className="text-xs text-gray-600">Avg Cluster Size</div>
                </div>
                <div className={`p-3 rounded-lg ${validation.smallClusters.length === 0
                        ? 'bg-green-50'
                        : 'bg-red-50'
                    }`}>
                    <div className={`text-2xl font-bold ${validation.smallClusters.length === 0
                            ? 'text-green-700'
                            : 'text-red-700'
                        }`}>
                        {validation.smallClusters.length}
                    </div>
                    <div className="text-xs text-gray-600">Small Clusters (&lt;{MINIMUM_CLUSTER_SIZE})</div>
                </div>
            </div>

            {/* Validation Status */}
            {validation.hasPrivacyRisk ? (
                <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                    <div className="flex items-start gap-3">
                        <AlertTriangle className="text-red-600 flex-shrink-0 mt-0.5" size={20} />
                        <div>
                            <h4 className="font-semibold text-red-900 mb-1">
                                Re-identification Risk Detected
                            </h4>
                            <p className="text-sm text-red-800 mb-2">
                                {validation.smallClusters.length} persona(s) have fewer than {MINIMUM_CLUSTER_SIZE} participants,
                                which may allow re-identification of individuals.
                            </p>
                            <div className="space-y-1">
                                {validation.smallClusters.map(p => (
                                    <div key={p.persona_id} className="text-xs bg-red-100 px-2 py-1 rounded inline-block mr-2">
                                        {p.name || p.persona_id}: n={p.n_participants}
                                    </div>
                                ))}
                            </div>
                            <p className="text-xs text-red-700 mt-2 italic">
                                Recommendation: Merge small clusters or exclude them from LLM experiments.
                            </p>
                        </div>
                    </div>
                </div>
            ) : (
                <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                    <div className="flex items-center gap-3">
                        <CheckCircle className="text-green-600" size={20} />
                        <div>
                            <h4 className="font-semibold text-green-900">
                                Privacy Validation Passed
                            </h4>
                            <p className="text-sm text-green-800">
                                All personas have sufficient cluster size (≥{MINIMUM_CLUSTER_SIZE}) to prevent re-identification.
                            </p>
                        </div>
                    </div>
                </div>
            )}
        </Card>
    );
};

// ============================================================================
// ORGANIZATIONAL PROFILING WARNING
// ============================================================================

const OrganizationalProfilingWarning = ({ personas }) => {
    if (!personas || personas.length === 0) return null;

    // Calculate vulnerability profile
    const highRiskCount = personas.filter(p =>
        p.risk_level === 'CRITICAL' || p.risk_level === 'HIGH'
    ).length;
    const highRiskPct = (highRiskCount / personas.length * 100).toFixed(0);

    return (
        <Card className="border-amber-200 bg-amber-50">
            <div className="flex items-start gap-3">
                <AlertTriangle className="text-amber-600 flex-shrink-0 mt-0.5" size={24} />
                <div>
                    <h3 className="text-lg font-semibold text-amber-900 mb-2">
                        Organizational Profiling Risk
                    </h3>
                    <p className="text-sm text-amber-800 mb-3">
                        Your {personas.length} personas form a <strong>behavioral fingerprint</strong> of the workforce.
                        If leaked, attackers could learn:
                    </p>
                    <ul className="text-sm text-amber-800 space-y-1 mb-3">
                        <li>• <strong>{highRiskPct}%</strong> of personas are high-risk for phishing susceptibility</li>
                        <li>• Which psychological manipulation tactics are likely to work</li>
                        <li>• Overall security awareness posture of the organization</li>
                    </ul>
                    <div className="p-3 bg-amber-100 rounded-lg">
                        <h4 className="font-semibold text-amber-900 text-sm mb-1">Required Safeguards:</h4>
                        <ul className="text-xs text-amber-800 space-y-1">
                            <li>✓ Treat persona specs as confidential security information</li>
                            <li>✓ Encrypt at rest and in transit</li>
                            <li>✓ Implement strict access control (CISO team only)</li>
                            <li>✓ Maintain audit logs of all access</li>
                            <li>✓ Use enterprise LLM endpoints with data retention controls</li>
                        </ul>
                    </div>
                </div>
            </div>
        </Card>
    );
};

// ============================================================================
// PRIVACY MODE SELECTOR
// ============================================================================

const PrivacyModeSelector = ({ selectedMode, onModeChange }) => (
    <Card>
        <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
            <Eye className="text-indigo-600" size={20} />
            LLM Data Exposure Mode
        </h3>
        <p className="text-sm text-gray-600 mb-4">
            Select how much persona data is sent to LLM providers during experiments.
        </p>

        <div className="space-y-3">
            {Object.entries(PRIVACY_MODES).map(([key, mode]) => (
                <label
                    key={key}
                    className={`flex items-start gap-3 p-4 rounded-lg border-2 cursor-pointer transition-all ${selectedMode === key
                            ? 'border-indigo-500 bg-indigo-50'
                            : 'border-gray-200 hover:border-gray-300'
                        }`}
                >
                    <input
                        type="radio"
                        name="privacyMode"
                        value={key}
                        checked={selectedMode === key}
                        onChange={() => onModeChange(key)}
                        className="mt-1"
                    />
                    <div className="flex-1">
                        <div className="flex items-center gap-2">
                            <span className="font-semibold text-gray-900">{mode.label}</span>
                            {mode.recommended && <Badge color="green">Recommended</Badge>}
                            <Badge color={
                                mode.risk === 'low' ? 'green' :
                                    mode.risk === 'medium' ? 'yellow' : 'red'
                            }>
                                {mode.risk} exposure
                            </Badge>
                        </div>
                        <p className="text-sm text-gray-600 mt-1">{mode.description}</p>
                    </div>
                </label>
            ))}
        </div>

        {/* What each mode includes */}
        <div className="mt-4 p-3 bg-gray-50 rounded-lg">
            <h4 className="text-sm font-semibold text-gray-900 mb-2">Data sent to LLMs by mode:</h4>
            <div className="grid grid-cols-3 gap-2 text-xs">
                <div>
                    <strong className="text-gray-700">Full:</strong>
                    <ul className="text-gray-600 mt-1 space-y-0.5">
                        <li>• All trait z-scores</li>
                        <li>• Behavioral statistics</li>
                        <li>• Boundary conditions</li>
                        <li>• Reasoning examples</li>
                        <li>• Vulnerability profile</li>
                    </ul>
                </div>
                <div>
                    <strong className="text-gray-700">Standard:</strong>
                    <ul className="text-gray-600 mt-1 space-y-0.5">
                        <li>• Key traits (high/low)</li>
                        <li>• Target accuracy</li>
                        <li>• Cognitive style</li>
                        <li>• Reasoning examples</li>
                    </ul>
                </div>
                <div>
                    <strong className="text-gray-700">Minimal:</strong>
                    <ul className="text-gray-600 mt-1 space-y-0.5">
                        <li>• Description text</li>
                        <li>• Risk level</li>
                        <li>• Archetype name</li>
                    </ul>
                </div>
            </div>
        </div>
    </Card>
);

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const OverviewTab = ({
    personas,
    emails,
    models,
    providers,
    experiments,
    usage,
    onImportPersonas,
    onLoadEmails,
    onRefresh,
    hasImportedFromPhase1 = false,
    privacyMode = 'standard',
    onPrivacyModeChange
}) => {
    const [importing, setImporting] = useState(false);
    const [loadingEmails, setLoadingEmails] = useState(false);
    const [importResult, setImportResult] = useState(null);
    const [showUploadOption, setShowUploadOption] = useState(false);
    const fileInputRef = useRef(null);
    const emailFileInputRef = useRef(null);

    // Calculate stats
    const modelsByTier = models || {};
    const totalModels = Object.values(modelsByTier).flat().length;
    const availableModels = Object.values(modelsByTier).flat().filter(m => m.is_available).length;
    const configuredProviders = Array.isArray(providers)
        ? providers.filter(p => p.is_configured).length
        : Object.values(providers).filter(p => p?.configured).length;
    const completedExperiments = experiments.filter(e => e.status === 'completed').length;

    // Handle persona import
    const handleFileUpload = async (event) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setImporting(true);
        setImportResult(null);

        try {
            const text = await file.text();
            const data = JSON.parse(text);
            const result = await onImportPersonas(data);
            setImportResult(result);
            setShowUploadOption(false);
        } catch (error) {
            setImportResult({ success: false, error: error.message });
        } finally {
            setImporting(false);
        }
    };

    // Handle email loading
    const handleEmailUpload = async (event) => {
        const file = event.target.files?.[0];
        if (!file) return;

        setLoadingEmails(true);

        try {
            const text = await file.text();
            let data;
            if (file.name.endsWith('.json')) {
                data = JSON.parse(text);
            } else {
                const lines = text.split('\n');
                const headers = lines[0].split(',').map(h => h.trim());
                data = lines.slice(1).filter(l => l.trim()).map(line => {
                    const values = line.split(',');
                    const obj = {};
                    headers.forEach((h, i) => obj[h] = values[i]?.trim());
                    return obj;
                });
            }
            await onLoadEmails({ emails: data });
        } catch (error) {
            console.error('Failed to load emails:', error);
        } finally {
            setLoadingEmails(false);
        }
    };

    return (
        <div className="space-y-6">
            {/* Data Flow Visualization */}
            <DataFlowVisualization />

            {/* Quick Stats */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <StatCard
                    icon={<Users className="text-purple-600" />}
                    label="Personas Loaded"
                    value={personas.length}
                    sublabel={personas.length > 0 ? "Ready for simulation" : "Import from Phase 1"}
                    status={personas.length > 0 ? 'success' : 'warning'}
                />
                <StatCard
                    icon={<Mail className="text-blue-600" />}
                    label="Email Stimuli"
                    value={emails.length}
                    sublabel={emails.length >= 16 ? "Full factorial design" : "Load email data"}
                    status={emails.length >= 16 ? 'success' : emails.length > 0 ? 'warning' : 'error'}
                />
                <StatCard
                    icon={<Cpu className="text-green-600" />}
                    label="Available Models"
                    value={`${availableModels}/${totalModels}`}
                    sublabel={`${configuredProviders} providers configured`}
                    status={availableModels > 0 ? 'success' : 'error'}
                />
                <StatCard
                    icon={<TrendingUp className="text-amber-600" />}
                    label="Experiments"
                    value={experiments.length}
                    sublabel={`${completedExperiments} completed`}
                    status={completedExperiments > 0 ? 'success' : 'neutral'}
                />
            </div>

            {/* Organizational Profiling Warning (if personas loaded) */}
            {personas.length > 0 && <OrganizationalProfilingWarning personas={personas} />}

            {/* Persona Privacy Validation */}
            {personas.length > 0 && <PersonaPrivacyValidation personas={personas} />}

            {/* Privacy Mode Selector */}
            <PrivacyModeSelector
                selectedMode={privacyMode}
                onModeChange={onPrivacyModeChange}
            />

            {/* Import/Load Section */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Import Personas */}
                <Card>
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold flex items-center gap-2">
                            <Users className="text-purple-600" size={20} />
                            Personas
                        </h3>
                        {personas.length === 0 && (
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                disabled={importing}
                                className="px-4 py-2 bg-purple-600 text-white rounded-lg text-sm hover:bg-purple-700 disabled:opacity-50 flex items-center gap-2"
                            >
                                {importing ? (
                                    <RefreshCw className="animate-spin" size={16} />
                                ) : (
                                    <Upload size={16} />
                                )}
                                {importing ? 'Importing...' : 'Import JSON'}
                            </button>
                        )}
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept=".json"
                            onChange={handleFileUpload}
                            className="hidden"
                        />
                    </div>

                    {importResult && (
                        <div className={`p-3 rounded-lg mb-4 ${importResult.success ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'}`}>
                            {importResult.success
                                ? `Successfully imported ${importResult.count} personas`
                                : `Import failed: ${importResult.error}`
                            }
                        </div>
                    )}

                    {personas.length > 0 ? (
                        <div className="space-y-3">
                            {hasImportedFromPhase1 && (
                                <div className="flex items-center gap-2 px-3 py-2 bg-green-50 border border-green-200 rounded-lg text-sm text-green-700">
                                    <CheckCircle size={16} />
                                    <span>Automatically transferred from Phase 1</span>
                                </div>
                            )}
                            <div className="text-sm text-gray-500 mb-2">
                                {personas.length} personas loaded:
                            </div>
                            <div className="max-h-48 overflow-y-auto space-y-2">
                                {personas.map(p => (
                                    <div key={p.persona_id} className="flex items-center justify-between p-2 bg-gray-50 rounded">
                                        <div>
                                            <span className="font-medium text-gray-900">{p.name || p.persona_id}</span>
                                            <span className="text-xs text-gray-500 ml-2">n={p.n_participants || '?'}</span>
                                        </div>
                                        <div className="flex items-center gap-2">
                                            {(p.n_participants || 0) < MINIMUM_CLUSTER_SIZE && (
                                                <AlertTriangle size={14} className="text-amber-500" title="Small cluster - privacy risk" />
                                            )}
                                            <Badge color={
                                                p.risk_level === 'CRITICAL' ? 'red' :
                                                    p.risk_level === 'HIGH' ? 'red' :
                                                        p.risk_level === 'MEDIUM' ? 'yellow' : 'green'
                                            }>
                                                {p.risk_level || 'UNKNOWN'}
                                            </Badge>
                                        </div>
                                    </div>
                                ))}
                            </div>
                            {/* Option to replace */}
                            <div className="pt-2 border-t border-gray-100">
                                {showUploadOption ? (
                                    <div className="flex items-center gap-2">
                                        <label className="flex-1 inline-flex items-center justify-center gap-2 px-3 py-2 bg-gray-100 text-gray-700 rounded-lg cursor-pointer hover:bg-gray-200 transition-colors text-xs font-medium">
                                            <Upload size={14} />
                                            Choose File
                                            <input
                                                type="file"
                                                accept=".json"
                                                onChange={handleFileUpload}
                                                disabled={importing}
                                                className="hidden"
                                            />
                                        </label>
                                        <button
                                            onClick={() => setShowUploadOption(false)}
                                            className="px-3 py-2 text-xs text-gray-500 hover:text-gray-700"
                                        >
                                            Cancel
                                        </button>
                                    </div>
                                ) : (
                                    <button
                                        onClick={() => setShowUploadOption(true)}
                                        className="text-xs text-gray-500 hover:text-indigo-600 transition-colors"
                                    >
                                        Replace with different personas...
                                    </button>
                                )}
                            </div>
                        </div>
                    ) : (
                        <div className="text-center py-8 text-gray-500">
                            <Users className="mx-auto mb-2 text-gray-300" size={32} />
                            <p className="font-medium">No personas loaded</p>
                            <p className="text-sm">Complete Phase 1 to auto-transfer, or upload exported JSON</p>
                        </div>
                    )}
                </Card>

                {/* Load Emails */}
                <Card>
                    <div className="flex items-center justify-between mb-4">
                        <h3 className="text-lg font-semibold flex items-center gap-2">
                            <Mail className="text-blue-600" size={20} />
                            Email Stimuli
                        </h3>
                        <button
                            onClick={() => emailFileInputRef.current?.click()}
                            disabled={loadingEmails}
                            className="px-4 py-2 bg-blue-600 text-white rounded-lg text-sm hover:bg-blue-700 disabled:opacity-50 flex items-center gap-2"
                        >
                            {loadingEmails ? (
                                <RefreshCw className="animate-spin" size={16} />
                            ) : (
                                <Upload size={16} />
                            )}
                            Load Emails
                        </button>
                        <input
                            ref={emailFileInputRef}
                            type="file"
                            accept=".json,.csv"
                            onChange={handleEmailUpload}
                            className="hidden"
                        />
                    </div>

                    {emails.length > 0 ? (
                        <div className="space-y-3">
                            <div className="flex items-center justify-between text-sm text-gray-500">
                                <span>{emails.length} emails loaded</span>
                                <div className="flex gap-2">
                                    <span className="px-2 py-0.5 bg-red-100 text-red-700 rounded text-xs">
                                        {emails.filter(e => e.email_type === 'phishing').length} Phishing
                                    </span>
                                    <span className="px-2 py-0.5 bg-green-100 text-green-700 rounded text-xs">
                                        {emails.filter(e => e.email_type === 'legitimate').length} Legitimate
                                    </span>
                                </div>
                            </div>
                            <div className="max-h-48 overflow-y-auto space-y-1">
                                {emails.map((e) => (
                                    <div
                                        key={e.email_id}
                                        className="flex items-center gap-2 p-2 bg-gray-50 rounded text-sm"
                                    >
                                        <span
                                            className={`px-1.5 py-0.5 rounded text-xs font-medium shrink-0 ${
                                                e.email_type === "phishing"
                                                    ? "bg-red-100 text-red-700"
                                                    : "bg-green-100 text-green-700"
                                            }`}
                                        >
                                            {e.email_type === "phishing" ? "P" : "L"}
                                        </span>
                                        <span className="truncate flex-1 text-gray-700">
                                            {e.subject_line || e.email_id}
                                        </span>
                                        <span
                                            className={`px-1.5 py-0.5 rounded text-[10px] shrink-0 ${
                                                e.urgency_level === "high"
                                                    ? "bg-orange-100 text-orange-700"
                                                    : "bg-blue-100 text-blue-700"
                                            }`}
                                        >
                                            {e.urgency_level === "high" ? "HI" : "LO"}
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    ) : (
                        <div className="text-center py-8 text-gray-500">
                            <Mail className="mx-auto mb-2 text-gray-300" size={32} />
                            <p>No emails loaded yet</p>
                            <p className="text-sm">Load email_stimuli.csv or JSON file</p>
                        </div>
                    )}
                </Card>
            </div>

            {/* Usage Summary */}
            <Card>
                <h3 className="text-lg font-semibold flex items-center gap-2 mb-4">
                    <DollarSign className="text-green-600" size={20} />
                    Usage Summary
                </h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 bg-green-50 rounded-lg">
                        <div className="text-sm text-gray-500">Total Cost</div>
                        <div className="text-2xl font-bold text-green-600">
                            ${usage.total_cost?.toFixed(2) || '0.00'}
                        </div>
                    </div>
                    <div className="p-4 bg-blue-50 rounded-lg">
                        <div className="text-sm text-gray-500">Total Requests</div>
                        <div className="text-2xl font-bold text-blue-600">
                            {usage.total_requests?.toLocaleString() || 0}
                        </div>
                    </div>
                    <div className="p-4 bg-purple-50 rounded-lg">
                        <div className="text-sm text-gray-500">Total Tokens</div>
                        <div className="text-2xl font-bold text-purple-600">
                            {((usage.total_tokens || 0) / 1000).toFixed(1)}K
                        </div>
                    </div>
                    <div className="p-4 bg-amber-50 rounded-lg">
                        <div className="text-sm text-gray-500">Experiments Run</div>
                        <div className="text-2xl font-bold text-amber-600">
                            {completedExperiments}
                        </div>
                    </div>
                </div>
            </Card>
        </div>
    );
};

export default OverviewTab;