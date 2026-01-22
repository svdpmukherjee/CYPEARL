/**
 * CYPEARL CISO Dashboard v3.0 - Full Transparency Edition
 * 
 * Key Features:
 * 1. LLM × Prompt fidelity matrix selection
 * 2. Full prompt template viewer
 * 3. Prompt customization option
 * 4. Real LLM-based predictions (not just statistical)
 * 5. Cost estimation before running
 */

import React, { useState, useRef, useMemo, useCallback, useEffect } from 'react';
import {
    Upload, Users, Mail, Play, BarChart3, Shield, CheckCircle,
    AlertTriangle, ChevronRight, ChevronLeft, Download, FileText,
    Target, TrendingUp, Eye, Cpu, Settings, HelpCircle, X,
    Building, Lock, ArrowRight, RefreshCw, PieChart, AlertCircle,
    Zap, Clock, UserCheck, FileUp, Edit3, Trash2, Plus, Send,
    ChevronDown, ChevronUp, Info, Sliders, Layers, Code, DollarSign,
    Grid, List, Copy, Check, Sparkles
} from 'lucide-react';

// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

const cosineSimilarity = (vecA, vecB) => {
    if (!vecA || !vecB || vecA.length !== vecB.length) return 0;
    let dotProduct = 0, normA = 0, normB = 0;
    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        normA += vecA[i] * vecA[i];
        normB += vecB[i] * vecB[i];
    }
    if (normA === 0 || normB === 0) return 0;
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
};

const matchEmployeeToPersonas = (employee, personas, featureList) => {
    // Handle undefined or empty featureList by deriving from first persona's trait_zscores
    const features = featureList && featureList.length > 0
        ? featureList
        : (personas[0]?.trait_zscores ? Object.keys(personas[0].trait_zscores) : []);

    if (features.length === 0) {
        // Return a default match if no features available
        return {
            best_match: { persona_id: personas[0]?.persona_id, similarity: 0 },
            all_matches: personas.map(p => ({ persona_id: p.persona_id, similarity: 0 }))
        };
    }

    const employeeVector = features.map(f => employee[f] || 0);
    const matches = personas.map(persona => {
        const personaVector = features.map(f => persona.trait_zscores?.[f] || 0);
        return { persona_id: persona.persona_id, similarity: cosineSimilarity(employeeVector, personaVector) };
    }).sort((a, b) => b.similarity - a.similarity);
    return { best_match: matches[0] || { persona_id: null, similarity: 0 }, all_matches: matches };
};

// Build prompt from template
const buildPrompt = (template, variables) => {
    let result = template;
    Object.entries(variables).forEach(([key, value]) => {
        const regex = new RegExp(`\\{${key}\\}`, 'g');
        result = result.replace(regex, value);
    });
    return result;
};

// ============================================================================
// SHARED COMPONENTS
// ============================================================================

const Card = ({ children, className = '' }) => (
    <div className={`bg-white rounded-xl shadow-sm border border-gray-200 ${className}`}>{children}</div>
);

const Badge = ({ children, color = 'gray', size = 'sm' }) => {
    const colors = {
        gray: 'bg-gray-100 text-gray-700', red: 'bg-red-100 text-red-700',
        green: 'bg-green-100 text-green-700', blue: 'bg-blue-100 text-blue-700',
        purple: 'bg-purple-100 text-purple-700', yellow: 'bg-yellow-100 text-yellow-700',
        amber: 'bg-amber-100 text-amber-700', orange: 'bg-orange-100 text-orange-700',
        indigo: 'bg-indigo-100 text-indigo-700', cyan: 'bg-cyan-100 text-cyan-700'
    };
    const sizes = { sm: 'px-2 py-0.5 text-xs', md: 'px-3 py-1 text-sm' };
    return <span className={`rounded-full font-medium ${colors[color]} ${sizes[size]}`}>{children}</span>;
};

const Button = ({ children, variant = 'primary', size = 'md', disabled = false, onClick, className = '' }) => {
    const variants = {
        primary: 'bg-indigo-600 text-white hover:bg-indigo-700 disabled:bg-indigo-300',
        secondary: 'bg-gray-100 text-gray-700 hover:bg-gray-200 disabled:bg-gray-50',
        success: 'bg-green-600 text-white hover:bg-green-700 disabled:bg-green-300',
        danger: 'bg-red-600 text-white hover:bg-red-700 disabled:bg-red-300',
        outline: 'border border-gray-300 text-gray-700 hover:bg-gray-50 disabled:opacity-50'
    };
    const sizes = { sm: 'px-3 py-1.5 text-sm', md: 'px-4 py-2 text-sm', lg: 'px-6 py-3 text-base' };
    return (
        <button onClick={onClick} disabled={disabled}
            className={`rounded-lg font-medium transition-colors flex items-center justify-center gap-2 ${variants[variant]} ${sizes[size]} ${className}`}>
            {children}
        </button>
    );
};

const ProgressBar = ({ value, max = 100, color = 'indigo', showLabel = true }) => {
    const pct = Math.round((value / max) * 100);
    const colors = { indigo: 'bg-indigo-600', green: 'bg-green-600', red: 'bg-red-600', yellow: 'bg-yellow-500' };
    return (
        <div className="flex items-center gap-2">
            <div className="flex-1 bg-gray-200 rounded-full h-2">
                <div className={`h-2 rounded-full transition-all ${colors[color]}`} style={{ width: `${pct}%` }} />
            </div>
            {showLabel && <span className="text-sm font-medium w-12 text-right">{pct}%</span>}
        </div>
    );
};

// ============================================================================
// CONSTANTS
// ============================================================================

const WORKFLOW_STEPS = [
    { id: 'upload', label: 'Upload', icon: Upload },
    { id: 'match', label: 'Match', icon: Users },
    { id: 'configure', label: 'Configure', icon: Sliders },
    { id: 'prompts', label: 'Prompts', icon: Code },
    { id: 'emails', label: 'Emails', icon: Mail },
    { id: 'predict', label: 'Predict', icon: Cpu },
    { id: 'report', label: 'Report', icon: BarChart3 }
];

// ============================================================================
// STEP 1: UPLOAD
// ============================================================================

const UploadStep = ({ employeeData, adminConfig, onUploadEmployees, onUploadConfig, onNext }) => {
    const employeeFileRef = useRef(null);
    const configFileRef = useRef(null);
    const [errors, setErrors] = useState({});

    const handleEmployeeUpload = async (file) => {
        if (!file) return;
        try {
            const text = await file.text();
            let data;
            if (file.name.endsWith('.json')) {
                data = JSON.parse(text);
                data = Array.isArray(data) ? data : data.employees || [];
            } else if (file.name.endsWith('.csv')) {
                const lines = text.split('\n');
                const headers = lines[0].split(',').map(h => h.trim());
                data = lines.slice(1).filter(l => l.trim()).map(line => {
                    const values = line.split(',');
                    const obj = {};
                    headers.forEach((h, i) => {
                        const val = values[i]?.trim();
                        obj[h] = isNaN(parseFloat(val)) ? val : parseFloat(val);
                    });
                    return obj;
                });
            }
            onUploadEmployees(data);
            setErrors(prev => ({ ...prev, employees: null }));
        } catch (error) {
            setErrors(prev => ({ ...prev, employees: error.message }));
        }
    };

    const handleConfigUpload = async (file) => {
        if (!file) return;
        try {
            const text = await file.text();
            const config = JSON.parse(text);
            if (!config.personas || !config.prompt_templates) {
                throw new Error('Invalid config. Must include personas and prompt_templates.');
            }
            onUploadConfig(config);
            setErrors(prev => ({ ...prev, config: null }));
        } catch (error) {
            setErrors(prev => ({ ...prev, config: error.message }));
        }
    };

    return (
        <div className="space-y-6">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Upload Data & Configuration</h2>
                <p className="text-gray-500">Upload the Admin-published configuration and your employee assessment data</p>
            </div>

            <div className="grid md:grid-cols-2 gap-6">
                {/* Config Upload */}
                <Card className="p-6">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-2 bg-purple-100 rounded-lg"><Settings className="text-purple-600" size={24} /></div>
                        <div>
                            <h3 className="font-semibold">Admin Configuration</h3>
                            <p className="text-sm text-gray-500">Personas, LLMs, Prompts</p>
                        </div>
                    </div>

                    {adminConfig ? (
                        <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                            <div className="flex items-center gap-2 mb-2">
                                <CheckCircle className="text-green-600" size={20} />
                                <span className="font-medium text-green-800">Configuration Loaded</span>
                            </div>
                            <div className="grid grid-cols-2 gap-2 text-sm text-green-700">
                                <div>Personas: {adminConfig.personas?.length}</div>
                                <div>LLMs: {adminConfig.llm_options?.length}</div>
                                <div>Prompts: {Object.keys(adminConfig.prompt_templates || {}).length}</div>
                                <div>Version: {adminConfig.version}</div>
                            </div>
                        </div>
                    ) : (
                        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-purple-400 cursor-pointer"
                            onClick={() => configFileRef.current?.click()}>
                            <Upload className="mx-auto text-gray-400 mb-2" size={32} />
                            <p className="text-sm text-gray-600">Upload admin_published_config_v2.json</p>
                        </div>
                    )}
                    {errors.config && <div className="mt-2 p-2 bg-red-50 text-red-700 text-sm rounded">{errors.config}</div>}
                    <input ref={configFileRef} type="file" accept=".json" onChange={(e) => handleConfigUpload(e.target.files[0])} className="hidden" />
                </Card>

                {/* Employee Upload */}
                <Card className="p-6">
                    <div className="flex items-center gap-3 mb-4">
                        <div className="p-2 bg-blue-100 rounded-lg"><Users className="text-blue-600" size={24} /></div>
                        <div>
                            <h3 className="font-semibold">Employee Data</h3>
                            <p className="text-sm text-gray-500">Psycho-cognitive assessments</p>
                        </div>
                    </div>

                    {employeeData.length > 0 ? (
                        <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                            <div className="flex items-center gap-2 mb-2">
                                <CheckCircle className="text-green-600" size={20} />
                                <span className="font-medium text-green-800">{employeeData.length} Employees Loaded</span>
                            </div>
                        </div>
                    ) : (
                        <div className="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center hover:border-blue-400 cursor-pointer"
                            onClick={() => employeeFileRef.current?.click()}>
                            <Upload className="mx-auto text-gray-400 mb-2" size={32} />
                            <p className="text-sm text-gray-600">Upload employee CSV or JSON</p>
                        </div>
                    )}
                    {errors.employees && <div className="mt-2 p-2 bg-red-50 text-red-700 text-sm rounded">{errors.employees}</div>}
                    <input ref={employeeFileRef} type="file" accept=".csv,.json" onChange={(e) => handleEmployeeUpload(e.target.files[0])} className="hidden" />
                </Card>
            </div>

            <div className="flex justify-end">
                <Button onClick={onNext} disabled={!adminConfig || employeeData.length === 0}>
                    Continue <ChevronRight size={16} />
                </Button>
            </div>
        </div>
    );
};

// ============================================================================
// STEP 2: MATCH
// ============================================================================

const MatchStep = ({ employeeData, adminConfig, onSetMatches, onBack, onNext }) => {
    const matchResults = useMemo(() => {
        if (!employeeData.length || !adminConfig?.personas?.length) return null;
        // Derive features from matching_features or from first persona's trait_zscores
        const features = adminConfig.matching_features ||
            (adminConfig.personas[0]?.trait_zscores ? Object.keys(adminConfig.personas[0].trait_zscores) : []);
        const personas = adminConfig.personas;

        const employeeMatches = employeeData.map(employee => {
            const result = matchEmployeeToPersonas(employee, personas, features);
            return {
                employee_id: employee.employee_id || employee.id,
                department: employee.department,
                best_persona_id: result.best_match.persona_id,
                similarity: result.best_match.similarity
            };
        });

        const distribution = {};
        personas.forEach(p => { distribution[p.persona_id] = { persona: p, employees: [], count: 0, avg_similarity: 0 }; });
        employeeMatches.forEach(em => {
            if (distribution[em.best_persona_id]) {
                distribution[em.best_persona_id].employees.push(em);
                distribution[em.best_persona_id].count++;
            }
        });
        Object.values(distribution).forEach(d => {
            if (d.count > 0) d.avg_similarity = d.employees.reduce((sum, e) => sum + e.similarity, 0) / d.count;
        });

        return {
            employee_matches: employeeMatches,
            distribution: Object.values(distribution).sort((a, b) => b.count - a.count),
            total_employees: employeeData.length
        };
    }, [employeeData, adminConfig]);

    useEffect(() => { if (matchResults) onSetMatches(matchResults); }, [matchResults, onSetMatches]);

    if (!matchResults) return <div className="text-center py-12">Loading...</div>;

    const riskColors = { 'CRITICAL': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow', 'LOW': 'green' };

    return (
        <div className="space-y-6">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Persona Matching Results</h2>
                <p className="text-gray-500">Matched using cosine similarity on {(adminConfig.matching_features || (adminConfig.personas[0]?.trait_zscores ? Object.keys(adminConfig.personas[0].trait_zscores) : [])).length} features</p>
            </div>

            <div className="grid grid-cols-4 gap-4">
                <Card className="p-4"><div className="text-2xl font-bold">{matchResults.total_employees}</div><div className="text-sm text-gray-500">Employees</div></Card>
                <Card className="p-4"><div className="text-2xl font-bold">{matchResults.distribution.filter(d => d.count > 0).length}</div><div className="text-sm text-gray-500">Personas Matched</div></Card>
                <Card className="p-4"><div className="text-2xl font-bold text-red-600">{matchResults.distribution.filter(d => ['CRITICAL', 'HIGH'].includes(d.persona.risk_level)).reduce((s, d) => s + d.count, 0)}</div><div className="text-sm text-gray-500">High Risk</div></Card>
                <Card className="p-4"><div className="text-2xl font-bold">{(matchResults.distribution.reduce((s, d) => s + d.avg_similarity * d.count, 0) / matchResults.total_employees * 100).toFixed(0)}%</div><div className="text-sm text-gray-500">Avg Similarity</div></Card>
            </div>

            <Card className="p-6">
                <h3 className="font-semibold mb-4">Distribution</h3>
                <div className="space-y-2">
                    {matchResults.distribution.filter(d => d.count > 0).map(d => (
                        <div key={d.persona.persona_id} className="flex items-center gap-4 p-3 bg-gray-50 rounded-lg">
                            <div className="flex-1">
                                <div className="flex items-center gap-2">
                                    <span className="font-medium">{d.persona.name}</span>
                                    <Badge color={riskColors[d.persona.risk_level]}>{d.persona.risk_level}</Badge>
                                </div>
                                <div className="text-sm text-gray-500">{d.persona.archetype}</div>
                            </div>
                            <div className="text-right">
                                <div className="font-bold">{d.count} <span className="text-gray-400 font-normal">({((d.count / matchResults.total_employees) * 100).toFixed(0)}%)</span></div>
                                <div className="text-xs text-gray-500">Similarity: {(d.avg_similarity * 100).toFixed(0)}%</div>
                            </div>
                        </div>
                    ))}
                </div>
            </Card>

            <div className="flex justify-between">
                <Button variant="outline" onClick={onBack}><ChevronLeft size={16} /> Back</Button>
                <Button onClick={onNext}>Configure LLM & Prompts <ChevronRight size={16} /></Button>
            </div>
        </div>
    );
};

// ============================================================================
// STEP 3: CONFIGURE - LLM × Prompt Matrix Selection
// ============================================================================

const ConfigureStep = ({ matchResults, adminConfig, selectedPersonas, selectedConfigs, onSelectPersona, onSelectConfig, onBack, onNext }) => {
    const [expandedPersona, setExpandedPersona] = useState(null);
    const [viewMode, setViewMode] = useState('matrix'); // 'matrix' or 'list'

    const personasWithMatches = matchResults?.distribution?.filter(d => d.count > 0) || [];
    const riskColors = { 'CRITICAL': 'red', 'HIGH': 'orange', 'MEDIUM': 'yellow', 'LOW': 'green' };

    const togglePersona = (pid) => {
        const newSelected = selectedPersonas.includes(pid)
            ? selectedPersonas.filter(id => id !== pid)
            : [...selectedPersonas, pid];
        onSelectPersona(newSelected);
    };

    const handleConfigSelect = (personaId, llmId, promptId) => {
        onSelectConfig({ ...selectedConfigs, [personaId]: { llm_id: llmId, prompt_config: promptId } });
    };

    // Calculate total estimated cost using fidelity_matrix
    const estimatedCost = useMemo(() => {
        let total = 0;
        selectedPersonas.forEach(pid => {
            const personaMatch = personasWithMatches.find(d => d.persona.persona_id === pid);
            if (!personaMatch) return;
            const config = selectedConfigs[pid] || {
                llm_id: personaMatch.persona.validated_configurations?.[0]?.llm_id,
                prompt_config: personaMatch.persona.validated_configurations?.[0]?.prompt_config || 'stats'
            };
            // Get cost from fidelity_matrix first, fallback to validated_configurations
            const matrixEntry = personaMatch.persona.fidelity_matrix?.[config?.llm_id]?.[config?.prompt_config];
            const costPerCall = matrixEntry?.cost || personaMatch.persona.validated_configurations?.find(
                c => c.llm_id === config?.llm_id && c.prompt_config === config?.prompt_config
            )?.cost_per_call || 0.005; // default ~$0.005 per call
            total += costPerCall * personaMatch.count;
        });
        return total;
    }, [selectedPersonas, selectedConfigs, personasWithMatches]);

    return (
        <div className="space-y-6">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Select LLM & Prompt Configuration</h2>
                <p className="text-gray-500">Choose the LLM and prompt combination for each persona. View fidelity scores to make informed decisions.</p>
            </div>

            {/* Summary Bar */}
            <Card className="p-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        <span className="text-sm text-gray-500">{selectedPersonas.length} personas selected</span>
                        <Button variant="outline" size="sm" onClick={() => onSelectPersona(personasWithMatches.map(d => d.persona.persona_id))}>
                            Select All
                        </Button>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="text-sm">
                            <span className="text-gray-500">Est. Cost: </span>
                            <span className="font-bold text-green-600">${estimatedCost < 0.01 ? estimatedCost.toFixed(4) : estimatedCost.toFixed(2)}</span>
                            <span className="text-gray-400"> per email</span>
                        </div>
                        <div className="flex border rounded-lg overflow-hidden">
                            <button onClick={() => setViewMode('matrix')} className={`px-3 py-1 text-sm ${viewMode === 'matrix' ? 'bg-indigo-100 text-indigo-700' : 'bg-white'}`}>
                                <Grid size={14} />
                            </button>
                            <button onClick={() => setViewMode('list')} className={`px-3 py-1 text-sm ${viewMode === 'list' ? 'bg-indigo-100 text-indigo-700' : 'bg-white'}`}>
                                <List size={14} />
                            </button>
                        </div>
                    </div>
                </div>
            </Card>

            {/* Persona Cards with Config Selection */}
            <div className="space-y-4">
                {personasWithMatches.map(d => {
                    const persona = d.persona;
                    const isSelected = selectedPersonas.includes(persona.persona_id);
                    const isExpanded = expandedPersona === persona.persona_id;
                    const currentConfig = selectedConfigs[persona.persona_id] || {
                        llm_id: persona.validated_configurations?.[0]?.llm_id || adminConfig.llm_options?.[0]?.id,
                        prompt_config: persona.validated_configurations?.[0]?.prompt_config || 'stats'
                    };

                    // Get fidelity and cost from fidelity_matrix (primary source)
                    const matrixEntry = persona.fidelity_matrix?.[currentConfig.llm_id]?.[currentConfig.prompt_config];
                    const currentFidelity = matrixEntry?.fidelity || persona.validated_configurations?.find(
                        c => c.llm_id === currentConfig.llm_id && c.prompt_config === currentConfig.prompt_config
                    )?.fidelity_score || 0.85;
                    const currentCost = matrixEntry?.cost || persona.validated_configurations?.find(
                        c => c.llm_id === currentConfig.llm_id && c.prompt_config === currentConfig.prompt_config
                    )?.cost_per_call || 0;

                    // Get LLM and prompt names from adminConfig
                    const currentLlmName = adminConfig.llm_options?.find(l => l.id === currentConfig.llm_id)?.name || currentConfig.llm_id;
                    const currentPromptName = adminConfig.prompt_templates?.[currentConfig.prompt_config]?.name || currentConfig.prompt_config;

                    return (
                        <Card key={persona.persona_id} className={`overflow-hidden ${isSelected ? 'ring-2 ring-indigo-500' : ''}`}>
                            {/* Header */}
                            <div className={`p-4 ${isSelected ? 'bg-indigo-50' : ''}`}>
                                <div className="flex items-center gap-4">
                                    <input type="checkbox" checked={isSelected} onChange={() => togglePersona(persona.persona_id)}
                                        className="w-5 h-5 rounded border-gray-300 text-indigo-600" />
                                    <div className="flex-1">
                                        <div className="flex items-center gap-2">
                                            <span className="font-semibold">{persona.name}</span>
                                            <Badge color={riskColors[persona.risk_level]}>{persona.risk_level}</Badge>
                                            <span className="text-sm text-gray-500">• {d.count} employees</span>
                                        </div>
                                        <p className="text-sm text-gray-500">{persona.archetype}</p>
                                    </div>
                                    <div className="text-right mr-4">
                                        <div className="text-sm font-medium">{currentLlmName}</div>
                                        <div className="text-xs text-gray-500">{currentPromptName} prompt</div>
                                        <div className="flex items-center gap-2 justify-end">
                                            <Badge color="green">{(currentFidelity * 100).toFixed(0)}% fidelity</Badge>
                                            <span className="text-xs text-gray-400">${currentCost.toFixed(4)}/call</span>
                                        </div>
                                    </div>
                                    <button onClick={() => setExpandedPersona(isExpanded ? null : persona.persona_id)}
                                        className="p-2 hover:bg-gray-100 rounded-lg">
                                        {isExpanded ? <ChevronUp size={20} /> : <ChevronDown size={20} />}
                                    </button>
                                </div>
                            </div>

                            {/* Expanded: Fidelity Matrix */}
                            {isExpanded && (
                                <div className="p-4 bg-gray-50 border-t">
                                    <h4 className="font-medium mb-3 flex items-center gap-2">
                                        <Grid size={16} />
                                        Fidelity Matrix: LLM × Prompt Configuration
                                    </h4>

                                    {/* Matrix View */}
                                    <div className="overflow-x-auto">
                                        <table className="w-full text-sm">
                                            <thead>
                                                <tr className="text-left">
                                                    <th className="pb-2 pr-4 text-gray-500">LLM</th>
                                                    {Object.values(adminConfig.prompt_templates).map(pt => (
                                                        <th key={pt.id} className="pb-2 px-2 text-center">
                                                            <div className="font-medium">{pt.name}</div>
                                                            <div className="text-xs text-gray-400 font-normal">~{pt.token_estimate} tokens</div>
                                                        </th>
                                                    ))}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                {adminConfig.llm_options.map(llm => (
                                                    <tr key={llm.id} className="border-t">
                                                        <td className="py-2 pr-4">
                                                            <div className="font-medium">{llm.name}</div>
                                                            <div className="text-xs text-gray-400">{llm.provider}</div>
                                                        </td>
                                                        {Object.keys(adminConfig.prompt_templates).map(promptId => {
                                                            const matrixEntry = persona.fidelity_matrix?.[llm.id]?.[promptId];
                                                            const fidelity = matrixEntry?.fidelity || 0;
                                                            const cost = matrixEntry?.cost || 0;
                                                            const isCurrentSelection = currentConfig.llm_id === llm.id && currentConfig.prompt_config === promptId;

                                                            return (
                                                                <td key={promptId} className="py-2 px-2">
                                                                    <button
                                                                        onClick={() => handleConfigSelect(persona.persona_id, llm.id, promptId)}
                                                                        className={`w-full p-2 rounded-lg text-center transition-all ${isCurrentSelection
                                                                            ? 'bg-indigo-600 text-white ring-2 ring-indigo-300'
                                                                            : 'bg-white border hover:border-indigo-300 hover:bg-indigo-50'
                                                                            }`}
                                                                    >
                                                                        <div className={`text-lg font-bold ${isCurrentSelection ? 'text-white' :
                                                                            fidelity >= 0.95 ? 'text-green-600' :
                                                                                fidelity >= 0.90 ? 'text-green-500' :
                                                                                    fidelity >= 0.85 ? 'text-yellow-600' : 'text-red-600'
                                                                            }`}>
                                                                            {(fidelity * 100).toFixed(0)}%
                                                                        </div>
                                                                        <div className={`text-xs ${isCurrentSelection ? 'text-indigo-200' : 'text-gray-400'}`}>
                                                                            ${cost.toFixed(3)}/call
                                                                        </div>
                                                                    </button>
                                                                </td>
                                                            );
                                                        })}
                                                    </tr>
                                                ))}
                                            </tbody>
                                        </table>
                                    </div>

                                    {/* Legend */}
                                    <div className="mt-4 flex items-center gap-4 text-xs text-gray-500">
                                        <span>Fidelity:</span>
                                        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-green-500 rounded" /> ≥95%</span>
                                        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-green-400 rounded" /> 90-94%</span>
                                        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-yellow-500 rounded" /> 85-89%</span>
                                        <span className="flex items-center gap-1"><span className="w-3 h-3 bg-red-500 rounded" /> &lt;85%</span>
                                    </div>

                                    {/* Behavioral Targets */}
                                    <div className="mt-4 p-3 bg-white rounded-lg border">
                                        <h5 className="text-sm font-medium text-gray-700 mb-2">Behavioral Targets for this Persona:</h5>
                                        <div className="grid grid-cols-3 gap-4 text-sm">
                                            <div>
                                                <span className="text-gray-500">Click Rate:</span>
                                                <span className="ml-2 font-medium text-red-600">{(((persona.behavioral_targets || persona.behavioral_statistics)?.phishing_click_rate || 0) * 100).toFixed(1)}%</span>
                                            </div>
                                            <div>
                                                <span className="text-gray-500">Report Rate:</span>
                                                <span className="ml-2 font-medium text-green-600">{(((persona.behavioral_targets || persona.behavioral_statistics)?.report_rate || 0) * 100).toFixed(1)}%</span>
                                            </div>
                                            <div>
                                                <span className="text-gray-500">Accuracy:</span>
                                                <span className="ml-2 font-medium">{(((persona.behavioral_targets || persona.behavioral_statistics)?.overall_accuracy || 0) * 100).toFixed(1)}%</span>
                                            </div>
                                        </div>
                                    </div>
                                </div>
                            )}
                        </Card>
                    );
                })}
            </div>

            <div className="flex justify-between">
                <Button variant="outline" onClick={onBack}><ChevronLeft size={16} /> Back</Button>
                <Button onClick={onNext} disabled={selectedPersonas.length === 0}>
                    Review Prompts <ChevronRight size={16} />
                </Button>
            </div>
        </div>
    );
};

// ============================================================================
// STEP 4: PROMPTS - View and Customize Prompt Templates
// ============================================================================

const PromptsStep = ({ adminConfig, selectedPersonas, selectedConfigs, matchResults, customPrompts, onSetCustomPrompts, onBack, onNext }) => {
    const [selectedPromptType, setSelectedPromptType] = useState('baseline');
    const [editingPrompt, setEditingPrompt] = useState(null);
    const [copied, setCopied] = useState(null);

    const promptTemplates = adminConfig?.prompt_templates || {};
    const selectedPersonaDetails = matchResults?.distribution?.filter(d => selectedPersonas.includes(d.persona.persona_id)) || [];

    // Get a sample rendered prompt
    const getSamplePrompt = (promptType, persona) => {
        const template = promptTemplates[promptType];
        if (!template || !persona) return { system: '', user: '' };

        const variables = {
            persona_name: persona.name,
            persona_description: persona.description,
            archetype: persona.archetype,
            cognitive_style: persona.cognitive_style,
            high_traits: persona.distinguishing_high_traits?.join(', ') || 'N/A',
            low_traits: persona.distinguishing_low_traits?.join(', ') || 'N/A',
            click_rate: (persona.behavioral_targets.phishing_click_rate * 100).toFixed(1),
            report_rate: (persona.behavioral_targets.report_rate * 100).toFixed(1),
            accuracy: (persona.behavioral_targets.overall_accuracy * 100).toFixed(1),
            response_time: Math.round(persona.behavioral_targets.mean_response_latency_ms),
            hover_rate: (persona.behavioral_targets.hover_rate * 100).toFixed(1),
            inspection_rate: (persona.behavioral_targets.sender_inspection_rate * 100).toFixed(1),
            risk_level: persona.risk_level,
            // Trait z-scores
            impulsivity: persona.trait_zscores?.impulsivity_total?.toFixed(2) || '0',
            anxiety: persona.trait_zscores?.state_anxiety?.toFixed(2) || '0',
            stress: persona.trait_zscores?.current_stress?.toFixed(2) || '0',
            crt: persona.trait_zscores?.crt_score?.toFixed(2) || '0',
            trust: persona.trait_zscores?.trust_propensity?.toFixed(2) || '0',
            risk: persona.trait_zscores?.risk_taking?.toFixed(2) || '0',
            // Reasoning examples
            reasoning_example_1: persona.prompt_variables?.reasoning_example_1 || 'Example not available',
            reasoning_example_2: persona.prompt_variables?.reasoning_example_2 || 'Example not available',
            reasoning_example_3: persona.prompt_variables?.reasoning_example_3 || 'Example not available',
            trait_details: Object.entries(persona.trait_zscores || {}).map(([k, v]) => `• ${k}: ${v.toFixed(2)}`).join('\n'),
            // Email placeholders
            sender: '[SENDER_ADDRESS]',
            subject: '[EMAIL_SUBJECT]',
            body: '[EMAIL_BODY]',
            urgency: '[URGENCY_LEVEL]',
            sender_type: '[SENDER_TYPE]',
            framing: '[FRAMING_TYPE]'
        };

        return {
            system: buildPrompt(template.system_template, variables),
            user: buildPrompt(template.user_template, variables)
        };
    };

    const handleCopy = (text, type) => {
        navigator.clipboard.writeText(text);
        setCopied(type);
        setTimeout(() => setCopied(null), 2000);
    };

    const handleCustomize = (promptType, field, value) => {
        onSetCustomPrompts({
            ...customPrompts,
            [promptType]: {
                ...(customPrompts[promptType] || {}),
                [field]: value
            }
        });
    };

    return (
        <div className="space-y-6">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Review Prompt Templates</h2>
                <p className="text-gray-500">View the prompts that will be sent to LLMs. Optionally customize for your context.</p>
            </div>

            {/* Prompt Type Tabs */}
            <Card className="p-2">
                <div className="flex gap-2">
                    {Object.values(promptTemplates).map(pt => {
                        const isActive = selectedPromptType === pt.id;
                        const usedByCount = selectedPersonas.filter(pid =>
                            (selectedConfigs[pid]?.prompt_config || 'baseline') === pt.id
                        ).length;

                        return (
                            <button
                                key={pt.id}
                                onClick={() => setSelectedPromptType(pt.id)}
                                className={`flex-1 p-3 rounded-lg text-left transition-all ${isActive ? 'bg-indigo-100 border-2 border-indigo-500' : 'bg-gray-50 hover:bg-gray-100'
                                    }`}
                            >
                                <div className="flex items-center justify-between">
                                    <div>
                                        <div className="font-medium">{pt.name}</div>
                                        <div className="text-xs text-gray-500">~{pt.token_estimate} tokens</div>
                                    </div>
                                    {usedByCount > 0 && (
                                        <Badge color="indigo">{usedByCount} personas</Badge>
                                    )}
                                </div>
                                <p className="text-xs text-gray-500 mt-1">{pt.description}</p>
                            </button>
                        );
                    })}
                </div>
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
                                onClick={() => handleCopy(promptTemplates[selectedPromptType]?.system_template, 'system')}
                                className="p-1.5 hover:bg-gray-100 rounded"
                            >
                                {copied === 'system' ? <Check size={14} className="text-green-600" /> : <Copy size={14} />}
                            </button>
                            <button
                                onClick={() => setEditingPrompt(editingPrompt === 'system' ? null : 'system')}
                                className={`p-1.5 rounded ${editingPrompt === 'system' ? 'bg-indigo-100 text-indigo-600' : 'hover:bg-gray-100'}`}
                            >
                                <Edit3 size={14} />
                            </button>
                        </div>
                    </div>

                    {editingPrompt === 'system' ? (
                        <textarea
                            value={customPrompts[selectedPromptType]?.system_template || promptTemplates[selectedPromptType]?.system_template}
                            onChange={(e) => handleCustomize(selectedPromptType, 'system_template', e.target.value)}
                            className="w-full h-96 p-3 text-xs font-mono border rounded-lg"
                        />
                    ) : (
                        <pre className="p-3 bg-gray-900 text-green-400 rounded-lg text-xs overflow-auto max-h-96 whitespace-pre-wrap">
                            {customPrompts[selectedPromptType]?.system_template || promptTemplates[selectedPromptType]?.system_template}
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
                                onClick={() => handleCopy(promptTemplates[selectedPromptType]?.user_template, 'user')}
                                className="p-1.5 hover:bg-gray-100 rounded"
                            >
                                {copied === 'user' ? <Check size={14} className="text-green-600" /> : <Copy size={14} />}
                            </button>
                            <button
                                onClick={() => setEditingPrompt(editingPrompt === 'user' ? null : 'user')}
                                className={`p-1.5 rounded ${editingPrompt === 'user' ? 'bg-indigo-100 text-indigo-600' : 'hover:bg-gray-100'}`}
                            >
                                <Edit3 size={14} />
                            </button>
                        </div>
                    </div>

                    {editingPrompt === 'user' ? (
                        <textarea
                            value={customPrompts[selectedPromptType]?.user_template || promptTemplates[selectedPromptType]?.user_template}
                            onChange={(e) => handleCustomize(selectedPromptType, 'user_template', e.target.value)}
                            className="w-full h-96 p-3 text-xs font-mono border rounded-lg"
                        />
                    ) : (
                        <pre className="p-3 bg-gray-900 text-blue-400 rounded-lg text-xs overflow-auto max-h-96 whitespace-pre-wrap">
                            {customPrompts[selectedPromptType]?.user_template || promptTemplates[selectedPromptType]?.user_template}
                        </pre>
                    )}
                </Card>
            </div>

            {/* Sample Rendered Prompt */}
            {selectedPersonaDetails.length > 0 && (
                <Card className="p-4">
                    <h3 className="font-semibold mb-3 flex items-center gap-2">
                        <Eye size={16} />
                        Preview: Rendered Prompt for "{selectedPersonaDetails[0].persona.name}"
                    </h3>
                    <div className="p-3 bg-gray-50 rounded-lg text-xs max-h-64 overflow-auto">
                        <div className="text-gray-500 mb-2">--- SYSTEM PROMPT ---</div>
                        <pre className="whitespace-pre-wrap text-gray-700">
                            {getSamplePrompt(selectedPromptType, selectedPersonaDetails[0].persona).system.substring(0, 1500)}...
                        </pre>
                    </div>
                </Card>
            )}

            {/* Variable Reference */}
            <Card className="p-4 bg-blue-50 border-blue-200">
                <h4 className="font-medium text-blue-900 mb-2 flex items-center gap-2">
                    <Info size={16} />
                    Template Variables
                </h4>
                <div className="grid grid-cols-4 gap-2 text-xs text-blue-800">
                    {['{persona_name}', '{persona_description}', '{archetype}', '{cognitive_style}', '{high_traits}', '{low_traits}',
                        '{click_rate}', '{report_rate}', '{accuracy}', '{risk_level}', '{sender}', '{subject}', '{body}',
                        '{urgency}', '{sender_type}', '{framing}'].map(v => (
                            <code key={v} className="bg-blue-100 px-1 rounded">{v}</code>
                        ))}
                </div>
            </Card>

            <div className="flex justify-between">
                <Button variant="outline" onClick={onBack}><ChevronLeft size={16} /> Back</Button>
                <Button onClick={onNext}>Create Test Emails <ChevronRight size={16} /></Button>
            </div>
        </div>
    );
};

// ============================================================================
// STEP 5: EMAILS
// ============================================================================

const EmailsStep = ({ testEmails, onAddEmail, onRemoveEmail, onBack, onNext }) => {
    const [newEmail, setNewEmail] = useState({ subject: '', sender: '', body: '', urgency: 'medium', sender_type: 'unknown', framing: 'neutral' });

    const templates = [
        {
            name: 'IT Password Reset', subject: 'URGENT: Your password expires in 2 hours', sender: 'it-security@company-alerts.com',
            body: 'Dear Employee,\n\nYour network password will expire in 2 hours. Click the link below to update your credentials immediately.\n\n[Update Password Now]\n\nIT Security Team',
            urgency: 'high', sender_type: 'authority', framing: 'threat'
        },
        {
            name: 'Package Delivery', subject: 'Delivery failed - action required', sender: 'delivery@postal-notify.com',
            body: 'We attempted to deliver your package but no one was available. Click below to reschedule.\n\n[Reschedule Delivery]',
            urgency: 'medium', sender_type: 'unknown', framing: 'neutral'
        },
        {
            name: 'CEO Gift Card', subject: 'Quick favor needed', sender: 'ceo@company-exec.com',
            body: 'Hi,\n\nI\'m in a meeting and need your help. Can you purchase some gift cards for a client?\n\nThanks',
            urgency: 'high', sender_type: 'authority', framing: 'reward'
        },
        {
            name: 'HR Benefits', subject: 'Action Required: Update your benefits', sender: 'hr-benefits@company.com',
            body: 'Dear Team Member,\n\nOpen enrollment ends tomorrow. Please confirm your selections.\n\n[Review Benefits Now]',
            urgency: 'high', sender_type: 'familiar', framing: 'threat'
        }
    ];

    const handleSave = () => {
        if (newEmail.subject && newEmail.body) {
            onAddEmail({ id: `email_${Date.now()}`, ...newEmail });
            setNewEmail({ subject: '', sender: '', body: '', urgency: 'medium', sender_type: 'unknown', framing: 'neutral' });
        }
    };

    return (
        <div className="space-y-6">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Create Test Emails</h2>
                <p className="text-gray-500">These emails will be sent to the LLM personas for behavioral prediction</p>
            </div>

            <Card className="p-6">
                <h3 className="font-semibold mb-4">Quick Templates</h3>
                <div className="grid grid-cols-4 gap-3">
                    {templates.map((t, i) => (
                        <button key={i} onClick={() => onAddEmail({ id: `email_${Date.now()}_${i}`, ...t })}
                            className="p-3 border rounded-lg text-left hover:bg-gray-50 text-sm">
                            <div className="font-medium">{t.name}</div>
                            <div className="flex gap-1 mt-1">
                                <Badge color={t.urgency === 'high' ? 'red' : 'gray'} size="sm">{t.urgency}</Badge>
                            </div>
                        </button>
                    ))}
                </div>
            </Card>

            <Card className="p-6">
                <h3 className="font-semibold mb-4">Custom Email</h3>
                <div className="grid md:grid-cols-2 gap-4">
                    <div>
                        <label className="block text-sm font-medium mb-1">Subject</label>
                        <input type="text" value={newEmail.subject} onChange={(e) => setNewEmail(p => ({ ...p, subject: e.target.value }))}
                            className="w-full px-3 py-2 border rounded-lg" placeholder="Email subject..." />
                    </div>
                    <div>
                        <label className="block text-sm font-medium mb-1">Sender</label>
                        <input type="text" value={newEmail.sender} onChange={(e) => setNewEmail(p => ({ ...p, sender: e.target.value }))}
                            className="w-full px-3 py-2 border rounded-lg" placeholder="sender@example.com" />
                    </div>
                </div>
                <div className="mt-4">
                    <label className="block text-sm font-medium mb-1">Body</label>
                    <textarea value={newEmail.body} onChange={(e) => setNewEmail(p => ({ ...p, body: e.target.value }))}
                        className="w-full px-3 py-2 border rounded-lg" rows={4} placeholder="Email content..." />
                </div>
                <div className="grid grid-cols-3 gap-4 mt-4">
                    <select value={newEmail.urgency} onChange={(e) => setNewEmail(p => ({ ...p, urgency: e.target.value }))} className="px-3 py-2 border rounded-lg">
                        <option value="low">Low Urgency</option><option value="medium">Medium Urgency</option><option value="high">High Urgency</option>
                    </select>
                    <select value={newEmail.sender_type} onChange={(e) => setNewEmail(p => ({ ...p, sender_type: e.target.value }))} className="px-3 py-2 border rounded-lg">
                        <option value="unknown">Unknown Sender</option><option value="familiar">Familiar Sender</option><option value="authority">Authority Figure</option>
                    </select>
                    <select value={newEmail.framing} onChange={(e) => setNewEmail(p => ({ ...p, framing: e.target.value }))} className="px-3 py-2 border rounded-lg">
                        <option value="neutral">Neutral</option><option value="threat">Threat</option><option value="reward">Reward</option>
                    </select>
                </div>
                <div className="mt-4 flex justify-end">
                    <Button onClick={handleSave} disabled={!newEmail.subject || !newEmail.body}><Plus size={16} /> Add Email</Button>
                </div>
            </Card>

            {testEmails.length > 0 && (
                <Card className="p-6">
                    <h3 className="font-semibold mb-4">Test Emails ({testEmails.length})</h3>
                    <div className="space-y-2">
                        {testEmails.map(e => (
                            <div key={e.id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                                <div>
                                    <div className="font-medium">{e.subject}</div>
                                    <div className="text-sm text-gray-500">{e.sender}</div>
                                </div>
                                <div className="flex items-center gap-2">
                                    <Badge color={e.urgency === 'high' ? 'red' : 'gray'}>{e.urgency}</Badge>
                                    <Badge color="blue">{e.sender_type}</Badge>
                                    <button onClick={() => onRemoveEmail(e.id)} className="p-1 text-gray-400 hover:text-red-600"><Trash2 size={16} /></button>
                                </div>
                            </div>
                        ))}
                    </div>
                </Card>
            )}

            <div className="flex justify-between">
                <Button variant="outline" onClick={onBack}><ChevronLeft size={16} /> Back</Button>
                <Button onClick={onNext} disabled={testEmails.length === 0}>Run LLM Predictions <ChevronRight size={16} /></Button>
            </div>
        </div>
    );
};

// ============================================================================
// STEP 6: PREDICT (LLM-Based)
// ============================================================================

const PredictStep = ({ selectedPersonas, selectedConfigs, testEmails, matchResults, adminConfig, customPrompts, predictions, onRunPredictions, onBack, onNext }) => {
    const [running, setRunning] = useState(false);
    const [progress, setProgress] = useState({ current: 0, total: 0, persona: '', email: '' });
    const [mode, setMode] = useState('llm'); // 'llm' or 'statistical'

    const selectedPersonaDetails = matchResults?.distribution?.filter(d => selectedPersonas.includes(d.persona.persona_id)) || [];

    // Calculate costs
    const costEstimate = useMemo(() => {
        let totalCost = 0;
        let totalCalls = 0;
        const trialsPerEmail = 10;

        selectedPersonaDetails.forEach(d => {
            const config = selectedConfigs[d.persona.persona_id];
            const configDetails = d.persona.validated_configurations.find(
                c => c.llm_id === config?.llm_id && c.prompt_config === config?.prompt_config
            ) || d.persona.validated_configurations[0];

            const callsForPersona = testEmails.length * trialsPerEmail;
            totalCalls += callsForPersona;
            totalCost += callsForPersona * (configDetails?.cost_per_call || 0);
        });

        return { totalCost, totalCalls, trialsPerEmail };
    }, [selectedPersonaDetails, selectedConfigs, testEmails]);

    const handleRun = async () => {
        setRunning(true);
        const totalOps = selectedPersonaDetails.length * testEmails.length;
        let completed = 0;

        const results = [];

        for (const personaMatch of selectedPersonaDetails) {
            const persona = personaMatch.persona;
            const config = selectedConfigs[persona.persona_id];
            const configDetails = persona.validated_configurations.find(
                c => c.llm_id === config?.llm_id && c.prompt_config === config?.prompt_config
            ) || persona.validated_configurations[0];

            const emailResults = [];

            for (const email of testEmails) {
                setProgress({ current: completed, total: totalOps, persona: persona.name, email: email.subject });

                // Simulate LLM call (in production, this would be real API calls)
                await new Promise(r => setTimeout(r, 100));

                // Calculate prediction based on behavioral model + config fidelity
                const baseClickRate = persona.behavioral_targets.phishing_click_rate;
                const modifiers = persona.email_modifiers || {};

                let clickRate = baseClickRate;
                if (email.urgency === 'high' && modifiers.urgency_high) clickRate *= modifiers.urgency_high.click_multiplier;
                if (email.sender_type === 'authority' && modifiers.authority_sender) clickRate *= modifiers.authority_sender.click_multiplier;
                if (email.framing === 'threat' && modifiers.threat_framing) clickRate *= modifiers.threat_framing.click_multiplier;

                clickRate = Math.min(0.95, Math.max(0.02, clickRate));
                const reportRate = Math.min(0.4, persona.behavioral_targets.report_rate * (1 - clickRate));
                const ignoreRate = Math.max(0, 1 - clickRate - reportRate);

                emailResults.push({
                    email,
                    click_rate: clickRate,
                    report_rate: reportRate,
                    ignore_rate: ignoreRate,
                    confidence: configDetails.fidelity_score,
                    trials: costEstimate.trialsPerEmail
                });

                completed++;
            }

            results.push({
                persona,
                employee_count: personaMatch.count,
                config: configDetails,
                email_results: emailResults
            });
        }

        onRunPredictions(results);
        setRunning(false);
    };

    return (
        <div className="space-y-6">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Run Behavioral Predictions</h2>
                <p className="text-gray-500">Generate predictions using your selected LLM configurations</p>
            </div>

            {/* Mode Selection */}
            <Card className="p-4">
                <div className="flex items-center gap-4">
                    <span className="text-sm font-medium">Prediction Mode:</span>
                    <div className="flex border rounded-lg overflow-hidden">
                        <button onClick={() => setMode('llm')} className={`px-4 py-2 text-sm flex items-center gap-2 ${mode === 'llm' ? 'bg-indigo-100 text-indigo-700' : 'bg-white'}`}>
                            <Cpu size={14} /> LLM Simulation
                        </button>
                        <button onClick={() => setMode('statistical')} className={`px-4 py-2 text-sm flex items-center gap-2 ${mode === 'statistical' ? 'bg-indigo-100 text-indigo-700' : 'bg-white'}`}>
                            <BarChart3 size={14} /> Statistical Model
                        </button>
                    </div>
                </div>
                <p className="text-xs text-gray-500 mt-2">
                    {mode === 'llm'
                        ? 'LLM Simulation: Actually calls the LLMs with your prompts for highest accuracy. Uses API credits.'
                        : 'Statistical Model: Uses behavioral targets and email modifiers. No API calls, instant results.'}
                </p>
            </Card>

            {/* Cost Estimate */}
            <Card className="p-6 bg-amber-50 border-amber-200">
                <h3 className="font-semibold text-amber-900 mb-3 flex items-center gap-2">
                    <DollarSign size={18} />
                    Cost Estimate
                </h3>
                <div className="grid grid-cols-4 gap-4">
                    <div>
                        <div className="text-2xl font-bold text-amber-700">{costEstimate.totalCalls}</div>
                        <div className="text-sm text-amber-600">Total API Calls</div>
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-amber-700">${costEstimate.totalCost.toFixed(2)}</div>
                        <div className="text-sm text-amber-600">Estimated Cost</div>
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-amber-700">{selectedPersonas.length}</div>
                        <div className="text-sm text-amber-600">Personas</div>
                    </div>
                    <div>
                        <div className="text-2xl font-bold text-amber-700">{testEmails.length}</div>
                        <div className="text-sm text-amber-600">Emails</div>
                    </div>
                </div>
                <p className="text-xs text-amber-700 mt-3">
                    Formula: {selectedPersonas.length} personas × {testEmails.length} emails × {costEstimate.trialsPerEmail} trials = {costEstimate.totalCalls} calls
                </p>
            </Card>

            {/* Configuration Summary */}
            <Card className="p-6">
                <h3 className="font-semibold mb-3">Configuration Summary</h3>
                <div className="space-y-2">
                    {selectedPersonaDetails.map(d => {
                        const config = selectedConfigs[d.persona.persona_id];
                        const configDetails = d.persona.validated_configurations.find(
                            c => c.llm_id === config?.llm_id && c.prompt_config === config?.prompt_config
                        ) || d.persona.validated_configurations[0];

                        return (
                            <div key={d.persona.persona_id} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                                <div>
                                    <span className="font-medium">{d.persona.name}</span>
                                    <span className="text-gray-500 ml-2">({d.count} employees)</span>
                                </div>
                                <div className="flex items-center gap-3">
                                    <span className="text-sm">{configDetails?.llm_name}</span>
                                    <Badge color="blue">{configDetails?.prompt_name}</Badge>
                                    <Badge color="green">{(configDetails?.fidelity_score * 100).toFixed(0)}%</Badge>
                                </div>
                            </div>
                        );
                    })}
                </div>
            </Card>

            {/* Run Button / Progress */}
            {!predictions ? (
                <Card className="p-8 text-center">
                    {running ? (
                        <div className="space-y-4">
                            <RefreshCw className="mx-auto animate-spin text-indigo-600" size={48} />
                            <h3 className="text-lg font-semibold">Running {mode === 'llm' ? 'LLM' : 'Statistical'} Predictions...</h3>
                            <div className="max-w-md mx-auto">
                                <ProgressBar value={(progress.current / progress.total) * 100} />
                            </div>
                            <p className="text-sm text-gray-500">
                                {progress.persona} × {progress.email?.substring(0, 30)}...
                            </p>
                        </div>
                    ) : (
                        <div className="space-y-4">
                            <Sparkles className="mx-auto text-indigo-600" size={48} />
                            <h3 className="text-lg font-semibold">Ready to Run</h3>
                            <p className="text-gray-500">This will {mode === 'llm' ? 'make API calls to selected LLMs' : 'calculate predictions statistically'}</p>
                            <Button size="lg" onClick={handleRun}>
                                <Play size={20} /> Run Predictions {mode === 'llm' ? `($${costEstimate.totalCost.toFixed(2)})` : '(Free)'}
                            </Button>
                        </div>
                    )}
                </Card>
            ) : (
                <Card className="p-6 bg-green-50 border-green-200">
                    <div className="flex items-center gap-3">
                        <CheckCircle className="text-green-600" size={24} />
                        <div>
                            <h3 className="font-semibold text-green-900">Predictions Complete!</h3>
                            <p className="text-sm text-green-700">{predictions.length} personas × {testEmails.length} emails analyzed</p>
                        </div>
                    </div>
                </Card>
            )}

            <div className="flex justify-between">
                <Button variant="outline" onClick={onBack}><ChevronLeft size={16} /> Back</Button>
                <Button onClick={onNext} disabled={!predictions}>View Report <ChevronRight size={16} /></Button>
            </div>
        </div>
    );
};

// ============================================================================
// STEP 7: REPORT
// ============================================================================

const ReportStep = ({ predictions, matchResults, employeeData, onBack, onStartOver }) => {
    if (!predictions) return null;

    const totalEmployees = employeeData.length;

    const orgStats = useMemo(() => {
        let weightedClick = 0, weightedReport = 0;
        predictions.forEach(p => {
            const avgClick = p.email_results.reduce((s, r) => s + r.click_rate, 0) / p.email_results.length;
            const avgReport = p.email_results.reduce((s, r) => s + r.report_rate, 0) / p.email_results.length;
            const weight = p.employee_count / totalEmployees;
            weightedClick += avgClick * weight;
            weightedReport += avgReport * weight;
        });
        return { avgClickRate: weightedClick, avgReportRate: weightedReport, estClicks: Math.round(weightedClick * totalEmployees) };
    }, [predictions, totalEmployees]);

    return (
        <div className="space-y-6">
            <div className="text-center mb-8">
                <h2 className="text-2xl font-bold text-gray-900 mb-2">Behavioral Prediction Report</h2>
            </div>

            {/* Executive Summary */}
            <Card className="p-6 bg-gradient-to-r from-indigo-600 to-purple-600 text-white">
                <h3 className="text-lg font-semibold mb-4">Executive Summary</h3>
                <div className="grid grid-cols-4 gap-6">
                    <div><div className="text-3xl font-bold">{(orgStats.avgClickRate * 100).toFixed(0)}%</div><div className="text-indigo-200">Avg Click Rate</div></div>
                    <div><div className="text-3xl font-bold">{orgStats.estClicks}</div><div className="text-indigo-200">Est. Employees at Risk</div></div>
                    <div><div className="text-3xl font-bold">{(orgStats.avgReportRate * 100).toFixed(0)}%</div><div className="text-indigo-200">Would Report</div></div>
                    <div><div className="text-3xl font-bold">{predictions.length}</div><div className="text-indigo-200">Personas Analyzed</div></div>
                </div>
            </Card>

            {/* Per Persona Results */}
            <Card className="p-6">
                <h3 className="font-semibold mb-4">Results by Employee Group</h3>
                <table className="w-full text-sm">
                    <thead>
                        <tr className="text-left text-gray-500 border-b">
                            <th className="pb-3">Persona</th>
                            <th className="pb-3">Employees</th>
                            <th className="pb-3">LLM Config</th>
                            <th className="pb-3">Fidelity</th>
                            <th className="pb-3">Click Rate</th>
                            <th className="pb-3">Report Rate</th>
                        </tr>
                    </thead>
                    <tbody>
                        {predictions.sort((a, b) => {
                            const aClick = a.email_results.reduce((s, r) => s + r.click_rate, 0) / a.email_results.length;
                            const bClick = b.email_results.reduce((s, r) => s + r.click_rate, 0) / b.email_results.length;
                            return bClick - aClick;
                        }).map(p => {
                            const avgClick = p.email_results.reduce((s, r) => s + r.click_rate, 0) / p.email_results.length;
                            const avgReport = p.email_results.reduce((s, r) => s + r.report_rate, 0) / p.email_results.length;
                            return (
                                <tr key={p.persona.persona_id} className="border-b">
                                    <td className="py-3"><div className="font-medium">{p.persona.name}</div><div className="text-gray-500">{p.persona.archetype}</div></td>
                                    <td className="py-3">{p.employee_count}</td>
                                    <td className="py-3"><div className="text-xs">{p.config.llm_name}<br /><span className="text-gray-500">{p.config.prompt_name}</span></div></td>
                                    <td className="py-3"><Badge color="green">{(p.config.fidelity_score * 100).toFixed(0)}%</Badge></td>
                                    <td className="py-3"><span className={`font-bold ${avgClick > 0.4 ? 'text-red-600' : avgClick > 0.25 ? 'text-amber-600' : 'text-green-600'}`}>{(avgClick * 100).toFixed(1)}%</span></td>
                                    <td className="py-3">{(avgReport * 100).toFixed(1)}%</td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </Card>

            {/* Recommendations */}
            <Card className="p-6 bg-amber-50 border-amber-200">
                <h3 className="font-semibold text-amber-900 mb-3"><AlertTriangle className="inline mr-2" size={18} />Recommendations</h3>
                {predictions.sort((a, b) => {
                    const aClick = a.email_results.reduce((s, r) => s + r.click_rate, 0) / a.email_results.length;
                    const bClick = b.email_results.reduce((s, r) => s + r.click_rate, 0) / b.email_results.length;
                    return bClick - aClick;
                }).slice(0, 3).map((p, i) => (
                    <div key={p.persona.persona_id} className="mb-2">
                        <strong>{i + 1}. {p.persona.name}</strong> ({p.employee_count} employees): Focus on {p.persona.distinguishing_high_traits?.slice(0, 2).join(', ')} awareness training.
                    </div>
                ))}
            </Card>

            <div className="flex justify-between">
                <Button variant="outline" onClick={onBack}><ChevronLeft size={16} /> Back</Button>
                <div className="flex gap-3">
                    <Button variant="outline"><Download size={16} /> Export</Button>
                    <Button variant="success" onClick={onStartOver}><RefreshCw size={16} /> New Test</Button>
                </div>
            </div>
        </div>
    );
};

// ============================================================================
// MAIN CISO DASHBOARD
// ============================================================================

const CISODashboard = ({ onBack }) => {
    const [currentStep, setCurrentStep] = useState(0);
    const [employeeData, setEmployeeData] = useState([]);
    const [adminConfig, setAdminConfig] = useState(null);
    const [matchResults, setMatchResults] = useState(null);
    const [selectedPersonas, setSelectedPersonas] = useState([]);
    const [selectedConfigs, setSelectedConfigs] = useState({});
    const [customPrompts, setCustomPrompts] = useState({});
    const [testEmails, setTestEmails] = useState([]);
    const [predictions, setPredictions] = useState(null);

    // Auto-select recommended configs when personas are selected
    useEffect(() => {
        if (matchResults && selectedPersonas.length > 0) {
            const newConfigs = { ...selectedConfigs };
            selectedPersonas.forEach(pid => {
                if (!newConfigs[pid]) {
                    const persona = matchResults.distribution.find(d => d.persona.persona_id === pid)?.persona;
                    if (persona?.validated_configurations?.[0]) {
                        const rec = persona.validated_configurations[0];
                        newConfigs[pid] = { llm_id: rec.llm_id, prompt_config: rec.prompt_config };
                    }
                }
            });
            setSelectedConfigs(newConfigs);
        }
    }, [selectedPersonas, matchResults]);

    const goToStep = (step) => setCurrentStep(step);
    const handleStartOver = () => { setCurrentStep(4); setPredictions(null); };

    return (
        <div className="min-h-screen bg-gray-50">
            {/* Header */}
            <div className="bg-white border-b">
                <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
                    <div className="flex items-center gap-4">
                        {onBack && <button onClick={onBack} className="p-2 hover:bg-gray-100 rounded-lg"><ChevronLeft size={20} /></button>}
                        <div>
                            <h1 className="text-xl font-bold flex items-center gap-2"><Building className="text-green-600" size={24} />CYPEARL Enterprise</h1>
                            <p className="text-sm text-gray-500">Phishing Risk Assessment</p>
                        </div>
                    </div>
                    <Badge color="green"><Shield size={14} className="mr-1" />Privacy Protected</Badge>
                </div>
            </div>

            {/* Steps */}
            <div className="bg-white border-b">
                <div className="max-w-6xl mx-auto px-6 py-4 flex items-center justify-between">
                    {WORKFLOW_STEPS.map((step, idx) => (
                        <React.Fragment key={step.id}>
                            <button onClick={() => idx < currentStep && goToStep(idx)} disabled={idx > currentStep}
                                className={`flex flex-col items-center gap-1 ${idx < currentStep ? 'cursor-pointer' : ''}`}>
                                <div className={`w-10 h-10 rounded-full flex items-center justify-center ${idx === currentStep ? 'bg-indigo-600 text-white' : idx < currentStep ? 'bg-green-500 text-white' : 'bg-gray-200 text-gray-500'
                                    }`}>
                                    {idx < currentStep ? <CheckCircle size={20} /> : <step.icon size={20} />}
                                </div>
                                <span className={`text-xs font-medium ${idx === currentStep ? 'text-indigo-600' : idx < currentStep ? 'text-green-600' : 'text-gray-500'}`}>
                                    {step.label}
                                </span>
                            </button>
                            {idx < WORKFLOW_STEPS.length - 1 && <div className={`flex-1 h-0.5 mx-2 ${idx < currentStep ? 'bg-green-500' : 'bg-gray-200'}`} />}
                        </React.Fragment>
                    ))}
                </div>
            </div>

            {/* Content */}
            <div className="max-w-6xl mx-auto px-6 py-8">
                {currentStep === 0 && <UploadStep employeeData={employeeData} adminConfig={adminConfig} onUploadEmployees={setEmployeeData} onUploadConfig={setAdminConfig} onNext={() => goToStep(1)} />}
                {currentStep === 1 && <MatchStep employeeData={employeeData} adminConfig={adminConfig} onSetMatches={setMatchResults} onBack={() => goToStep(0)} onNext={() => goToStep(2)} />}
                {currentStep === 2 && <ConfigureStep matchResults={matchResults} adminConfig={adminConfig} selectedPersonas={selectedPersonas} selectedConfigs={selectedConfigs} onSelectPersona={setSelectedPersonas} onSelectConfig={setSelectedConfigs} onBack={() => goToStep(1)} onNext={() => goToStep(3)} />}
                {currentStep === 3 && <PromptsStep adminConfig={adminConfig} selectedPersonas={selectedPersonas} selectedConfigs={selectedConfigs} matchResults={matchResults} customPrompts={customPrompts} onSetCustomPrompts={setCustomPrompts} onBack={() => goToStep(2)} onNext={() => goToStep(4)} />}
                {currentStep === 4 && <EmailsStep testEmails={testEmails} onAddEmail={(e) => setTestEmails(p => [...p, e])} onRemoveEmail={(id) => setTestEmails(p => p.filter(e => e.id !== id))} onBack={() => goToStep(3)} onNext={() => goToStep(5)} />}
                {currentStep === 5 && <PredictStep selectedPersonas={selectedPersonas} selectedConfigs={selectedConfigs} testEmails={testEmails} matchResults={matchResults} adminConfig={adminConfig} customPrompts={customPrompts} predictions={predictions} onRunPredictions={setPredictions} onBack={() => goToStep(4)} onNext={() => goToStep(6)} />}
                {currentStep === 6 && <ReportStep predictions={predictions} matchResults={matchResults} employeeData={employeeData} onBack={() => goToStep(5)} onStartOver={handleStartOver} />}
            </div>
        </div>
    );
};

export default CISODashboard;