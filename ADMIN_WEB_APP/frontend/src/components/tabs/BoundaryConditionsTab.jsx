/**
 * Phase 2 Boundary Conditions Tab - FIXED
 * 
 * Analysis of AI failure patterns and boundary conditions.
 * 
 * FIXES:
 * - Display API-detected boundary conditions with persona names
 * - Handle both fidelity_results and fidelity field names
 * - Show severity grouping from API
 */

import React, { useMemo } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
    ResponsiveContainer, Cell, RadarChart, PolarGrid,
    PolarAngleAxis, PolarRadiusAxis, Radar
} from 'recharts';
import {
    AlertTriangle, Brain, Clock, Zap, Users, TrendingDown,
    Shield, AlertCircle, Info, ChevronDown, ChevronRight
} from 'lucide-react';

// =============================================================================
// BOUNDARY CONDITION TYPES
// =============================================================================

const BOUNDARY_CONDITIONS = {
    over_deliberation: {
        name: 'Over-Deliberation',
        description: 'AI analyzes too much, humans act on intuition',
        icon: Brain,
        severity: 'high',
        mitigation: 'Use shorter prompts or reduce reasoning traces'
    },
    emotional_unresponsive: {
        name: 'Emotional/Urgency Unresponsive',
        description: 'AI fails to simulate emotional manipulation effects',
        icon: Zap,
        severity: 'high',
        mitigation: 'Add urgency susceptibility examples to prompt'
    },
    over_clicking: {
        name: 'Over-Clicking',
        description: 'AI clicks more than human baseline',
        icon: AlertTriangle,
        severity: 'medium',
        mitigation: 'Add more suspicion cues to persona prompt'
    },
    trust_calibration: {
        name: 'Trust Calibration Error',
        description: 'AI misjudges trust levels for familiar senders',
        icon: Users,
        severity: 'low',
        mitigation: 'Include trust propensity examples in prompt'
    },
    fast_heuristic: {
        name: 'Fast Heuristic Mismatch',
        description: 'AI over-deliberates where humans make instant decisions',
        icon: Clock,
        severity: 'high',
        mitigation: 'Fundamental LLM limitation - use human testing'
    }
};

const SEVERITY_COLORS = {
    high: { bg: 'bg-red-50', border: 'border-red-200', text: 'text-red-700', dot: '#ef4444' },
    medium: { bg: 'bg-amber-50', border: 'border-amber-200', text: 'text-amber-700', dot: '#f59e0b' },
    low: { bg: 'bg-yellow-50', border: 'border-yellow-200', text: 'text-yellow-700', dot: '#eab308' },
};

// =============================================================================
// MAIN COMPONENT
// =============================================================================

export const BoundaryConditionsTab = ({
    experiments,
    currentExperiment,
    results,
    personas = []
}) => {
    // Extract boundary conditions from results (API-detected)
    const apiBoundaryData = results?.boundary_conditions || [];
    const bySeverity = results?.by_severity || { high: [], medium: [], low: [] };
    const summary = results?.summary || {};

    // Extract fidelity data (handle both field names)
    const fidelityData = results?.fidelity_results || results?.fidelity || [];

    // Calculate persona-level failure patterns from fidelity data
    const personaFailures = useMemo(() => {
        if (fidelityData.length === 0) return [];

        const byPersona = {};
        fidelityData.forEach(d => {
            if (!byPersona[d.persona_id]) {
                byPersona[d.persona_id] = {
                    persona_id: d.persona_id,
                    persona_name: d.persona_name || d.persona_id,
                    total: 0,
                    failures: 0,
                    fidelity_sum: 0,
                    click_diff_sum: 0
                };
            }
            const accuracy = d.normalized_accuracy || 0;
            if (!isNaN(accuracy)) {
                byPersona[d.persona_id].total++;
                byPersona[d.persona_id].fidelity_sum += accuracy;
                byPersona[d.persona_id].click_diff_sum += Math.abs(d.click_rate_diff || 0);
                if (accuracy < 0.85) {
                    byPersona[d.persona_id].failures++;
                }
            }
        });

        return Object.values(byPersona).map(p => ({
            ...p,
            avg_fidelity: p.total > 0 ? p.fidelity_sum / p.total : 0,
            avg_click_diff: p.total > 0 ? p.click_diff_sum / p.total : 0,
            failure_rate: p.total > 0 ? p.failures / p.total : 0
        })).sort((a, b) => b.failure_rate - a.failure_rate);
    }, [fidelityData]);

    // Match personas with API-detected boundary conditions
    const personaBoundaryAnalysis = useMemo(() => {
        // Group API boundary conditions by persona
        const byPersona = {};
        apiBoundaryData.forEach(bc => {
            const personaId = bc.persona_id;
            if (!byPersona[personaId]) {
                byPersona[personaId] = {
                    persona_id: personaId,
                    persona_name: bc.persona_name || personaId,
                    conditions: [],
                    risk_level: null
                };
            }
            byPersona[personaId].conditions.push(bc);
        });

        // Add failure rate data
        const result = Object.values(byPersona).map(p => {
            const failureData = personaFailures.find(pf => pf.persona_id === p.persona_id);
            const personaInfo = personas.find(per => per.persona_id === p.persona_id);
            return {
                ...p,
                risk_level: personaInfo?.risk_level || 'UNKNOWN',
                actual_failure_rate: failureData?.failure_rate || 0,
                avg_fidelity: failureData?.avg_fidelity || 0
            };
        });

        return result.sort((a, b) => b.conditions.length - a.conditions.length);
    }, [apiBoundaryData, personaFailures, personas]);

    // Count by condition type
    const conditionTypeCounts = useMemo(() => {
        const counts = {};
        apiBoundaryData.forEach(bc => {
            const type = bc.type || 'unknown';
            counts[type] = (counts[type] || 0) + 1;
        });
        return counts;
    }, [apiBoundaryData]);

    // No results state
    if (!results || (fidelityData.length === 0 && apiBoundaryData.length === 0)) {
        return (
            <div className="space-y-6">
                <div className="bg-white rounded-xl border p-6">
                    <div className="text-center py-8 text-gray-500">
                        <AlertTriangle className="mx-auto mb-2 text-gray-300" size={32} />
                        <p>No boundary condition data available</p>
                        <p className="text-sm">Run an experiment and check Results first</p>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="space-y-6">
            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                <SummaryCard
                    icon={<AlertTriangle className="text-red-600" />}
                    label="Total Boundary Conditions"
                    value={apiBoundaryData.length}
                    status={apiBoundaryData.length > 0 ? 'warning' : 'success'}
                    sublabel="Detected failure patterns"
                />
                <SummaryCard
                    icon={<AlertCircle className="text-red-600" />}
                    label="High Severity"
                    value={bySeverity.high?.length || summary.high_severity || 0}
                    status={(bySeverity.high?.length || 0) > 0 ? 'error' : 'success'}
                    sublabel="Requires attention"
                />
                <SummaryCard
                    icon={<AlertCircle className="text-amber-600" />}
                    label="Medium Severity"
                    value={bySeverity.medium?.length || summary.medium_severity || 0}
                    status="warning"
                    sublabel="Should review"
                />
                <SummaryCard
                    icon={<Info className="text-yellow-600" />}
                    label="Low Severity"
                    value={bySeverity.low?.length || summary.low_severity || 0}
                    status="neutral"
                    sublabel="Minor issues"
                />
            </div>

            {/* API-Detected Boundary Conditions */}
            {apiBoundaryData.length > 0 && (
                <div className="bg-white rounded-xl border p-6">
                    <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                        <AlertTriangle className="text-amber-600" size={20} />
                        Boundary Conditions (Where AI Fails)
                    </h3>
                    <div className="space-y-4">
                        {apiBoundaryData.map((bc, i) => {
                            const conditionInfo = BOUNDARY_CONDITIONS[bc.type] || {};
                            const colors = SEVERITY_COLORS[bc.severity || 'medium'];
                            const Icon = conditionInfo.icon || AlertTriangle;

                            return (
                                <div
                                    key={i}
                                    className={`p-4 rounded-lg border ${colors.bg} ${colors.border}`}
                                >
                                    <div className="flex items-start gap-3">
                                        <Icon className={colors.text} size={20} />
                                        <div className="flex-1">
                                            <div className="flex items-center gap-2 mb-1">
                                                <span className="font-medium">
                                                    {conditionInfo.name || bc.type}
                                                </span>
                                                <span className={`px-2 py-0.5 rounded text-xs ${colors.text} ${colors.bg}`}>
                                                    {bc.severity || 'medium'}
                                                </span>
                                            </div>
                                            <p className="text-sm text-gray-700 mb-2">
                                                {bc.description}
                                            </p>
                                            <div className="text-xs text-gray-500">
                                                <span className="font-medium">Affected persona:</span>{' '}
                                                {bc.persona_name || bc.persona_id || 'N/A'}
                                            </div>
                                            {bc.recommendation && (
                                                <div className="mt-2 text-xs text-blue-600">
                                                    <span className="font-medium">Mitigation:</span> {bc.recommendation}
                                                </div>
                                            )}
                                        </div>
                                    </div>
                                </div>
                            );
                        })}
                    </div>
                </div>
            )}

            {/* Charts Grid */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {/* Failure Rate by Persona */}
                <div className="bg-white rounded-xl border p-6">
                    <h3 className="text-lg font-semibold mb-4">Failure Rate by Persona</h3>
                    {personaFailures.length > 0 ? (
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={personaFailures.slice(0, 8)} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis type="number" domain={[0, 1]} tickFormatter={(v) => `${(v * 100).toFixed(0)}%`} />
                                <YAxis
                                    dataKey="persona_name"
                                    type="category"
                                    width={100}
                                    tick={{ fontSize: 11 }}
                                />
                                <Tooltip
                                    formatter={(v) => `${(v * 100).toFixed(1)}%`}
                                    labelFormatter={(label) => `Persona: ${label}`}
                                />
                                <Bar dataKey="failure_rate" name="Failure Rate">
                                    {personaFailures.slice(0, 8).map((entry, index) => (
                                        <Cell
                                            key={`cell-${index}`}
                                            fill={entry.failure_rate > 0.5 ? '#ef4444' :
                                                entry.failure_rate > 0.2 ? '#f59e0b' : '#22c55e'}
                                        />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    ) : (
                        <div className="h-[300px] flex items-center justify-center text-gray-400">
                            No failure data available
                        </div>
                    )}
                </div>

                {/* Boundary Condition Type Counts */}
                <div className="bg-white rounded-xl border p-6">
                    <h3 className="text-lg font-semibold mb-4">Condition Types Detected</h3>
                    {Object.keys(conditionTypeCounts).length > 0 ? (
                        <div className="space-y-3">
                            {Object.entries(conditionTypeCounts).map(([type, count]) => {
                                const info = BOUNDARY_CONDITIONS[type] || { name: type };
                                const Icon = info.icon || AlertTriangle;
                                return (
                                    <div key={type} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                                        <div className="flex items-center gap-2">
                                            <Icon size={18} className="text-gray-600" />
                                            <span className="text-sm">{info.name || type}</span>
                                        </div>
                                        <span className="font-bold text-lg">{count}</span>
                                    </div>
                                );
                            })}
                        </div>
                    ) : (
                        <div className="h-[300px] flex items-center justify-center text-gray-400">
                            <div className="text-center">
                                <Shield className="mx-auto mb-2 text-green-400" size={32} />
                                <p>No boundary conditions detected</p>
                                <p className="text-sm">AI performs well for all personas</p>
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Detailed Persona Analysis Table */}
            {personaBoundaryAnalysis.length > 0 && (
                <div className="bg-white rounded-xl border p-6">
                    <h3 className="text-lg font-semibold mb-4">Persona Boundary Condition Analysis</h3>
                    <div className="overflow-x-auto">
                        <table className="w-full text-sm">
                            <thead className="bg-gray-50">
                                <tr>
                                    <th className="px-4 py-2 text-left">Persona</th>
                                    <th className="px-4 py-2 text-left">Risk Level</th>
                                    <th className="px-4 py-2 text-left">Detected Conditions</th>
                                    <th className="px-4 py-2 text-right">Avg Fidelity</th>
                                    <th className="px-4 py-2 text-left">Recommendation</th>
                                </tr>
                            </thead>
                            <tbody>
                                {personaBoundaryAnalysis.map((pa, i) => (
                                    <tr key={i} className="border-t hover:bg-gray-50">
                                        <td className="px-4 py-2 font-medium">
                                            {pa.persona_name}
                                        </td>
                                        <td className="px-4 py-2">
                                            <span className={`px-2 py-0.5 rounded text-xs ${pa.risk_level === 'CRITICAL' ? 'bg-red-100 text-red-700' :
                                                    pa.risk_level === 'HIGH' ? 'bg-orange-100 text-orange-700' :
                                                        pa.risk_level === 'MEDIUM' ? 'bg-yellow-100 text-yellow-700' :
                                                            'bg-green-100 text-green-700'
                                                }`}>
                                                {pa.risk_level}
                                            </span>
                                        </td>
                                        <td className="px-4 py-2">
                                            <div className="flex flex-wrap gap-1">
                                                {pa.conditions.map((cond, j) => {
                                                    const colors = SEVERITY_COLORS[cond.severity || 'medium'];
                                                    const info = BOUNDARY_CONDITIONS[cond.type] || {};
                                                    return (
                                                        <span
                                                            key={j}
                                                            className={`px-1.5 py-0.5 rounded text-xs ${colors.bg} ${colors.text}`}
                                                            title={cond.description}
                                                        >
                                                            {info.name?.split(' ')[0] || cond.type}
                                                        </span>
                                                    );
                                                })}
                                            </div>
                                        </td>
                                        <td className="px-4 py-2 text-right font-mono">
                                            <span className={pa.avg_fidelity >= 0.85 ? 'text-green-600' : 'text-red-600'}>
                                                {(pa.avg_fidelity * 100).toFixed(1)}%
                                            </span>
                                        </td>
                                        <td className="px-4 py-2 text-xs text-gray-600">
                                            {pa.conditions.length >= 2
                                                ? '✗ Use human testing'
                                                : pa.conditions.length === 1
                                                    ? '⚠ Validate with subset'
                                                    : '✓ AI suitable'
                                            }
                                        </td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                </div>
            )}

            {/* Recommendations */}
            <div className="bg-white rounded-xl border p-6">
                <h3 className="text-lg font-semibold mb-4 flex items-center gap-2">
                    <Shield className="text-blue-600" size={20} />
                    Deployment Recommendations
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="p-4 bg-green-50 border border-green-200 rounded-lg">
                        <h4 className="font-medium text-green-800 mb-2">✓ AI Suitable</h4>
                        <p className="text-sm text-green-700 mb-2">
                            Personas with high fidelity (≥85%) and no boundary conditions:
                        </p>
                        <ul className="text-sm text-green-600 list-disc list-inside">
                            {personaFailures
                                .filter(p => p.avg_fidelity >= 0.85)
                                .filter(p => !personaBoundaryAnalysis.find(pa =>
                                    pa.persona_id === p.persona_id && pa.conditions.length > 0
                                ))
                                .slice(0, 5)
                                .map(p => (
                                    <li key={p.persona_id}>{p.persona_name} ({(p.avg_fidelity * 100).toFixed(0)}%)</li>
                                ))
                            }
                            {personaFailures.filter(p => p.avg_fidelity >= 0.85).length === 0 && (
                                <li className="text-gray-500">None meet threshold yet</li>
                            )}
                        </ul>
                    </div>
                    <div className="p-4 bg-red-50 border border-red-200 rounded-lg">
                        <h4 className="font-medium text-red-800 mb-2">✗ Human Testing Required</h4>
                        <p className="text-sm text-red-700 mb-2">
                            Personas with significant boundary conditions:
                        </p>
                        <ul className="text-sm text-red-600 list-disc list-inside">
                            {personaBoundaryAnalysis
                                .filter(p => p.conditions.length >= 1)
                                .slice(0, 5)
                                .map(p => (
                                    <li key={p.persona_id}>
                                        {p.persona_name} ({p.conditions.length} condition{p.conditions.length > 1 ? 's' : ''})
                                    </li>
                                ))
                            }
                            {personaBoundaryAnalysis.filter(p => p.conditions.length >= 1).length === 0 && (
                                <li className="text-gray-500">No problematic personas</li>
                            )}
                        </ul>
                    </div>
                </div>
            </div>
        </div>
    );
};

// =============================================================================
// HELPER COMPONENTS
// =============================================================================

const SummaryCard = ({ icon, label, value, status, sublabel }) => {
    const statusColors = {
        success: 'bg-green-50 border-green-200',
        warning: 'bg-amber-50 border-amber-200',
        error: 'bg-red-50 border-red-200',
        neutral: 'bg-gray-50 border-gray-200'
    };

    return (
        <div className={`p-4 rounded-xl border ${statusColors[status]}`}>
            <div className="flex items-center gap-2 mb-1">
                {icon}
                <span className="text-sm text-gray-600">{label}</span>
            </div>
            <div className="text-2xl font-bold">{value}</div>
            <div className="text-xs text-gray-500">{sublabel}</div>
        </div>
    );
};

export default BoundaryConditionsTab;