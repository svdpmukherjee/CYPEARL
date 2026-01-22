/**
 * Behavioral Validation Tab
 * 
 * Shows validation metrics for clustering results:
 * - Effect size summary (η²)
 * - Prediction strength by outcome
 * - Click rate comparison
 * - Behavioral profile comparison
 * - Cluster sizes
 * - Predicted behaviors table
 */

import React from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
    Legend, ResponsiveContainer, ReferenceLine, Cell
} from 'recharts';
import { Target, CheckCircle, AlertTriangle, AlertCircle } from 'lucide-react';
import { Card, ChartCard, EmptyState, CustomTooltip } from '../common';
import { OUTCOME_LABELS, RISK_COLORS } from '../../constants';

export const ValidationTab = ({ clusteringResult, minClusterSize, personaLabels }) => {
    if (!clusteringResult) {
        return (
            <EmptyState
                icon={<Target size={48} />}
                title="Run clustering first"
                description="Behavioral validation requires clustering results. Go to the Clustering tab first."
            />
        );
    }

    // Helper to get persona display name
    const getPersonaName = (cluster) => {
        const label = personaLabels?.[cluster.cluster_id];
        if (label?.name) return label.name;
        return `Persona ${cluster.cluster_id + 1}`;
    };

    const eta2ByOutcome = clusteringResult.metrics?.eta_squared_by_outcome || {};
    const eta2Data = Object.entries(eta2ByOutcome).map(([outcome, value]) => ({
        outcome: OUTCOME_LABELS[outcome] || outcome,
        key: outcome,
        eta2: value,
        isClickRate: outcome.includes('click')
    })).sort((a, b) => b.eta2 - a.eta2);

    const clusters = Object.values(clusteringResult.clusters || {});

    // Prepare behavioral comparison data
    const behaviorComparisonData = clusters.map(c => ({
        name: getPersonaName(c),
        clickRate: (c.phishing_click_rate || 0) * 100,
        reportRate: (c.behavioral_outcomes?.report_rate?.mean || 0) * 100,
        accuracy: (c.behavioral_outcomes?.overall_accuracy?.mean || 0) * 100,
        hoverRate: (c.behavioral_outcomes?.hover_rate?.mean || 0) * 100,
        risk: c.risk_level
    })).sort((a, b) => b.clickRate - a.clickRate);

    const sizeData = clusters.map(c => ({
        name: getPersonaName(c),
        size: c.n_participants,
        meetsMin: c.n_participants >= minClusterSize
    }));

    const effectSizeMean = clusteringResult.metrics?.eta_squared_mean || 0;

    return (
        <div className="space-y-6">
            {/* Effect Size Summary */}
            <div className={`p-4 rounded-lg border ${effectSizeMean >= 0.14
                ? 'bg-green-50 border-green-200'
                : effectSizeMean >= 0.06
                    ? 'bg-yellow-50 border-yellow-200'
                    : 'bg-red-50 border-red-200'
                }`}>
                <div className="flex items-center gap-3">
                    {effectSizeMean >= 0.14 ? (
                        <CheckCircle className="text-green-600" size={24} />
                    ) : effectSizeMean >= 0.06 ? (
                        <AlertTriangle className="text-yellow-600" size={24} />
                    ) : (
                        <AlertCircle className="text-red-600" size={24} />
                    )}
                    <div>
                        <span className="font-semibold text-lg">
                            {effectSizeMean >= 0.14 ? 'Large' :
                                effectSizeMean >= 0.06 ? 'Medium' : 'Small'} Effect Size
                        </span>
                        <p className="text-gray-600">
                            Clusters explain <strong>{(effectSizeMean * 100).toFixed(1)}%</strong> of variance in phishing behavior on average
                        </p>
                    </div>
                </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* η² by Outcome */}
                <ChartCard title="Prediction Strength by Outcome" subtitle="How well clusters predict each behavioral metric (η²)">
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={eta2Data} layout="vertical">
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis type="number" stroke="#6b7280" fontSize={12} domain={[0, 'auto']} />
                            <YAxis type="category" dataKey="outcome" stroke="#6b7280" fontSize={11} width={140} />
                            <RechartsTooltip content={<CustomTooltip />} />
                            <ReferenceLine x={0.06} stroke="#f59e0b" strokeDasharray="5 5" label={{ value: "Medium", fontSize: 10, position: 'top' }} />
                            <ReferenceLine x={0.14} stroke="#22c55e" strokeDasharray="5 5" label={{ value: "Large", fontSize: 10, position: 'top' }} />
                            <Bar dataKey="eta2" name="η²">
                                {eta2Data.map((entry, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill={entry.eta2 >= 0.14 ? '#22c55e' : entry.eta2 >= 0.06 ? '#f59e0b' : '#94a3b8'}
                                    />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </ChartCard>

                {/* Click Rate by Cluster */}
                <ChartCard title="Phishing Click Rate by Persona" subtitle="Higher = more vulnerable to phishing">
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={behaviorComparisonData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis dataKey="name" stroke="#6b7280" fontSize={12} />
                            <YAxis stroke="#6b7280" fontSize={12} unit="%" />
                            <RechartsTooltip content={<CustomTooltip />} />
                            <Bar dataKey="clickRate" name="Click Rate %">
                                {behaviorComparisonData.map((entry, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill={RISK_COLORS[entry.risk]?.border || '#6b7280'}
                                    />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </ChartCard>

                {/* Behavioral Comparison */}
                <ChartCard title="Behavioral Profile Comparison" subtitle="Multiple metrics across personas">
                    <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={behaviorComparisonData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis dataKey="name" stroke="#6b7280" fontSize={12} />
                            <YAxis stroke="#6b7280" fontSize={12} unit="%" />
                            <RechartsTooltip content={<CustomTooltip />} />
                            <Legend />
                            <Bar dataKey="clickRate" name="Click Rate" fill="#ef4444" />
                            <Bar dataKey="reportRate" name="Report Rate" fill="#22c55e" />
                            <Bar dataKey="accuracy" name="Accuracy" fill="#3b82f6" />
                        </BarChart>
                    </ResponsiveContainer>
                </ChartCard>

                {/* Cluster Sizes */}
                <ChartCard title="Cluster Sizes" subtitle={`Minimum required: ${minClusterSize} participants`}>
                    <ResponsiveContainer width="100%" height={200}>
                        <BarChart data={sizeData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                            <XAxis dataKey="name" stroke="#6b7280" fontSize={12} />
                            <YAxis stroke="#6b7280" fontSize={12} />
                            <RechartsTooltip content={<CustomTooltip />} />
                            <ReferenceLine y={minClusterSize} stroke="#ef4444" strokeDasharray="5 5" label={{ value: `Min: ${minClusterSize}`, fontSize: 10 }} />
                            <Bar dataKey="size" name="Participants">
                                {sizeData.map((entry, index) => (
                                    <Cell
                                        key={`cell-${index}`}
                                        fill={entry.meetsMin ? '#4f46e5' : '#ef4444'}
                                    />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </ChartCard>
            </div>

            {/* Predicted Behaviors Table */}
            <Card>
                <h3 className="text-lg font-semibold mb-4">Predicted Behavioral Outcomes by Persona</h3>
                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-4 py-3 text-left font-medium text-gray-700">Persona</th>
                                <th className="px-4 py-3 text-left font-medium text-gray-700">Risk Level</th>
                                <th className="px-4 py-3 text-right font-medium text-gray-700">Click Rate</th>
                                <th className="px-4 py-3 text-right font-medium text-gray-700">Report Rate</th>
                                <th className="px-4 py-3 text-right font-medium text-gray-700">Accuracy</th>
                                <th className="px-4 py-3 text-right font-medium text-gray-700">Hover Rate</th>
                                <th className="px-4 py-3 text-left font-medium text-gray-700">Behavioral Prediction</th>
                            </tr>
                        </thead>
                        <tbody>
                            {clusters.map(c => (
                                <tr key={c.cluster_id} className="border-t hover:bg-gray-50">
                                    <td className="px-4 py-3 font-medium">{getPersonaName(c)}</td>
                                    <td className="px-4 py-3">
                                        <span
                                            className="px-2 py-1 rounded text-xs font-medium"
                                            style={{
                                                backgroundColor: RISK_COLORS[c.risk_level]?.bg,
                                                color: RISK_COLORS[c.risk_level]?.text
                                            }}
                                        >
                                            {c.risk_level}
                                        </span>
                                    </td>
                                    <td className="px-4 py-3 text-right font-mono text-red-600">
                                        {((c.phishing_click_rate || 0) * 100).toFixed(1)}%
                                    </td>
                                    <td className="px-4 py-3 text-right font-mono text-green-600">
                                        {((c.behavioral_outcomes?.report_rate?.mean || 0) * 100).toFixed(1)}%
                                    </td>
                                    <td className="px-4 py-3 text-right font-mono text-blue-600">
                                        {((c.behavioral_outcomes?.overall_accuracy?.mean || 0) * 100).toFixed(1)}%
                                    </td>
                                    <td className="px-4 py-3 text-right font-mono">
                                        {((c.behavioral_outcomes?.hover_rate?.mean || 0) * 100).toFixed(1)}%
                                    </td>
                                    <td className="px-4 py-3 text-gray-600 text-xs max-w-xs">
                                        {c.risk_level === 'CRITICAL' || c.risk_level === 'HIGH'
                                            ? 'Likely to click on phishing emails, especially under urgency'
                                            : c.risk_level === 'MEDIUM'
                                                ? 'Moderate risk, behavior varies by email type'
                                                : 'Cautious, likely to report suspicious emails'
                                        }
                                    </td>
                                </tr>
                            ))}
                        </tbody>
                    </table>
                </div>
            </Card>

            {/* Statistical Summary */}
            <Card>
                <h3 className="text-lg font-semibold mb-4">Statistical Quality Metrics</h3>
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                    <div className="p-4 bg-gray-50 rounded-lg">
                        <div className="text-sm text-gray-500">Calinski-Harabasz</div>
                        <div className="text-2xl font-bold">{clusteringResult.metrics?.calinski_harabasz?.toFixed(1) || '-'}</div>
                        <div className="text-xs text-gray-400">Higher = better defined clusters</div>
                    </div>
                    <div className="p-4 bg-gray-50 rounded-lg">
                        <div className="text-sm text-gray-500">Davies-Bouldin</div>
                        <div className="text-2xl font-bold">{clusteringResult.metrics?.davies_bouldin?.toFixed(3) || '-'}</div>
                        <div className="text-xs text-gray-400">Lower = better separation</div>
                    </div>
                    <div className="p-4 bg-gray-50 rounded-lg">
                        <div className="text-sm text-gray-500">Significant Outcomes</div>
                        <div className="text-2xl font-bold">{clusteringResult.metrics?.anova_significant_outcomes || '-'}</div>
                        <div className="text-xs text-gray-400">Outcomes with p &lt; 0.05</div>
                    </div>
                    <div className="p-4 bg-gray-50 rounded-lg">
                        <div className="text-sm text-gray-500">Average η²</div>
                        <div className="text-2xl font-bold">{clusteringResult.metrics?.eta_squared_mean?.toFixed(3) || '-'}</div>
                        <div className="text-xs text-gray-400">Mean effect size</div>
                    </div>
                </div>
            </Card>
        </div>
    );
};

export default ValidationTab;