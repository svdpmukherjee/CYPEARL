/**
 * Cross-Industry Analysis Tab
 * 
 * Analyzes whether personas are consistent across industries.
 * Tests domain transferability (RQ3 from proposal).
 */

import React from 'react';
import { Building2, BarChart2, RefreshCw, AlertCircle } from 'lucide-react';
import { Card, EmptyState } from '../common';
import { CLUSTER_COLORS } from '../../constants';

export const CrossIndustryTab = ({
    clusteringResult,
    industryAnalysis,
    loading,
    error,
    onLoadAnalysis,
    personaLabels
}) => {
    // Helper to get persona display name
    const getPersonaName = (clusterId) => {
        const label = personaLabels?.[clusterId];
        if (label?.name) return label.name;
        return `P${parseInt(clusterId) + 1}`;
    };
    if (!clusteringResult) {
        return (
            <EmptyState
                icon={<Building2 size={48} />}
                title="Run clustering first"
                description="Cross-industry analysis requires clustering results."
            />
        );
    }

    return (
        <div className="space-y-6">
            <Card>
                <div className="flex items-center justify-between mb-4">
                    <div>
                        <h3 className="text-lg font-semibold">Industry Analysis</h3>
                        <p className="text-sm text-gray-500">
                            Analyze whether personas are consistent across different industries
                        </p>
                    </div>
                    <button
                        onClick={onLoadAnalysis}
                        disabled={loading}
                        className="px-4 py-2 bg-indigo-600 text-white rounded-lg hover:bg-indigo-700 disabled:opacity-50 flex items-center gap-2"
                    >
                        {loading ? <RefreshCw className="animate-spin" size={16} /> : <BarChart2 size={16} />}
                        {loading ? 'Analyzing...' : 'Analyze Industries'}
                    </button>
                </div>

                {/* Error Message */}
                {error && (
                    <div className="mb-4 p-4 bg-red-50 border border-red-200 rounded-lg flex items-start gap-3">
                        <AlertCircle className="text-red-600 mt-0.5" size={20} />
                        <div>
                            <p className="font-medium text-red-900">Analysis Failed</p>
                            <p className="text-sm text-red-700">{error}</p>
                            <p className="text-xs text-red-600 mt-2">
                                Tip: Make sure the backend is running and you've run clustering first.
                            </p>
                        </div>
                    </div>
                )}

                {industryAnalysis ? (
                    <div className="space-y-6">
                        {/* Note if mock data */}
                        {industryAnalysis.note && (
                            <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg text-sm text-blue-700">
                                {industryAnalysis.note}
                            </div>
                        )}

                        {/* Statistical Results */}
                        <div className="grid grid-cols-3 gap-4">
                            <div className="p-4 bg-gray-50 rounded-lg">
                                <div className="text-sm text-gray-500">Cramér's V</div>
                                <div className="text-2xl font-bold">
                                    {industryAnalysis.statistical_tests?.cramers_v?.toFixed(3) || '-'}
                                </div>
                                <div className="text-xs text-gray-400">Industry × Cluster association</div>
                            </div>
                            <div className="p-4 bg-gray-50 rounded-lg">
                                <div className="text-sm text-gray-500">χ² p-value</div>
                                <div className="text-2xl font-bold">
                                    {industryAnalysis.statistical_tests?.p_value?.toFixed(4) || '-'}
                                </div>
                                <div className="text-xs text-gray-400">
                                    {industryAnalysis.statistical_tests?.p_value > 0.05
                                        ? 'Not significant (industry-independent)'
                                        : 'Significant (industry effects)'}
                                </div>
                            </div>
                            <div className="p-4 bg-gray-50 rounded-lg">
                                <div className="text-sm text-gray-500">Domain Transferability</div>
                                <div className={`text-2xl font-bold ${industryAnalysis.domain_transferable ? 'text-green-600' : 'text-amber-600'
                                    }`}>
                                    {industryAnalysis.domain_transferable ? 'HIGH' : 'MODERATE'}
                                </div>
                                <div className="text-xs text-gray-400">Persona generalizability</div>
                            </div>
                        </div>

                        {/* Interpretation */}
                        <div className={`p-4 rounded-lg ${industryAnalysis.domain_transferable
                            ? 'bg-green-50 border border-green-200'
                            : 'bg-amber-50 border border-amber-200'
                            }`}>
                            <p className="text-sm">{industryAnalysis.interpretation}</p>
                        </div>

                        {/* Industry Distribution */}
                        <div>
                            <h4 className="font-medium mb-3">Cluster Distribution by Industry</h4>
                            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                                {Object.entries(industryAnalysis.distribution || {}).map(([ind, data]) => (
                                    <div key={ind} className="p-4 border rounded-lg">
                                        <div className="font-medium capitalize mb-2">{ind}</div>
                                        <div className="text-sm text-gray-500 mb-2">n = {data.n}</div>
                                        <div className="flex flex-wrap gap-1">
                                            {Object.entries(data.percentages || {}).map(([c, pct]) => (
                                                <span
                                                    key={c}
                                                    className="px-2 py-0.5 text-xs rounded"
                                                    style={{
                                                        backgroundColor: CLUSTER_COLORS[parseInt(c) % CLUSTER_COLORS.length] + '20',
                                                        color: CLUSTER_COLORS[parseInt(c) % CLUSTER_COLORS.length]
                                                    }}
                                                >
                                                    {getPersonaName(c)}: {typeof pct === 'number' ? pct.toFixed(0) : pct}%
                                                </span>
                                            ))}
                                        </div>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>
                ) : !error && (
                    <div className="text-center py-12">
                        <Building2 className="mx-auto text-gray-300 mb-4" size={48} />
                        <p className="text-gray-500">Click "Analyze Industries" to run cross-domain analysis</p>
                        <p className="text-sm text-gray-400 mt-2">
                            This will test whether your personas are consistent across different industries
                        </p>
                    </div>
                )}
            </Card>
        </div>
    );
};

export default CrossIndustryTab;