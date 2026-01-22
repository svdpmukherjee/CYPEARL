/**
 * Persona Profiles Tab - Redesigned for Comparison
 * 
 * Focus: Help admins compare ALL personas at a glance before diving into details
 * 
 * Sections:
 * 1. Persona Overview Table - Side-by-side comparison of key metrics
 * 2. Trait Profile Heatmap - Visual comparison of psychological traits
 * 3. Behavioral Outcomes Comparison - Bar chart comparison
 * 4. Selected Persona Details - Expandable detail panel (optional)
 */

import React, { useState, useMemo } from 'react';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip as RechartsTooltip,
    Legend, ResponsiveContainer, Cell
} from 'recharts';
import { Users, ChevronDown, ChevronRight, EyeOff } from 'lucide-react';
import { Card, EmptyState, CustomTooltip } from '../common';
import {
    TRAIT_CATEGORIES, TRAIT_LABELS, CATEGORY_LABELS,
    OUTCOME_LABELS
} from '../../constants';

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

const getZScoreColor = (z) => {
    if (z >= 1.5) return '#dc2626';  // Strong red
    if (z >= 1.0) return '#ef4444';  // Red
    if (z >= 0.5) return '#fca5a5';  // Light red
    if (z <= -1.5) return '#1d4ed8'; // Strong blue
    if (z <= -1.0) return '#3b82f6'; // Blue
    if (z <= -0.5) return '#93c5fd'; // Light blue
    return '#f3f4f6';                // Neutral gray
};

const getRiskBadge = (riskLevel) => {
    const colors = {
        'CRITICAL': 'bg-red-100 text-red-700 border-red-300',
        'HIGH': 'bg-orange-100 text-orange-700 border-orange-300',
        'MEDIUM': 'bg-yellow-100 text-yellow-700 border-yellow-300',
        'LOW': 'bg-green-100 text-green-700 border-green-300'
    };
    return colors[riskLevel] || colors.MEDIUM;
};

// ============================================================================
// SUB-COMPONENTS
// ============================================================================

/**
 * Compact Trait Heatmap - Clusters as columns, traits as rows
 */
const TraitHeatmap = ({ clusters, selectedCluster, onSelectCluster, personaLabels }) => {
    const [expandedCategories, setExpandedCategories] = useState({});

    const toggleCategory = (cat) => {
        setExpandedCategories(prev => ({ ...prev, [cat]: !prev[cat] }));
    };

    return (
        <div className="overflow-x-auto">
            <table className="w-full text-xs">
                <thead className="sticky top-0 z-10">
                    <tr className="bg-gray-50">
                        <th className="sticky left-0 bg-gray-50 px-3 py-2 text-left font-medium text-gray-500 border-b min-w-[140px]">
                            Trait / Cluster →
                        </th>
                        {clusters.map(c => {
                            const label = personaLabels?.[c.cluster_id];
                            const displayName = label?.name || `Persona ${c.cluster_id + 1}`;
                            // Truncate long names for header
                            const shortName = displayName.length > 12
                                ? displayName.substring(0, 10) + '...'
                                : displayName;
                            return (
                                <th
                                    key={c.cluster_id}
                                    onClick={() => onSelectCluster(selectedCluster === c.cluster_id ? null : c.cluster_id)}
                                    className={`px-2 py-2 text-center font-medium border-b cursor-pointer transition min-w-[56px] ${selectedCluster === c.cluster_id
                                        ? 'bg-indigo-100 text-indigo-700'
                                        : 'hover:bg-gray-100 text-gray-700'
                                        }`}
                                    title={displayName}
                                >
                                    <div className="text-xs">{shortName}</div>
                                    <div className={`text-[9px] font-normal px-1 py-0.5 rounded mt-0.5 ${getRiskBadge(c.risk_level)}`}>
                                        {c.risk_level}
                                    </div>
                                </th>
                            );
                        })}
                    </tr>
                </thead>
                <tbody>
                    {Object.entries(TRAIT_CATEGORIES).map(([category, traits]) => (
                        <React.Fragment key={category}>
                            {/* Category header - clickable to expand/collapse */}
                            <tr
                                className="cursor-pointer hover:bg-gray-100"
                                onClick={() => toggleCategory(category)}
                            >
                                <td
                                    colSpan={clusters.length + 1}
                                    className="sticky left-0 bg-gray-100 px-3 py-1.5 text-[10px] font-bold uppercase text-gray-600 tracking-wide"
                                >
                                    <div className="flex items-center gap-2">
                                        {expandedCategories[category] ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
                                        {CATEGORY_LABELS[category]} ({traits.length})
                                    </div>
                                </td>
                            </tr>
                            {/* Trait rows - only show if expanded */}
                            {expandedCategories[category] && traits.map(trait => (
                                <tr key={trait} className="hover:bg-gray-50">
                                    <td className="sticky left-0 bg-white px-3 py-1 text-gray-600 border-b whitespace-nowrap">
                                        {TRAIT_LABELS[trait] || trait}
                                    </td>
                                    {clusters.map(c => {
                                        const z = c.trait_zscores?.[trait] || 0;
                                        return (
                                            <td
                                                key={c.cluster_id}
                                                className={`px-1 py-1 text-center border-b ${selectedCluster === c.cluster_id ? 'ring-1 ring-indigo-400' : ''
                                                    }`}
                                                style={{ backgroundColor: getZScoreColor(z) }}
                                                title={`${TRAIT_LABELS[trait]}: z=${z.toFixed(2)}`}
                                            >
                                                <span className={Math.abs(z) > 0.75 ? 'text-white font-medium' : 'text-gray-700'}>
                                                    {z.toFixed(1)}
                                                </span>
                                            </td>
                                        );
                                    })}
                                </tr>
                            ))}
                        </React.Fragment>
                    ))}
                </tbody>
            </table>
        </div>
    );
};

/**
 * Behavioral Outcomes Comparison Chart
 */
const BehavioralComparisonChart = ({ clusters, personaLabels }) => {
    const chartData = useMemo(() => {
        return clusters.map(c => {
            const label = personaLabels?.[c.cluster_id];
            const displayName = label?.name || `P${c.cluster_id + 1}`;
            // Truncate for chart axis
            const shortName = displayName.length > 15
                ? displayName.substring(0, 12) + '...'
                : displayName;
            return {
                name: shortName,
                fullName: displayName,
                risk: c.risk_level,
                clickRate: (c.phishing_click_rate || 0) * 100,
                reportRate: (c.behavioral_outcomes?.report_rate?.mean || 0) * 100,
                accuracy: (c.behavioral_outcomes?.overall_accuracy?.mean || 0) * 100,
            };
        }).sort((a, b) => b.clickRate - a.clickRate);
    }, [clusters, personaLabels]);

    return (
        <ResponsiveContainer width="100%" height={280}>
            <BarChart data={chartData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                <XAxis type="number" stroke="#6b7280" fontSize={11} unit="%" domain={[0, 100]} />
                <YAxis type="category" dataKey="name" stroke="#6b7280" fontSize={11} width={40} />
                <RechartsTooltip content={<CustomTooltip />} />
                <Legend wrapperStyle={{ fontSize: '11px' }} />
                <Bar dataKey="clickRate" name="Click Rate" fill="#ef4444" />
                <Bar dataKey="reportRate" name="Report Rate" fill="#22c55e" />
                <Bar dataKey="accuracy" name="Accuracy" fill="#3b82f6" />
            </BarChart>
        </ResponsiveContainer>
    );
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

export const ProfilesTab = ({ clusteringResult, personaLabels, onSaveLabel }) => {
    const [selectedCluster, setSelectedCluster] = useState(null);
    const [showDetailPanel, setShowDetailPanel] = useState(false);

    if (!clusteringResult) {
        return (
            <EmptyState
                icon={<Users size={48} />}
                title="Run clustering first"
                description="Persona profiles require clustering results. Go to the Clustering tab to generate personas."
            />
        );
    }

    const clusters = Object.values(clusteringResult.clusters || {});
    const selectedData = selectedCluster !== null
        ? clusters.find(c => c.cluster_id === selectedCluster)
        : null;

    return (
        <div className="space-y-6">
            {/* Section 1: Persona Overview Comparison Table */}
            <Card>
                <div className="flex items-center justify-between mb-4">
                    <div>
                        <h3 className="text-lg font-semibold text-gray-900">Persona Overview Comparison</h3>
                        <p className="text-sm text-gray-500">Click any row to see detailed profile</p>
                    </div>
                    <div className="flex items-center gap-4 text-xs text-gray-500">
                        <span>{clusters.length} personas</span>
                        <span>•</span>
                        <span>{clusteringResult.metrics?.eta_squared_mean?.toFixed(3)} avg η²</span>
                    </div>
                </div>

                <div className="overflow-x-auto">
                    <table className="w-full text-sm">
                        <thead className="bg-gray-50">
                            <tr>
                                <th className="px-3 py-3 text-left font-medium text-gray-700">Persona</th>
                                <th className="px-3 py-3 text-center font-medium text-gray-700">Risk</th>
                                <th className="px-3 py-3 text-center font-medium text-gray-700">Size</th>
                                <th className="px-3 py-3 text-center font-medium text-gray-700">Click Rate</th>
                                <th className="px-3 py-3 text-center font-medium text-gray-700">Report Rate</th>
                                <th className="px-3 py-3 text-center font-medium text-gray-700">Accuracy</th>
                                <th className="px-3 py-3 text-left font-medium text-gray-700">Key High Traits</th>
                                <th className="px-3 py-3 text-left font-medium text-gray-700">Key Low Traits</th>
                            </tr>
                        </thead>
                        <tbody>
                            {clusters.sort((a, b) => b.phishing_click_rate - a.phishing_click_rate).map(c => {
                                const isSelected = selectedCluster === c.cluster_id;
                                const label = personaLabels?.[c.cluster_id];

                                return (
                                    <tr
                                        key={c.cluster_id}
                                        onClick={() => {
                                            setSelectedCluster(isSelected ? null : c.cluster_id);
                                            setShowDetailPanel(true);
                                        }}
                                        className={`border-t cursor-pointer transition ${isSelected
                                            ? 'bg-indigo-50 border-l-4 border-l-indigo-500'
                                            : 'hover:bg-gray-50'
                                            }`}
                                    >
                                        <td className="px-3 py-3">
                                            <div className="font-medium text-gray-900">
                                                {label?.name || `Cluster ${c.cluster_id + 1}`}
                                            </div>
                                            {label?.archetype && (
                                                <div className="text-xs text-gray-500">{label.archetype}</div>
                                            )}
                                        </td>
                                        <td className="px-3 py-3 text-center">
                                            <span className={`px-2 py-1 rounded text-xs font-medium ${getRiskBadge(c.risk_level)}`}>
                                                {c.risk_level}
                                            </span>
                                        </td>
                                        <td className="px-3 py-3 text-center">
                                            <div className="font-medium">{c.n_participants}</div>
                                            <div className="text-xs text-gray-400">{c.pct_of_population?.toFixed(1)}%</div>
                                        </td>
                                        <td className="px-3 py-3 text-center">
                                            <div className="flex items-center justify-center gap-2">
                                                <div
                                                    className="h-2 rounded-full bg-red-200"
                                                    style={{ width: '60px' }}
                                                >
                                                    <div
                                                        className="h-2 rounded-full bg-red-500"
                                                        style={{ width: `${(c.phishing_click_rate || 0) * 100}%` }}
                                                    />
                                                </div>
                                                <span className="font-mono text-red-600 w-12">
                                                    {((c.phishing_click_rate || 0) * 100).toFixed(1)}%
                                                </span>
                                            </div>
                                        </td>
                                        <td className="px-3 py-3 text-center">
                                            <span className="font-mono text-green-600">
                                                {((c.behavioral_outcomes?.report_rate?.mean || 0) * 100).toFixed(1)}%
                                            </span>
                                        </td>
                                        <td className="px-3 py-3 text-center">
                                            <span className="font-mono text-blue-600">
                                                {((c.behavioral_outcomes?.overall_accuracy?.mean || 0) * 100).toFixed(1)}%
                                            </span>
                                        </td>
                                        <td className="px-3 py-3">
                                            <div className="flex flex-wrap gap-1">
                                                {(c.top_high_traits || []).slice(0, 2).map(([trait, z]) => (
                                                    <span key={trait} className="px-1.5 py-0.5 bg-red-100 text-red-700 rounded text-xs">
                                                        ↑{TRAIT_LABELS[trait]?.split(' ')[0] || trait}
                                                    </span>
                                                ))}
                                            </div>
                                        </td>
                                        <td className="px-3 py-3">
                                            <div className="flex flex-wrap gap-1">
                                                {(c.top_low_traits || []).slice(0, 2).map(([trait, z]) => (
                                                    <span key={trait} className="px-1.5 py-0.5 bg-blue-100 text-blue-700 rounded text-xs">
                                                        ↓{TRAIT_LABELS[trait]?.split(' ')[0] || trait}
                                                    </span>
                                                ))}
                                            </div>
                                        </td>
                                    </tr>
                                );
                            })}
                        </tbody>
                    </table>
                </div>
            </Card>

            {/* Section 2: Visual Comparisons - Side by Side */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Behavioral Outcomes Chart */}
                <Card>
                    <h3 className="text-md font-semibold text-gray-900 mb-1">Behavioral Outcomes Comparison</h3>
                    <p className="text-xs text-gray-500 mb-4">Key metrics across all personas (sorted by click rate)</p>
                    <BehavioralComparisonChart clusters={clusters} personaLabels={personaLabels} />
                </Card>

                {/* Risk Distribution Summary */}
                <Card>
                    <h3 className="text-md font-semibold text-gray-900 mb-1">Risk Distribution Summary</h3>
                    <p className="text-xs text-gray-500 mb-4">Persona count by risk level</p>

                    <div className="space-y-4">
                        {['CRITICAL', 'HIGH', 'MEDIUM', 'LOW'].map(risk => {
                            const count = clusters.filter(c => c.risk_level === risk).length;
                            const personas = clusters.filter(c => c.risk_level === risk);
                            const totalParticipants = personas.reduce((sum, c) => sum + c.n_participants, 0);
                            const pct = clusters.length > 0 ? (count / clusters.length) * 100 : 0;

                            return (
                                <div key={risk} className="flex items-center gap-3">
                                    <span className={`px-2 py-1 rounded text-xs font-medium w-20 text-center ${getRiskBadge(risk)}`}>
                                        {risk}
                                    </span>
                                    <div className="flex-1">
                                        <div className="h-4 bg-gray-100 rounded-full overflow-hidden">
                                            <div
                                                className={`h-full rounded-full ${risk === 'CRITICAL' ? 'bg-red-500' :
                                                    risk === 'HIGH' ? 'bg-orange-500' :
                                                        risk === 'MEDIUM' ? 'bg-yellow-500' : 'bg-green-500'
                                                    }`}
                                                style={{ width: `${pct}%` }}
                                            />
                                        </div>
                                    </div>
                                    <div className="text-right w-32">
                                        <span className="font-medium">{count} persona{count !== 1 ? 's' : ''}</span>
                                        <span className="text-xs text-gray-500 ml-2">({totalParticipants} users)</span>
                                    </div>
                                </div>
                            );
                        })}
                    </div>

                    {/* Action Summary */}
                    <div className="mt-6 pt-4 border-t border-gray-100">
                        <h4 className="text-sm font-medium text-gray-700 mb-2">Recommended Actions</h4>
                        <div className="space-y-2 text-xs">
                            {clusters.filter(c => c.risk_level === 'CRITICAL' || c.risk_level === 'HIGH').length > 0 && (
                                <div className="flex items-start gap-2 p-2 bg-red-50 rounded">
                                    <span className="text-red-500">⚠</span>
                                    <span className="text-red-700">
                                        {clusters.filter(c => c.risk_level === 'CRITICAL' || c.risk_level === 'HIGH').length} high-risk
                                        personas need targeted training interventions
                                    </span>
                                </div>
                            )}
                            {clusters.filter(c => c.risk_level === 'LOW').length > 0 && (
                                <div className="flex items-start gap-2 p-2 bg-green-50 rounded">
                                    <span className="text-green-500">✓</span>
                                    <span className="text-green-700">
                                        {clusters.filter(c => c.risk_level === 'LOW').length} low-risk
                                        personas can serve as security champions
                                    </span>
                                </div>
                            )}
                        </div>
                    </div>
                </Card>
            </div>

            {/* Section 3: Trait Profile Heatmap */}
            <Card>
                <div className="flex items-center justify-between mb-4">
                    <div>
                        <h3 className="text-md font-semibold text-gray-900">Psychological Trait Heatmap</h3>
                        <p className="text-xs text-gray-500">Z-scores relative to population mean (click category to expand/collapse)</p>
                    </div>
                    <div className="flex items-center gap-4 text-xs">
                        <div className="flex items-center gap-1">
                            <span className="w-4 h-3 rounded" style={{ backgroundColor: '#3b82f6' }}></span>
                            <span>Below avg</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <span className="w-4 h-3 rounded bg-gray-200"></span>
                            <span>Average</span>
                        </div>
                        <div className="flex items-center gap-1">
                            <span className="w-4 h-3 rounded" style={{ backgroundColor: '#ef4444' }}></span>
                            <span>Above avg</span>
                        </div>
                    </div>
                </div>

                <TraitHeatmap
                    clusters={clusters}
                    selectedCluster={selectedCluster}
                    onSelectCluster={(id) => {
                        setSelectedCluster(id);
                        if (id !== null) setShowDetailPanel(true);
                    }}
                    personaLabels={personaLabels}
                />
            </Card>

            {/* Section 4: Selected Persona Detail Panel */}
            {selectedData && showDetailPanel && (
                <Card>
                    <div className="flex items-start justify-between mb-4">
                        <div className="flex items-center gap-3">
                            <h3 className="text-lg font-semibold text-gray-900">
                                {personaLabels?.[selectedCluster]?.name || `Cluster ${selectedCluster}`} Details
                            </h3>
                            <span className={`px-2 py-1 rounded text-xs font-medium ${getRiskBadge(selectedData.risk_level)}`}>
                                {selectedData.risk_level}
                            </span>
                        </div>
                        <button
                            onClick={() => setShowDetailPanel(false)}
                            className="text-sm text-gray-500 hover:text-gray-700 flex items-center gap-1"
                        >
                            <EyeOff size={14} /> Hide details
                        </button>
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                        {/* Overview Stats */}
                        <div className="space-y-3">
                            <h4 className="text-sm font-medium text-gray-700">Overview</h4>
                            <div className="grid grid-cols-2 gap-2 text-sm">
                                <div className="p-2 bg-gray-50 rounded text-center">
                                    <div className="text-xl font-bold text-gray-900">{selectedData.n_participants}</div>
                                    <div className="text-xs text-gray-500">Participants</div>
                                </div>
                                <div className="p-2 bg-red-50 rounded text-center">
                                    <div className="text-xl font-bold text-red-600">
                                        {((selectedData.phishing_click_rate || 0) * 100).toFixed(1)}%
                                    </div>
                                    <div className="text-xs text-gray-500">Click Rate</div>
                                </div>
                            </div>
                        </div>

                        {/* High Traits */}
                        <div>
                            <h4 className="text-sm font-medium text-gray-700 mb-2">↑ High Traits</h4>
                            <div className="space-y-1">
                                {(selectedData.top_high_traits || []).map(([trait, z]) => (
                                    <div key={trait} className="flex items-center justify-between text-xs">
                                        <span className="text-gray-600">{TRAIT_LABELS[trait] || trait}</span>
                                        <span className="font-mono text-red-600">+{z.toFixed(2)}σ</span>
                                    </div>
                                ))}
                                {!selectedData.top_high_traits?.length && (
                                    <span className="text-xs text-gray-400">None significant</span>
                                )}
                            </div>
                        </div>

                        {/* Low Traits */}
                        <div>
                            <h4 className="text-sm font-medium text-gray-700 mb-2">↓ Low Traits</h4>
                            <div className="space-y-1">
                                {(selectedData.top_low_traits || []).map(([trait, z]) => (
                                    <div key={trait} className="flex items-center justify-between text-xs">
                                        <span className="text-gray-600">{TRAIT_LABELS[trait] || trait}</span>
                                        <span className="font-mono text-blue-600">{z.toFixed(2)}σ</span>
                                    </div>
                                ))}
                                {!selectedData.top_low_traits?.length && (
                                    <span className="text-xs text-gray-400">None significant</span>
                                )}
                            </div>
                        </div>

                        {/* Behavioral Outcomes */}
                        <div>
                            <h4 className="text-sm font-medium text-gray-700 mb-2">Behavioral Outcomes</h4>
                            <div className="space-y-1 text-xs">
                                {Object.entries(selectedData.behavioral_outcomes || {}).slice(0, 5).map(([key, val]) => (
                                    <div key={key} className="flex justify-between">
                                        <span className="text-gray-600">{OUTCOME_LABELS[key] || key}</span>
                                        <span className="font-mono">
                                            {key.includes('rate') || key.includes('accuracy')
                                                ? `${((val?.mean || 0) * 100).toFixed(1)}%`
                                                : (val?.mean || 0).toFixed(1)
                                            }
                                        </span>
                                    </div>
                                ))}
                            </div>
                        </div>
                    </div>

                    {/* Description */}
                    <div className="mt-4 pt-4 border-t border-gray-100">
                        <p className="text-sm text-gray-600">
                            <span className="font-medium">Description: </span>
                            {personaLabels?.[selectedCluster]?.description || selectedData.description}
                        </p>
                    </div>
                </Card>
            )}
        </div>
    );
};

export default ProfilesTab;
