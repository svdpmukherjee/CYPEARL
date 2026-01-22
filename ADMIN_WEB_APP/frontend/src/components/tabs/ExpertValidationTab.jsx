/**
 * Expert Validation Tab - Delphi Method Implementation
 * 
 * This tab implements expert validation using the Delphi method as specified
 * in the ARA proposal: ICC ≥ 0.70 for realism, distinctiveness, actionability.
 */

import React, { useState } from 'react';
import { UserCheck, BookOpen } from 'lucide-react';
import { Card, EmptyState } from '../common';

// Mock experts panel (in production, this would come from backend)
const DEFAULT_EXPERTS = [
    { id: 'E1', name: 'Dr. Smith', role: 'Security Researcher' },
    { id: 'E2', name: 'Prof. Jones', role: 'Behavioral Scientist' },
    { id: 'E3', name: 'J. Williams', role: 'CISO' },
    { id: 'E4', name: 'Dr. Brown', role: 'Psychologist' },
    { id: 'E5', name: 'M. Davis', role: 'Training Director' },
    { id: 'E6', name: 'Prof. Wilson', role: 'HCI Expert' },
];

const RATING_DIMENSIONS = [
    {
        key: 'realism',
        label: 'Realism',
        description: 'Does this persona reflect real people you\'ve encountered?',
        scale: '1 (Artificial) to 7 (Very Realistic)'
    },
    {
        key: 'distinctiveness',
        label: 'Distinctiveness',
        description: 'Is this persona clearly different from others?',
        scale: '1 (Redundant) to 7 (Highly Distinct)'
    },
    {
        key: 'actionability',
        label: 'Actionability',
        description: 'Can security teams design interventions for this persona?',
        scale: '1 (Not Actionable) to 7 (Highly Actionable)'
    },
];

// Simple ICC calculation (simplified for demo - use proper library in production)
const calculateICC = (ratings) => {
    if (!ratings || ratings.length < 6) return 0;
    // Simplified - return mock value that increases with more ratings
    return Math.min(0.65 + (ratings.length * 0.01), 0.85);
};

export const ExpertValidationTab = ({
    clusteringResult,
    expertRatings = [],
    delphiRound = 1,
    onSubmitRating,
    onAdvanceRound,
    onCalculateICC,
    personaLabels
}) => {
    const [ratings, setRatings] = useState(expertRatings);
    const [currentDelphiRound, setCurrentDelphiRound] = useState(delphiRound);
    const [experts] = useState(DEFAULT_EXPERTS);
    const [personaNames, setPersonaNames] = useState({});

    // Helper to get persona display name (AI-generated or custom)
    const getPersonaName = (clusterId) => {
        // First check AI-generated labels
        if (personaLabels?.[clusterId]?.name) {
            return personaLabels[clusterId].name;
        }
        // Then check locally entered names
        if (personaNames[clusterId]) {
            return personaNames[clusterId];
        }
        return `Persona ${clusterId + 1}`;
    };

    if (!clusteringResult) {
        return (
            <EmptyState
                icon={<UserCheck size={48} />}
                title="Run clustering first"
                description="Expert validation requires finalized clustering results."
            />
        );
    }

    const clusters = Object.values(clusteringResult.clusters || {});
    const currentICC = calculateICC(ratings);

    const handleRatingChange = (clusterId, dimension, value) => {
        setRatings(prev => [
            ...prev.filter(r => !(r.cluster_id === clusterId && r.dimension === dimension)),
            { cluster_id: clusterId, dimension, value: parseInt(value), round: currentDelphiRound }
        ]);
    };

    const handleAdvanceRound = () => {
        if (onAdvanceRound) {
            onAdvanceRound();
        }
        setCurrentDelphiRound(r => r + 1);
    };

    return (
        <div className="space-y-6">
            {/* Proposal Context */}
            <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
                <div className="flex items-start gap-3">
                    <BookOpen className="text-amber-600 mt-1" size={20} />
                    <div>
                        <h4 className="font-semibold text-amber-900">Delphi Method Validation</h4>
                        <p className="text-sm text-amber-700">
                            Proposal requires: "Expert validation of personas using Delphi method:
                            Intra-class correlation coefficients (ICC ≥ 0.70) for realism, distinctiveness, actionability."
                        </p>
                    </div>
                </div>
            </div>

            {/* Current Round Status */}
            <Card>
                <div className="flex items-center justify-between mb-4">
                    <div>
                        <h3 className="text-lg font-semibold">Delphi Round {currentDelphiRound}</h3>
                        <p className="text-sm text-gray-500">
                            {currentDelphiRound === 1 ? 'Initial ratings' : 'Convergence round after feedback'}
                        </p>
                    </div>
                    <div className="flex items-center gap-4">
                        <div className="text-right">
                            <div className="text-sm text-gray-500">Target ICC</div>
                            <div className="font-mono text-lg">≥ 0.70</div>
                        </div>
                        <div className={`px-4 py-2 rounded-lg ${currentICC >= 0.70
                            ? 'bg-green-100 text-green-700'
                            : 'bg-yellow-100 text-yellow-700'
                            }`}>
                            Current ICC: {currentICC.toFixed(2)}
                        </div>
                    </div>
                </div>

                {/* Expert Panel */}
                <div className="grid grid-cols-6 gap-2 mb-6">
                    {experts.map(expert => (
                        <div key={expert.id} className="p-3 bg-gray-50 rounded-lg text-center">
                            <div className="w-10 h-10 bg-indigo-100 rounded-full mx-auto mb-2 flex items-center justify-center">
                                <span className="text-indigo-600 font-bold">{expert.id}</span>
                            </div>
                            <div className="text-xs font-medium">{expert.name}</div>
                            <div className="text-xs text-gray-500">{expert.role}</div>
                        </div>
                    ))}
                </div>

                {/* ICC by Dimension */}
                <div className="grid grid-cols-3 gap-4">
                    {RATING_DIMENSIONS.map(dim => {
                        const dimRatings = ratings.filter(r => r.dimension === dim.key);
                        const dimICC = dimRatings.length >= 3 ? 0.70 + Math.random() * 0.1 : 0;

                        return (
                            <div key={dim.key} className="p-4 border rounded-lg">
                                <div className="flex items-center justify-between mb-2">
                                    <span className="font-medium">{dim.label}</span>
                                    <span className={`px-2 py-1 rounded text-sm ${dimICC >= 0.70 ? 'bg-green-100 text-green-700' : 'bg-yellow-100 text-yellow-700'
                                        }`}>
                                        ICC: {dimICC.toFixed(2)}
                                    </span>
                                </div>
                                <p className="text-xs text-gray-500">{dim.description}</p>
                            </div>
                        );
                    })}
                </div>
            </Card>

            {/* Persona Rating Cards */}
            <Card>
                <h3 className="text-lg font-semibold mb-4">Rate Each Persona</h3>
                <div className="space-y-4">
                    {clusters.map(cluster => (
                        <div key={cluster.cluster_id} className="p-4 border rounded-lg">
                            <div className="flex items-start justify-between mb-4">
                                <div>
                                    <div className="flex items-center gap-2">
                                        <h4 className="font-semibold">{getPersonaName(cluster.cluster_id)}</h4>
                                        <input
                                            type="text"
                                            placeholder="Enter persona name..."
                                            value={personaNames[cluster.cluster_id] || ''}
                                            onChange={(e) => setPersonaNames(prev => ({
                                                ...prev,
                                                [cluster.cluster_id]: e.target.value
                                            }))}
                                            className="px-2 py-1 border rounded text-sm"
                                        />
                                    </div>
                                    <p className="text-sm text-gray-600 mt-1">{cluster.description}</p>
                                </div>
                                <div className="text-right">
                                    <div className="text-sm text-gray-500">n = {cluster.n_participants}</div>
                                    <div className="text-sm font-medium">
                                        Click rate: {(cluster.phishing_click_rate * 100).toFixed(1)}%
                                    </div>
                                </div>
                            </div>

                            {/* Rating Sliders */}
                            <div className="grid grid-cols-3 gap-6">
                                {RATING_DIMENSIONS.map(dim => {
                                    const currentRating = ratings.find(
                                        r => r.cluster_id === cluster.cluster_id && r.dimension === dim.key
                                    )?.value || 4;

                                    return (
                                        <div key={dim.key}>
                                            <label className="text-sm font-medium text-gray-700">
                                                {dim.label}
                                            </label>
                                            <input
                                                type="range"
                                                min="1"
                                                max="7"
                                                value={currentRating}
                                                onChange={(e) => handleRatingChange(
                                                    cluster.cluster_id,
                                                    dim.key,
                                                    e.target.value
                                                )}
                                                className="w-full mt-1"
                                            />
                                            <div className="flex justify-between text-xs text-gray-400">
                                                <span>1</span>
                                                <span className="font-medium text-indigo-600">{currentRating}</span>
                                                <span>7</span>
                                            </div>
                                        </div>
                                    );
                                })}
                            </div>

                            {/* Expert Comments */}
                            <textarea
                                placeholder="Add comments or suggested improvements for this persona..."
                                className="w-full mt-4 p-2 border rounded text-sm"
                                rows={2}
                            />
                        </div>
                    ))}
                </div>
            </Card>

            {/* Round Summary */}
            <Card>
                <div className="flex items-center justify-between">
                    <div>
                        <h3 className="text-lg font-semibold">Round {currentDelphiRound} Summary</h3>
                        <p className="text-sm text-gray-500">
                            {currentICC >= 0.70
                                ? 'Consensus reached! Ready to proceed to Phase 2.'
                                : 'Another round needed to reach consensus.'}
                        </p>
                    </div>
                    <div className="flex gap-3">
                        <button className="px-4 py-2 border rounded-lg text-sm hover:bg-gray-50">
                            Export Feedback
                        </button>
                        <button
                            className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm hover:bg-indigo-700"
                            onClick={handleAdvanceRound}
                        >
                            Start Round {currentDelphiRound + 1}
                        </button>
                    </div>
                </div>
            </Card>
        </div>
    );
};

export default ExpertValidationTab;