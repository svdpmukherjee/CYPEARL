/**
 * Dataset Description Tab
 * 
 * Shows overview of the dataset with expandable sections for:
 * - Total test subjects with demographics
 * - Features used for clustering
 * - Behavioral outcome measures
 * - Email factorial design (2×2×2×2)
 * - Industry coverage
 */

import React, { useMemo } from 'react';
import { Users, Brain, Target, Mail, Building2, CheckCircle } from 'lucide-react';
import { Card, ExpandableCard } from '../common';
import { TRAIT_CATEGORIES, TRAIT_LABELS, CATEGORY_LABELS, OUTCOME_FEATURES, OUTCOME_LABELS } from '../../constants';

export const DatasetDescriptionTab = ({ summary, dataQuality }) => {
    // Demographics calculation
    const demographics = useMemo(() => {
        if (!summary) return null;
        return {
            avgAge: summary.demographics?.avg_age || 35,
            genderDist: summary.demographics?.gender || { Male: 52, Female: 46, Other: 2 },
            educationDist: summary.demographics?.education || { 'Bachelor': 45, 'Master': 30, 'High School': 15, 'PhD': 10 }
        };
    }, [summary]);

    // Feature categorization
    const featuresByCategory = useMemo(() => {
        const result = {};
        let totalClustering = 0;
        Object.entries(TRAIT_CATEGORIES).forEach(([cat, features]) => {
            result[cat] = features.map(f => ({
                key: f,
                label: TRAIT_LABELS[f] || f,
                available: summary?.clustering_features?.includes(f) || true
            }));
            totalClustering += features.length;
        });
        return { categories: result, totalClustering };
    }, [summary]);

    // Email factorial design
    const emailDesign = {
        factors: [
            { name: 'Email Type', levels: ['Phishing', 'Legitimate'] },
            { name: 'Urgency', levels: ['High', 'Low'] },
            { name: 'Sender Familiarity', levels: ['Familiar', 'Unfamiliar'] },
            { name: 'Content Framing', levels: ['Threat', 'Reward'] }
        ],
        totalCombinations: summary?.n_emails || 0,
        description: summary?.n_emails + ' emails'
    };

    // Industry data
    const industryData = useMemo(() => {
        if (summary?.industry_counts) {
            const total = Object.values(summary.industry_counts).reduce((a, b) => a + b, 0);
            return Object.entries(summary.industry_counts).map(([name, count]) => ({
                name,
                count,
                pct: Math.round((count / total) * 100)
            })).sort((a, b) => b.count - a.count);
        }
        return [
            { name: 'Technology', count: 280, pct: 28 },
            { name: 'Healthcare', count: 180, pct: 18 },
            { name: 'Finance', count: 160, pct: 16 },
            { name: 'Education', count: 150, pct: 15 },
            { name: 'Retail', count: 120, pct: 12 },
            { name: 'Other', count: 110, pct: 11 }
        ];
    }, [summary]);

    return (
        <div className="space-y-4">
            <h2 className="text-xl font-semibold text-gray-900 mb-4">Description of the Dataset</h2>

            {/* Total Test Subjects */}
            <ExpandableCard
                title="Total Test Subjects"
                value={summary?.n_participants || 1000}
                icon={<Users size={20} />}
            >
                <div className="space-y-4">
                    <div>
                        <h4 className="font-medium text-gray-700 mb-2">Demographics Summary</h4>
                        <div className="grid grid-cols-3 gap-4">
                            <div className="p-3 bg-indigo-50 rounded-lg text-center">
                                <div className="text-2xl font-bold text-indigo-600">
                                    {demographics?.avgAge || 35}
                                </div>
                                <div className="text-sm text-gray-600">Average Age</div>
                            </div>
                            <div className="p-3 bg-green-50 rounded-lg">
                                <div className="text-sm font-medium text-gray-700 mb-1">Gender Distribution</div>
                                <div className="text-xs space-y-1">
                                    {Object.entries(demographics?.genderDist || {}).map(([g, pct]) => (
                                        <div key={g} className="flex justify-between">
                                            <span>{g}</span>
                                            <span className="font-medium">{pct}%</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                            <div className="p-3 bg-amber-50 rounded-lg">
                                <div className="text-sm font-medium text-gray-700 mb-1">Education Level</div>
                                <div className="text-xs space-y-1">
                                    {Object.entries(demographics?.educationDist || {}).map(([e, pct]) => (
                                        <div key={e} className="flex justify-between">
                                            <span>{e}</span>
                                            <span className="font-medium">{pct}%</span>
                                        </div>
                                    ))}
                                </div>
                            </div>
                        </div>
                    </div>
                </div>
            </ExpandableCard>

            {/* Features Used for Clustering */}
            <ExpandableCard
                title="Features Used for Clustering"
                value={`${featuresByCategory.totalClustering} features`}
                icon={<Brain size={20} />}
            >
                <div className="space-y-4">
                    <p className="text-sm text-gray-600">
                        Features used for clustering test subjects into behavioral personas:
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                        {Object.entries(featuresByCategory.categories).map(([cat, features]) => (
                            <div key={cat} className="border rounded-lg p-3">
                                <h5 className="font-medium text-indigo-700 mb-2 flex items-center gap-2">
                                    <span className="w-2 h-2 rounded-full bg-indigo-500"></span>
                                    {CATEGORY_LABELS[cat]}
                                </h5>
                                <ul className="text-sm space-y-1">
                                    {features.map(f => (
                                        <li key={f.key} className="text-gray-600 flex items-center gap-2">
                                            <CheckCircle size={12} className="text-green-500" />
                                            {f.label}
                                        </li>
                                    ))}
                                </ul>
                            </div>
                        ))}
                    </div>
                </div>
            </ExpandableCard>

            {/* Behavioral Outcomes */}
            <ExpandableCard
                title="Behavioral Outcome Measures"
                value={`${OUTCOME_FEATURES.length} outcomes`}
                icon={<Target size={20} />}
            >
                <div className="space-y-3">
                    <p className="text-sm text-gray-600">
                        Behavioral outcomes measured for each test subject:
                    </p>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        {OUTCOME_FEATURES.map(f => (
                            <div key={f} className="p-3 bg-gray-50 rounded-lg">
                                <div className="text-sm font-medium text-gray-900">
                                    {OUTCOME_LABELS[f]}
                                </div>
                                <div className="text-xs text-gray-500 mt-1">
                                    {f.includes('rate') || f.includes('accuracy') ? 'Percentage' :
                                        f.includes('latency') ? 'Milliseconds' : 'Score'}
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </ExpandableCard>

            {/* Email Factorial Design */}
            <ExpandableCard
                title="Email Stimulus"
                value={emailDesign.description}
                icon={<Mail size={20} />}
            >
                <div className="space-y-4">
                    <p className="text-sm text-gray-600">
                        {emailDesign.totalCombinations} unique email conditions from factorial combination:
                    </p>
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        {emailDesign.factors.map((factor, i) => (
                            <div key={factor.name} className="border rounded-lg p-3">
                                <div className="text-sm font-medium text-gray-900 mb-2">
                                    {factor.name}
                                </div>
                                <div className="flex gap-2">
                                    {factor.levels.map(level => (
                                        <span key={level} className="px-2 py-1 bg-indigo-100 text-indigo-700 text-xs rounded">
                                            {level}
                                        </span>
                                    ))}
                                </div>
                            </div>
                        ))}
                    </div>
                    <div className="p-3 bg-blue-50 rounded-lg">
                        <div className="text-sm text-blue-800">
                            <strong>Design:</strong> 2 (Type) × 2 (Urgency) × 2 (Familiarity) × 2 (Framing) = 16 conditions
                        </div>
                        <div className="text-xs text-blue-600 mt-1">
                            Each test subject responded to all 16 email stimuli
                        </div>
                    </div>
                </div>
            </ExpandableCard>

            {/* Industry Coverage */}
            <ExpandableCard
                title="Industry Coverage"
                value={`${industryData.length} industries`}
                icon={<Building2 size={20} />}
            >
                <div className="space-y-4">
                    <p className="text-sm text-gray-600">
                        Distribution of test subjects across industries:
                    </p>
                    <div className="space-y-2">
                        {industryData.map(ind => (
                            <div key={ind.name} className="flex items-center gap-3">
                                <div className="w-28 text-sm font-medium text-gray-700">
                                    {ind.name}
                                </div>
                                <div className="flex-1 h-6 bg-gray-100 rounded-full overflow-hidden">
                                    <div
                                        className="h-full bg-indigo-500 rounded-full"
                                        style={{ width: `${ind.pct}%` }}
                                    />
                                </div>
                                <div className="w-20 text-right">
                                    <span className="font-medium">{ind.count}</span>
                                    <span className="text-gray-500 text-sm ml-1">({ind.pct}%)</span>
                                </div>
                            </div>
                        ))}
                    </div>
                </div>
            </ExpandableCard>
        </div>
    );
};

export default DatasetDescriptionTab;