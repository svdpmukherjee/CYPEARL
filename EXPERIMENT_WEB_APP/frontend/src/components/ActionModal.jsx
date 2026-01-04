/**
 * ActionModal Component - Updated for CYPEARL Experiment
 * 
 * Collects both confidence and suspicion ratings as required by
 * phishing_study_responses.csv
 * 
 * Changes:
 * - Added suspicion_rating slider (1-10)
 * - Restructured to separate observational prompts from self-report
 * - Better validation before submit
 */

import React, { useState, useEffect } from 'react';

const ActionModal = ({ isOpen, onClose, onSubmit, actionType }) => {
    // Qualitative self-report fields
    const [detailsNoticed, setDetailsNoticed] = useState('');
    const [stepsTaken, setStepsTaken] = useState('');
    const [decisionReason, setDecisionReason] = useState('');
    const [confidenceReason, setConfidenceReason] = useState('');
    const [unsureAbout, setUnsureAbout] = useState('');

    // Numeric ratings (REQUIRED for CSV)
    const [confidence, setConfidence] = useState(5);
    const [suspicion, setSuspicion] = useState(5);  // NEW: Required field

    // Reset form when modal opens
    useEffect(() => {
        if (isOpen) {
            setDetailsNoticed('');
            setStepsTaken('');
            setDecisionReason('');
            setConfidenceReason('');
            setUnsureAbout('');
            setConfidence(5);
            setSuspicion(5);
        }
    }, [isOpen]);

    if (!isOpen) return null;

    const handleSubmit = (e) => {
        e.preventDefault();

        // Combine qualitative fields into structured reason
        const combinedReason = [
            `DETAILS_NOTICED: ${detailsNoticed}`,
            `STEPS_TAKEN: ${stepsTaken}`,
            `DECISION_REASON: ${decisionReason}`,
            `CONFIDENCE_REASON: ${confidenceReason}`,
            `UNSURE_ABOUT: ${unsureAbout}`
        ].join('\n');

        // Submit with both ratings
        onSubmit({
            reason: combinedReason,
            confidence,
            suspicion  // NEW: Include suspicion rating
        });
    };

    const getDecisionPrompt = () => {
        switch (actionType) {
            case 'safe':
                return "I decided to mark this email as safe because...";
            case 'report':
                return "I decided to report this email as suspicious because...";
            case 'delete':
                return "I decided to delete this email because...";
            case 'ignore':
                return "I decided to ignore this email because...";
            default:
                return "I decided to take this action because...";
        }
    };

    const getActionTitle = () => {
        switch (actionType) {
            case 'report': return 'Report Email as Suspicious';
            case 'delete': return 'Delete Email';
            case 'safe': return 'Mark as Safe';
            case 'ignore': return 'Ignore Email';
            default: return 'Action Confirmation';
        }
    };

    const getActionColor = () => {
        switch (actionType) {
            case 'report': return 'red';
            case 'delete': return 'gray';
            case 'safe': return 'green';
            case 'ignore': return 'yellow';
            default: return 'blue';
        }
    };

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 overflow-y-auto py-4">
            <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl p-8 my-auto max-h-[90vh] overflow-y-auto">
                {/* Header */}
                <div className="mb-6">
                    <h2 className="text-2xl font-semibold text-gray-900">
                        {getActionTitle()}
                    </h2>
                    <p className="text-gray-600 mt-2">
                        Please answer the following questions to help us understand your decision.
                    </p>
                </div>

                <form onSubmit={handleSubmit} className="space-y-5">
                    {/* Section 1: What You Noticed */}
                    <div className="bg-gray-50 p-4 rounded-lg space-y-4">
                        <h3 className="font-semibold text-gray-800 text-sm uppercase tracking-wide">
                            Your Observations
                        </h3>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                The specific details I noticed...
                            </label>
                            <input
                                type="text"
                                className="w-full border border-gray-300 rounded-md p-3 focus:ring-blue-500 focus:border-blue-500"
                                value={detailsNoticed}
                                onChange={(e) => setDetailsNoticed(e.target.value)}
                                placeholder="e.g., sender address, urgent language, links..."
                                required
                                autoFocus
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                Steps I took to evaluate this email...
                            </label>
                            <input
                                type="text"
                                className="w-full border border-gray-300 rounded-md p-3 focus:ring-blue-500 focus:border-blue-500"
                                value={stepsTaken}
                                onChange={(e) => setStepsTaken(e.target.value)}
                                placeholder="e.g., checked the sender, hovered over links..."
                                required
                            />
                        </div>
                    </div>

                    {/* Section 2: Your Decision */}
                    <div className="bg-blue-50 p-4 rounded-lg space-y-4">
                        <h3 className="font-semibold text-gray-800 text-sm uppercase tracking-wide">
                            Your Decision
                        </h3>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                {getDecisionPrompt()}
                            </label>
                            <input
                                type="text"
                                className="w-full border border-gray-300 rounded-md p-3 focus:ring-blue-500 focus:border-blue-500"
                                value={decisionReason}
                                onChange={(e) => setDecisionReason(e.target.value)}
                                placeholder="Explain your reasoning..."
                                required
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                What made me confident in this decision...
                            </label>
                            <input
                                type="text"
                                className="w-full border border-gray-300 rounded-md p-3 focus:ring-blue-500 focus:border-blue-500"
                                value={confidenceReason}
                                onChange={(e) => setConfidenceReason(e.target.value)}
                                placeholder="What confirmed your decision?"
                                required
                            />
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-1">
                                What I was unsure about...
                            </label>
                            <input
                                type="text"
                                className="w-full border border-gray-300 rounded-md p-3 focus:ring-blue-500 focus:border-blue-500"
                                value={unsureAbout}
                                onChange={(e) => setUnsureAbout(e.target.value)}
                                placeholder="e.g., nothing, or I wasn't sure about the logo..."
                                required
                            />
                        </div>
                    </div>

                    {/* Section 3: Ratings (REQUIRED) */}
                    <div className="bg-purple-50 p-4 rounded-lg space-y-6">
                        <h3 className="font-semibold text-gray-800 text-sm uppercase tracking-wide">
                            Rate Your Response
                        </h3>

                        {/* Suspicion Rating (NEW - REQUIRED) */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                How suspicious did this email seem to you? (1-10)
                            </label>
                            <div className="flex items-center space-x-4">
                                <input
                                    type="range"
                                    min="1"
                                    max="10"
                                    value={suspicion}
                                    onChange={(e) => setSuspicion(parseInt(e.target.value))}
                                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-purple-600"
                                />
                                <span className="font-bold text-purple-600 w-8 text-center text-lg bg-white px-2 py-1 rounded">
                                    {suspicion}
                                </span>
                            </div>
                            <div className="flex justify-between text-xs text-gray-500 mt-1">
                                <span>Not suspicious at all</span>
                                <span>Extremely suspicious</span>
                            </div>
                        </div>

                        {/* Confidence Rating */}
                        <div>
                            <label className="block text-sm font-medium text-gray-700 mb-2">
                                How confident are you in your decision? (1-10)
                            </label>
                            <div className="flex items-center space-x-4">
                                <input
                                    type="range"
                                    min="1"
                                    max="10"
                                    value={confidence}
                                    onChange={(e) => setConfidence(parseInt(e.target.value))}
                                    className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                                />
                                <span className="font-bold text-blue-600 w-8 text-center text-lg bg-white px-2 py-1 rounded">
                                    {confidence}
                                </span>
                            </div>
                            <div className="flex justify-between text-xs text-gray-500 mt-1">
                                <span>Very unsure</span>
                                <span>Very certain</span>
                            </div>
                        </div>
                    </div>

                    {/* Actions */}
                    <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200">
                        <button
                            type="button"
                            onClick={onClose}
                            className="px-6 py-2.5 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 font-medium transition-colors"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            className={`px-6 py-2.5 rounded-md text-white font-medium shadow-sm transition-colors
                                ${actionType === 'report' ? 'bg-red-600 hover:bg-red-700' : ''}
                                ${actionType === 'delete' ? 'bg-gray-600 hover:bg-gray-700' : ''}
                                ${actionType === 'safe' ? 'bg-green-600 hover:bg-green-700' : ''}
                                ${actionType === 'ignore' ? 'bg-yellow-500 hover:bg-yellow-600' : ''}
                                ${!['report', 'delete', 'safe', 'ignore'].includes(actionType) ? 'bg-blue-600 hover:bg-blue-700' : ''}
                            `}
                        >
                            Submit Response
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default ActionModal;