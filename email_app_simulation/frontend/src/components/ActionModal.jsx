import React, { useState, useEffect } from 'react';

const ActionModal = ({ isOpen, onClose, onSubmit, actionType }) => {
    // 5 State variables for the 5 questions
    const [detailsNoticed, setDetailsNoticed] = useState('');
    const [stepsTaken, setStepsTaken] = useState('');
    const [decisionReason, setDecisionReason] = useState('');
    const [confidenceReason, setConfidenceReason] = useState('');
    const [unsureAbout, setUnsureAbout] = useState('');

    const [confidence, setConfidence] = useState(5);

    // Reset form when modal opens
    useEffect(() => {
        if (isOpen) {
            setDetailsNoticed('');
            setStepsTaken('');
            setDecisionReason('');
            setConfidenceReason('');
            setUnsureAbout('');
            setConfidence(5);
        }
    }, [isOpen]);

    if (!isOpen) return null;

    const handleSubmit = (e) => {
        e.preventDefault();
        // Combine the 5 fields into a structured string
        const combinedReason = `DETAILS_NOTICED: ${detailsNoticed}\nSTEPS_TAKEN: ${stepsTaken}\nDECISION_REASON: ${decisionReason}\nCONFIDENCE_REASON: ${confidenceReason}\nUNSURE_ABOUT: ${unsureAbout}`;
        onSubmit({ reason: combinedReason, confidence });
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

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 overflow-y-auto py-4">
            <div className="bg-white rounded-lg shadow-xl w-full max-w-2xl p-8 my-auto">
                <h2 className="text-2xl font-semibold mb-2">
                    {actionType === 'report' ? 'Report Email' :
                        actionType === 'delete' ? 'Delete Email' :
                            actionType === 'safe' ? 'Mark as Safe' :
                                'Action Confirmation'}
                </h2>
                <p className="text-gray-600 mb-6">
                    Please answer the following questions to help us understand your decision.
                </p>

                <form onSubmit={handleSubmit} className="space-y-5">
                    {/* Question 1 */}
                    <div>
                        <label className="block text-sm font-bold text-gray-700 mb-1">
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

                    {/* Question 2 */}
                    <div>
                        <label className="block text-sm font-bold text-gray-700 mb-1">
                            What steps did I take to evaluate this email?
                        </label>
                        <input
                            type="text"
                            className="w-full border border-gray-300 rounded-md p-3 focus:ring-blue-500 focus:border-blue-500"
                            value={stepsTaken}
                            onChange={(e) => setStepsTaken(e.target.value)}
                            placeholder="e.g., checked the sender, hovered over links, compared with policy..."
                            required
                        />
                    </div>

                    {/* Question 3 (Dynamic) */}
                    <div>
                        <label className="block text-sm font-bold text-gray-700 mb-1">
                            {getDecisionPrompt()}
                        </label>
                        <input
                            type="text"
                            className="w-full border border-gray-300 rounded-md p-3 focus:ring-blue-500 focus:border-blue-500"
                            value={decisionReason}
                            onChange={(e) => setDecisionReason(e.target.value)}
                            required
                        />
                    </div>

                    {/* Question 4 */}
                    <div>
                        <label className="block text-sm font-bold text-gray-700 mb-1">
                            What made me confident in this decision...
                        </label>
                        <input
                            type="text"
                            className="w-full border border-gray-300 rounded-md p-3 focus:ring-blue-500 focus:border-blue-500"
                            value={confidenceReason}
                            onChange={(e) => setConfidenceReason(e.target.value)}
                            required
                        />
                    </div>

                    {/* Question 5 */}
                    <div>
                        <label className="block text-sm font-bold text-gray-700 mb-1">
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

                    <div className="pt-4 border-t border-gray-100">
                        <label className="block text-sm font-bold text-gray-700 mb-2">
                            I am confident in my decision (1-10)
                        </label>
                        <div className="flex items-center space-x-4">
                            <input
                                type="range"
                                min="1"
                                max="10"
                                value={confidence}
                                onChange={(e) => setConfidence(parseInt(e.target.value))}
                                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                            />
                            <span className="font-bold text-blue-600 w-8 text-center text-lg">{confidence}</span>
                        </div>
                        <div className="flex justify-between text-xs text-gray-500 mt-1">
                            <span>Unsure</span>
                            <span>Certain</span>
                        </div>
                    </div>

                    <div className="flex justify-end space-x-3 pt-4">
                        <button
                            type="button"
                            onClick={onClose}
                            className="px-6 py-2.5 border border-gray-300 rounded-md text-gray-700 hover:bg-gray-50 font-medium"
                        >
                            Cancel
                        </button>
                        <button
                            type="submit"
                            className="px-6 py-2.5 bg-blue-600 text-white rounded-md hover:bg-blue-700 font-medium shadow-sm"
                        >
                            Submit
                        </button>
                    </div>
                </form>
            </div>
        </div>
    );
};

export default ActionModal;
