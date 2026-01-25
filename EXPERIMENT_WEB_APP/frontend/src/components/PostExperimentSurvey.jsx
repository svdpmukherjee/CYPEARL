/**
 * CYPEARL Post-Experiment Survey Component
 *
 * Single-page questionnaire containing post-experiment state measures:
 * - State Anxiety (STAI-6) - 6 items
 * - Current Stress (PSS-4) - 4 items
 * - Fatigue Level - 1 item
 *
 * Administered AFTER the email experiment to capture current psychological state.
 *
 * References from Data_Collection_Guide.docx
 */

import React, { useState, useEffect } from 'react';
import { AlertCircle, CheckCircle } from 'lucide-react';

// Local storage key for persisting responses
const POST_SURVEY_RESPONSES_KEY = 'post_survey_responses_draft';

// ============================================================================
// SURVEY CONFIGURATION
// ============================================================================

const STAI_SCALE = [
    { value: 1, label: 'Not at all' },
    { value: 2, label: 'Somewhat' },
    { value: 3, label: 'Moderately' },
    { value: 4, label: 'Very much' }
];

const PSS_SCALE = [
    { value: 1, label: 'Never' },
    { value: 2, label: 'Almost never' },
    { value: 3, label: 'Sometimes' },
    { value: 4, label: 'Fairly often' },
    { value: 5, label: 'Very often' }
];

const FATIGUE_SCALE = [
    { value: 1, label: '1 - Not at all fatigued' },
    { value: 2, label: '2' },
    { value: 3, label: '3' },
    { value: 4, label: '4 - Moderately fatigued' },
    { value: 5, label: '5' },
    { value: 6, label: '6' },
    { value: 7, label: '7 - Extremely fatigued' }
];

// ============================================================================
// COMPONENTS
// ============================================================================

const LikertQuestion = ({ id, question, options, value, onChange, required = true }) => (
    <div className="mb-6 p-4 bg-white rounded-lg border border-gray-100">
        <p className="text-gray-800 mb-3">
            {question}
            {required && <span className="text-red-500 ml-1">*</span>}
        </p>
        <div className="flex flex-wrap gap-2">
            {options.map((option) => (
                <label
                    key={option.value}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer transition-colors ${
                        value === option.value
                            ? 'bg-blue-50 border-blue-500 text-blue-700'
                            : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                    }`}
                >
                    <input
                        type="radio"
                        name={id}
                        value={option.value}
                        checked={value === option.value}
                        onChange={() => onChange(id, option.value)}
                        className="sr-only"
                    />
                    <span className="text-sm">{option.label}</span>
                </label>
            ))}
        </div>
    </div>
);

// ============================================================================
// MAIN COMPONENT
// ============================================================================

const PostExperimentSurvey = ({ onComplete, participantId, bonusTotal, onGoToEmail }) => {
    // Load saved responses from localStorage on mount
    const [responses, setResponses] = useState(() => {
        const saved = localStorage.getItem(POST_SURVEY_RESPONSES_KEY);
        return saved ? JSON.parse(saved) : {};
    });
    const [errors, setErrors] = useState([]);
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Save responses to localStorage whenever they change
    useEffect(() => {
        localStorage.setItem(POST_SURVEY_RESPONSES_KEY, JSON.stringify(responses));
    }, [responses]);

    const handleChange = (id, value) => {
        setResponses(prev => ({ ...prev, [id]: value }));
    };

    // ========================================================================
    // VALIDATION & SUBMISSION
    // ========================================================================

    const validateForm = () => {
        const requiredFields = [
            // STAI-6
            'stai_1', 'stai_2', 'stai_3', 'stai_4', 'stai_5', 'stai_6',
            // PSS-4
            'pss_1', 'pss_2', 'pss_3', 'pss_4',
            // Fatigue
            'fatigue'
        ];

        const missing = requiredFields.filter(field =>
            responses[field] === undefined || responses[field] === ''
        );

        return missing;
    };

    const handleSubmit = async () => {
        // TESTING MODE: Validation disabled - uncomment below for production
        /*
        const missing = validateForm();

        if (missing.length > 0) {
            setErrors(missing);
            alert(`Please complete all required fields. ${missing.length} fields are missing.`);
            return;
        }
        */

        setIsSubmitting(true);

        try {
            // Calculate derived scores
            const processedData = calculateScores(responses);
            onComplete(processedData);
        } catch (error) {
            console.error('Error submitting survey:', error);
            alert('Failed to submit survey. Please try again.');
        } finally {
            setIsSubmitting(false);
        }
    };

    // ========================================================================
    // SCORE CALCULATION
    // ========================================================================

    const calculateScores = (data) => {
        // STAI-6 Score (mean after reverse-coding positive items)
        // Items 1, 4, 5 are positive (calm, relaxed, content) - need reverse coding
        // Items 2, 3, 6 are negative (tense, upset, worried) - no reverse
        const stai_items = ['stai_1', 'stai_2', 'stai_3', 'stai_4', 'stai_5', 'stai_6'];
        const stai_reverse = ['stai_1', 'stai_4', 'stai_5'];

        let stai_sum = 0;
        let stai_count = 0;
        stai_items.forEach(item => {
            let value = parseInt(data[item]);
            if (!isNaN(value)) {
                if (stai_reverse.includes(item)) {
                    value = 5 - value; // Reverse for 4-point scale (1→4, 2→3, 3→2, 4→1)
                }
                stai_sum += value;
                stai_count++;
            }
        });
        const state_anxiety = stai_count > 0 ? round2(stai_sum / stai_count) : 0;

        // PSS-4 Score (mean after reverse-coding positive items)
        // Items 2, 3 are positive - need reverse coding
        // Items 1, 4 are negative - no reverse
        const pss_items = ['pss_1', 'pss_2', 'pss_3', 'pss_4'];
        const pss_reverse = ['pss_2', 'pss_3'];

        let pss_sum = 0;
        let pss_count = 0;
        pss_items.forEach(item => {
            let value = parseInt(data[item]);
            if (!isNaN(value)) {
                if (pss_reverse.includes(item)) {
                    value = 6 - value; // Reverse for 5-point scale
                }
                pss_sum += value;
                pss_count++;
            }
        });
        const current_stress = pss_count > 0 ? round2(pss_sum / pss_count) : 0;

        // Fatigue Level (single item, 1-7 scale) - default to 4 (middle) if not provided
        const fatigue_raw = parseInt(data.fatigue);
        const fatigue_level = !isNaN(fatigue_raw) ? fatigue_raw : 4;

        return {
            state_anxiety,
            current_stress,
            fatigue_level,
            raw_responses: data,
            completed_at: new Date().toISOString()
        };
    };

    const round2 = (num) => Math.round(num * 100) / 100;

    // ========================================================================
    // RENDER
    // ========================================================================

    return (
        <div className="min-h-screen bg-gray-50 py-8 px-4">
            <div className="max-w-3xl mx-auto">
                {/* Header */}
                <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
                    <div className="flex items-center gap-3 mb-4">
                        <CheckCircle className="w-8 h-8 text-green-600" />
                        <h1 className="text-2xl font-bold text-green-700">Email Evaluation Complete!</h1>
                    </div>
                    <p className="text-gray-600 mb-4">
                        Thank you for completing the email evaluation task. You have reviewed all 16 emails.
                    </p>

                    {bonusTotal !== undefined && (
                        <div className="p-4 bg-green-50 rounded-lg border border-green-200 mb-4">
                            <p className="text-green-800 font-medium">
                                Your performance bonus: <span className="text-xl">{bonusTotal >= 0 ? '+' : ''}{bonusTotal} points</span>
                            </p>
                            <p className="text-sm text-green-700 mt-1">
                                This bonus was calculated based on your link-clicking decisions during the study.
                            </p>
                        </div>
                    )}

                    <div className="p-3 bg-blue-50 rounded-lg flex items-start gap-2">
                        <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5" />
                        <p className="text-sm text-blue-800">
                            Please complete this final brief questionnaire about how you're feeling right now.
                            This takes approximately 2-3 minutes.
                        </p>
                    </div>
                </div>

                {/* Section 1: State Anxiety (STAI-6) */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <div className="p-4 bg-gray-50 border-b border-gray-200">
                        <h3 className="font-semibold text-gray-800">1. Current Feelings</h3>
                        <p className="text-sm text-gray-500">Rate how you feel RIGHT NOW</p>
                    </div>
                    <div className="p-6">
                        <LikertQuestion
                            id="stai_1"
                            question="I feel calm."
                            options={STAI_SCALE}
                            value={responses.stai_1}
                            onChange={handleChange}
                        />
                        <LikertQuestion
                            id="stai_2"
                            question="I am tense."
                            options={STAI_SCALE}
                            value={responses.stai_2}
                            onChange={handleChange}
                        />
                        <LikertQuestion
                            id="stai_3"
                            question="I feel upset."
                            options={STAI_SCALE}
                            value={responses.stai_3}
                            onChange={handleChange}
                        />
                        <LikertQuestion
                            id="stai_4"
                            question="I am relaxed."
                            options={STAI_SCALE}
                            value={responses.stai_4}
                            onChange={handleChange}
                        />
                        <LikertQuestion
                            id="stai_5"
                            question="I feel content."
                            options={STAI_SCALE}
                            value={responses.stai_5}
                            onChange={handleChange}
                        />
                        <LikertQuestion
                            id="stai_6"
                            question="I am worried."
                            options={STAI_SCALE}
                            value={responses.stai_6}
                            onChange={handleChange}
                        />
                    </div>
                </div>

                {/* Section 2: Perceived Stress (PSS-4) */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <div className="p-4 bg-gray-50 border-b border-gray-200">
                        <h3 className="font-semibold text-gray-800">2. Recent Experiences</h3>
                        <p className="text-sm text-gray-500">In the last hour during this study...</p>
                    </div>
                    <div className="p-6">
                        <LikertQuestion
                            id="pss_1"
                            question="How often have you felt that you were unable to control the important things?"
                            options={PSS_SCALE}
                            value={responses.pss_1}
                            onChange={handleChange}
                        />
                        <LikertQuestion
                            id="pss_2"
                            question="How often have you felt confident about your ability to handle problems?"
                            options={PSS_SCALE}
                            value={responses.pss_2}
                            onChange={handleChange}
                        />
                        <LikertQuestion
                            id="pss_3"
                            question="How often have you felt that things were going your way?"
                            options={PSS_SCALE}
                            value={responses.pss_3}
                            onChange={handleChange}
                        />
                        <LikertQuestion
                            id="pss_4"
                            question="How often have you felt difficulties were piling up so high that you could not overcome them?"
                            options={PSS_SCALE}
                            value={responses.pss_4}
                            onChange={handleChange}
                        />
                    </div>
                </div>

                {/* Section 3: Fatigue Level */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <div className="p-4 bg-gray-50 border-b border-gray-200">
                        <h3 className="font-semibold text-gray-800">3. Mental Fatigue</h3>
                        <p className="text-sm text-gray-500">Your current energy level</p>
                    </div>
                    <div className="p-6">
                        <LikertQuestion
                            id="fatigue"
                            question="How mentally fatigued or tired do you feel right now?"
                            options={FATIGUE_SCALE}
                            value={responses.fatigue}
                            onChange={handleChange}
                        />
                    </div>
                </div>

                {/* Submit Button */}
                <div className="bg-white rounded-lg shadow-sm p-6">
                    <div className="flex items-center justify-between">
                        <div className="text-gray-600">
                            <p className="font-medium">Almost done!</p>
                            <p className="text-sm">Click the button to complete the study.</p>
                        </div>
                        <button
                            onClick={handleSubmit}
                            disabled={isSubmitting}
                            className={`px-8 py-3 rounded-lg font-semibold transition-colors ${
                                isSubmitting
                                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                    : 'bg-green-600 text-white hover:bg-green-700'
                            }`}
                        >
                            {isSubmitting ? 'Submitting...' : 'Complete Study'}
                        </button>
                    </div>
                </div>
            </div>

            {/* TESTING ONLY: Navigation button to go back to email */}
            {onGoToEmail && (
                <div className="fixed bottom-0 left-0 right-0 px-4 py-2 flex justify-start items-center z-50 pointer-events-none">
                    <button
                        onClick={onGoToEmail}
                        className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium pointer-events-auto"
                    >
                        ← Go to Email Experiment
                    </button>
                </div>
            )}
        </div>
    );
};

export default PostExperimentSurvey;
