/**
 * CYPEARL Post-Experiment Survey Component
 *
 * Administered AFTER the email experiment. Contains:
 *
 * Part A — Experience measures (original post-experiment items):
 * - Mental Fatigue (1 item, 1-7 scale)
 * - Perceived Realism (1 item, 1-7 scale)
 * - Perceived Work Relevance Global (1 item, 1-7 scale)
 * - Count of Relevant Emails (1 item, slider 0-16)
 * - Open-text Feedback (optional)
 *
 * Part B — Phishing-specific measures (moved from pre-survey to avoid priming):
 * - Phishing Self-Efficacy (4 items)
 * - Perceived Risk (4 items)
 * - Security Attitudes SA-6 (6 items)
 * - Phishing Knowledge Quiz (10 items)
 * - Prior Experience (3 items)
 * - Link-Click Tendency (1 item, moved from Email Habits)
 *
 * References from Validated_Instruments_Complete.xlsx
 */

import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp, AlertCircle, CheckCircle } from 'lucide-react';

// Local storage key for persisting responses
const POST_SURVEY_RESPONSES_KEY = 'post_survey_responses_draft';

// ============================================================================
// SURVEY CONFIGURATION
// ============================================================================

const FATIGUE_SCALE = [
    { value: 1, label: '1 - Not at all fatigued' },
    { value: 2, label: '2' },
    { value: 3, label: '3' },
    { value: 4, label: '4 - Moderately fatigued' },
    { value: 5, label: '5' },
    { value: 6, label: '6' },
    { value: 7, label: '7 - Extremely fatigued' }
];

const REALISM_SCALE = [
    { value: 1, label: '1 - Not at all realistic' },
    { value: 2, label: '2' },
    { value: 3, label: '3' },
    { value: 4, label: '4 - Moderately realistic' },
    { value: 5, label: '5' },
    { value: 6, label: '6' },
    { value: 7, label: '7 - Extremely realistic' }
];

const RELEVANCE_SCALE = [
    { value: 1, label: '1 - Not at all relevant' },
    { value: 2, label: '2' },
    { value: 3, label: '3' },
    { value: 4, label: '4 - Moderately relevant' },
    { value: 5, label: '5' },
    { value: 6, label: '6' },
    { value: 7, label: '7 - Extremely relevant' }
];

const LIKERT_5_AGREE = [
    { value: 1, label: 'Strongly Disagree' },
    { value: 2, label: 'Disagree' },
    { value: 3, label: 'Neutral' },
    { value: 4, label: 'Agree' },
    { value: 5, label: 'Strongly Agree' }
];

// ============================================================================
// SECTION COMPONENTS
// ============================================================================

const SectionHeader = ({ title, description, isOpen, onToggle, isComplete }) => (
    <div
        className="flex items-center justify-between p-4 bg-gray-50 border border-gray-200 rounded-lg cursor-pointer hover:bg-gray-100 transition-colors"
        onClick={onToggle}
    >
        <div className="flex items-center gap-3">
            {isComplete ? (
                <CheckCircle className="w-5 h-5 text-green-600" />
            ) : (
                <div className="w-5 h-5 rounded-full border-2 border-gray-300" />
            )}
            <div>
                <h3 className="font-semibold text-gray-800">{title}</h3>
                <p className="text-sm text-gray-500">{description}</p>
            </div>
        </div>
        {isOpen ? <ChevronUp className="w-5 h-5 text-gray-500" /> : <ChevronDown className="w-5 h-5 text-gray-500" />}
    </div>
);

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

const SelectQuestion = ({ id, question, options, value, onChange, required = true }) => (
    <div className="mb-4">
        <label className="block text-gray-800 mb-2">
            {question}
            {required && <span className="text-red-500 ml-1">*</span>}
        </label>
        <select
            value={value || ''}
            onChange={(e) => onChange(id, e.target.value)}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
            <option value="">Select...</option>
            {options.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
            ))}
        </select>
    </div>
);

const TextInputQuestion = ({ id, question, value, onChange, type = "text", placeholder = "", required = true }) => (
    <div className="mb-4">
        <label className="block text-gray-800 mb-2">
            {question}
            {required && <span className="text-red-500 ml-1">*</span>}
        </label>
        <input
            type={type}
            value={value || ''}
            onChange={(e) => onChange(id, type === 'number' ? parseInt(e.target.value) || '' : e.target.value)}
            placeholder={placeholder}
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
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
    const [openSections, setOpenSections] = useState({ experience: true });
    const [errors, setErrors] = useState([]);
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Save responses to localStorage whenever they change
    useEffect(() => {
        localStorage.setItem(POST_SURVEY_RESPONSES_KEY, JSON.stringify(responses));
    }, [responses]);

    const handleChange = (id, value) => {
        setResponses(prev => ({ ...prev, [id]: value }));
    };

    const toggleSection = (section) => {
        setOpenSections(prev => ({ ...prev, [section]: !prev[section] }));
    };

    const isSectionComplete = (requiredFields) => {
        return requiredFields.every(field => responses[field] !== undefined && responses[field] !== '');
    };

    // ========================================================================
    // VALIDATION & SUBMISSION
    // ========================================================================

    const validateForm = () => {
        const requiredFields = [
            // Experience measures
            'fatigue', 'perceived_realism', 'perceived_work_relevance_global', 'emails_felt_relevant_count',
            // Phishing Self-Efficacy
            'pse_1', 'pse_2', 'pse_3', 'pse_4',
            // Perceived Risk
            'pr_1', 'pr_2', 'pr_3', 'pr_4',
            // Security Attitudes SA-6
            'sa_1', 'sa_2', 'sa_3', 'sa_4', 'sa_5', 'sa_6',
            // Phishing Knowledge
            'pk_1', 'pk_2', 'pk_3', 'pk_4', 'pk_5', 'pk_6', 'pk_7', 'pk_8', 'pk_9', 'pk_10',
            // Prior Experience
            'prior_victimization', 'security_training', 'years_email_use',
            // Link-Click Tendency
            'link_click_tendency'
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

    const calculateMean = (data, items, reverseItems) => {
        let sum = 0;
        let count = 0;
        items.forEach(item => {
            if (data[item] !== undefined) {
                let value = parseInt(data[item]);
                if (reverseItems.includes(item)) {
                    value = 6 - value; // Reverse for 5-point scale
                }
                sum += value;
                count++;
            }
        });
        return count > 0 ? sum / count : 0;
    };

    const round2 = (num) => Math.round(num * 100) / 100;

    const calculateScores = (data) => {
        // Experience measures
        const fatigue_raw = parseInt(data.fatigue);
        const fatigue_level = !isNaN(fatigue_raw) ? fatigue_raw : 4;

        const realism_raw = parseInt(data.perceived_realism);
        const perceived_realism = !isNaN(realism_raw) ? realism_raw : null;

        const relevance_raw = parseInt(data.perceived_work_relevance_global);
        const perceived_work_relevance_global = !isNaN(relevance_raw) ? relevance_raw : null;

        const count_raw = parseInt(data.emails_felt_relevant_count);
        const emails_felt_relevant_count = !isNaN(count_raw) ? count_raw : null;

        const open_text_feedback = data.open_text_feedback || '';

        // Phishing Self-Efficacy (mean of 4 items)
        const phishing_self_efficacy = calculateMean(data, ['pse_1', 'pse_2', 'pse_3', 'pse_4'], []);

        // Perceived Risk (mean of 4 items)
        const perceived_risk = calculateMean(data, ['pr_1', 'pr_2', 'pr_3', 'pr_4'], []);

        // Security Attitudes SA-6 (mean of 6 items)
        const security_attitudes = calculateMean(data, ['sa_1', 'sa_2', 'sa_3', 'sa_4', 'sa_5', 'sa_6'], []);

        // Phishing Knowledge (count correct out of 10)
        const pk_correct = { pk_1: 2, pk_2: 1, pk_3: 3, pk_4: 2, pk_5: 1, pk_6: 2, pk_7: 3, pk_8: 1, pk_9: 2, pk_10: 3 };
        let phishing_knowledge = 0;
        Object.keys(pk_correct).forEach(key => {
            if (parseInt(data[key]) === pk_correct[key]) phishing_knowledge++;
        });

        return {
            // Experience measures (original post-experiment)
            fatigue_level,
            perceived_realism,
            perceived_work_relevance_global,
            emails_felt_relevant_count,
            open_text_feedback,

            // Security attitudes (moved from pre-survey)
            phishing_self_efficacy: round2(phishing_self_efficacy),
            perceived_risk: round2(perceived_risk),
            security_attitudes: round2(security_attitudes),

            // Knowledge & experience (moved from pre-survey)
            phishing_knowledge,
            prior_victimization: data.prior_victimization,
            security_training: data.security_training === 'yes' ? 1 : 0,
            years_email_use: parseInt(data.years_email_use),

            // Link-click tendency (moved from pre-survey Email Habits)
            link_click_tendency: round2(parseFloat(data.link_click_tendency)),

            // Store raw responses for debugging
            raw_responses: data,
            completed_at: new Date().toISOString()
        };
    };

    // ========================================================================
    // RENDER
    // ========================================================================

    return (
        <div className="min-h-screen bg-gray-50 py-8 px-4">
            <div className="max-w-4xl mx-auto">
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
                            Please complete this final questionnaire about your experience and background.
                            This takes approximately 10-12 minutes.
                        </p>
                    </div>
                </div>

                {/* ============================================================ */}
                {/* PART A: Experience Measures (original post-experiment items)  */}
                {/* ============================================================ */}

                <div className="bg-white rounded-lg shadow-sm p-4 mb-4">
                    <h2 className="text-lg font-bold text-gray-700">Part A: Your Experience</h2>
                    <p className="text-sm text-gray-500">Questions about the email task you just completed</p>
                </div>

                {/* Section 1: Mental Fatigue + Perceived Realism + Relevance */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="1. Experience & Perceptions"
                        description="How the email task felt"
                        isOpen={openSections.experience}
                        onToggle={() => toggleSection('experience')}
                        isComplete={isSectionComplete(['fatigue', 'perceived_realism', 'perceived_work_relevance_global', 'emails_felt_relevant_count'])}
                    />
                    {openSections.experience && (
                        <div className="p-6">
                            <LikertQuestion
                                id="fatigue"
                                question="How mentally fatigued or tired do you feel right now?"
                                options={FATIGUE_SCALE}
                                value={responses.fatigue}
                                onChange={handleChange}
                            />
                            <LikertQuestion
                                id="perceived_realism"
                                question="How realistic did the email inbox experience feel?"
                                options={REALISM_SCALE}
                                value={responses.perceived_realism}
                                onChange={handleChange}
                            />
                            <LikertQuestion
                                id="perceived_work_relevance_global"
                                question="Overall, how relevant were the emails to the types you typically receive in your actual job?"
                                options={RELEVANCE_SCALE}
                                value={responses.perceived_work_relevance_global}
                                onChange={handleChange}
                            />
                            <div className="mb-6 p-4 bg-white rounded-lg border border-gray-100">
                                <p className="text-gray-800 mb-3">
                                    How many of the 16 emails felt directly related to your daily work?
                                    <span className="text-red-500 ml-1">*</span>
                                </p>
                                <div className="flex items-center space-x-4">
                                    <input
                                        type="range"
                                        min="0"
                                        max="16"
                                        value={responses.emails_felt_relevant_count ?? 8}
                                        onChange={(e) => handleChange('emails_felt_relevant_count', parseInt(e.target.value))}
                                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-blue-600"
                                    />
                                    <span className="font-bold text-blue-600 w-8 text-center text-lg bg-white px-2 py-1 rounded border border-gray-200">
                                        {responses.emails_felt_relevant_count ?? 8}
                                    </span>
                                </div>
                                <div className="flex justify-between text-xs text-gray-500 mt-1">
                                    <span>0</span>
                                    <span>16</span>
                                </div>
                            </div>
                        </div>
                    )}
                </div>

                {/* ============================================================ */}
                {/* PART B: Individual-Difference Measures (moved from pre-survey)*/}
                {/* ============================================================ */}

                <div className="bg-white rounded-lg shadow-sm p-4 mb-4">
                    <h2 className="text-lg font-bold text-gray-700">Part B: About You</h2>
                    <p className="text-sm text-gray-500">Questions about your attitudes, knowledge, and experience</p>
                </div>

                {/* Section 2: Phishing Self-Efficacy */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="2. Email Detection Confidence"
                        description="How confident you feel about identifying suspicious emails"
                        isOpen={openSections.pse}
                        onToggle={() => toggleSection('pse')}
                        isComplete={isSectionComplete(['pse_1', 'pse_2', 'pse_3', 'pse_4'])}
                    />
                    {openSections.pse && (
                        <div className="p-6">
                            <LikertQuestion id="pse_1" question="I am confident I can identify a phishing email." options={LIKERT_5_AGREE} value={responses.pse_1} onChange={handleChange} />
                            <LikertQuestion id="pse_2" question="I know what to look for to detect suspicious emails." options={LIKERT_5_AGREE} value={responses.pse_2} onChange={handleChange} />
                            <LikertQuestion id="pse_3" question="I can protect myself from email scams." options={LIKERT_5_AGREE} value={responses.pse_3} onChange={handleChange} />
                            <LikertQuestion id="pse_4" question="I would recognize if an email was trying to trick me." options={LIKERT_5_AGREE} value={responses.pse_4} onChange={handleChange} />
                        </div>
                    )}
                </div>

                {/* Section 3: Perceived Risk */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="3. Perceived Risk"
                        description="Your perception of email threats"
                        isOpen={openSections.pr}
                        onToggle={() => toggleSection('pr')}
                        isComplete={isSectionComplete(['pr_1', 'pr_2', 'pr_3', 'pr_4'])}
                    />
                    {openSections.pr && (
                        <div className="p-6">
                            <LikertQuestion id="pr_1" question="I believe phishing attacks are a serious threat." options={LIKERT_5_AGREE} value={responses.pr_1} onChange={handleChange} />
                            <LikertQuestion id="pr_2" question="Falling for a phishing email could cause me significant harm." options={LIKERT_5_AGREE} value={responses.pr_2} onChange={handleChange} />
                            <LikertQuestion id="pr_3" question="The consequences of clicking a phishing link are severe." options={LIKERT_5_AGREE} value={responses.pr_3} onChange={handleChange} />
                            <LikertQuestion id="pr_4" question="I am likely to receive phishing emails." options={LIKERT_5_AGREE} value={responses.pr_4} onChange={handleChange} />
                        </div>
                    )}
                </div>

                {/* Section 4: Security Attitudes SA-6 */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="4. Security Attitudes"
                        description="SA-6 Scale"
                        isOpen={openSections.sa}
                        onToggle={() => toggleSection('sa')}
                        isComplete={isSectionComplete(['sa_1', 'sa_2', 'sa_3', 'sa_4', 'sa_5', 'sa_6'])}
                    />
                    {openSections.sa && (
                        <div className="p-6">
                            <LikertQuestion id="sa_1" question="I seek out information about security threats." options={LIKERT_5_AGREE} value={responses.sa_1} onChange={handleChange} />
                            <LikertQuestion id="sa_2" question="I am extremely motivated to take all the steps needed to keep my online accounts secure." options={LIKERT_5_AGREE} value={responses.sa_2} onChange={handleChange} />
                            <LikertQuestion id="sa_3" question="Generally, I diligently follow a routine about security practices." options={LIKERT_5_AGREE} value={responses.sa_3} onChange={handleChange} />
                            <LikertQuestion id="sa_4" question="I always pay attention to experts' advice about the steps I need to take to keep my data safe." options={LIKERT_5_AGREE} value={responses.sa_4} onChange={handleChange} />
                            <LikertQuestion id="sa_5" question="I am extremely knowledgeable about all the steps needed to keep my online accounts secure." options={LIKERT_5_AGREE} value={responses.sa_5} onChange={handleChange} />
                            <LikertQuestion id="sa_6" question="I often am interested in articles about security threats." options={LIKERT_5_AGREE} value={responses.sa_6} onChange={handleChange} />
                        </div>
                    )}
                </div>

                {/* Section 5: Phishing Knowledge Quiz */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="5. Email Security Knowledge"
                        description="10-item quiz"
                        isOpen={openSections.pk}
                        onToggle={() => toggleSection('pk')}
                        isComplete={isSectionComplete(['pk_1', 'pk_2', 'pk_3', 'pk_4', 'pk_5', 'pk_6', 'pk_7', 'pk_8', 'pk_9', 'pk_10'])}
                    />
                    {openSections.pk && (
                        <div className="p-6">
                            <p className="text-gray-600 mb-4">Answer these questions about phishing and email security.</p>

                            <SelectQuestion
                                id="pk_1"
                                question="Which of the following is a common indicator of a phishing email?"
                                options={[
                                    { value: 1, label: 'A personalized greeting using your name' },
                                    { value: 2, label: 'Urgent requests for immediate action' },
                                    { value: 3, label: 'An email from your company\'s IT department' }
                                ]}
                                value={responses.pk_1}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="pk_2"
                                question="What should you do if you receive an email asking you to click a link to verify your account?"
                                options={[
                                    { value: 1, label: 'Go directly to the company\'s website by typing the URL yourself' },
                                    { value: 2, label: 'Click the link if the email looks legitimate' },
                                    { value: 3, label: 'Reply to the email asking for more information' }
                                ]}
                                value={responses.pk_2}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="pk_3"
                                question="Which URL is most likely to be a phishing attempt?"
                                options={[
                                    { value: 1, label: 'https://www.paypal.com/login' },
                                    { value: 2, label: 'https://secure.paypal.com.verify-account.net' },
                                    { value: 3, label: 'https://paypal.com/account/security' }
                                ]}
                                value={responses.pk_3}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="pk_4"
                                question="What is 'spear phishing'?"
                                options={[
                                    { value: 1, label: 'Phishing attempts sent to millions of random people' },
                                    { value: 2, label: 'Targeted phishing attacks aimed at specific individuals or organizations' },
                                    { value: 3, label: 'Phishing through phone calls' }
                                ]}
                                value={responses.pk_4}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="pk_5"
                                question="A legitimate company will typically:"
                                options={[
                                    { value: 1, label: 'Never ask for sensitive information via email' },
                                    { value: 2, label: 'Request your password to verify your account' },
                                    { value: 3, label: 'Send urgent emails requiring immediate action' }
                                ]}
                                value={responses.pk_5}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="pk_6"
                                question="Hovering over a link in an email allows you to:"
                                options={[
                                    { value: 1, label: 'Automatically scan it for viruses' },
                                    { value: 2, label: 'See the actual URL before clicking' },
                                    { value: 3, label: 'Report it to your email provider' }
                                ]}
                                value={responses.pk_6}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="pk_7"
                                question="Which of the following sender addresses is most suspicious?"
                                options={[
                                    { value: 1, label: 'support@microsoft.com' },
                                    { value: 2, label: 'support@micr0soft-help.com' },
                                    { value: 3, label: 'noreply@microsoft.com' }
                                ]}
                                value={responses.pk_7}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="pk_8"
                                question="What is 'social engineering' in the context of cybersecurity?"
                                options={[
                                    { value: 1, label: 'Manipulating people into revealing confidential information' },
                                    { value: 2, label: 'Building secure social media platforms' },
                                    { value: 3, label: 'Engineering social networking features' }
                                ]}
                                value={responses.pk_8}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="pk_9"
                                question="If you accidentally clicked a phishing link, you should:"
                                options={[
                                    { value: 1, label: 'Ignore it if you didn\'t enter any information' },
                                    { value: 2, label: 'Report it and change relevant passwords immediately' },
                                    { value: 3, label: 'Wait to see if anything bad happens' }
                                ]}
                                value={responses.pk_9}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="pk_10"
                                question="HTTPS in a URL indicates:"
                                options={[
                                    { value: 1, label: 'The website is definitely safe and trustworthy' },
                                    { value: 2, label: 'The website is from a large company' },
                                    { value: 3, label: 'The connection is encrypted, but doesn\'t guarantee legitimacy' }
                                ]}
                                value={responses.pk_10}
                                onChange={handleChange}
                            />
                        </div>
                    )}
                </div>

                {/* Section 6: Prior Experience */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="6. Prior Experience"
                        description="Your security background"
                        isOpen={openSections.exp}
                        onToggle={() => toggleSection('exp')}
                        isComplete={isSectionComplete(['prior_victimization', 'security_training', 'years_email_use'])}
                    />
                    {openSections.exp && (
                        <div className="p-6">
                            <SelectQuestion
                                id="security_training"
                                question="Have you ever completed formal security awareness training?"
                                options={[
                                    { value: 'yes', label: 'Yes' },
                                    { value: 'no', label: 'No' }
                                ]}
                                value={responses.security_training}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="prior_victimization"
                                question="How many times have you fallen victim to a phishing attack or online scam?"
                                options={[
                                    { value: 'never', label: 'Never' },
                                    { value: '1_time', label: 'Once' },
                                    { value: '2_plus_times', label: '2 or more times' }
                                ]}
                                value={responses.prior_victimization}
                                onChange={handleChange}
                            />
                            <TextInputQuestion
                                id="years_email_use"
                                question="How many years have you been using email?"
                                type="number"
                                value={responses.years_email_use}
                                onChange={handleChange}
                                placeholder="e.g., 15"
                            />
                        </div>
                    )}
                </div>

                {/* Section 7: Link-Click Tendency */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="7. Email Link Habits"
                        description="Your typical email behavior"
                        isOpen={openSections.link}
                        onToggle={() => toggleSection('link')}
                        isComplete={isSectionComplete(['link_click_tendency'])}
                    />
                    {openSections.link && (
                        <div className="p-6">
                            <SelectQuestion
                                id="link_click_tendency"
                                question="How often do you click links in emails without checking where they lead?"
                                options={[
                                    { value: 1, label: '1 - Never' },
                                    { value: 2, label: '2 - Rarely' },
                                    { value: 3, label: '3 - Sometimes' },
                                    { value: 4, label: '4 - Often' },
                                    { value: 5, label: '5 - Always' }
                                ]}
                                value={responses.link_click_tendency}
                                onChange={handleChange}
                            />
                        </div>
                    )}
                </div>

                {/* Section 8: Open-text Feedback */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <div className="p-4 bg-gray-50 border-b border-gray-200">
                        <h3 className="font-semibold text-gray-800">8. Additional Comments</h3>
                        <p className="text-sm text-gray-500">Optional feedback</p>
                    </div>
                    <div className="p-6">
                        <div className="mb-6 p-4 bg-white rounded-lg border border-gray-100">
                            <p className="text-gray-800 mb-3">
                                Any additional comments about the study? (Optional)
                            </p>
                            <textarea
                                value={responses.open_text_feedback || ''}
                                onChange={(e) => handleChange('open_text_feedback', e.target.value)}
                                placeholder="Share any thoughts, feedback, or observations about your experience..."
                                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-y"
                                rows={4}
                            />
                        </div>
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
