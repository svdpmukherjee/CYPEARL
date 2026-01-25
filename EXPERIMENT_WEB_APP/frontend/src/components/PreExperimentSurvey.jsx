/**
 * CYPEARL Pre-Experiment Survey Component
 *
 * Single-page questionnaire containing all pre-experiment validated instruments:
 * - Demographics (7 items)
 * - CRT-7 Cognitive Reflection Test (7 items)
 * - Need for Cognition NFC-6 Short Form (6 items)
 * - Working Memory Digit Span (simplified web version)
 * - Big Five BFI-10 (10 items)
 * - Impulsivity UPPS-P Short Form (4 items)
 * - Sensation Seeking BSSS-4 (4 items)
 * - Trust Propensity (6 items)
 * - Risk Taking DOSPERT-6 (6 items)
 * - Phishing Self-Efficacy (4 items)
 * - Perceived Risk (4 items)
 * - Security Attitudes SA-6 (6 items)
 * - Privacy Concern IUIPC-3 (3 items)
 * - Phishing Knowledge Quiz (10 items)
 * - Technical Expertise & Experience (5 items)
 * - Email Habits (4 items)
 * - Influence Susceptibility (9 items - 3 per construct)
 *
 * References from Validated_Instruments_Complete.xlsx
 */

import React, { useState, useEffect } from 'react';
import { ChevronDown, ChevronUp, AlertCircle, CheckCircle } from 'lucide-react';

// Local storage key for persisting responses
const PRE_SURVEY_RESPONSES_KEY = 'pre_survey_responses_draft';

// ============================================================================
// SURVEY CONFIGURATION
// ============================================================================

const LIKERT_5_AGREE = [
    { value: 1, label: 'Strongly Disagree' },
    { value: 2, label: 'Disagree' },
    { value: 3, label: 'Neutral' },
    { value: 4, label: 'Agree' },
    { value: 5, label: 'Strongly Agree' }
];

const LIKERT_5_CHARACTERISTIC = [
    { value: 1, label: 'Extremely Uncharacteristic' },
    { value: 2, label: 'Somewhat Uncharacteristic' },
    { value: 3, label: 'Neutral' },
    { value: 4, label: 'Somewhat Characteristic' },
    { value: 5, label: 'Extremely Characteristic' }
];

const LIKERT_5_LIKELY = [
    { value: 1, label: 'Extremely Unlikely' },
    { value: 2, label: 'Unlikely' },
    { value: 3, label: 'Neutral' },
    { value: 4, label: 'Likely' },
    { value: 5, label: 'Extremely Likely' }
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

// ============================================================================
// MAIN COMPONENT
// ============================================================================

const PreExperimentSurvey = ({ onComplete, onGoToEmail }) => {
    // Load saved responses from localStorage on mount
    const [responses, setResponses] = useState(() => {
        const saved = localStorage.getItem(PRE_SURVEY_RESPONSES_KEY);
        return saved ? JSON.parse(saved) : {};
    });
    const [openSections, setOpenSections] = useState({ demographics: true });
    const [errors, setErrors] = useState([]);
    const [isSubmitting, setIsSubmitting] = useState(false);

    // Save responses to localStorage whenever they change
    useEffect(() => {
        localStorage.setItem(PRE_SURVEY_RESPONSES_KEY, JSON.stringify(responses));
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
            // Demographics
            'age', 'gender', 'education', 'technical_field', 'employment', 'industry',
            // CRT
            'crt_1', 'crt_2', 'crt_3', 'crt_4', 'crt_5', 'crt_6', 'crt_7',
            // NFC
            'nfc_1', 'nfc_2', 'nfc_3', 'nfc_4', 'nfc_5', 'nfc_6',
            // Working Memory
            'digit_span',
            // Big Five
            'bfi_1', 'bfi_2', 'bfi_3', 'bfi_4', 'bfi_5', 'bfi_6', 'bfi_7', 'bfi_8', 'bfi_9', 'bfi_10',
            // Impulsivity
            'imp_1', 'imp_2', 'imp_3', 'imp_4',
            // Sensation Seeking
            'ss_1', 'ss_2', 'ss_3', 'ss_4',
            // Trust
            'trust_1', 'trust_2', 'trust_3', 'trust_4', 'trust_5', 'trust_6',
            // Risk Taking
            'risk_1', 'risk_2', 'risk_3', 'risk_4', 'risk_5', 'risk_6',
            // Phishing Self-Efficacy
            'pse_1', 'pse_2', 'pse_3', 'pse_4',
            // Perceived Risk
            'pr_1', 'pr_2', 'pr_3', 'pr_4',
            // Security Attitudes SA-6
            'sa_1', 'sa_2', 'sa_3', 'sa_4', 'sa_5', 'sa_6',
            // Privacy Concern
            'pc_1', 'pc_2', 'pc_3',
            // Phishing Knowledge
            'pk_1', 'pk_2', 'pk_3', 'pk_4', 'pk_5', 'pk_6', 'pk_7', 'pk_8', 'pk_9', 'pk_10',
            // Experience
            'technical_expertise', 'prior_victimization', 'security_training', 'years_email_use',
            // Email Habits
            'daily_email_volume', 'email_check_frequency', 'link_click_tendency', 'social_media_usage',
            // Influence Susceptibility
            'auth_1', 'auth_2', 'auth_3', 'urg_1', 'urg_2', 'urg_3', 'scar_1', 'scar_2', 'scar_3'
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
        // CRT Score (count of correct answers)
        const crtCorrect = {
            crt_1: 5,    // 5 cents
            crt_2: 5,    // 5 minutes
            crt_3: 47,   // 47 days
            crt_4: 4,    // 4 days
            crt_5: 29,   // 29 students
            crt_6: 20,   // $20
            crt_7: 3     // is behind (option 3)
        };

        let crt_score = 0;
        Object.keys(crtCorrect).forEach(key => {
            if (parseInt(data[key]) === crtCorrect[key]) crt_score++;
        });

        // NFC Score (mean of 6 items, some reverse coded)
        const nfc_items = ['nfc_1', 'nfc_2', 'nfc_3', 'nfc_4', 'nfc_5', 'nfc_6'];
        const nfc_reverse = ['nfc_3', 'nfc_4'];
        const need_for_cognition = calculateMean(data, nfc_items, nfc_reverse);

        // Big Five Scores (2 items each, some reverse coded)
        const big5_extraversion = calculateMean(data, ['bfi_1', 'bfi_6'], ['bfi_1']);
        const big5_agreeableness = calculateMean(data, ['bfi_2', 'bfi_7'], ['bfi_7']);
        const big5_conscientiousness = calculateMean(data, ['bfi_3', 'bfi_8'], ['bfi_3']);
        const big5_neuroticism = calculateMean(data, ['bfi_4', 'bfi_9'], ['bfi_9']);
        const big5_openness = calculateMean(data, ['bfi_5', 'bfi_10'], ['bfi_5']);

        // Impulsivity (mean of 4 items)
        const impulsivity_total = calculateMean(data, ['imp_1', 'imp_2', 'imp_3', 'imp_4'], ['imp_1']);

        // Sensation Seeking (mean of 4 items)
        const sensation_seeking = calculateMean(data, ['ss_1', 'ss_2', 'ss_3', 'ss_4'], []);

        // Trust Propensity (mean of 6 items)
        const trust_propensity = calculateMean(data, ['trust_1', 'trust_2', 'trust_3', 'trust_4', 'trust_5', 'trust_6'], []);

        // Risk Taking (mean of 6 items)
        const risk_taking = calculateMean(data, ['risk_1', 'risk_2', 'risk_3', 'risk_4', 'risk_5', 'risk_6'], []);

        // Phishing Self-Efficacy (mean of 4 items)
        const phishing_self_efficacy = calculateMean(data, ['pse_1', 'pse_2', 'pse_3', 'pse_4'], []);

        // Perceived Risk (mean of 4 items)
        const perceived_risk = calculateMean(data, ['pr_1', 'pr_2', 'pr_3', 'pr_4'], []);

        // Security Attitudes SA-6 (mean of 6 items)
        const security_attitudes = calculateMean(data, ['sa_1', 'sa_2', 'sa_3', 'sa_4', 'sa_5', 'sa_6'], []);

        // Privacy Concern (mean of 3 items)
        const privacy_concern = calculateMean(data, ['pc_1', 'pc_2', 'pc_3'], []);

        // Phishing Knowledge (count correct out of 10)
        const pk_correct = { pk_1: 2, pk_2: 1, pk_3: 3, pk_4: 2, pk_5: 1, pk_6: 2, pk_7: 3, pk_8: 1, pk_9: 2, pk_10: 3 };
        let phishing_knowledge = 0;
        Object.keys(pk_correct).forEach(key => {
            if (parseInt(data[key]) === pk_correct[key]) phishing_knowledge++;
        });

        // Influence Susceptibility Scales (mean of 3 items each)
        const authority_susceptibility = calculateMean(data, ['auth_1', 'auth_2', 'auth_3'], []);
        const urgency_susceptibility = calculateMean(data, ['urg_1', 'urg_2', 'urg_3'], []);
        const scarcity_susceptibility = calculateMean(data, ['scar_1', 'scar_2', 'scar_3'], []);

        // Education numeric
        const education_map = { 'high_school': 1, 'some_college': 2, 'bachelor': 3, 'master': 4, 'doctorate': 5 };
        const education_numeric = education_map[data.education] || 0;

        // Email volume numeric
        const email_volume_map = { '0-10': 1, '11-25': 2, '26-50': 3, '51-100': 4, '100+': 5 };
        const email_volume_numeric = email_volume_map[data.daily_email_volume] || 0;

        return {
            // Raw demographics
            age: parseInt(data.age),
            gender: data.gender,
            education: data.education,
            education_numeric,
            technical_field: data.technical_field === 'yes' ? 1 : 0,
            employment: data.employment,
            industry: data.industry,

            // Cognitive scores
            crt_score,
            need_for_cognition: round2(need_for_cognition),
            working_memory: parseInt(data.digit_span),

            // Big Five
            big5_extraversion: round2(big5_extraversion),
            big5_agreeableness: round2(big5_agreeableness),
            big5_conscientiousness: round2(big5_conscientiousness),
            big5_neuroticism: round2(big5_neuroticism),
            big5_openness: round2(big5_openness),

            // Personality
            impulsivity_total: round2(impulsivity_total),
            sensation_seeking: round2(sensation_seeking),
            trust_propensity: round2(trust_propensity),
            risk_taking: round2(risk_taking),

            // Security attitudes
            phishing_self_efficacy: round2(phishing_self_efficacy),
            perceived_risk: round2(perceived_risk),
            security_attitudes: round2(security_attitudes),
            privacy_concern: round2(privacy_concern),

            // Knowledge & experience
            phishing_knowledge,
            technical_expertise: parseInt(data.technical_expertise),
            prior_victimization: parseInt(data.prior_victimization),
            security_training: data.security_training === 'yes' ? 1 : 0,
            years_email_use: parseInt(data.years_email_use),

            // Email habits
            daily_email_volume: data.daily_email_volume,
            email_volume_numeric,
            email_check_frequency: data.email_check_frequency,
            link_click_tendency: round2(parseFloat(data.link_click_tendency)),
            social_media_usage: parseFloat(data.social_media_usage),

            // Influence susceptibility
            authority_susceptibility: round2(authority_susceptibility),
            urgency_susceptibility: round2(urgency_susceptibility),
            scarcity_susceptibility: round2(scarcity_susceptibility),

            // Store raw responses for debugging
            raw_responses: data,
            completed_at: new Date().toISOString()
        };
    };

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

    // ========================================================================
    // RENDER
    // ========================================================================

    return (
        <div className="min-h-screen bg-gray-50 py-8 px-4">
            <div className="max-w-4xl mx-auto">
                {/* Header */}
                <div className="bg-white rounded-lg shadow-sm p-6 mb-6">
                    <h1 className="text-2xl font-bold text-[#0078d4] mb-2">Pre-Experiment Questionnaire</h1>
                    <p className="text-gray-600">
                        Please complete all sections below before proceeding to the email evaluation task.
                        This survey takes approximately 15-20 minutes.
                    </p>
                    <div className="mt-4 p-3 bg-blue-50 rounded-lg flex items-start gap-2">
                        <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5" />
                        <p className="text-sm text-blue-800">
                            All responses are confidential and will only be used for research purposes.
                            Please answer honestly - there are no right or wrong answers.
                        </p>
                    </div>
                </div>

                {/* Section 1: Demographics */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="1. Demographics"
                        description="Basic information about yourself"
                        isOpen={openSections.demographics}
                        onToggle={() => toggleSection('demographics')}
                        isComplete={isSectionComplete(['age', 'gender', 'education', 'technical_field', 'employment', 'industry'])}
                    />
                    {openSections.demographics && (
                        <div className="p-6">
                            <TextInputQuestion
                                id="age"
                                question="What is your age in years?"
                                type="number"
                                value={responses.age}
                                onChange={handleChange}
                                placeholder="e.g., 35"
                            />
                            <SelectQuestion
                                id="gender"
                                question="What is your gender?"
                                options={[
                                    { value: 'male', label: 'Male' },
                                    { value: 'female', label: 'Female' },
                                    { value: 'non_binary', label: 'Non-binary' },
                                    { value: 'prefer_not', label: 'Prefer not to say' }
                                ]}
                                value={responses.gender}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="education"
                                question="What is your highest level of education completed?"
                                options={[
                                    { value: 'high_school', label: 'High school' },
                                    { value: 'some_college', label: 'Some college' },
                                    { value: 'bachelor', label: 'Bachelor\'s degree' },
                                    { value: 'master', label: 'Master\'s degree' },
                                    { value: 'doctorate', label: 'Doctorate' }
                                ]}
                                value={responses.education}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="technical_field"
                                question="Is your educational or professional background in a technical field? (e.g., Computer Science, IT, Engineering)"
                                options={[
                                    { value: 'yes', label: 'Yes' },
                                    { value: 'no', label: 'No' }
                                ]}
                                value={responses.technical_field}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="employment"
                                question="What is your current employment status?"
                                options={[
                                    { value: 'student', label: 'Student' },
                                    { value: 'employed_full', label: 'Employed full-time' },
                                    { value: 'employed_part', label: 'Employed part-time' },
                                    { value: 'self_employed', label: 'Self-employed' },
                                    { value: 'unemployed', label: 'Unemployed' },
                                    { value: 'retired', label: 'Retired' }
                                ]}
                                value={responses.employment}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="industry"
                                question="What industry do you work in?"
                                options={[
                                    { value: 'technology', label: 'Technology' },
                                    { value: 'finance', label: 'Finance' },
                                    { value: 'healthcare', label: 'Healthcare' },
                                    { value: 'education', label: 'Education' },
                                    { value: 'government', label: 'Government' },
                                    { value: 'manufacturing', label: 'Manufacturing' },
                                    { value: 'retail', label: 'Retail' },
                                    { value: 'other', label: 'Other' }
                                ]}
                                value={responses.industry}
                                onChange={handleChange}
                            />
                        </div>
                    )}
                </div>

                {/* Section 2: Cognitive Reflection Test */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="2. Problem Solving"
                        description="Cognitive Reflection Test (CRT-7)"
                        isOpen={openSections.crt}
                        onToggle={() => toggleSection('crt')}
                        isComplete={isSectionComplete(['crt_1', 'crt_2', 'crt_3', 'crt_4', 'crt_5', 'crt_6', 'crt_7'])}
                    />
                    {openSections.crt && (
                        <div className="p-6">
                            <p className="text-gray-600 mb-4">Please answer the following questions. Take your time to think carefully.</p>

                            <TextInputQuestion
                                id="crt_1"
                                question="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? (in cents)"
                                type="number"
                                value={responses.crt_1}
                                onChange={handleChange}
                                placeholder="Enter number only"
                            />
                            <TextInputQuestion
                                id="crt_2"
                                question="If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? (in minutes)"
                                type="number"
                                value={responses.crt_2}
                                onChange={handleChange}
                                placeholder="Enter number only"
                            />
                            <TextInputQuestion
                                id="crt_3"
                                question="In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake? (in days)"
                                type="number"
                                value={responses.crt_3}
                                onChange={handleChange}
                                placeholder="Enter number only"
                            />
                            <TextInputQuestion
                                id="crt_4"
                                question="If John can drink one barrel of water in 6 days, and Mary can drink one barrel of water in 12 days, how long would it take them to drink one barrel of water together? (in days)"
                                type="number"
                                value={responses.crt_4}
                                onChange={handleChange}
                                placeholder="Enter number only"
                            />
                            <TextInputQuestion
                                id="crt_5"
                                question="Jerry received both the 15th highest and the 15th lowest mark in the class. How many students are in the class?"
                                type="number"
                                value={responses.crt_5}
                                onChange={handleChange}
                                placeholder="Enter number only"
                            />
                            <TextInputQuestion
                                id="crt_6"
                                question="A man buys a pig for $60, sells it for $70, buys it back for $80, and sells it finally for $90. How much has he made? (in dollars)"
                                type="number"
                                value={responses.crt_6}
                                onChange={handleChange}
                                placeholder="Enter number only"
                            />
                            <SelectQuestion
                                id="crt_7"
                                question="Simon decided to invest $8,000 in the stock market one day early in 2008. Six months after he invested, on July 17, the stocks he had purchased were down 50%. Fortunately for Simon, from July 17 to October 17, the stocks he had purchased went up 75%. At this point, Simon:"
                                options={[
                                    { value: 1, label: 'Has broken even in the stock market' },
                                    { value: 2, label: 'Is ahead of where he began' },
                                    { value: 3, label: 'Is behind where he began' }
                                ]}
                                value={responses.crt_7}
                                onChange={handleChange}
                            />
                        </div>
                    )}
                </div>

                {/* Section 3: Need for Cognition */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="3. Thinking Style"
                        description="Need for Cognition (NFC-6)"
                        isOpen={openSections.nfc}
                        onToggle={() => toggleSection('nfc')}
                        isComplete={isSectionComplete(['nfc_1', 'nfc_2', 'nfc_3', 'nfc_4', 'nfc_5', 'nfc_6'])}
                    />
                    {openSections.nfc && (
                        <div className="p-6">
                            <p className="text-gray-600 mb-4">How characteristic are these statements of you?</p>

                            <LikertQuestion
                                id="nfc_1"
                                question="I would prefer complex to simple problems."
                                options={LIKERT_5_CHARACTERISTIC}
                                value={responses.nfc_1}
                                onChange={handleChange}
                            />
                            <LikertQuestion
                                id="nfc_2"
                                question="I like to have the responsibility of handling a situation that requires a lot of thinking."
                                options={LIKERT_5_CHARACTERISTIC}
                                value={responses.nfc_2}
                                onChange={handleChange}
                            />
                            <LikertQuestion
                                id="nfc_3"
                                question="Thinking is not my idea of fun."
                                options={LIKERT_5_CHARACTERISTIC}
                                value={responses.nfc_3}
                                onChange={handleChange}
                            />
                            <LikertQuestion
                                id="nfc_4"
                                question="I would rather do something that requires little thought than something that is sure to challenge my thinking abilities."
                                options={LIKERT_5_CHARACTERISTIC}
                                value={responses.nfc_4}
                                onChange={handleChange}
                            />
                            <LikertQuestion
                                id="nfc_5"
                                question="I really enjoy a task that involves coming up with new solutions to problems."
                                options={LIKERT_5_CHARACTERISTIC}
                                value={responses.nfc_5}
                                onChange={handleChange}
                            />
                            <LikertQuestion
                                id="nfc_6"
                                question="I prefer to think about small, daily projects rather than long-term ones."
                                options={LIKERT_5_CHARACTERISTIC}
                                value={responses.nfc_6}
                                onChange={handleChange}
                            />
                        </div>
                    )}
                </div>

                {/* Section 4: Working Memory */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="4. Working Memory"
                        description="Digit Span Self-Assessment"
                        isOpen={openSections.wm}
                        onToggle={() => toggleSection('wm')}
                        isComplete={isSectionComplete(['digit_span'])}
                    />
                    {openSections.wm && (
                        <div className="p-6">
                            <p className="text-gray-600 mb-4">
                                This measures your working memory capacity. Think about how many digits you can reliably remember in sequence.
                            </p>
                            <SelectQuestion
                                id="digit_span"
                                question="What is the longest sequence of random digits you can reliably remember and repeat back in order? (e.g., if someone reads out 7 digits, can you repeat them?)"
                                options={[
                                    { value: 3, label: '3 digits (e.g., 4-7-2)' },
                                    { value: 4, label: '4 digits (e.g., 3-9-1-6)' },
                                    { value: 5, label: '5 digits (e.g., 8-2-5-7-3)' },
                                    { value: 6, label: '6 digits (e.g., 1-4-9-2-6-8)' },
                                    { value: 7, label: '7 digits (e.g., 5-3-8-1-7-4-2)' },
                                    { value: 8, label: '8 digits' },
                                    { value: 9, label: '9 or more digits' }
                                ]}
                                value={responses.digit_span}
                                onChange={handleChange}
                            />
                        </div>
                    )}
                </div>

                {/* Section 5: Big Five Personality */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="5. Personality Traits"
                        description="Big Five Inventory (BFI-10)"
                        isOpen={openSections.bfi}
                        onToggle={() => toggleSection('bfi')}
                        isComplete={isSectionComplete(['bfi_1', 'bfi_2', 'bfi_3', 'bfi_4', 'bfi_5', 'bfi_6', 'bfi_7', 'bfi_8', 'bfi_9', 'bfi_10'])}
                    />
                    {openSections.bfi && (
                        <div className="p-6">
                            <p className="text-gray-600 mb-4">I see myself as someone who...</p>

                            <LikertQuestion id="bfi_1" question="...is reserved" options={LIKERT_5_AGREE} value={responses.bfi_1} onChange={handleChange} />
                            <LikertQuestion id="bfi_2" question="...is generally trusting" options={LIKERT_5_AGREE} value={responses.bfi_2} onChange={handleChange} />
                            <LikertQuestion id="bfi_3" question="...tends to be lazy" options={LIKERT_5_AGREE} value={responses.bfi_3} onChange={handleChange} />
                            <LikertQuestion id="bfi_4" question="...is relaxed, handles stress well" options={LIKERT_5_AGREE} value={responses.bfi_4} onChange={handleChange} />
                            <LikertQuestion id="bfi_5" question="...has few artistic interests" options={LIKERT_5_AGREE} value={responses.bfi_5} onChange={handleChange} />
                            <LikertQuestion id="bfi_6" question="...is outgoing, sociable" options={LIKERT_5_AGREE} value={responses.bfi_6} onChange={handleChange} />
                            <LikertQuestion id="bfi_7" question="...tends to find fault with others" options={LIKERT_5_AGREE} value={responses.bfi_7} onChange={handleChange} />
                            <LikertQuestion id="bfi_8" question="...does a thorough job" options={LIKERT_5_AGREE} value={responses.bfi_8} onChange={handleChange} />
                            <LikertQuestion id="bfi_9" question="...gets nervous easily" options={LIKERT_5_AGREE} value={responses.bfi_9} onChange={handleChange} />
                            <LikertQuestion id="bfi_10" question="...has an active imagination" options={LIKERT_5_AGREE} value={responses.bfi_10} onChange={handleChange} />
                        </div>
                    )}
                </div>

                {/* Section 6: Impulsivity */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="6. Impulsivity"
                        description="UPPS-P Short Form"
                        isOpen={openSections.imp}
                        onToggle={() => toggleSection('imp')}
                        isComplete={isSectionComplete(['imp_1', 'imp_2', 'imp_3', 'imp_4'])}
                    />
                    {openSections.imp && (
                        <div className="p-6">
                            <LikertQuestion id="imp_1" question="I have a reserved and cautious attitude toward life." options={LIKERT_5_AGREE} value={responses.imp_1} onChange={handleChange} />
                            <LikertQuestion id="imp_2" question="I generally seek new and exciting experiences and sensations." options={LIKERT_5_AGREE} value={responses.imp_2} onChange={handleChange} />
                            <LikertQuestion id="imp_3" question="When I am upset I often act without thinking." options={LIKERT_5_AGREE} value={responses.imp_3} onChange={handleChange} />
                            <LikertQuestion id="imp_4" question="I finish what I start." options={LIKERT_5_AGREE} value={responses.imp_4} onChange={handleChange} />
                        </div>
                    )}
                </div>

                {/* Section 7: Sensation Seeking */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="7. Sensation Seeking"
                        description="BSSS-4"
                        isOpen={openSections.ss}
                        onToggle={() => toggleSection('ss')}
                        isComplete={isSectionComplete(['ss_1', 'ss_2', 'ss_3', 'ss_4'])}
                    />
                    {openSections.ss && (
                        <div className="p-6">
                            <LikertQuestion id="ss_1" question="I would like to explore strange places." options={LIKERT_5_AGREE} value={responses.ss_1} onChange={handleChange} />
                            <LikertQuestion id="ss_2" question="I like to do frightening things." options={LIKERT_5_AGREE} value={responses.ss_2} onChange={handleChange} />
                            <LikertQuestion id="ss_3" question="I like wild parties." options={LIKERT_5_AGREE} value={responses.ss_3} onChange={handleChange} />
                            <LikertQuestion id="ss_4" question="I would like to take off on a trip with no pre-planned routes or timetables." options={LIKERT_5_AGREE} value={responses.ss_4} onChange={handleChange} />
                        </div>
                    )}
                </div>

                {/* Section 8: Trust Propensity */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="8. Trust Propensity"
                        description="General Trust Scale"
                        isOpen={openSections.trust}
                        onToggle={() => toggleSection('trust')}
                        isComplete={isSectionComplete(['trust_1', 'trust_2', 'trust_3', 'trust_4', 'trust_5', 'trust_6'])}
                    />
                    {openSections.trust && (
                        <div className="p-6">
                            <LikertQuestion id="trust_1" question="Most people are basically honest." options={LIKERT_5_AGREE} value={responses.trust_1} onChange={handleChange} />
                            <LikertQuestion id="trust_2" question="Most people are trustworthy." options={LIKERT_5_AGREE} value={responses.trust_2} onChange={handleChange} />
                            <LikertQuestion id="trust_3" question="Most people are basically good and kind." options={LIKERT_5_AGREE} value={responses.trust_3} onChange={handleChange} />
                            <LikertQuestion id="trust_4" question="Most people are trustful of others." options={LIKERT_5_AGREE} value={responses.trust_4} onChange={handleChange} />
                            <LikertQuestion id="trust_5" question="I am trustful." options={LIKERT_5_AGREE} value={responses.trust_5} onChange={handleChange} />
                            <LikertQuestion id="trust_6" question="Most people will respond in kind when they are trusted by others." options={LIKERT_5_AGREE} value={responses.trust_6} onChange={handleChange} />
                        </div>
                    )}
                </div>

                {/* Section 9: Risk Taking */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="9. Risk Taking"
                        description="DOSPERT-6"
                        isOpen={openSections.risk}
                        onToggle={() => toggleSection('risk')}
                        isComplete={isSectionComplete(['risk_1', 'risk_2', 'risk_3', 'risk_4', 'risk_5', 'risk_6'])}
                    />
                    {openSections.risk && (
                        <div className="p-6">
                            <p className="text-gray-600 mb-4">How likely are you to engage in the following activities?</p>

                            <LikertQuestion id="risk_1" question="Betting a day's income at a poker game." options={LIKERT_5_LIKELY} value={responses.risk_1} onChange={handleChange} />
                            <LikertQuestion id="risk_2" question="Investing 10% of your annual income in a moderate growth mutual fund." options={LIKERT_5_LIKELY} value={responses.risk_2} onChange={handleChange} />
                            <LikertQuestion id="risk_3" question="Downloading pirated software or media." options={LIKERT_5_LIKELY} value={responses.risk_3} onChange={handleChange} />
                            <LikertQuestion id="risk_4" question="Revealing a friend's secret to someone else." options={LIKERT_5_LIKELY} value={responses.risk_4} onChange={handleChange} />
                            <LikertQuestion id="risk_5" question="Taking a job that you enjoy over one that is secure but less enjoyable." options={LIKERT_5_LIKELY} value={responses.risk_5} onChange={handleChange} />
                            <LikertQuestion id="risk_6" question="Disagreeing with an authority figure on a major issue." options={LIKERT_5_LIKELY} value={responses.risk_6} onChange={handleChange} />
                        </div>
                    )}
                </div>

                {/* Section 10: Phishing Self-Efficacy */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="10. Phishing Detection Confidence"
                        description="Phishing Self-Efficacy"
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

                {/* Section 11: Perceived Risk */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="11. Perceived Risk"
                        description="Perceived phishing threat"
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

                {/* Section 12: Security Attitudes */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="12. Security Attitudes"
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

                {/* Section 13: Privacy Concern */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="13. Privacy Concern"
                        description="IUIPC-3"
                        isOpen={openSections.pc}
                        onToggle={() => toggleSection('pc')}
                        isComplete={isSectionComplete(['pc_1', 'pc_2', 'pc_3'])}
                    />
                    {openSections.pc && (
                        <div className="p-6">
                            <LikertQuestion id="pc_1" question="I am concerned about threats to my personal information online." options={LIKERT_5_AGREE} value={responses.pc_1} onChange={handleChange} />
                            <LikertQuestion id="pc_2" question="It usually bothers me when websites ask me for personal information." options={LIKERT_5_AGREE} value={responses.pc_2} onChange={handleChange} />
                            <LikertQuestion id="pc_3" question="I am concerned that online companies are collecting too much personal information about me." options={LIKERT_5_AGREE} value={responses.pc_3} onChange={handleChange} />
                        </div>
                    )}
                </div>

                {/* Section 14: Phishing Knowledge Quiz */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="14. Phishing Knowledge"
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

                {/* Section 15: Technical Expertise & Experience */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="15. Technical Expertise & Experience"
                        description="Your background with technology"
                        isOpen={openSections.exp}
                        onToggle={() => toggleSection('exp')}
                        isComplete={isSectionComplete(['technical_expertise', 'prior_victimization', 'security_training', 'years_email_use'])}
                    />
                    {openSections.exp && (
                        <div className="p-6">
                            <SelectQuestion
                                id="technical_expertise"
                                question="How would you rate your overall computer and internet skills?"
                                options={[
                                    { value: 1, label: '1 - Novice (basic tasks only)' },
                                    { value: 2, label: '2 - Beginner' },
                                    { value: 3, label: '3 - Intermediate' },
                                    { value: 4, label: '4 - Competent' },
                                    { value: 5, label: '5 - Proficient' },
                                    { value: 6, label: '6 - Advanced' },
                                    { value: 7, label: '7 - Expert' }
                                ]}
                                value={responses.technical_expertise}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="prior_victimization"
                                question="How many times have you fallen victim to a phishing attack or online scam?"
                                options={[
                                    { value: 0, label: 'Never (0 times)' },
                                    { value: 1, label: 'Once (1 time)' },
                                    { value: 2, label: 'Twice (2 times)' },
                                    { value: 3, label: '3 times' },
                                    { value: 4, label: '4 times' },
                                    { value: 5, label: '5 or more times' }
                                ]}
                                value={responses.prior_victimization}
                                onChange={handleChange}
                            />
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

                {/* Section 16: Email Habits */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="16. Email Habits"
                        description="Your email usage patterns"
                        isOpen={openSections.email}
                        onToggle={() => toggleSection('email')}
                        isComplete={isSectionComplete(['daily_email_volume', 'email_check_frequency', 'link_click_tendency', 'social_media_usage'])}
                    />
                    {openSections.email && (
                        <div className="p-6">
                            <SelectQuestion
                                id="daily_email_volume"
                                question="How many emails do you typically receive per day?"
                                options={[
                                    { value: '0-10', label: '0-10 emails' },
                                    { value: '11-25', label: '11-25 emails' },
                                    { value: '26-50', label: '26-50 emails' },
                                    { value: '51-100', label: '51-100 emails' },
                                    { value: '100+', label: 'More than 100 emails' }
                                ]}
                                value={responses.daily_email_volume}
                                onChange={handleChange}
                            />
                            <SelectQuestion
                                id="email_check_frequency"
                                question="How often do you check your email?"
                                options={[
                                    { value: 'few_times_week', label: 'A few times per week' },
                                    { value: 'once_daily', label: 'Once daily' },
                                    { value: 'few_times_day', label: 'A few times per day' },
                                    { value: 'hourly', label: 'Hourly' },
                                    { value: 'constantly', label: 'Constantly / Always open' }
                                ]}
                                value={responses.email_check_frequency}
                                onChange={handleChange}
                            />
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
                            <TextInputQuestion
                                id="social_media_usage"
                                question="On average, how many hours per day do you spend on social media?"
                                type="number"
                                value={responses.social_media_usage}
                                onChange={handleChange}
                                placeholder="e.g., 2.5"
                            />
                        </div>
                    )}
                </div>

                {/* Section 17: Influence Susceptibility */}
                <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
                    <SectionHeader
                        title="17. Influence Susceptibility"
                        description="How you respond to persuasion tactics"
                        isOpen={openSections.influence}
                        onToggle={() => toggleSection('influence')}
                        isComplete={isSectionComplete(['auth_1', 'auth_2', 'auth_3', 'urg_1', 'urg_2', 'urg_3', 'scar_1', 'scar_2', 'scar_3'])}
                    />
                    {openSections.influence && (
                        <div className="p-6">
                            <h4 className="font-medium text-gray-700 mb-3">Authority Susceptibility</h4>
                            <LikertQuestion id="auth_1" question="I tend to comply with requests from people in positions of authority." options={LIKERT_5_AGREE} value={responses.auth_1} onChange={handleChange} />
                            <LikertQuestion id="auth_2" question="When someone with a title asks me to do something, I usually do it." options={LIKERT_5_AGREE} value={responses.auth_2} onChange={handleChange} />
                            <LikertQuestion id="auth_3" question="I respect and follow instructions from my superiors." options={LIKERT_5_AGREE} value={responses.auth_3} onChange={handleChange} />

                            <h4 className="font-medium text-gray-700 mb-3 mt-6">Urgency Susceptibility</h4>
                            <LikertQuestion id="urg_1" question="When something is urgent, I act quickly without overthinking." options={LIKERT_5_AGREE} value={responses.urg_1} onChange={handleChange} />
                            <LikertQuestion id="urg_2" question="Deadlines make me act faster." options={LIKERT_5_AGREE} value={responses.urg_2} onChange={handleChange} />
                            <LikertQuestion id="urg_3" question="I feel anxious when asked to respond immediately and usually comply quickly." options={LIKERT_5_AGREE} value={responses.urg_3} onChange={handleChange} />

                            <h4 className="font-medium text-gray-700 mb-3 mt-6">Scarcity Susceptibility</h4>
                            <LikertQuestion id="scar_1" question="Limited time offers make me more likely to act." options={LIKERT_5_AGREE} value={responses.scar_1} onChange={handleChange} />
                            <LikertQuestion id="scar_2" question="I worry about missing out on opportunities." options={LIKERT_5_AGREE} value={responses.scar_2} onChange={handleChange} />
                            <LikertQuestion id="scar_3" question="I am influenced by claims that something is rare or exclusive." options={LIKERT_5_AGREE} value={responses.scar_3} onChange={handleChange} />
                        </div>
                    )}
                </div>

                {/* Submit Button */}
                <div className="bg-white rounded-lg shadow-sm p-6">
                    <div className="flex items-center justify-between">
                        <div className="text-gray-600">
                            <p className="font-medium">Ready to proceed?</p>
                            <p className="text-sm">Please ensure all sections are complete.</p>
                        </div>
                        <button
                            onClick={handleSubmit}
                            disabled={isSubmitting}
                            className={`px-8 py-3 rounded-lg font-semibold transition-colors ${
                                isSubmitting
                                    ? 'bg-gray-300 text-gray-500 cursor-not-allowed'
                                    : 'bg-[#0078d4] text-white hover:bg-[#106ebe]'
                            }`}
                        >
                            {isSubmitting ? 'Submitting...' : 'Continue to Study'}
                        </button>
                    </div>
                </div>
            </div>

            {/* TESTING ONLY: Navigation button to go to email experiment */}
            {onGoToEmail && (
                <div className="fixed bottom-0 left-0 right-0 px-4 py-2 flex justify-end items-center z-50 pointer-events-none">
                    <button
                        onClick={onGoToEmail}
                        className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium pointer-events-auto"
                    >
                        Go to Email Experiment →
                    </button>
                </div>
            )}
        </div>
    );
};

export default PreExperimentSurvey;
