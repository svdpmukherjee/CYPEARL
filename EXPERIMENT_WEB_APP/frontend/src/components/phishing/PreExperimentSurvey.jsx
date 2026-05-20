/**
 * CYPEARL Pre-Experiment Survey Component
 *
 * Contains ONLY non-priming instruments administered BEFORE the email experiment.
 * Phishing-specific measures (self-efficacy, perceived risk, SA-6, knowledge quiz,
 * prior experience, link-click tendency) are moved to PostExperimentSurvey to avoid
 * priming participants with phishing-related content before the behavioral task.
 *
 * Pre-experiment instruments:
 * - Demographics (7 items including sub_role; industry captured in job cluster selection)
 * - CRT-7 Cognitive Reflection Test (7 items)
 * - Big Five BFI-10 (10 items)
 * - Impulsivity UPPS-P (8 items: 4 Lack of Premeditation + 4 Negative Urgency)
 * - Trust Propensity (6 items)
 * - Email Habits (3 items — volume, frequency, social media; link-click tendency moved to post)
 *
 * References from Validated_Instruments_Complete.xlsx
 */

import React, { useState, useEffect } from "react";
import { ChevronDown, ChevronUp, AlertCircle, CheckCircle } from "lucide-react";

// Local storage key for persisting responses
const PRE_SURVEY_RESPONSES_KEY = "pre_survey_responses_draft";

// ============================================================================
// SURVEY CONFIGURATION
// ============================================================================

const LIKERT_5_AGREE = [
  { value: 1, label: "Strongly Disagree" },
  { value: 2, label: "Disagree" },
  { value: 3, label: "Neutral" },
  { value: 4, label: "Agree" },
  { value: 5, label: "Strongly Agree" },
];

const LIKERT_5_CHARACTERISTIC = [
  { value: 1, label: "Extremely Uncharacteristic" },
  { value: 2, label: "Somewhat Uncharacteristic" },
  { value: 3, label: "Neutral" },
  { value: 4, label: "Somewhat Characteristic" },
  { value: 5, label: "Extremely Characteristic" },
];

const LIKERT_5_LIKELY = [
  { value: 1, label: "Extremely Unlikely" },
  { value: 2, label: "Unlikely" },
  { value: 3, label: "Neutral" },
  { value: 4, label: "Likely" },
  { value: 5, label: "Extremely Likely" },
];

// ============================================================================
// SECTION COMPONENTS
// ============================================================================

const SectionHeader = ({
  title,
  description,
  isOpen,
  onToggle,
  isComplete,
}) => (
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
    {isOpen ? (
      <ChevronUp className="w-5 h-5 text-gray-500" />
    ) : (
      <ChevronDown className="w-5 h-5 text-gray-500" />
    )}
  </div>
);

const LikertQuestion = ({
  id,
  question,
  options,
  value,
  onChange,
  required = true,
}) => (
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
              ? "bg-blue-50 border-blue-500 text-blue-700"
              : "bg-gray-50 border-gray-200 hover:bg-gray-100"
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

const TextInputQuestion = ({
  id,
  question,
  value,
  onChange,
  type = "text",
  placeholder = "",
  required = true,
}) => (
  <div className="mb-4">
    <label className="block text-gray-800 mb-2">
      {question}
      {required && <span className="text-red-500 ml-1">*</span>}
    </label>
    <input
      type={type}
      value={value || ""}
      onChange={(e) =>
        onChange(
          id,
          type === "number" ? parseInt(e.target.value) || "" : e.target.value,
        )
      }
      placeholder={placeholder}
      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
    />
  </div>
);

const SelectQuestion = ({
  id,
  question,
  options,
  value,
  onChange,
  required = true,
}) => (
  <div className="mb-4">
    <label className="block text-gray-800 mb-2">
      {question}
      {required && <span className="text-red-500 ml-1">*</span>}
    </label>
    <select
      value={value || ""}
      onChange={(e) => onChange(id, e.target.value)}
      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
    >
      <option value="">Select...</option>
      {options.map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
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
    setResponses((prev) => ({ ...prev, [id]: value }));
  };

  const toggleSection = (section) => {
    setOpenSections((prev) => ({ ...prev, [section]: !prev[section] }));
  };

  const isSectionComplete = (requiredFields) => {
    return requiredFields.every(
      (field) => responses[field] !== undefined && responses[field] !== "",
    );
  };

  // ========================================================================
  // VALIDATION & SUBMISSION
  // ========================================================================

  const validateForm = () => {
    const requiredFields = [
      // Demographics
      "age",
      "gender",
      "education",
      "technical_field",
      "employment",
      "sub_role",
      // CRT
      "crt_1",
      "crt_2",
      "crt_3",
      "crt_4",
      "crt_5",
      "crt_6",
      "crt_7",
      // Big Five
      "bfi_1",
      "bfi_2",
      "bfi_3",
      "bfi_4",
      "bfi_5",
      "bfi_6",
      "bfi_7",
      "bfi_8",
      "bfi_9",
      "bfi_10",
      // Impulsivity - Lack of Premeditation
      "imp_lp_1",
      "imp_lp_2",
      "imp_lp_3",
      "imp_lp_4",
      // Impulsivity - Negative Urgency
      "imp_nu_1",
      "imp_nu_2",
      "imp_nu_3",
      "imp_nu_4",
      // Trust
      "trust_1",
      "trust_2",
      "trust_3",
      "trust_4",
      "trust_5",
      "trust_6",
      // Email Habits (link_click_tendency moved to post-survey to avoid priming)
      "daily_email_volume",
      "email_check_frequency",
      "social_media_usage",
    ];

    const missing = requiredFields.filter(
      (field) => responses[field] === undefined || responses[field] === "",
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
      console.error("Error submitting survey:", error);
      alert("Failed to submit survey. Please try again.");
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
      crt_1: 5, // 5 cents
      crt_2: 5, // 5 minutes
      crt_3: 47, // 47 days
      crt_4: 4, // 4 days
      crt_5: 29, // 29 students
      crt_6: 20, // $20
      crt_7: 3, // is behind (option 3)
    };

    let crt_score = 0;
    Object.keys(crtCorrect).forEach((key) => {
      if (parseInt(data[key]) === crtCorrect[key]) crt_score++;
    });

    // Big Five Scores (2 items each, some reverse coded)
    const big5_extraversion = calculateMean(
      data,
      ["bfi_1", "bfi_6"],
      ["bfi_1"],
    );
    const big5_agreeableness = calculateMean(
      data,
      ["bfi_2", "bfi_7"],
      ["bfi_7"],
    );
    const big5_conscientiousness = calculateMean(
      data,
      ["bfi_3", "bfi_8"],
      ["bfi_3"],
    );
    const big5_neuroticism = calculateMean(data, ["bfi_4", "bfi_9"], ["bfi_9"]);
    const big5_openness = calculateMean(data, ["bfi_5", "bfi_10"], ["bfi_5"]);

    // Impulsivity - Lack of Premeditation (ALL 4 reverse-coded: subtract from scale max+1 = 6)
    const imp_lp_items = ["imp_lp_1", "imp_lp_2", "imp_lp_3", "imp_lp_4"];
    const lack_of_premeditation = calculateMean(
      data,
      imp_lp_items,
      imp_lp_items,
    );

    // Impulsivity - Negative Urgency (none reverse-coded)
    const imp_nu_items = ["imp_nu_1", "imp_nu_2", "imp_nu_3", "imp_nu_4"];
    const negative_urgency = calculateMean(data, imp_nu_items, []);

    // Trust Propensity (mean of 6 items)
    const trust_propensity = calculateMean(
      data,
      ["trust_1", "trust_2", "trust_3", "trust_4", "trust_5", "trust_6"],
      [],
    );

    // NOTE: Phishing Self-Efficacy, Perceived Risk, SA-6, Phishing Knowledge,
    // and Prior Experience are now measured in PostExperimentSurvey to avoid priming.

    // Education numeric
    const education_map = {
      high_school: 1,
      some_college: 2,
      bachelor: 3,
      master: 4,
      doctorate: 5,
    };
    const education_numeric = education_map[data.education] || 0;

    // Email volume numeric
    const email_volume_map = {
      "0-10": 1,
      "11-25": 2,
      "26-50": 3,
      "51-100": 4,
      "100+": 5,
    };
    const email_volume_numeric = email_volume_map[data.daily_email_volume] || 0;

    return {
      // Raw demographics
      age: parseInt(data.age),
      gender: data.gender,
      education: data.education,
      education_numeric,
      technical_field: data.technical_field === "yes" ? 1 : 0,
      employment: data.employment,
      sub_role: data.sub_role,

      // Cognitive scores
      crt_score,

      // Big Five
      big5_extraversion: round2(big5_extraversion),
      big5_agreeableness: round2(big5_agreeableness),
      big5_conscientiousness: round2(big5_conscientiousness),
      big5_neuroticism: round2(big5_neuroticism),
      big5_openness: round2(big5_openness),

      // Impulsivity subscales
      lack_of_premeditation: round2(lack_of_premeditation),
      negative_urgency: round2(negative_urgency),

      // Personality
      trust_propensity: round2(trust_propensity),

      // Email habits (link_click_tendency moved to post-survey)
      daily_email_volume: data.daily_email_volume,
      email_volume_numeric,
      email_check_frequency: data.email_check_frequency,
      social_media_usage: parseFloat(data.social_media_usage),

      // Store raw responses for debugging
      raw_responses: data,
      completed_at: new Date().toISOString(),
    };
  };

  const calculateMean = (data, items, reverseItems) => {
    let sum = 0;
    let count = 0;
    items.forEach((item) => {
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
          <h1 className="text-2xl font-bold text-[#0078d4] mb-2">
            Pre-Experiment Questionnaire
          </h1>
          <p className="text-gray-600">
            Please complete all sections below before proceeding to the email
            evaluation task. This survey takes approximately 8-10 minutes.
          </p>
          <p className="text-sm text-orange-600 mt-2">
            For demo purpose, you can move to the next section without filling
            out the survey. In production, all questions will be required.
          </p>
          <div className="mt-4 p-3 bg-blue-50 rounded-lg flex items-start gap-2">
            <AlertCircle className="w-5 h-5 text-blue-600 mt-0.5" />
            <p className="text-sm text-blue-800">
              All responses are confidential and will only be used for research
              purposes. Please answer honestly - there are no right or wrong
              answers.
            </p>
          </div>
        </div>

        {/* Section 1: Demographics */}
        <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
          <SectionHeader
            title="1. Demographics"
            description="Basic information about yourself"
            isOpen={openSections.demographics}
            onToggle={() => toggleSection("demographics")}
            isComplete={isSectionComplete([
              "age",
              "gender",
              "education",
              "technical_field",
              "employment",
              "sub_role",
            ])}
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
                  { value: "male", label: "Male" },
                  { value: "female", label: "Female" },
                  { value: "non_binary", label: "Non-binary" },
                  { value: "prefer_not", label: "Prefer not to say" },
                ]}
                value={responses.gender}
                onChange={handleChange}
              />
              <SelectQuestion
                id="education"
                question="What is your highest level of education completed?"
                options={[
                  { value: "high_school", label: "High school" },
                  { value: "some_college", label: "Some college" },
                  { value: "bachelor", label: "Bachelor's degree" },
                  { value: "master", label: "Master's degree" },
                  { value: "doctorate", label: "Doctorate" },
                ]}
                value={responses.education}
                onChange={handleChange}
              />
              <SelectQuestion
                id="technical_field"
                question="Is your educational or professional background in a technical field? (e.g., Computer Science, IT, Engineering)"
                options={[
                  { value: "yes", label: "Yes" },
                  { value: "no", label: "No" },
                ]}
                value={responses.technical_field}
                onChange={handleChange}
              />
              <SelectQuestion
                id="employment"
                question="What is your current employment status?"
                options={[
                  { value: "student", label: "Student" },
                  { value: "employed_full", label: "Employed full-time" },
                  { value: "employed_part", label: "Employed part-time" },
                  { value: "self_employed", label: "Self-employed" },
                  { value: "unemployed", label: "Unemployed" },
                  { value: "retired", label: "Retired" },
                ]}
                value={responses.employment}
                onChange={handleChange}
              />
              {/* Functional Sub-Role */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Which best describes your primary functional role?
                </label>
                <select
                  className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-teal-500 focus:border-transparent"
                  value={responses.sub_role || ""}
                  onChange={(e) => handleChange("sub_role", e.target.value)}
                >
                  <option value="">Select your functional role</option>
                  <option value="Technical/Engineering">
                    Technical/Engineering
                  </option>
                  <option value="Managerial">Managerial</option>
                  <option value="Client-facing">Client-facing</option>
                  <option value="Administrative">Administrative</option>
                  <option value="Research">Research</option>
                  <option value="Teaching/Training">Teaching/Training</option>
                  <option value="Other">Other</option>
                </select>
              </div>
            </div>
          )}
        </div>

        {/* Section 2: Cognitive Reflection Test */}
        <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
          <SectionHeader
            title="2. Problem Solving"
            description="Cognitive Reflection Test (CRT-7)"
            isOpen={openSections.crt}
            onToggle={() => toggleSection("crt")}
            isComplete={isSectionComplete([
              "crt_1",
              "crt_2",
              "crt_3",
              "crt_4",
              "crt_5",
              "crt_6",
              "crt_7",
            ])}
          />
          {openSections.crt && (
            <div className="p-6">
              <p className="text-gray-600 mb-4">
                Please answer the following questions. Take your time to think
                carefully.
              </p>

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
                  { value: 1, label: "Has broken even in the stock market" },
                  { value: 2, label: "Is ahead of where he began" },
                  { value: 3, label: "Is behind where he began" },
                ]}
                value={responses.crt_7}
                onChange={handleChange}
              />
            </div>
          )}
        </div>

        {/* Section 3: Big Five Personality */}
        <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
          <SectionHeader
            title="3. Personality Traits"
            description="Big Five Inventory (BFI-10)"
            isOpen={openSections.bfi}
            onToggle={() => toggleSection("bfi")}
            isComplete={isSectionComplete([
              "bfi_1",
              "bfi_2",
              "bfi_3",
              "bfi_4",
              "bfi_5",
              "bfi_6",
              "bfi_7",
              "bfi_8",
              "bfi_9",
              "bfi_10",
            ])}
          />
          {openSections.bfi && (
            <div className="p-6">
              <p className="text-gray-600 mb-4">
                I see myself as someone who...
              </p>

              <LikertQuestion
                id="bfi_1"
                question="...is reserved"
                options={LIKERT_5_AGREE}
                value={responses.bfi_1}
                onChange={handleChange}
              />
              <LikertQuestion
                id="bfi_2"
                question="...is generally trusting"
                options={LIKERT_5_AGREE}
                value={responses.bfi_2}
                onChange={handleChange}
              />
              <LikertQuestion
                id="bfi_3"
                question="...tends to be lazy"
                options={LIKERT_5_AGREE}
                value={responses.bfi_3}
                onChange={handleChange}
              />
              <LikertQuestion
                id="bfi_4"
                question="...is relaxed, handles stress well"
                options={LIKERT_5_AGREE}
                value={responses.bfi_4}
                onChange={handleChange}
              />
              <LikertQuestion
                id="bfi_5"
                question="...has few artistic interests"
                options={LIKERT_5_AGREE}
                value={responses.bfi_5}
                onChange={handleChange}
              />
              <LikertQuestion
                id="bfi_6"
                question="...is outgoing, sociable"
                options={LIKERT_5_AGREE}
                value={responses.bfi_6}
                onChange={handleChange}
              />
              <LikertQuestion
                id="bfi_7"
                question="...tends to find fault with others"
                options={LIKERT_5_AGREE}
                value={responses.bfi_7}
                onChange={handleChange}
              />
              <LikertQuestion
                id="bfi_8"
                question="...does a thorough job"
                options={LIKERT_5_AGREE}
                value={responses.bfi_8}
                onChange={handleChange}
              />
              <LikertQuestion
                id="bfi_9"
                question="...gets nervous easily"
                options={LIKERT_5_AGREE}
                value={responses.bfi_9}
                onChange={handleChange}
              />
              <LikertQuestion
                id="bfi_10"
                question="...has an active imagination"
                options={LIKERT_5_AGREE}
                value={responses.bfi_10}
                onChange={handleChange}
              />
            </div>
          )}
        </div>

        {/* Section 4: Impulsivity */}
        <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
          <SectionHeader
            title="4. Impulsivity"
            description="UPPS-P Short Form"
            isOpen={openSections.imp}
            onToggle={() => toggleSection("imp")}
            isComplete={isSectionComplete([
              "imp_lp_1",
              "imp_lp_2",
              "imp_lp_3",
              "imp_lp_4",
              "imp_nu_1",
              "imp_nu_2",
              "imp_nu_3",
              "imp_nu_4",
            ])}
          />
          {openSections.imp && (
            <div className="p-6">
              <h4 className="font-medium text-gray-700 mb-3">
                Lack of Premeditation
              </h4>
              <LikertQuestion
                id="imp_lp_1"
                question="I have a reserved and cautious attitude toward life."
                options={LIKERT_5_AGREE}
                value={responses.imp_lp_1}
                onChange={handleChange}
              />
              <LikertQuestion
                id="imp_lp_2"
                question="My thinking is usually careful and purposeful."
                options={LIKERT_5_AGREE}
                value={responses.imp_lp_2}
                onChange={handleChange}
              />
              <LikertQuestion
                id="imp_lp_3"
                question="I usually think before I act."
                options={LIKERT_5_AGREE}
                value={responses.imp_lp_3}
                onChange={handleChange}
              />
              <LikertQuestion
                id="imp_lp_4"
                question="Before I get into a new situation I like to find out what to expect from it."
                options={LIKERT_5_AGREE}
                value={responses.imp_lp_4}
                onChange={handleChange}
              />

              <h4 className="font-medium text-gray-700 mb-3 mt-6">
                Negative Urgency
              </h4>
              <LikertQuestion
                id="imp_nu_1"
                question="When I am upset I often act without thinking."
                options={LIKERT_5_AGREE}
                value={responses.imp_nu_1}
                onChange={handleChange}
              />
              <LikertQuestion
                id="imp_nu_2"
                question="When I feel bad, I will often do things I later regret in order to make myself feel better now."
                options={LIKERT_5_AGREE}
                value={responses.imp_nu_2}
                onChange={handleChange}
              />
              <LikertQuestion
                id="imp_nu_3"
                question="It is hard for me to resist acting on my feelings."
                options={LIKERT_5_AGREE}
                value={responses.imp_nu_3}
                onChange={handleChange}
              />
              <LikertQuestion
                id="imp_nu_4"
                question="Sometimes when I feel bad, I can't seem to stop what I am doing even though it is making me feel worse."
                options={LIKERT_5_AGREE}
                value={responses.imp_nu_4}
                onChange={handleChange}
              />
            </div>
          )}
        </div>

        {/* Section 5: Trust Propensity */}
        <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
          <SectionHeader
            title="5. Trust Propensity"
            description="General Trust Scale"
            isOpen={openSections.trust}
            onToggle={() => toggleSection("trust")}
            isComplete={isSectionComplete([
              "trust_1",
              "trust_2",
              "trust_3",
              "trust_4",
              "trust_5",
              "trust_6",
            ])}
          />
          {openSections.trust && (
            <div className="p-6">
              <LikertQuestion
                id="trust_1"
                question="Most people are basically honest."
                options={LIKERT_5_AGREE}
                value={responses.trust_1}
                onChange={handleChange}
              />
              <LikertQuestion
                id="trust_2"
                question="Most people are trustworthy."
                options={LIKERT_5_AGREE}
                value={responses.trust_2}
                onChange={handleChange}
              />
              <LikertQuestion
                id="trust_3"
                question="Most people are basically good and kind."
                options={LIKERT_5_AGREE}
                value={responses.trust_3}
                onChange={handleChange}
              />
              <LikertQuestion
                id="trust_4"
                question="Most people are trustful of others."
                options={LIKERT_5_AGREE}
                value={responses.trust_4}
                onChange={handleChange}
              />
              <LikertQuestion
                id="trust_5"
                question="I am trustful."
                options={LIKERT_5_AGREE}
                value={responses.trust_5}
                onChange={handleChange}
              />
              <LikertQuestion
                id="trust_6"
                question="Most people will respond in kind when they are trusted by others."
                options={LIKERT_5_AGREE}
                value={responses.trust_6}
                onChange={handleChange}
              />
            </div>
          )}
        </div>

        {/* Sections 6-10 (Phishing Self-Efficacy, Perceived Risk, SA-6, Knowledge Quiz,
                    Prior Experience) moved to PostExperimentSurvey to avoid priming */}

        {/* Section 6: Email Habits */}
        <div className="bg-white rounded-lg shadow-sm mb-4 overflow-hidden">
          <SectionHeader
            title="6. Email Habits"
            description="Your email usage patterns"
            isOpen={openSections.email}
            onToggle={() => toggleSection("email")}
            isComplete={isSectionComplete([
              "daily_email_volume",
              "email_check_frequency",
              "social_media_usage",
            ])}
          />
          {openSections.email && (
            <div className="p-6">
              <SelectQuestion
                id="daily_email_volume"
                question="How many emails do you typically receive per day?"
                options={[
                  { value: "0-10", label: "0-10 emails" },
                  { value: "11-25", label: "11-25 emails" },
                  { value: "26-50", label: "26-50 emails" },
                  { value: "51-100", label: "51-100 emails" },
                  { value: "100+", label: "More than 100 emails" },
                ]}
                value={responses.daily_email_volume}
                onChange={handleChange}
              />
              <SelectQuestion
                id="email_check_frequency"
                question="How often do you check your email?"
                options={[
                  { value: "few_times_week", label: "A few times per week" },
                  { value: "once_daily", label: "Once daily" },
                  { value: "few_times_day", label: "A few times per day" },
                  { value: "hourly", label: "Hourly" },
                  { value: "constantly", label: "Constantly / Always open" },
                ]}
                value={responses.email_check_frequency}
                onChange={handleChange}
              />
              {/* link_click_tendency moved to post-survey to avoid priming */}
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

        {/* Submit Button */}
        <div className="bg-white rounded-lg shadow-sm p-6">
          <div className="flex items-center justify-between">
            <div className="text-gray-600">
              <p className="font-medium">Ready to proceed?</p>
              <p className="text-sm">
                Please ensure all sections are complete.
              </p>
            </div>
            <button
              onClick={handleSubmit}
              disabled={isSubmitting}
              className={`px-8 py-3 rounded-lg font-semibold transition-colors ${
                isSubmitting
                  ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                  : "bg-[#0078d4] text-white hover:bg-[#106ebe]"
              }`}
            >
              {isSubmitting ? "Submitting..." : "Continue to Study"}
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
