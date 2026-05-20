/**
 * CYPEARL Dark Patterns Pre-Experiment Survey Component
 *
 * Single-page questionnaire containing all pre-experiment validated instruments:
 *
 * CORE TRAIT SPINE (Shared with Phishing - 53 items):
 * - Demographics (7 items)
 * - CRT-7 Cognitive Reflection Test (7 items)
 * - Need for Cognition NFC-6 Short Form (6 items)
 * - Working Memory (1 item)
 * - Big Five BFI-10 (10 items)
 * - Impulsivity UPPS-P Short Form (4 items)
 * - Sensation Seeking BSSS-4 (4 items)
 * - Trust Propensity (6 items)
 * - Risk Taking DOSPERT-6 (6 items)
 * - Influence Susceptibility (9 items)
 *
 * DARK PATTERNS SPECIFIC (13 items):
 * - Digital Literacy Scale (4 items)
 * - Impulse Buying Tendency (4 items)
 * - Online Shopping Experience (2 items)
 * - Dark Pattern Awareness (3 items)
 *
 * MODIFIED FROM PHISHING:
 * - Manipulation Detection Efficacy (4 items)
 * - Perceived Online Risk (4 items)
 */

import React, { useState, useEffect } from "react";
import {
  ChevronDown,
  ChevronUp,
  CheckCircle,
  MousePointer2,
  ArrowRight,
} from "lucide-react";

const PRE_SURVEY_RESPONSES_KEY = "dp_pre_survey_responses_draft";

// ============================================================================
// SURVEY SCALES
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
// REUSABLE COMPONENTS
// ============================================================================

const SectionHeader = ({
  title,
  description,
  isOpen,
  onToggle,
  isComplete,
}) => (
  <div
    className="flex items-center justify-between p-4 bg-purple-50 border border-purple-200 rounded-lg cursor-pointer hover:bg-purple-100 transition-colors"
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
              ? "bg-purple-50 border-purple-500 text-purple-700"
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
      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
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
      className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
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

const PreSurveyDarkPatterns = ({
  onComplete,
  onGoToExperiment,
  onGoToScenarioSelector,
}) => {
  const [responses, setResponses] = useState(() => {
    const saved = localStorage.getItem(PRE_SURVEY_RESPONSES_KEY);
    return saved ? JSON.parse(saved) : {};
  });
  const [openSections, setOpenSections] = useState({ demographics: true });
  const [errors, setErrors] = useState([]);
  const [isSubmitting, setIsSubmitting] = useState(false);

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
      // CRT (simplified - 3 items for brevity)
      "crt_1",
      "crt_2",
      "crt_3",
      // NFC (4 items)
      "nfc_1",
      "nfc_2",
      "nfc_3",
      "nfc_4",
      // Big Five (10 items)
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
      // Impulsivity (4 items)
      "imp_1",
      "imp_2",
      "imp_3",
      "imp_4",
      // Trust (4 items)
      "trust_1",
      "trust_2",
      "trust_3",
      "trust_4",
      // Influence Susceptibility (6 items)
      "auth_1",
      "auth_2",
      "urg_1",
      "urg_2",
      "scar_1",
      "scar_2",
      // Digital Literacy (4 items)
      "dl_1",
      "dl_2",
      "dl_3",
      "dl_4",
      // Impulse Buying (4 items)
      "ib_1",
      "ib_2",
      "ib_3",
      "ib_4",
      // Online Shopping Experience
      "shopping_frequency",
      "years_online_shopping",
      // Dark Pattern Awareness
      "dpa_1",
      "dpa_2",
      "dpa_3",
      // Manipulation Detection
      "mde_1",
      "mde_2",
      "mde_3",
      "mde_4",
    ];

    const missing = requiredFields.filter(
      (field) => responses[field] === undefined || responses[field] === "",
    );
    return missing;
  };

  const computeScores = () => {
    // CRT Score (count correct)
    const crt_answers = { crt_1: 5, crt_2: 5, crt_3: 47 }; // bat/ball, machines, lily pad
    const crt_score = ["crt_1", "crt_2", "crt_3"].filter(
      (id) => parseInt(responses[id]) === crt_answers[id],
    ).length;

    // NFC Score (average)
    const nfc_items = ["nfc_1", "nfc_2", "nfc_3", "nfc_4"];
    const nfc_score =
      nfc_items.reduce((sum, id) => sum + (responses[id] || 0), 0) /
      nfc_items.length;

    // Big Five (simplified - just extraversion and conscientiousness for now)
    const big5_extraversion =
      ((responses["bfi_1"] || 3) + (6 - (responses["bfi_6"] || 3))) / 2;
    const big5_conscientiousness =
      ((responses["bfi_3"] || 3) + (6 - (responses["bfi_8"] || 3))) / 2;
    const big5_agreeableness =
      (6 - (responses["bfi_2"] || 3) + (responses["bfi_7"] || 3)) / 2;
    const big5_neuroticism =
      ((responses["bfi_4"] || 3) + (6 - (responses["bfi_9"] || 3))) / 2;
    const big5_openness =
      ((responses["bfi_5"] || 3) + (responses["bfi_10"] || 3)) / 2;

    // Impulsivity
    const imp_items = ["imp_1", "imp_2", "imp_3", "imp_4"];
    const impulsivity_total =
      imp_items.reduce((sum, id) => sum + (responses[id] || 0), 0) /
      imp_items.length;

    // Trust Propensity
    const trust_items = ["trust_1", "trust_2", "trust_3", "trust_4"];
    const trust_propensity =
      trust_items.reduce((sum, id) => sum + (responses[id] || 0), 0) /
      trust_items.length;

    // Influence Susceptibility
    const authority_susceptibility =
      ((responses["auth_1"] || 0) + (responses["auth_2"] || 0)) / 2;
    const urgency_susceptibility =
      ((responses["urg_1"] || 0) + (responses["urg_2"] || 0)) / 2;
    const scarcity_susceptibility =
      ((responses["scar_1"] || 0) + (responses["scar_2"] || 0)) / 2;

    // Digital Literacy
    const dl_items = ["dl_1", "dl_2", "dl_3", "dl_4"];
    const digital_literacy =
      dl_items.reduce((sum, id) => sum + (responses[id] || 0), 0) /
      dl_items.length;

    // Impulse Buying
    const ib_items = ["ib_1", "ib_2", "ib_3", "ib_4"];
    const impulse_buying =
      ib_items.reduce((sum, id) => sum + (responses[id] || 0), 0) /
      ib_items.length;

    // Manipulation Detection Efficacy
    const mde_items = ["mde_1", "mde_2", "mde_3", "mde_4"];
    const manipulation_detection_efficacy =
      mde_items.reduce((sum, id) => sum + (responses[id] || 0), 0) /
      mde_items.length;

    return {
      crt_score,
      need_for_cognition: nfc_score,
      big5_extraversion,
      big5_agreeableness,
      big5_conscientiousness,
      big5_neuroticism,
      big5_openness,
      impulsivity_total,
      trust_propensity,
      authority_susceptibility,
      urgency_susceptibility,
      scarcity_susceptibility,
      digital_literacy,
      impulse_buying,
      manipulation_detection_efficacy,
    };
  };

  const handleSubmit = async () => {
    // TESTING MODE: Validation disabled - uncomment below for production
    /*
    const missingFields = validateForm();
    if (missingFields.length > 0) {
      setErrors(missingFields);
      alert(`Please complete all required fields. Missing: ${missingFields.length} items.`);
      return;
    }
    */

    setIsSubmitting(true);
    setErrors([]);

    const scores = computeScores();
    const submissionData = {
      ...responses,
      ...scores,
      submitted_at: new Date().toISOString(),
    };

    localStorage.removeItem(PRE_SURVEY_RESPONSES_KEY);
    setIsSubmitting(false);
    onComplete(submissionData);
  };

  // ========================================================================
  // RENDER
  // ========================================================================

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-slate-100 py-8 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 bg-purple-600 rounded-xl flex items-center justify-center">
              <MousePointer2 className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-gray-800">
              Pre-Experiment Survey
            </h1>
          </div>
          <p className="text-gray-600">
            Please answer the following questions honestly. There are no right
            or wrong answers.
          </p>
          <p className="text-sm text-purple-600 mt-2">
            For demo purpose, you can move to the next section without filling
            out the survey. In production, all questions will be required.
          </p>
        </div>

        {/* Survey Sections */}
        <div className="space-y-4">
          {/* Demographics */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="About You"
              description="Basic demographic information"
              isOpen={openSections.demographics}
              onToggle={() => toggleSection("demographics")}
              isComplete={isSectionComplete([
                "age",
                "gender",
                "education",
                "employment",
              ])}
            />
            {openSections.demographics && (
              <div className="p-6">
                <TextInputQuestion
                  id="age"
                  question="What is your age?"
                  type="number"
                  value={responses.age}
                  onChange={handleChange}
                  placeholder="Enter your age"
                />
                <SelectQuestion
                  id="gender"
                  question="What is your gender?"
                  value={responses.gender}
                  onChange={handleChange}
                  options={[
                    { value: "male", label: "Male" },
                    { value: "female", label: "Female" },
                    { value: "non_binary", label: "Non-binary" },
                    { value: "prefer_not", label: "Prefer not to say" },
                    { value: "other", label: "Other" },
                  ]}
                />
                <SelectQuestion
                  id="education"
                  question="What is your highest level of education?"
                  value={responses.education}
                  onChange={handleChange}
                  options={[
                    { value: "high_school", label: "High School" },
                    { value: "some_college", label: "Some College" },
                    { value: "bachelors", label: "Bachelor's Degree" },
                    { value: "masters", label: "Master's Degree" },
                    { value: "doctorate", label: "Doctorate" },
                  ]}
                />
                <SelectQuestion
                  id="technical_field"
                  question="Do you work in a technical field (IT, engineering, etc.)?"
                  value={responses.technical_field}
                  onChange={handleChange}
                  options={[
                    { value: "yes", label: "Yes" },
                    { value: "no", label: "No" },
                  ]}
                />
                <SelectQuestion
                  id="employment"
                  question="What is your current employment status?"
                  value={responses.employment}
                  onChange={handleChange}
                  options={[
                    { value: "full_time", label: "Full-time employed" },
                    { value: "part_time", label: "Part-time employed" },
                    { value: "self_employed", label: "Self-employed" },
                    { value: "student", label: "Student" },
                    { value: "unemployed", label: "Unemployed" },
                    { value: "retired", label: "Retired" },
                  ]}
                />
              </div>
            )}
          </div>

          {/* Cognitive Reflection Test */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="Problem Solving"
              description="Three quick thinking puzzles"
              isOpen={openSections.crt}
              onToggle={() => toggleSection("crt")}
              isComplete={isSectionComplete(["crt_1", "crt_2", "crt_3"])}
            />
            {openSections.crt && (
              <div className="p-6">
                <p className="text-gray-600 mb-4 text-sm">
                  Please answer these questions. Take your time and think
                  carefully.
                </p>
                <TextInputQuestion
                  id="crt_1"
                  question="A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? (in cents)"
                  type="number"
                  value={responses.crt_1}
                  onChange={handleChange}
                  placeholder="Enter amount in cents"
                />
                <TextInputQuestion
                  id="crt_2"
                  question="If it takes 5 machines 5 minutes to make 5 widgets, how long would it take 100 machines to make 100 widgets? (in minutes)"
                  type="number"
                  value={responses.crt_2}
                  onChange={handleChange}
                  placeholder="Enter time in minutes"
                />
                <TextInputQuestion
                  id="crt_3"
                  question="In a lake, there is a patch of lily pads. Every day, the patch doubles in size. If it takes 48 days for the patch to cover the entire lake, how long would it take for the patch to cover half of the lake? (in days)"
                  type="number"
                  value={responses.crt_3}
                  onChange={handleChange}
                  placeholder="Enter number of days"
                />
              </div>
            )}
          </div>

          {/* Need for Cognition */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="Thinking Style"
              description="How you approach mental tasks"
              isOpen={openSections.nfc}
              onToggle={() => toggleSection("nfc")}
              isComplete={isSectionComplete([
                "nfc_1",
                "nfc_2",
                "nfc_3",
                "nfc_4",
              ])}
            />
            {openSections.nfc && (
              <div className="p-6">
                <LikertQuestion
                  id="nfc_1"
                  question="I would prefer complex to simple problems."
                  options={LIKERT_5_AGREE}
                  value={responses.nfc_1}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="nfc_2"
                  question="I like to have the responsibility of handling a situation that requires a lot of thinking."
                  options={LIKERT_5_AGREE}
                  value={responses.nfc_2}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="nfc_3"
                  question="Thinking is not my idea of fun."
                  options={LIKERT_5_AGREE}
                  value={responses.nfc_3}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="nfc_4"
                  question="I prefer my life to be filled with puzzles that I must solve."
                  options={LIKERT_5_AGREE}
                  value={responses.nfc_4}
                  onChange={handleChange}
                />
              </div>
            )}
          </div>

          {/* Big Five Personality */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="Personality"
              description="How you see yourself (BFI-10)"
              isOpen={openSections.bigfive}
              onToggle={() => toggleSection("bigfive")}
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
            {openSections.bigfive && (
              <div className="p-6">
                <p className="text-gray-600 mb-4">
                  I see myself as someone who...
                </p>
                {[
                  { id: "bfi_1", q: "is reserved" },
                  { id: "bfi_2", q: "is generally trusting" },
                  { id: "bfi_3", q: "tends to be lazy" },
                  { id: "bfi_4", q: "is relaxed, handles stress well" },
                  { id: "bfi_5", q: "has few artistic interests" },
                  { id: "bfi_6", q: "is outgoing, sociable" },
                  { id: "bfi_7", q: "tends to find fault with others" },
                  { id: "bfi_8", q: "does a thorough job" },
                  { id: "bfi_9", q: "gets nervous easily" },
                  { id: "bfi_10", q: "has an active imagination" },
                ].map((item) => (
                  <LikertQuestion
                    key={item.id}
                    id={item.id}
                    question={`...${item.q}`}
                    options={LIKERT_5_AGREE}
                    value={responses[item.id]}
                    onChange={handleChange}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Impulsivity */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="Behavioral Tendencies"
              description="How you typically act"
              isOpen={openSections.impulsivity}
              onToggle={() => toggleSection("impulsivity")}
              isComplete={isSectionComplete([
                "imp_1",
                "imp_2",
                "imp_3",
                "imp_4",
              ])}
            />
            {openSections.impulsivity && (
              <div className="p-6">
                <LikertQuestion
                  id="imp_1"
                  question="I tend to act on the spur of the moment without being aware of the risks."
                  options={LIKERT_5_CHARACTERISTIC}
                  value={responses.imp_1}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="imp_2"
                  question="When I feel bad, I will often do things I later regret in order to make myself feel better now."
                  options={LIKERT_5_CHARACTERISTIC}
                  value={responses.imp_2}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="imp_3"
                  question="I often get involved in things I later wish I could get out of."
                  options={LIKERT_5_CHARACTERISTIC}
                  value={responses.imp_3}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="imp_4"
                  question="I finish what I start."
                  options={LIKERT_5_CHARACTERISTIC}
                  value={responses.imp_4}
                  onChange={handleChange}
                />
              </div>
            )}
          </div>

          {/* Trust Propensity */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="Trust"
              description="How you trust others and organizations"
              isOpen={openSections.trust}
              onToggle={() => toggleSection("trust")}
              isComplete={isSectionComplete([
                "trust_1",
                "trust_2",
                "trust_3",
                "trust_4",
              ])}
            />
            {openSections.trust && (
              <div className="p-6">
                <LikertQuestion
                  id="trust_1"
                  question="Most people can be counted on to do what they say they will do."
                  options={LIKERT_5_AGREE}
                  value={responses.trust_1}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="trust_2"
                  question="Most companies are honest in their dealings with customers."
                  options={LIKERT_5_AGREE}
                  value={responses.trust_2}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="trust_3"
                  question="Websites generally have my best interests at heart."
                  options={LIKERT_5_AGREE}
                  value={responses.trust_3}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="trust_4"
                  question="Most online businesses operate fairly."
                  options={LIKERT_5_AGREE}
                  value={responses.trust_4}
                  onChange={handleChange}
                />
              </div>
            )}
          </div>

          {/* Influence Susceptibility */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="Decision Making"
              description="What influences your decisions"
              isOpen={openSections.influence}
              onToggle={() => toggleSection("influence")}
              isComplete={isSectionComplete([
                "auth_1",
                "auth_2",
                "urg_1",
                "urg_2",
                "scar_1",
                "scar_2",
              ])}
            />
            {openSections.influence && (
              <div className="p-6">
                <p className="text-gray-600 mb-4 text-sm">
                  How likely are you to be influenced by each of the following?
                </p>
                <LikertQuestion
                  id="auth_1"
                  question="A message from an authority figure (e.g., 'Recommended by experts')"
                  options={LIKERT_5_LIKELY}
                  value={responses.auth_1}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="auth_2"
                  question="An official-looking seal or certification badge"
                  options={LIKERT_5_LIKELY}
                  value={responses.auth_2}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="urg_1"
                  question="A countdown timer showing limited time remaining"
                  options={LIKERT_5_LIKELY}
                  value={responses.urg_1}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="urg_2"
                  question="A message saying 'Act now before it's too late'"
                  options={LIKERT_5_LIKELY}
                  value={responses.urg_2}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="scar_1"
                  question="A message saying 'Only 2 items left in stock'"
                  options={LIKERT_5_LIKELY}
                  value={responses.scar_1}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="scar_2"
                  question="A notification that others are viewing the same item"
                  options={LIKERT_5_LIKELY}
                  value={responses.scar_2}
                  onChange={handleChange}
                />
              </div>
            )}
          </div>

          {/* Digital Literacy (Dark Patterns Specific) */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden border-2 border-purple-200">
            <SectionHeader
              title="Digital Skills"
              description="Your experience with websites and apps"
              isOpen={openSections.digital}
              onToggle={() => toggleSection("digital")}
              isComplete={isSectionComplete(["dl_1", "dl_2", "dl_3", "dl_4"])}
            />
            {openSections.digital && (
              <div className="p-6">
                <LikertQuestion
                  id="dl_1"
                  question="I know how to adjust privacy settings on websites and apps."
                  options={LIKERT_5_AGREE}
                  value={responses.dl_1}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="dl_2"
                  question="I can identify when a website is trying to manipulate me into clicking something."
                  options={LIKERT_5_AGREE}
                  value={responses.dl_2}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="dl_3"
                  question="I understand what happens when I accept cookies on a website."
                  options={LIKERT_5_AGREE}
                  value={responses.dl_3}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="dl_4"
                  question="I know how to cancel online subscriptions."
                  options={LIKERT_5_AGREE}
                  value={responses.dl_4}
                  onChange={handleChange}
                />
              </div>
            )}
          </div>

          {/* Impulse Buying (Dark Patterns Specific) */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden border-2 border-purple-200">
            <SectionHeader
              title="Shopping Habits"
              description="How you make purchasing decisions"
              isOpen={openSections.impulse}
              onToggle={() => toggleSection("impulse")}
              isComplete={isSectionComplete(["ib_1", "ib_2", "ib_3", "ib_4"])}
            />
            {openSections.impulse && (
              <div className="p-6">
                <LikertQuestion
                  id="ib_1"
                  question="I often buy things spontaneously."
                  options={LIKERT_5_AGREE}
                  value={responses.ib_1}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="ib_2"
                  question="'Just do it' describes the way I buy things."
                  options={LIKERT_5_AGREE}
                  value={responses.ib_2}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="ib_3"
                  question="I often buy things without thinking."
                  options={LIKERT_5_AGREE}
                  value={responses.ib_3}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="ib_4"
                  question="'Buy now, think about it later' describes me."
                  options={LIKERT_5_AGREE}
                  value={responses.ib_4}
                  onChange={handleChange}
                />
              </div>
            )}
          </div>

          {/* Online Shopping Experience */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden border-2 border-purple-200">
            <SectionHeader
              title="Online Shopping Experience"
              description="Your experience with online shopping"
              isOpen={openSections.shopping}
              onToggle={() => toggleSection("shopping")}
              isComplete={isSectionComplete([
                "shopping_frequency",
                "years_online_shopping",
              ])}
            />
            {openSections.shopping && (
              <div className="p-6">
                <SelectQuestion
                  id="shopping_frequency"
                  question="How often do you shop online?"
                  value={responses.shopping_frequency}
                  onChange={handleChange}
                  options={[
                    { value: "never", label: "Never" },
                    { value: "few_times_year", label: "A few times a year" },
                    { value: "monthly", label: "Monthly" },
                    { value: "weekly", label: "Weekly" },
                    { value: "daily", label: "Daily" },
                  ]}
                />
                <TextInputQuestion
                  id="years_online_shopping"
                  question="How many years have you been shopping online?"
                  type="number"
                  value={responses.years_online_shopping}
                  onChange={handleChange}
                  placeholder="Enter number of years"
                />
              </div>
            )}
          </div>

          {/* Dark Pattern Awareness */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden border-2 border-purple-200">
            <SectionHeader
              title="Website Design Awareness"
              description="Your experience with manipulative designs"
              isOpen={openSections.awareness}
              onToggle={() => toggleSection("awareness")}
              isComplete={isSectionComplete(["dpa_1", "dpa_2", "dpa_3"])}
            />
            {openSections.awareness && (
              <div className="p-6">
                <SelectQuestion
                  id="dpa_1"
                  question="Have you heard of the term 'dark patterns' in website/app design?"
                  value={responses.dpa_1}
                  onChange={handleChange}
                  options={[
                    { value: "yes", label: "Yes" },
                    { value: "no", label: "No" },
                    { value: "not_sure", label: "Not sure" },
                  ]}
                />
                <SelectQuestion
                  id="dpa_2"
                  question="Have you ever felt tricked by a website into doing something you didn't intend?"
                  value={responses.dpa_2}
                  onChange={handleChange}
                  options={[
                    { value: "never", label: "Never" },
                    { value: "once_twice", label: "Once or twice" },
                    { value: "several_times", label: "Several times" },
                    { value: "many_times", label: "Many times" },
                  ]}
                />
                <LikertQuestion
                  id="dpa_3"
                  question="How confident are you in recognizing manipulative website designs?"
                  options={LIKERT_5_AGREE}
                  value={responses.dpa_3}
                  onChange={handleChange}
                />
              </div>
            )}
          </div>

          {/* Manipulation Detection Efficacy */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden border-2 border-purple-200">
            <SectionHeader
              title="Manipulation Detection"
              description="Your confidence in spotting tricks"
              isOpen={openSections.detection}
              onToggle={() => toggleSection("detection")}
              isComplete={isSectionComplete([
                "mde_1",
                "mde_2",
                "mde_3",
                "mde_4",
              ])}
            />
            {openSections.detection && (
              <div className="p-6">
                <LikertQuestion
                  id="mde_1"
                  question="I can tell when a website is trying to manipulate me."
                  options={LIKERT_5_AGREE}
                  value={responses.mde_1}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="mde_2"
                  question="I know how to avoid being tricked by misleading website designs."
                  options={LIKERT_5_AGREE}
                  value={responses.mde_2}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="mde_3"
                  question="I can protect myself from deceptive online practices."
                  options={LIKERT_5_AGREE}
                  value={responses.mde_3}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="mde_4"
                  question="I read the fine print before agreeing to anything online."
                  options={LIKERT_5_AGREE}
                  value={responses.mde_4}
                  onChange={handleChange}
                />
              </div>
            )}
          </div>
        </div>

        {/* Submit Button */}
        <div className="mt-8 flex justify-end items-center">
          <button
            onClick={handleSubmit}
            disabled={isSubmitting}
            className="bg-purple-600 text-white py-3 px-8 rounded-xl font-semibold hover:bg-purple-700 transition-colors flex items-center gap-2 disabled:opacity-50"
          >
            {isSubmitting ? "Submitting..." : "Continue to Experiment"}
            <ArrowRight size={18} />
          </button>
        </div>

        {/* Error Display */}
        {errors.length > 0 && (
          <div className="mt-4 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-600 text-sm">
              Please complete all required fields ({errors.length} missing).
            </p>
          </div>
        )}
      </div>

      {/* TESTING ONLY: Navigation buttons */}
      <div className="fixed bottom-0 left-0 right-0 px-4 py-2 flex justify-between items-center z-50 pointer-events-none">
        {onGoToScenarioSelector && (
          <button
            onClick={onGoToScenarioSelector}
            className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium pointer-events-auto"
          >
            ← Scenario Selector
          </button>
        )}
        {onGoToExperiment && (
          <button
            onClick={onGoToExperiment}
            className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium pointer-events-auto ml-auto"
          >
            Go to Dark Patterns Experiment →
          </button>
        )}
      </div>
    </div>
  );
};

export default PreSurveyDarkPatterns;
