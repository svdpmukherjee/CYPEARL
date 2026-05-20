/**
 * CYPEARL Fake News Pre-Experiment Survey Component
 *
 * Single-page questionnaire containing all pre-experiment validated instruments:
 *
 * CORE TRAIT SPINE (Shared - 53 items):
 * - Demographics (7 items)
 * - CRT-7 Cognitive Reflection Test (3 items simplified)
 * - Need for Cognition NFC-6 Short Form (4 items)
 * - Big Five BFI-10 (10 items)
 * - Impulsivity UPPS-P Short Form (4 items)
 * - Trust Propensity (4 items)
 * - Influence Susceptibility (6 items)
 *
 * FAKE NEWS SPECIFIC (23 items):
 * - Political Ideology (2 items) - CRITICAL for congruence calculation
 * - Conspiracy Mentality CMQ-5 (5 items)
 * - News Media Literacy NML-6 (6 items)
 * - Bullshit Receptivity BSR-4 (4 items)
 * - Trust in Media (3 items)
 * - Fear of Missing Out FoMOs-3 (3 items)
 *
 * MODIFIED FROM PHISHING:
 * - Fake News Detection Efficacy (4 items)
 * - News Consumption Habits (4 items)
 */

import React, { useState, useEffect } from "react";
import {
  ChevronDown,
  ChevronUp,
  CheckCircle,
  Newspaper,
  ArrowRight,
} from "lucide-react";

const PRE_SURVEY_RESPONSES_KEY = "fn_pre_survey_responses_draft";

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

const LIKERT_5_PROFOUND = [
  { value: 1, label: "Not at all profound" },
  { value: 2, label: "Slightly profound" },
  { value: 3, label: "Moderately profound" },
  { value: 4, label: "Very profound" },
  { value: 5, label: "Extremely profound" },
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
    className="flex items-center justify-between p-4 bg-gray-50 border border-blue-200 rounded-lg cursor-pointer hover:bg-blue-100 transition-colors"
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

const PreSurveyFakeNews = ({
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
      // Political Ideology (CRITICAL)
      "political_ideology",
      "party_id",
      // CRT
      "crt_1",
      "crt_2",
      "crt_3",
      // NFC
      "nfc_1",
      "nfc_2",
      "nfc_3",
      "nfc_4",
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
      // Impulsivity
      "imp_1",
      "imp_2",
      "imp_3",
      "imp_4",
      // Trust
      "trust_1",
      "trust_2",
      "trust_3",
      "trust_4",
      // Influence
      "auth_1",
      "auth_2",
      "urg_1",
      "urg_2",
      "scar_1",
      "scar_2",
      // Conspiracy Mentality
      "cmq_1",
      "cmq_2",
      "cmq_3",
      "cmq_4",
      "cmq_5",
      // News Media Literacy
      "nml_1",
      "nml_2",
      "nml_3",
      "nml_4",
      "nml_5",
      "nml_6",
      // Bullshit Receptivity
      "bsr_1",
      "bsr_2",
      "bsr_3",
      "bsr_4",
      // Trust in Media
      "tm_1",
      "tm_2",
      "tm_3",
      // FOMO
      "fomo_1",
      "fomo_2",
      "fomo_3",
      // Fake News Detection
      "fnde_1",
      "fnde_2",
      "fnde_3",
      "fnde_4",
      // News Consumption
      "news_freq",
      "news_source",
    ];

    const missing = requiredFields.filter(
      (field) => responses[field] === undefined || responses[field] === "",
    );
    return missing;
  };

  const computeScores = () => {
    // CRT Score
    const crt_answers = { crt_1: 5, crt_2: 5, crt_3: 47 };
    const crt_score = ["crt_1", "crt_2", "crt_3"].filter(
      (id) => parseInt(responses[id]) === crt_answers[id],
    ).length;

    // NFC Score
    const nfc_items = ["nfc_1", "nfc_2", "nfc_3", "nfc_4"];
    const nfc_score =
      nfc_items.reduce((sum, id) => sum + (responses[id] || 0), 0) /
      nfc_items.length;

    // Big Five
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

    // Conspiracy Mentality
    const cmq_items = ["cmq_1", "cmq_2", "cmq_3", "cmq_4", "cmq_5"];
    const conspiracy_mentality =
      cmq_items.reduce((sum, id) => sum + (responses[id] || 0), 0) /
      cmq_items.length;

    // News Media Literacy
    const nml_items = ["nml_1", "nml_2", "nml_3", "nml_4", "nml_5", "nml_6"];
    const news_media_literacy =
      nml_items.reduce((sum, id) => sum + (responses[id] || 0), 0) /
      nml_items.length;

    // Bullshit Receptivity
    const bsr_items = ["bsr_1", "bsr_2", "bsr_3", "bsr_4"];
    const bullshit_receptivity =
      bsr_items.reduce((sum, id) => sum + (responses[id] || 0), 0) /
      bsr_items.length;

    // Trust in Media
    const tm_items = ["tm_1", "tm_2", "tm_3"];
    const trust_in_media =
      tm_items.reduce((sum, id) => sum + (responses[id] || 0), 0) /
      tm_items.length;

    // FOMO
    const fomo_items = ["fomo_1", "fomo_2", "fomo_3"];
    const fomo =
      fomo_items.reduce((sum, id) => sum + (responses[id] || 0), 0) /
      fomo_items.length;

    // Fake News Detection Efficacy
    const fnde_items = ["fnde_1", "fnde_2", "fnde_3", "fnde_4"];
    const fake_news_detection_efficacy =
      fnde_items.reduce((sum, id) => sum + (responses[id] || 0), 0) /
      fnde_items.length;

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
      conspiracy_mentality,
      news_media_literacy,
      bullshit_receptivity,
      trust_in_media,
      fomo,
      fake_news_detection_efficacy,
    };
  };

  const handleSubmit = async () => {
    // TESTING MODE: Validation disabled - uncomment below for production
    /*
    const missingFields = validateForm();
    if (missingFields.length > 0) {
      setErrors(missingFields);
      alert(
        `Please complete all required fields. Missing: ${missingFields.length} items.`,
      );
      return;
    }
    */

    setIsSubmitting(true);
    setErrors([]);

    const scores = computeScores();
    const submissionData = {
      ...responses,
      ...scores,
      political_ideology: parseInt(responses.political_ideology),
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
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-slate-100 py-8 px-4">
      <div className="max-w-3xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center">
              <Newspaper className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-gray-800">
              Pre-Experiment Survey
            </h1>
          </div>
          <p className="text-gray-600">
            Please answer the following questions honestly. There are no right
            or wrong answers.
          </p>
          <p className="text-sm text-orange-600 mt-2">
            For demo purpose, you can move to the next section without filling
            out the survey. In production, all questions will be required.
          </p>
          {/* <p className="text-sm text-gray-600 mt-2">News Evaluation Study</p> */}
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

          {/* Political Ideology - CRITICAL */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="Political Views"
              description="Your political orientation"
              isOpen={openSections.political}
              onToggle={() => toggleSection("political")}
              isComplete={isSectionComplete(["political_ideology", "party_id"])}
            />
            {openSections.political && (
              <div className="p-6">
                <div className="mb-6">
                  <p className="text-gray-800 mb-3">
                    In terms of political views, where would you place yourself?
                    <span className="text-red-500 ml-1">*</span>
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {[
                      { value: 1, label: "Very Liberal" },
                      { value: 2, label: "Liberal" },
                      { value: 3, label: "Slightly Liberal" },
                      { value: 4, label: "Moderate" },
                      { value: 5, label: "Slightly Conservative" },
                      { value: 6, label: "Conservative" },
                      { value: 7, label: "Very Conservative" },
                    ].map((option) => (
                      <label
                        key={option.value}
                        className={`flex items-center gap-2 px-3 py-2 rounded-lg border cursor-pointer transition-colors ${
                          responses.political_ideology === option.value
                            ? "bg-orange-50 border-orange-500 text-orange-700"
                            : "bg-gray-50 border-gray-200 hover:bg-gray-100"
                        }`}
                      >
                        <input
                          type="radio"
                          name="political_ideology"
                          value={option.value}
                          checked={
                            responses.political_ideology === option.value
                          }
                          onChange={() =>
                            handleChange("political_ideology", option.value)
                          }
                          className="sr-only"
                        />
                        <span className="text-sm">{option.label}</span>
                      </label>
                    ))}
                  </div>
                </div>
                <SelectQuestion
                  id="party_id"
                  question="Generally speaking, do you usually think of yourself as a..."
                  value={responses.party_id}
                  onChange={handleChange}
                  options={[
                    { value: "democrat", label: "Democrat" },
                    { value: "republican", label: "Republican" },
                    { value: "independent", label: "Independent" },
                    { value: "other", label: "Other" },
                    { value: "none", label: "No preference" },
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
                  { id: "bfi_1", text: "...is reserved" },
                  { id: "bfi_2", text: "...is generally trusting" },
                  { id: "bfi_3", text: "...tends to be lazy" },
                  { id: "bfi_4", text: "...is relaxed, handles stress well" },
                  {
                    id: "bfi_5",
                    text: "...has few artistic interests",
                  },
                  { id: "bfi_6", text: "...is outgoing, sociable" },
                  {
                    id: "bfi_7",
                    text: "...tends to find fault with others",
                  },
                  { id: "bfi_8", text: "...does a thorough job" },
                  { id: "bfi_9", text: "...gets nervous easily" },
                  {
                    id: "bfi_10",
                    text: "...has an active imagination",
                  },
                ].map((item) => (
                  <LikertQuestion
                    key={item.id}
                    id={item.id}
                    question={item.text}
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
              title="Decision Making"
              description="How you make decisions"
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
                {[
                  {
                    id: "imp_1",
                    text: "I have trouble controlling my impulses.",
                  },
                  {
                    id: "imp_2",
                    text: "I often act without thinking through all the alternatives.",
                  },
                  {
                    id: "imp_3",
                    text: "I am always able to keep my feelings under control.",
                  },
                  {
                    id: "imp_4",
                    text: "I usually think carefully before doing anything.",
                  },
                ].map((item) => (
                  <LikertQuestion
                    key={item.id}
                    id={item.id}
                    question={item.text}
                    options={LIKERT_5_AGREE}
                    value={responses[item.id]}
                    onChange={handleChange}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Trust Propensity */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="Trust"
              description="Your general tendency to trust"
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
                {[
                  { id: "trust_1", text: "Most people can be trusted." },
                  {
                    id: "trust_2",
                    text: "Most people would try to take advantage of you if they got a chance.",
                  },
                  { id: "trust_3", text: "Most people try to be fair." },
                  {
                    id: "trust_4",
                    text: "You can't be too careful in dealing with people.",
                  },
                ].map((item) => (
                  <LikertQuestion
                    key={item.id}
                    id={item.id}
                    question={item.text}
                    options={LIKERT_5_AGREE}
                    value={responses[item.id]}
                    onChange={handleChange}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Influence Susceptibility */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="Influence Factors"
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
                <p className="text-gray-600 mb-4 font-medium">Authority</p>
                <LikertQuestion
                  id="auth_1"
                  question="I am more likely to believe information that comes from an expert or authority figure."
                  options={LIKERT_5_AGREE}
                  value={responses.auth_1}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="auth_2"
                  question="I tend to follow recommendations from official sources without much questioning."
                  options={LIKERT_5_AGREE}
                  value={responses.auth_2}
                  onChange={handleChange}
                />

                <p className="text-gray-600 mb-4 mt-6 font-medium">Urgency</p>
                <LikertQuestion
                  id="urg_1"
                  question="When something seems urgent, I tend to act quickly without thinking carefully."
                  options={LIKERT_5_AGREE}
                  value={responses.urg_1}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="urg_2"
                  question="Time pressure makes me more likely to make hasty decisions."
                  options={LIKERT_5_AGREE}
                  value={responses.urg_2}
                  onChange={handleChange}
                />

                <p className="text-gray-600 mb-4 mt-6 font-medium">Scarcity</p>
                <LikertQuestion
                  id="scar_1"
                  question="I feel more compelled to act when I think an opportunity is limited."
                  options={LIKERT_5_AGREE}
                  value={responses.scar_1}
                  onChange={handleChange}
                />
                <LikertQuestion
                  id="scar_2"
                  question="I am influenced by messages that suggest something is rare or exclusive."
                  options={LIKERT_5_AGREE}
                  value={responses.scar_2}
                  onChange={handleChange}
                />
              </div>
            )}
          </div>

          {/* Conspiracy Mentality CMQ-5 */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="Beliefs About the World"
              description="Your views on events and institutions"
              isOpen={openSections.conspiracy}
              onToggle={() => toggleSection("conspiracy")}
              isComplete={isSectionComplete([
                "cmq_1",
                "cmq_2",
                "cmq_3",
                "cmq_4",
                "cmq_5",
              ])}
            />
            {openSections.conspiracy && (
              <div className="p-6">
                {[
                  {
                    id: "cmq_1",
                    text: "I think that many very important things happen in the world, which the public is never informed about.",
                  },
                  {
                    id: "cmq_2",
                    text: "I think that politicians usually do not tell us the true motives for their decisions.",
                  },
                  {
                    id: "cmq_3",
                    text: "I think that government agencies closely monitor all citizens.",
                  },
                  {
                    id: "cmq_4",
                    text: "I think that events which superficially seem to lack a connection are often the result of secret activities.",
                  },
                  {
                    id: "cmq_5",
                    text: "I think that there are secret organizations that greatly influence political decisions.",
                  },
                ].map((item) => (
                  <LikertQuestion
                    key={item.id}
                    id={item.id}
                    question={item.text}
                    options={LIKERT_5_AGREE}
                    value={responses[item.id]}
                    onChange={handleChange}
                  />
                ))}
              </div>
            )}
          </div>

          {/* News Media Literacy NML-6 */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="News Media Skills"
              description="Your ability to evaluate news"
              isOpen={openSections.media_literacy}
              onToggle={() => toggleSection("media_literacy")}
              isComplete={isSectionComplete([
                "nml_1",
                "nml_2",
                "nml_3",
                "nml_4",
                "nml_5",
                "nml_6",
              ])}
            />
            {openSections.media_literacy && (
              <div className="p-6">
                {[
                  {
                    id: "nml_1",
                    text: "I know how to tell if a news source is reliable.",
                  },
                  {
                    id: "nml_2",
                    text: "I can identify when a news story is trying to influence my opinion rather than inform me.",
                  },
                  {
                    id: "nml_3",
                    text: "I understand how news organizations decide what stories to cover.",
                  },
                  {
                    id: "nml_4",
                    text: "I can recognize the difference between news and opinion pieces.",
                  },
                  {
                    id: "nml_5",
                    text: "I check multiple sources before believing a news story.",
                  },
                  {
                    id: "nml_6",
                    text: "I can identify misleading headlines.",
                  },
                ].map((item) => (
                  <LikertQuestion
                    key={item.id}
                    id={item.id}
                    question={item.text}
                    options={LIKERT_5_AGREE}
                    value={responses[item.id]}
                    onChange={handleChange}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Bullshit Receptivity BSR-4 */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="Statement Evaluation"
              description="Rate how profound these statements are"
              isOpen={openSections.bsr}
              onToggle={() => toggleSection("bsr")}
              isComplete={isSectionComplete([
                "bsr_1",
                "bsr_2",
                "bsr_3",
                "bsr_4",
              ])}
            />
            {openSections.bsr && (
              <div className="p-6">
                <p className="text-gray-600 mb-4 text-sm">
                  Please rate how profound you find each of the following
                  statements:
                </p>
                {[
                  {
                    id: "bsr_1",
                    text: '"Wholeness quiets infinite phenomena."',
                  },
                  {
                    id: "bsr_2",
                    text: '"The future explains irrational facts."',
                  },
                  {
                    id: "bsr_3",
                    text: '"Consciousness is the growth of coherence, and of us."',
                  },
                  {
                    id: "bsr_4",
                    text: '"Hidden meaning transforms unparalleled abstract beauty."',
                  },
                ].map((item) => (
                  <LikertQuestion
                    key={item.id}
                    id={item.id}
                    question={item.text}
                    options={LIKERT_5_PROFOUND}
                    value={responses[item.id]}
                    onChange={handleChange}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Trust in Media */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="Media Trust"
              description="Your trust in different media"
              isOpen={openSections.media_trust}
              onToggle={() => toggleSection("media_trust")}
              isComplete={isSectionComplete(["tm_1", "tm_2", "tm_3"])}
            />
            {openSections.media_trust && (
              <div className="p-6">
                <SelectQuestion
                  id="tm_1"
                  question="In general, how much trust and confidence do you have in the mass media?"
                  value={responses.tm_1}
                  onChange={handleChange}
                  options={[
                    { value: 1, label: "None at all" },
                    { value: 2, label: "Not very much" },
                    { value: 3, label: "A fair amount" },
                    { value: 4, label: "A great deal" },
                  ]}
                />
                <SelectQuestion
                  id="tm_2"
                  question="How often do you think mainstream news organizations report fairly?"
                  value={responses.tm_2}
                  onChange={handleChange}
                  options={[
                    { value: 1, label: "Never" },
                    { value: 2, label: "Rarely" },
                    { value: 3, label: "Sometimes" },
                    { value: 4, label: "Often" },
                    { value: 5, label: "Always" },
                  ]}
                />
                <SelectQuestion
                  id="tm_3"
                  question="How much do you trust information from social media?"
                  value={responses.tm_3}
                  onChange={handleChange}
                  options={[
                    { value: 1, label: "None at all" },
                    { value: 2, label: "Not very much" },
                    { value: 3, label: "A fair amount" },
                    { value: 4, label: "A great deal" },
                  ]}
                />
              </div>
            )}
          </div>

          {/* FOMO */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="Social Experiences"
              description="Your feelings about missing out"
              isOpen={openSections.fomo}
              onToggle={() => toggleSection("fomo")}
              isComplete={isSectionComplete(["fomo_1", "fomo_2", "fomo_3"])}
            />
            {openSections.fomo && (
              <div className="p-6">
                {[
                  {
                    id: "fomo_1",
                    text: "I fear others have more rewarding experiences than me.",
                  },
                  {
                    id: "fomo_2",
                    text: "I fear my friends have more rewarding experiences than me.",
                  },
                  {
                    id: "fomo_3",
                    text: "I get worried when I find out my friends are having fun without me.",
                  },
                ].map((item) => (
                  <LikertQuestion
                    key={item.id}
                    id={item.id}
                    question={item.text}
                    options={LIKERT_5_AGREE}
                    value={responses[item.id]}
                    onChange={handleChange}
                  />
                ))}
              </div>
            )}
          </div>

          {/* Fake News Detection Efficacy */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="Detection Confidence"
              description="Your confidence in spotting misinformation"
              isOpen={openSections.detection}
              onToggle={() => toggleSection("detection")}
              isComplete={isSectionComplete([
                "fnde_1",
                "fnde_2",
                "fnde_3",
                "fnde_4",
              ])}
            />
            {openSections.detection && (
              <div className="p-6">
                {[
                  {
                    id: "fnde_1",
                    text: "I am confident in my ability to identify fake news.",
                  },
                  {
                    id: "fnde_2",
                    text: "I can usually tell when a news headline is misleading.",
                  },
                  {
                    id: "fnde_3",
                    text: "I have the skills needed to spot misinformation online.",
                  },
                  {
                    id: "fnde_4",
                    text: "Compared to others, I am better at identifying false information.",
                  },
                ].map((item) => (
                  <LikertQuestion
                    key={item.id}
                    id={item.id}
                    question={item.text}
                    options={LIKERT_5_AGREE}
                    value={responses[item.id]}
                    onChange={handleChange}
                  />
                ))}
              </div>
            )}
          </div>

          {/* News Consumption Habits */}
          <div className="bg-white rounded-xl shadow-sm overflow-hidden">
            <SectionHeader
              title="News Habits"
              description="How you consume news"
              isOpen={openSections.news_habits}
              onToggle={() => toggleSection("news_habits")}
              isComplete={isSectionComplete(["news_freq", "news_source"])}
            />
            {openSections.news_habits && (
              <div className="p-6">
                <SelectQuestion
                  id="news_freq"
                  question="How often do you read or watch the news?"
                  value={responses.news_freq}
                  onChange={handleChange}
                  options={[
                    { value: "never", label: "Never" },
                    { value: "rarely", label: "Rarely (few times a month)" },
                    { value: "weekly", label: "Weekly" },
                    { value: "daily", label: "Daily" },
                    { value: "multiple_daily", label: "Multiple times a day" },
                  ]}
                />
                <SelectQuestion
                  id="news_source"
                  question="What is your primary source for news?"
                  value={responses.news_source}
                  onChange={handleChange}
                  options={[
                    { value: "tv", label: "Television" },
                    {
                      value: "newspaper",
                      label: "Newspapers (print or online)",
                    },
                    {
                      value: "social_media",
                      label: "Social Media (Facebook, Twitter, etc.)",
                    },
                    { value: "news_apps", label: "News apps/websites" },
                    { value: "radio", label: "Radio" },
                    { value: "word_of_mouth", label: "Word of mouth" },
                    { value: "other", label: "Other" },
                  ]}
                />
              </div>
            )}
          </div>
        </div>

        {/* Submit Button */}
        <div className="mt-8">
          <button
            onClick={handleSubmit}
            disabled={isSubmitting}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white py-4 px-6 rounded-xl font-semibold flex items-center justify-center gap-2 transition-colors disabled:bg-gray-400"
          >
            {isSubmitting ? (
              "Submitting..."
            ) : (
              <>
                Continue to News Evaluation
                <ArrowRight size={20} />
              </>
            )}
          </button>

          {errors.length > 0 && (
            <p className="text-red-600 text-sm mt-2 text-center">
              Please complete all required fields ({errors.length} remaining)
            </p>
          )}
        </div>
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
            Go to Fake News Experiment →
          </button>
        )}
      </div>
    </div>
  );
};

export default PreSurveyFakeNews;
