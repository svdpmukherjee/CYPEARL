/**
 * CYPEARL Fake News Post-Experiment Survey Component
 *
 * Post-experiment measures:
 * - State measures (anxiety, stress, fatigue)
 * - Fake news specific (feeling deceived, confidence in judgments)
 * - Fact-checking behavior
 * - Source memory test
 * - Open-ended feedback
 */

import React, { useState } from "react";
import { CheckCircle, Newspaper, ArrowRight } from "lucide-react";

const PostSurveyFakeNews = ({ onComplete, participantId, onGoToExperiment, onGoToScenarioSelector }) => {
  const [responses, setResponses] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (id, value) => {
    setResponses((prev) => ({ ...prev, [id]: value }));
  };

  const handleSubmit = async () => {
    setIsSubmitting(true);

    const submissionData = {
      ...responses,
      submitted_at: new Date().toISOString(),
    };

    onComplete(submissionData);
  };

  // Check if all required questions are answered
  const requiredFields = [
    "state_anxiety",
    "current_stress",
    "fatigue_level",
    "felt_deceived",
    "confidence_in_judgments",
    "fact_check_frequency",
  ];

  const canSubmit = requiredFields.every(
    (field) => responses[field] !== undefined && responses[field] !== "",
  );

  // Likert scale component
  const LikertScale = ({ id, question, leftLabel, rightLabel }) => (
    <div className="mb-6">
      <p className="text-gray-800 mb-3">
        {question}
        <span className="text-red-500 ml-1">*</span>
      </p>
      <div className="flex items-center justify-between gap-2">
        {[1, 2, 3, 4, 5, 6, 7].map((n) => (
          <button
            key={n}
            onClick={() => handleChange(id, n)}
            className={`w-10 h-10 rounded-full text-sm font-semibold transition-all
              ${
                responses[id] === n
                  ? "bg-blue-600 text-white scale-110 shadow-lg"
                  : "bg-gray-100 text-gray-600 hover:bg-blue-100 hover:text-blue-700"
              }`}
          >
            {n}
          </button>
        ))}
      </div>
      <div className="flex justify-between text-xs text-gray-500 mt-2">
        <span>{leftLabel}</span>
        <span>{rightLabel}</span>
      </div>
    </div>
  );

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-50 to-slate-100 py-8 px-4">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center">
              <Newspaper className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-gray-800">
              Final Questions
            </h1>
          </div>
          <p className="text-gray-600">
            Almost done! Please answer these final questions about your
            experience.
          </p>
        </div>

        {/* Survey Content */}
        <div className="bg-white rounded-xl shadow-md p-6 space-y-8">
          {/* State Measures */}
          <div>
            <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
              {/* <span className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center text-blue-600 text-sm font-bold">
                1
              </span> */}
              How are you feeling right now?
            </h2>

            <LikertScale
              id="state_anxiety"
              question="How anxious do you feel right now?"
              leftLabel="Not at all anxious"
              rightLabel="Extremely anxious"
            />

            <LikertScale
              id="current_stress"
              question="How stressed do you feel right now?"
              leftLabel="Not at all stressed"
              rightLabel="Extremely stressed"
            />

            <LikertScale
              id="fatigue_level"
              question="How mentally tired do you feel?"
              leftLabel="Not at all tired"
              rightLabel="Extremely tired"
            />
          </div>

          {/* Fake News Specific */}
          <div className="border-t pt-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
              {/* <span className="w-8 h-8 bg-gray-100 rounded-full flex items-center justify-center text-gray-600 text-sm font-bold">
                2
              </span> */}
              About the Headlines
            </h2>

            <LikertScale
              id="felt_deceived"
              question="Did you feel deceived by any of the headlines?"
              leftLabel="Not at all"
              rightLabel="Very much so"
            />

            <LikertScale
              id="confidence_in_judgments"
              question="How confident are you in your accuracy judgments overall?"
              leftLabel="Not at all confident"
              rightLabel="Extremely confident"
            />
          </div>

          {/* Fact-checking Behavior */}
          <div className="border-t pt-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
              {/* <span className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center text-orange-600 text-sm font-bold">
                3
              </span> */}
              Your Habits
            </h2>

            <div className="mb-6">
              <p className="text-gray-800 mb-3">
                In general, how often do you fact-check news before sharing it?
                <span className="text-red-500 ml-1">*</span>
              </p>
              <div className="flex flex-wrap gap-2">
                {[
                  { value: "never", label: "Never" },
                  { value: "rarely", label: "Rarely" },
                  { value: "sometimes", label: "Sometimes" },
                  { value: "often", label: "Often" },
                  { value: "always", label: "Always" },
                ].map((option) => (
                  <button
                    key={option.value}
                    onClick={() =>
                      handleChange("fact_check_frequency", option.value)
                    }
                    className={`px-4 py-2 rounded-lg text-sm font-medium transition-all ${
                      responses.fact_check_frequency === option.value
                        ? "bg-blue-600 text-white"
                        : "bg-gray-100 text-gray-600 hover:bg-blue-100"
                    }`}
                  >
                    {option.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Open-ended Questions */}
          <div className="border-t pt-6">
            <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
              {/* <span className="w-8 h-8 bg-orange-100 rounded-full flex items-center justify-center text-orange-600 text-sm font-bold">
                4
              </span> */}
              Your Thoughts
            </h2>

            <div className="mb-6">
              <p className="text-gray-800 mb-3">
                How do you typically decide if a news headline is true or false?
                (Optional)
              </p>
              <textarea
                value={responses.strategy_description || ""}
                onChange={(e) =>
                  handleChange("strategy_description", e.target.value)
                }
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                rows={4}
                placeholder="Describe your approach..."
              />
            </div>

            <div className="mb-6">
              <p className="text-gray-800 mb-3">
                Do you have any other feedback about this study? (Optional)
              </p>
              <textarea
                value={responses.general_feedback || ""}
                onChange={(e) =>
                  handleChange("general_feedback", e.target.value)
                }
                className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 resize-none"
                rows={3}
                placeholder="Any comments or suggestions..."
              />
            </div>
          </div>

          {/* Submit Button */}
          <div className="pt-4">
            <button
              onClick={handleSubmit}
              disabled={!canSubmit || isSubmitting}
              className={`w-full py-4 rounded-xl font-semibold flex items-center justify-center gap-2 transition-all ${
                canSubmit && !isSubmitting
                  ? "bg-blue-600 hover:bg-blue-700 text-white"
                  : "bg-gray-200 text-gray-400 cursor-not-allowed"
              }`}
            >
              {isSubmitting ? (
                "Submitting..."
              ) : (
                <>
                  Complete Study
                  <CheckCircle className="w-5 h-5" />
                </>
              )}
            </button>

            {!canSubmit && (
              <p className="text-center text-sm text-gray-500 mt-2">
                Please answer all required questions to continue
              </p>
            )}
          </div>
        </div>

        {/* Participant ID */}
        <div className="text-center mt-4">
          <p className="text-sm text-gray-500">
            Participant ID:{" "}
            <code className="bg-gray-100 px-2 py-1 rounded">
              {participantId}
            </code>
          </p>
        </div>
      </div>

      {/* TESTING ONLY: Navigation buttons */}
      <div className="fixed bottom-0 left-0 right-0 px-4 py-2 flex justify-between items-center z-50 pointer-events-none">
        {onGoToExperiment && (
          <button
            onClick={onGoToExperiment}
            className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium pointer-events-auto"
          >
            ← Go to Fake News Experiment
          </button>
        )}
        {onGoToScenarioSelector && (
          <button
            onClick={onGoToScenarioSelector}
            className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium pointer-events-auto ml-auto"
          >
            Scenario Selector →
          </button>
        )}
      </div>
    </div>
  );
};

export default PostSurveyFakeNews;
