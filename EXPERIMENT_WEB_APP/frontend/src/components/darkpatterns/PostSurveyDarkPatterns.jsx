/**
 * PostSurveyDarkPatterns - Post-experiment survey for dark patterns study
 *
 * Measures:
 * - State measures (anxiety, stress, fatigue)
 * - Dark pattern specific (frustration, felt manipulated, regret)
 * - Detection quiz
 * - Open-ended feedback
 */

import React, { useState } from 'react';
import { MousePointer2, ArrowRight, CheckCircle } from 'lucide-react';

const LikertScale = ({ id, question, value, onChange, leftLabel, rightLabel }) => (
  <div className="mb-6">
    <p className="text-gray-800 mb-3">{question}</p>
    <div className="flex gap-2 mb-1">
      {[1, 2, 3, 4, 5, 6, 7].map((n) => (
        <button
          key={n}
          onClick={() => onChange(id, n)}
          className={`w-10 h-10 rounded-lg border-2 font-medium transition-colors ${
            value === n
              ? 'bg-purple-100 border-purple-500 text-purple-700'
              : 'border-gray-300 hover:border-gray-400'
          }`}
        >
          {n}
        </button>
      ))}
    </div>
    <div className="flex justify-between text-xs text-gray-500">
      <span>{leftLabel}</span>
      <span>{rightLabel}</span>
    </div>
  </div>
);

export default function PostSurveyDarkPatterns({ onComplete, participantId, onGoToExperiment, onGoToScenarioSelector }) {
  const [responses, setResponses] = useState({});
  const [isSubmitting, setIsSubmitting] = useState(false);

  const handleChange = (id, value) => {
    setResponses(prev => ({ ...prev, [id]: value }));
  };

  const isComplete = () => {
    const required = [
      'state_anxiety', 'current_stress', 'fatigue_level',
      'frustration_level', 'felt_manipulated', 'regret_choices'
    ];
    return required.every(field => responses[field] !== undefined);
  };

  const handleSubmit = async () => {
    // TESTING MODE: Validation disabled - uncomment below for production
    /*
    if (!isComplete()) {
      alert('Please answer all required questions.');
      return;
    }
    */

    setIsSubmitting(true);
    const submissionData = {
      ...responses,
      submitted_at: new Date().toISOString()
    };
    onComplete(submissionData);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-slate-100 py-8 px-4">
      <div className="max-w-2xl mx-auto">
        {/* Header */}
        <div className="text-center mb-8">
          <div className="flex items-center justify-center gap-3 mb-4">
            <div className="w-12 h-12 bg-purple-600 rounded-xl flex items-center justify-center">
              <MousePointer2 className="w-7 h-7 text-white" />
            </div>
            <h1 className="text-2xl font-bold text-gray-800">Final Questions</h1>
          </div>
          <p className="text-gray-600">
            Almost done! Please answer these final questions about your experience.
          </p>
        </div>

        {/* Survey Content */}
        <div className="bg-white rounded-2xl shadow-lg p-8">
          {/* State Measures */}
          <h2 className="text-lg font-bold text-gray-800 mb-6">How are you feeling right now?</h2>

          <LikertScale
            id="state_anxiety"
            question="How anxious do you feel right now?"
            value={responses.state_anxiety}
            onChange={handleChange}
            leftLabel="Not at all"
            rightLabel="Extremely"
          />

          <LikertScale
            id="current_stress"
            question="How stressed do you feel right now?"
            value={responses.current_stress}
            onChange={handleChange}
            leftLabel="Not at all"
            rightLabel="Extremely"
          />

          <LikertScale
            id="fatigue_level"
            question="How mentally tired do you feel?"
            value={responses.fatigue_level}
            onChange={handleChange}
            leftLabel="Not at all"
            rightLabel="Extremely"
          />

          {/* Dark Pattern Specific */}
          <h2 className="text-lg font-bold text-gray-800 mb-6 mt-8 pt-6 border-t">
            About the tasks you completed
          </h2>

          <LikertScale
            id="frustration_level"
            question="How frustrated did you feel during the tasks?"
            value={responses.frustration_level}
            onChange={handleChange}
            leftLabel="Not at all"
            rightLabel="Extremely"
          />

          <LikertScale
            id="felt_manipulated"
            question="Did you feel manipulated by any of the interfaces?"
            value={responses.felt_manipulated}
            onChange={handleChange}
            leftLabel="Not at all"
            rightLabel="Very much"
          />

          <LikertScale
            id="regret_choices"
            question="Do you regret any choices you made during the tasks?"
            value={responses.regret_choices}
            onChange={handleChange}
            leftLabel="No regret"
            rightLabel="Strong regret"
          />

          {/* Open-ended */}
          <h2 className="text-lg font-bold text-gray-800 mb-4 mt-8 pt-6 border-t">
            Optional Feedback
          </h2>

          <div className="mb-6">
            <label className="block text-gray-700 mb-2">
              Did you notice any patterns or unusual elements in the interfaces?
            </label>
            <textarea
              value={responses.noticed_patterns || ''}
              onChange={(e) => handleChange('noticed_patterns', e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              rows={3}
              placeholder="Share your observations..."
            />
          </div>

          <div className="mb-8">
            <label className="block text-gray-700 mb-2">
              Any other feedback about the study?
            </label>
            <textarea
              value={responses.general_feedback || ''}
              onChange={(e) => handleChange('general_feedback', e.target.value)}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-purple-500"
              rows={3}
              placeholder="Your feedback helps us improve..."
            />
          </div>

          {/* Submit */}
          <button
            onClick={handleSubmit}
            disabled={!isComplete() || isSubmitting}
            className="w-full bg-purple-600 text-white py-4 rounded-xl font-semibold hover:bg-purple-700 transition-colors flex items-center justify-center gap-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isSubmitting ? (
              'Submitting...'
            ) : (
              <>
                Complete Study
                <CheckCircle className="w-5 h-5" />
              </>
            )}
          </button>
        </div>
      </div>

      {/* TESTING ONLY: Navigation buttons */}
      <div className="fixed bottom-0 left-0 right-0 px-4 py-2 flex justify-between items-center z-50 pointer-events-none">
        {onGoToExperiment && (
          <button
            onClick={onGoToExperiment}
            className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium pointer-events-auto"
          >
            ← Go to Dark Patterns Experiment
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
}
