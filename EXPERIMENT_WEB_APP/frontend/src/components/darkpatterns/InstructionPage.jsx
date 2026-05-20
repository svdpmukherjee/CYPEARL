/**
 * InstructionPage - Instructions shown after Prolific ID entry for Dark Patterns experiment
 *
 * Provides participants with:
 * - Overview of what they'll be doing
 * - Instructions on how to complete tasks
 * - Explanation of scenarios they'll encounter
 * - Consent and acknowledgment before starting
 */

import React, { useState } from 'react';
import { MousePointer2, CheckCircle, AlertTriangle, ArrowRight, Info, Clock, Target } from 'lucide-react';

export default function InstructionPage({ onContinue }) {
  const [acknowledged, setAcknowledged] = useState(false);
  const [currentPage, setCurrentPage] = useState(1);
  const totalPages = 3;

  const handleContinue = () => {
    if (currentPage < totalPages) {
      setCurrentPage(currentPage + 1);
    } else if (acknowledged) {
      onContinue();
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-slate-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-lg max-w-2xl w-full overflow-hidden">
        {/* Header */}
        <div className="bg-purple-600 px-8 py-6 text-white">
          <div className="flex items-center gap-3 mb-2">
            <MousePointer2 className="w-8 h-8" />
            <h1 className="text-2xl font-bold">Dark Patterns Study</h1>
          </div>
          <p className="text-purple-100">Instructions & Overview</p>
        </div>

        {/* Progress Indicator */}
        <div className="px-8 pt-6">
          <div className="flex items-center justify-between mb-2">
            <span className="text-sm text-gray-500">Page {currentPage} of {totalPages}</span>
            <div className="flex gap-1">
              {[1, 2, 3].map((page) => (
                <div
                  key={page}
                  className={`w-8 h-1 rounded-full transition-colors ${
                    page <= currentPage ? 'bg-purple-600' : 'bg-gray-200'
                  }`}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="px-8 py-6">
          {/* Page 1: Overview */}
          {currentPage === 1 && (
            <div className="space-y-6">
              <div className="flex items-start gap-4">
                <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center flex-shrink-0">
                  <Info className="w-6 h-6 text-purple-600" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-gray-800 mb-2">What is this study about?</h2>
                  <p className="text-gray-600 leading-relaxed">
                    In this study, you will interact with <strong>16 simulated website interfaces</strong>.
                    These interfaces represent common scenarios you might encounter while browsing the web,
                    such as cookie consent popups, subscription management, checkout processes, and more.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center flex-shrink-0">
                  <Target className="w-6 h-6 text-blue-600" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-gray-800 mb-2">Your Goal</h2>
                  <p className="text-gray-600 leading-relaxed">
                    For each interface, you'll be given a <strong>specific scenario</strong> that describes
                    what you want to accomplish. Your task is to make decisions as if you were actually
                    in that situation. There are no right or wrong answers - we want to understand
                    how people naturally interact with these designs.
                  </p>
                </div>
              </div>

              <div className="flex items-start gap-4">
                <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center flex-shrink-0">
                  <Clock className="w-6 h-6 text-green-600" />
                </div>
                <div>
                  <h2 className="text-xl font-semibold text-gray-800 mb-2">Duration</h2>
                  <p className="text-gray-600 leading-relaxed">
                    The study consists of a <strong>pre-survey</strong>, followed by <strong>16 interface tasks</strong>,
                    and ends with a brief <strong>post-survey</strong>. The entire study takes approximately
                    <strong> 20-25 minutes</strong> to complete.
                  </p>
                </div>
              </div>
            </div>
          )}

          {/* Page 2: How it works */}
          {currentPage === 2 && (
            <div className="space-y-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">How Each Task Works</h2>

              <div className="space-y-4">
                <div className="flex items-start gap-4 p-4 bg-gray-50 rounded-xl">
                  <div className="w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold flex-shrink-0">
                    1
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-800 mb-1">Read the Scenario</h3>
                    <p className="text-gray-600 text-sm">
                      Each task begins with a scenario description. This tells you the context -
                      for example, "You're trying to unsubscribe from a newsletter" or
                      "You want to delete your account from a service."
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4 p-4 bg-gray-50 rounded-xl">
                  <div className="w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold flex-shrink-0">
                    2
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-800 mb-1">Interact with the Interface</h3>
                    <p className="text-gray-600 text-sm">
                      You'll see a simulated website interface. Click buttons, make selections,
                      and navigate as you normally would. Take your time to read any text or options presented.
                    </p>
                  </div>
                </div>

                <div className="flex items-start gap-4 p-4 bg-gray-50 rounded-xl">
                  <div className="w-8 h-8 bg-purple-600 text-white rounded-full flex items-center justify-center font-bold flex-shrink-0">
                    3
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-800 mb-1">Answer Follow-up Questions</h3>
                    <p className="text-gray-600 text-sm">
                      After completing each task, you'll answer a few brief questions about your experience,
                      including your reasoning for the decisions you made.
                    </p>
                  </div>
                </div>
              </div>

              <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4 mt-4">
                <div className="flex items-start gap-3">
                  <AlertTriangle className="w-5 h-5 text-yellow-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <h4 className="font-semibold text-yellow-800 mb-1">Important Note</h4>
                    <p className="text-yellow-700 text-sm">
                      Some tasks may have a <strong>time limit</strong> shown at the top.
                      This simulates real-world time pressure. Do your best within the time given,
                      but don't worry if time runs out - simply complete your action.
                    </p>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Page 3: Final Instructions */}
          {currentPage === 3 && (
            <div className="space-y-6">
              <h2 className="text-xl font-semibold text-gray-800 mb-4">Before You Begin</h2>

              <div className="space-y-4">
                <div className="flex items-start gap-3 p-4 bg-green-50 rounded-xl border border-green-200">
                  <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <p className="text-green-800 text-sm">
                    <strong>Be natural:</strong> Respond as you would in real life. There are no trick questions.
                  </p>
                </div>

                <div className="flex items-start gap-3 p-4 bg-green-50 rounded-xl border border-green-200">
                  <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <p className="text-green-800 text-sm">
                    <strong>Take your time:</strong> Read each scenario carefully before making decisions.
                  </p>
                </div>

                <div className="flex items-start gap-3 p-4 bg-green-50 rounded-xl border border-green-200">
                  <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <p className="text-green-800 text-sm">
                    <strong>Be honest:</strong> Your genuine responses help us understand user behavior.
                  </p>
                </div>

                <div className="flex items-start gap-3 p-4 bg-green-50 rounded-xl border border-green-200">
                  <CheckCircle className="w-5 h-5 text-green-600 flex-shrink-0 mt-0.5" />
                  <p className="text-green-800 text-sm">
                    <strong>Explain your thinking:</strong> When asked why you made a decision, share your actual reasoning.
                  </p>
                </div>
              </div>

              {/* Acknowledgment Checkbox */}
              <div className="mt-6 p-4 bg-purple-50 rounded-xl border border-purple-200">
                <label className="flex items-start gap-3 cursor-pointer">
                  <input
                    type="checkbox"
                    checked={acknowledged}
                    onChange={(e) => setAcknowledged(e.target.checked)}
                    className="w-5 h-5 mt-0.5 rounded border-purple-300 text-purple-600 focus:ring-purple-500"
                  />
                  <span className="text-purple-800 text-sm">
                    I have read and understood the instructions. I understand that I will interact with
                    16 website interfaces and answer questions about my decisions. I am ready to begin.
                  </span>
                </label>
              </div>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="px-8 pb-8">
          <div className="flex justify-between items-center">
            {currentPage > 1 ? (
              <button
                onClick={() => setCurrentPage(currentPage - 1)}
                className="px-6 py-3 text-purple-600 font-medium hover:bg-purple-50 rounded-xl transition-colors"
              >
                Back
              </button>
            ) : (
              <div />
            )}

            <button
              onClick={handleContinue}
              disabled={currentPage === totalPages && !acknowledged}
              className={`px-8 py-3 rounded-xl font-semibold flex items-center gap-2 transition-colors ${
                currentPage === totalPages && !acknowledged
                  ? 'bg-gray-200 text-gray-400 cursor-not-allowed'
                  : 'bg-purple-600 text-white hover:bg-purple-700'
              }`}
            >
              {currentPage < totalPages ? 'Next' : 'Begin Pre-Survey'}
              <ArrowRight className="w-5 h-5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
