/**
 * InstructionPage - Introduction and instructions for the fake news experiment
 *
 * Provides:
 * - Overview of the study
 * - Instructions on how to evaluate news items
 * - What participants will be asked to do
 */

import React, { useState } from "react";
import {
  Newspaper,
  ArrowRight,
  CheckCircle,
  Eye,
  Share2,
  AlertCircle,
} from "lucide-react";

export default function InstructionPage({ onContinue }) {
  const [currentPage, setCurrentPage] = useState(0);

  const pages = [
    // Page 1: Welcome
    {
      title: "Welcome to the News Evaluation Study",
      content: (
        <div className="space-y-6">
          <p className="text-gray-600 leading-relaxed">
            In this study, you will be shown a series of news headlines as they
            might appear in a social media feed. Your task is to evaluate each
            headline and answer a few questions about it.
          </p>

          <div className="bg-orange-50 border border-orange-100 rounded-xl p-5">
            <div className="flex items-start gap-3">
              <Newspaper className="w-6 h-6 text-orange-600 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-orange-800 mb-2">
                  What You Will See
                </h3>
                <p className="text-orange-700 text-sm">
                  You will evaluate <strong>16 news headlines</strong> from
                  various sources. Each headline will be displayed in a format
                  similar to what you might see when scrolling through social
                  media.
                </p>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-xl p-5">
            <h3 className="font-semibold text-gray-800 mb-3">
              Study Structure
            </h3>
            <div className="space-y-3">
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center text-gray-600 font-semibold text-sm">
                  1
                </div>
                <span className="text-gray-600">
                  Brief questionnaire about yourself
                </span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center text-gray-600 font-semibold text-sm">
                  2
                </div>
                <span className="text-gray-600">
                  Evaluate 16 news headlines
                </span>
              </div>
              <div className="flex items-center gap-3">
                <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center text-gray-600 font-semibold text-sm">
                  3
                </div>
                <span className="text-gray-600">
                  Final questions about your experience
                </span>
              </div>
            </div>
          </div>
        </div>
      ),
    },
    // Page 2: How to Evaluate
    {
      title: "How to Evaluate Each Headline",
      content: (
        <div className="space-y-6">
          <p className="text-gray-600 leading-relaxed">
            For each headline, you will be asked to provide your honest
            assessment. Take as much time as you need to read and consider each
            headline carefully.
          </p>

          <div className="space-y-4">
            <div className="bg-blue-50 border border-blue-100 rounded-xl p-5">
              <div className="flex items-start gap-3">
                <Eye className="w-6 h-6 text-blue-600 mt-1 flex-shrink-0" />
                <div>
                  <h3 className="font-semibold text-blue-800 mb-2">
                    Accuracy Rating
                  </h3>
                  <p className="text-blue-700 text-sm">
                    You will rate how accurate you believe the headline is, from
                    "Definitely False" to "Definitely True". Trust your
                    instincts based on the information presented.
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-green-50 border border-green-100 rounded-xl p-5">
              <div className="flex items-start gap-3">
                <Share2 className="w-6 h-6 text-green-600 mt-1 flex-shrink-0" />
                <div>
                  <h3 className="font-semibold text-green-800 mb-2">
                    Sharing Intention
                  </h3>
                  <p className="text-green-700 text-sm">
                    You will indicate how likely you would be to share this
                    headline on social media, from "Definitely Would Not Share"
                    to "Definitely Would Share".
                  </p>
                </div>
              </div>
            </div>

            <div className="bg-purple-50 border border-purple-100 rounded-xl p-5">
              <div className="flex items-start gap-3">
                <CheckCircle className="w-6 h-6 text-purple-600 mt-1 flex-shrink-0" />
                <div>
                  <h3 className="font-semibold text-purple-800 mb-2">
                    Additional Questions
                  </h3>
                  <p className="text-purple-700 text-sm">
                    You will also indicate if you have seen the headline before
                    and rate your confidence in your assessment.
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      ),
    },
    // Page 3: Important Notes
    {
      title: "Before You Begin",
      content: (
        <div className="space-y-6">
          <div className="bg-amber-50 border border-amber-200 rounded-xl p-5">
            <div className="flex items-start gap-3">
              <AlertCircle className="w-6 h-6 text-amber-600 mt-1 flex-shrink-0" />
              <div>
                <h3 className="font-semibold text-amber-800 mb-2">
                  Important Notes
                </h3>
                <ul className="text-amber-700 text-sm space-y-2">
                  <li className="flex items-start gap-2">
                    <span className="mt-1.5 w-1.5 h-1.5 bg-amber-500 rounded-full flex-shrink-0" />
                    There are no right or wrong answers - we are interested in
                    your honest opinions
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="mt-1.5 w-1.5 h-1.5 bg-amber-500 rounded-full flex-shrink-0" />
                    Please do not use external sources to verify headlines
                    during the study
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="mt-1.5 w-1.5 h-1.5 bg-amber-500 rounded-full flex-shrink-0" />
                    Take your time with each headline - there is no time limit
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="mt-1.5 w-1.5 h-1.5 bg-amber-500 rounded-full flex-shrink-0" />
                    Your responses are completely anonymous and confidential
                  </li>
                </ul>
              </div>
            </div>
          </div>

          <div className="bg-gray-50 rounded-xl p-5">
            <h3 className="font-semibold text-gray-800 mb-3">
              Estimated Duration
            </h3>
            <div className="flex items-center gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-orange-600">~20</div>
                <div className="text-xs text-gray-500">minutes</div>
              </div>
              <div className="text-gray-600 text-sm">
                The study typically takes about 20 minutes to complete,
                including the initial questionnaire and final questions.
              </div>
            </div>
          </div>

          <div className="text-center pt-4">
            <p className="text-gray-500 text-sm mb-4">
              Ready to begin? Click the button below to start with a brief
              questionnaire.
            </p>
          </div>
        </div>
      ),
    },
  ];

  const isLastPage = currentPage === pages.length - 1;

  return (
    <div className="min-h-screen bg-gradient-to-br from-orange-50 to-slate-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-lg max-w-2xl w-full overflow-hidden">
        {/* Header */}
        <div className="bg-blue-600 px-8 py-6 text-white">
          <div className="flex items-center gap-3 mb-2">
            <Newspaper className="w-8 h-8" />
            <h1 className="text-2xl font-bold">News Evaluation Study</h1>
          </div>
          <p className="text-orange-100 text-sm">
            Understanding how people evaluate news and information online
          </p>
        </div>

        {/* Progress dots */}
        <div className="flex justify-center gap-2 py-4 bg-gray-50">
          {pages.map((_, index) => (
            <button
              key={index}
              onClick={() => setCurrentPage(index)}
              className={`w-3 h-3 rounded-full transition-all ${
                index === currentPage
                  ? "bg-blue-600 w-8"
                  : index < currentPage
                    ? "bg-blue-400"
                    : "bg-blue-200"
              }`}
            />
          ))}
        </div>

        {/* Content */}
        <div className="p-8">
          <h2 className="text-xl font-bold text-gray-800 mb-6">
            {pages[currentPage].title}
          </h2>
          {pages[currentPage].content}
        </div>

        {/* Navigation */}
        <div className="px-8 pb-8 flex justify-between items-center">
          <button
            onClick={() => setCurrentPage((prev) => Math.max(0, prev - 1))}
            className={`px-6 py-2 rounded-lg transition-colors ${
              currentPage === 0
                ? "text-gray-400 cursor-not-allowed"
                : "text-gray-600 hover:bg-gray-100"
            }`}
            disabled={currentPage === 0}
          >
            Back
          </button>

          <button
            onClick={() => {
              if (isLastPage) {
                onContinue();
              } else {
                setCurrentPage((prev) => prev + 1);
              }
            }}
            className="bg-blue-600 hover:bg-blue-700 text-white px-8 py-3 rounded-xl font-semibold flex items-center gap-2 transition-colors"
          >
            {isLastPage ? "Begin Study" : "Continue"}
            <ArrowRight size={18} />
          </button>
        </div>
      </div>
    </div>
  );
}
