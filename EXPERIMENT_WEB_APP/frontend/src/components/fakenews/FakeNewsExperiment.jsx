/**
 * FakeNewsExperiment - Main orchestrator for the fake news/misinformation experiment
 *
 * Flow:
 * 1. Instructions -> 2. Pre-survey -> 3. 16 News Items -> 4. Post-survey -> 5. Complete
 *
 * Handles:
 * - Loading current news item from backend
 * - Submitting evaluations and metrics
 * - Progress tracking
 * - Stage transitions
 * - Question order counterbalancing (accuracy_first vs sharing_first)
 */

import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { Newspaper, ArrowRight, CheckCircle, Loader, Info } from "lucide-react";

import InstructionPage from "./InstructionPage";
import PreSurveyFakeNews from "./PreSurveyFakeNews";
import PostSurveyFakeNews from "./PostSurveyFakeNews";
import NewsFeedScreen from "./NewsFeedScreen";

const API_URL = import.meta.env.VITE_API_URL || "/api";

// Stages: instructions -> presurvey -> experiment -> postsurvey -> completed
const getStage = () => localStorage.getItem("fn_study_stage") || "instructions";
const setStage = (stage) => localStorage.setItem("fn_study_stage", stage);

export default function FakeNewsExperiment({ participantId, onComplete, onGoToScenarioSelector }) {
  const [stage, setStageState] = useState(getStage());
  const [currentItem, setCurrentItem] = useState(null);
  const [progress, setProgress] = useState({ current: 0, total: 16 });
  const [questionOrder, setQuestionOrder] = useState("accuracy_first");
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Update both state and localStorage
  const updateStage = (newStage) => {
    setStage(newStage);
    setStageState(newStage);
  };

  // Fetch current news item from backend
  const fetchCurrentItem = useCallback(async () => {
    if (!participantId) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await axios.get(
        `${API_URL}/fake-news/items/current/${participantId}`
      );

      const data = response.data;

      if (data.is_finished) {
        setCurrentItem(null);
        setProgress({
          current: data.progress.total,
          total: data.progress.total,
        });
        updateStage("postsurvey");
      } else {
        setCurrentItem(data.item);
        setProgress(data.progress);
        if (data.question_order) {
          setQuestionOrder(data.question_order);
        }
      }
    } catch (err) {
      console.error("Error fetching news item:", err);
      setError("Failed to load news item. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }, [participantId]);

  // Load item when entering experiment stage
  useEffect(() => {
    if (stage === "experiment" && participantId) {
      fetchCurrentItem();
    }
  }, [stage, participantId, fetchCurrentItem]);

  // Handle pre-survey completion
  const handlePreSurveyComplete = async (data) => {
    try {
      // Save pre-survey data including political ideology
      await axios.post(`${API_URL}/fake-news/survey/pre`, {
        participant_id: participantId,
        ...data,
      });

      // Also update political ideology separately for congruence calculation
      if (data.political_ideology) {
        await axios.post(`${API_URL}/fake-news/ideology/${participantId}`, {
          political_ideology: data.political_ideology,
          party_id: data.party_id,
        });
      }

      updateStage("experiment");
    } catch (err) {
      console.error("Error saving pre-survey:", err);
      // Continue anyway - data is in localStorage
      updateStage("experiment");
    }
  };

  // Handle news item evaluation completion
  const handleItemComplete = async (evaluationData) => {
    setIsLoading(true);

    try {
      // Submit evaluation
      await axios.post(
        `${API_URL}/fake-news/action/${participantId}`,
        evaluationData
      );

      // Mark item complete and get next
      const completeResponse = await axios.post(
        `${API_URL}/fake-news/complete/${participantId}`
      );

      if (completeResponse.data.is_finished) {
        updateStage("postsurvey");
      } else {
        // Fetch next item and scroll to top
        await fetchCurrentItem();
        window.scrollTo({ top: 0, behavior: 'smooth' });
      }
    } catch (err) {
      console.error("Error completing item:", err);
      setError("Failed to save response. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Handle post-survey completion
  const handlePostSurveyComplete = async (data) => {
    try {
      await axios.post(
        `${API_URL}/fake-news/survey/post/${participantId}`,
        data
      );
      updateStage("completed");
      onComplete?.();
    } catch (err) {
      console.error("Error saving post-survey:", err);
      updateStage("completed");
      onComplete?.();
    }
  };

  // Reset handler (for testing)
  const handleReset = () => {
    if (confirm("Reset your progress? This will clear all responses.")) {
      localStorage.removeItem("fn_study_stage");
      localStorage.removeItem("fn_pre_survey_responses_draft");
      setStageState("instructions");
      setCurrentItem(null);
      setProgress({ current: 0, total: 16 });
    }
  };

  // =========================================================================
  // RENDER: Instructions
  // =========================================================================
  if (stage === "instructions") {
    return (
      <>
        <InstructionPage onContinue={() => updateStage("presurvey")} />
        <div className="fixed bottom-0 left-0 px-4 py-2 z-50">
          <button
            onClick={onGoToScenarioSelector}
            className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium"
          >
            ← Scenario Selector
          </button>
        </div>
      </>
    );
  }

  // =========================================================================
  // RENDER: Pre-Survey
  // =========================================================================
  if (stage === "presurvey") {
    return (
      <PreSurveyFakeNews
        onComplete={handlePreSurveyComplete}
        onGoToExperiment={() => updateStage("experiment")}
        onGoToScenarioSelector={onGoToScenarioSelector}
      />
    );
  }

  // =========================================================================
  // RENDER: Experiment (News Items)
  // =========================================================================
  if (stage === "experiment") {
    if (isLoading && !currentItem) {
      return (
        <div className="min-h-screen bg-gray-100 flex items-center justify-center">
          <div className="text-center">
            <Loader className="w-12 h-12 text-orange-600 animate-spin mx-auto mb-4" />
            <p className="text-gray-600">Loading news item...</p>
          </div>
        </div>
      );
    }

    if (error) {
      return (
        <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
          <div className="bg-white rounded-xl shadow-lg p-8 max-w-md w-full text-center">
            <p className="text-red-600 mb-4">{error}</p>
            <button
              onClick={fetchCurrentItem}
              className="bg-orange-600 text-white px-6 py-2 rounded-lg"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    if (currentItem) {
      return (
        <div className="relative">
          {/* Progress Bar */}
          <div className="fixed top-0 left-0 right-0 bg-white shadow-sm z-50">
            <div className="max-w-4xl mx-auto px-4 py-2 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <Newspaper className="w-5 h-5 text-orange-600" />
                <span className="font-semibold text-gray-800">
                  News Evaluation Study
                </span>
              </div>
              <div className="flex items-center gap-4">
                <span className="text-sm text-gray-500">
                  Item {progress.current + 1} of {progress.total}
                </span>
                <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-orange-600 transition-all duration-300"
                    style={{
                      width: `${((progress.current + 1) / progress.total) * 100}%`,
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* News Item Content */}
          <div className="pt-14">
            <NewsFeedScreen
              key={currentItem.item_id}
              newsItem={currentItem}
              onComplete={handleItemComplete}
              questionOrder={questionOrder}
              progress={progress}
            />
          </div>

          {/* Loading Overlay */}
          {isLoading && (
            <div className="fixed inset-0 bg-white/80 flex items-center justify-center z-50">
              <div className="text-center">
                <Loader className="w-10 h-10 text-orange-600 animate-spin mx-auto mb-3" />
                <p className="text-gray-600">Saving response...</p>
              </div>
            </div>
          )}

          {/* TESTING ONLY: Navigation buttons for quick access */}
          <div className="fixed bottom-0 left-0 right-0 px-4 py-2 flex justify-between items-center z-50 pointer-events-none">
            <button
              onClick={() => updateStage("presurvey")}
              className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium pointer-events-auto"
            >
              ← Go to Pre-Experiment Survey
            </button>
            <button
              onClick={onGoToScenarioSelector}
              className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium pointer-events-auto"
            >
              Scenario Selector
            </button>
            <button
              onClick={() => updateStage("postsurvey")}
              className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium pointer-events-auto"
            >
              Go to Post-Experiment Survey →
            </button>
          </div>
        </div>
      );
    }
  }

  // =========================================================================
  // RENDER: Post-Survey
  // =========================================================================
  if (stage === "postsurvey") {
    return (
      <PostSurveyFakeNews
        onComplete={handlePostSurveyComplete}
        participantId={participantId}
        onGoToExperiment={() => updateStage("experiment")}
        onGoToScenarioSelector={onGoToScenarioSelector}
      />
    );
  }

  // =========================================================================
  // RENDER: Completed
  // =========================================================================
  if (stage === "completed") {
    return (
      <div className="min-h-screen bg-gradient-to-br from-orange-50 to-slate-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-lg p-8 max-w-lg w-full text-center">
          <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <CheckCircle className="w-12 h-12 text-green-600" />
          </div>

          <h1 className="text-2xl font-bold text-gray-800 mb-4">
            Study Complete!
          </h1>

          <p className="text-gray-600 mb-6">
            Thank you for participating in this research study. Your responses
            have been recorded and will help us understand how people evaluate
            news and information online.
          </p>

          <div className="p-4 bg-orange-50 rounded-lg mb-6">
            <div className="flex items-start gap-3">
              <Info className="w-5 h-5 text-orange-600 mt-0.5 flex-shrink-0" />
              <div className="text-left">
                <p className="text-orange-800 text-sm font-medium mb-2">
                  What was this about?
                </p>
                <p className="text-orange-700 text-sm">
                  This study examined how people evaluate news headlines of
                  varying accuracy. Some headlines you saw were from verified
                  news sources and were factually accurate, while others
                  contained misinformation. Your responses help us understand
                  the factors that influence susceptibility to misinformation.
                </p>
              </div>
            </div>
          </div>

          <div className="p-4 bg-blue-50 rounded-lg mb-6">
            <p className="text-blue-800 text-sm">
              <strong>Tip:</strong> Before sharing news online, consider
              checking multiple sources and looking for verification from
              reputable fact-checking organizations.
            </p>
          </div>

          <p className="text-sm text-gray-500 mb-4">
            Participant ID:{" "}
            <code className="bg-gray-100 px-2 py-1 rounded">
              {participantId}
            </code>
          </p>

          <p className="text-sm text-gray-500">
            You may now close this window or return to Prolific.
          </p>

          <button
            onClick={onGoToScenarioSelector}
            className="mt-4 text-slate-500 hover:text-slate-700 text-sm underline"
          >
            ← Back to Scenario Selector
          </button>
        </div>
      </div>
    );
  }

  // Fallback
  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center">
      <p className="text-gray-600">Loading...</p>
    </div>
  );
}
