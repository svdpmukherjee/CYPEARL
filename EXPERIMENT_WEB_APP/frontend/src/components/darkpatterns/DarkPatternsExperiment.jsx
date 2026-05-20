/**
 * DarkPatternsExperiment - Main orchestrator for the dark patterns experiment
 *
 * Flow:
 * 1. Instructions → 2. Pre-survey → 3. 16 UI Tasks → 4. Post-survey → 5. Complete
 *
 * Handles:
 * - Loading current task from backend
 * - Submitting task responses
 * - Progress tracking
 * - Stage transitions
 */

import React, { useState, useEffect, useCallback } from "react";
import axios from "axios";
import { MousePointer2, ArrowRight, CheckCircle, Loader } from "lucide-react";

import InstructionPage from "./InstructionPage";
import PreSurveyDarkPatterns from "./PreSurveyDarkPatterns";
import PostSurveyDarkPatterns from "./PostSurveyDarkPatterns";
import TaskScreen from "./TaskScreen";

const API_URL = import.meta.env.VITE_API_URL || "/api";

// Stages: instructions → presurvey → experiment → postsurvey → completed
const getStage = () => localStorage.getItem("dp_study_stage") || "instructions";
const setStage = (stage) => localStorage.setItem("dp_study_stage", stage);

export default function DarkPatternsExperiment({ participantId, onComplete, onGoToScenarioSelector }) {
  const [stage, setStageState] = useState(getStage());
  const [currentTask, setCurrentTask] = useState(null);
  const [progress, setProgress] = useState({ current: 0, total: 16 });
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);

  // Update both state and localStorage
  const updateStage = (newStage) => {
    setStage(newStage);
    setStageState(newStage);
  };

  // Fetch current task from backend
  const fetchCurrentTask = useCallback(async () => {
    if (!participantId) return;

    setIsLoading(true);
    setError(null);

    try {
      const response = await axios.get(
        `${API_URL}/dark-patterns/tasks/current/${participantId}`,
      );

      const data = response.data;

      if (data.is_finished) {
        setCurrentTask(null);
        setProgress({
          current: data.progress.total,
          total: data.progress.total,
        });
        updateStage("postsurvey");
      } else {
        setCurrentTask(data.task);
        setProgress(data.progress);
      }
    } catch (err) {
      console.error("Error fetching task:", err);
      setError("Failed to load task. Please try again.");
    } finally {
      setIsLoading(false);
    }
  }, [participantId]);

  // Load task when entering experiment stage
  useEffect(() => {
    if (stage === "experiment" && participantId) {
      fetchCurrentTask();
    }
  }, [stage, participantId, fetchCurrentTask]);

  // Handle pre-survey completion
  const handlePreSurveyComplete = async (data) => {
    try {
      await axios.post(`${API_URL}/dark-patterns/survey/pre`, {
        participant_id: participantId,
        ...data,
      });
      updateStage("experiment");
    } catch (err) {
      console.error("Error saving pre-survey:", err);
      // Continue anyway - data is in localStorage
      updateStage("experiment");
    }
  };

  // Handle task completion
  const handleTaskComplete = async (taskData) => {
    setIsLoading(true);

    try {
      // Submit task action
      await axios.post(
        `${API_URL}/dark-patterns/action/${participantId}`,
        taskData,
      );

      // Mark task complete and get next
      const completeResponse = await axios.post(
        `${API_URL}/dark-patterns/complete/${participantId}`,
      );

      if (completeResponse.data.is_finished) {
        updateStage("postsurvey");
      } else {
        // Fetch next task
        await fetchCurrentTask();
      }
    } catch (err) {
      console.error("Error completing task:", err);
      setError("Failed to save response. Please try again.");
    } finally {
      setIsLoading(false);
    }
  };

  // Handle post-survey completion
  const handlePostSurveyComplete = async (data) => {
    try {
      await axios.post(
        `${API_URL}/dark-patterns/survey/post/${participantId}`,
        data,
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
      localStorage.removeItem("dp_study_stage");
      localStorage.removeItem("dp_pre_survey_responses_draft");
      setStageState("instructions");
      setCurrentTask(null);
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
      <PreSurveyDarkPatterns
        onComplete={handlePreSurveyComplete}
        onGoToExperiment={() => updateStage("experiment")}
        onGoToScenarioSelector={onGoToScenarioSelector}
      />
    );
  }

  // =========================================================================
  // RENDER: Experiment (Tasks)
  // =========================================================================
  if (stage === "experiment") {
    if (isLoading && !currentTask) {
      return (
        <div className="min-h-screen bg-gray-100 flex items-center justify-center">
          <div className="text-center">
            <Loader className="w-12 h-12 text-purple-600 animate-spin mx-auto mb-4" />
            <p className="text-gray-600">Loading task...</p>
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
              onClick={fetchCurrentTask}
              className="bg-purple-600 text-white px-6 py-2 rounded-lg"
            >
              Try Again
            </button>
          </div>
        </div>
      );
    }

    if (currentTask) {
      return (
        <div className="relative">
          {/* Progress Bar */}
          <div className="fixed top-0 left-0 right-0 bg-white shadow-sm z-50">
            <div className="max-w-4xl mx-auto px-4 py-2 flex items-center justify-between">
              <div className="flex items-center gap-2">
                <MousePointer2 className="w-5 h-5 text-purple-600" />
                <span className="font-semibold text-gray-800">
                  Dark Patterns Study
                </span>
              </div>
              <div className="flex items-center gap-4">
                <span className="text-sm font-medium text-gray-700">
                  {Math.round(((progress.current + 1) / progress.total) * 100)}%
                </span>
                <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                  <div
                    className="h-full bg-purple-600 transition-all duration-300"
                    style={{
                      width: `${((progress.current + 1) / progress.total) * 100}%`,
                    }}
                  />
                </div>
              </div>
            </div>
          </div>

          {/* Task Content */}
          <div className="pt-14">
            <TaskScreen
              key={currentTask.task_id}
              task={currentTask}
              onComplete={handleTaskComplete}
              participantId={participantId}
              progress={progress}
            />
          </div>

          {/* Loading Overlay */}
          {isLoading && (
            <div className="fixed inset-0 bg-white/80 flex items-center justify-center z-50">
              <div className="text-center">
                <Loader className="w-10 h-10 text-purple-600 animate-spin mx-auto mb-3" />
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
      <PostSurveyDarkPatterns
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
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-slate-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-lg p-8 max-w-lg w-full text-center">
          <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-6">
            <CheckCircle className="w-12 h-12 text-green-600" />
          </div>

          <h1 className="text-2xl font-bold text-gray-800 mb-4">
            Study Complete!
          </h1>

          <p className="text-gray-600 mb-6">
            Thank you for participating in this research study. Your responses
            have been recorded and will help us understand how people interact
            with website interfaces.
          </p>

          <div className="p-4 bg-purple-50 rounded-lg mb-6">
            <p className="text-purple-700 text-sm">
              <strong>What was this about?</strong>
              <br />
              This study examined how different website designs affect user
              decisions. Some interfaces used "dark patterns" - design choices
              that may nudge users toward certain actions.
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
