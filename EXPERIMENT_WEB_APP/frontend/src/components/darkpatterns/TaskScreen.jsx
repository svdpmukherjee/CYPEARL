/**
 * TaskScreen - Main container for displaying and tracking dark pattern tasks
 *
 * Handles:
 * - Loading the appropriate task component based on task context
 * - Tracking all user interactions (scroll, hover, click, time)
 * - Collecting metrics and submitting to backend
 */

import React, { useState, useEffect, useRef, useCallback } from "react";
import { Clock, AlertCircle, Info } from "lucide-react";

// Import task components
import CookieConsent from "./tasks/CookieConsent";
import NewsletterUnsubscribe from "./tasks/NewsletterUnsubscribe";
import FreeTrial from "./tasks/FreeTrial";
import AccountDeletion from "./tasks/AccountDeletion";
import CheckoutAddon from "./tasks/CheckoutAddon";
import ShippingUpgrade from "./tasks/ShippingUpgrade";
import PrivacySettings from "./tasks/PrivacySettings";
import DeclineOffer from "./tasks/DeclineOffer";

// Map task context to component
const TASK_COMPONENTS = {
  cookie_consent: CookieConsent,
  newsletter_unsubscribe: NewsletterUnsubscribe,
  free_trial: FreeTrial,
  account_deletion: AccountDeletion,
  checkout_addon: CheckoutAddon,
  shipping_upgrade: ShippingUpgrade,
  privacy_settings: PrivacySettings,
  decline_offer: DeclineOffer,
};

// Simple task instructions (shown in header) - derived from goal in scenario_info
const getTaskInstruction = (context) => {
  const instructions = {
    cookie_consent: "Respond to the cookie consent popup.",
    newsletter_unsubscribe: "Complete the unsubscription process.",
    free_trial: "Review the offer and make your decision.",
    account_deletion: "Complete the account deletion process.",
    checkout_addon: "Review and complete your checkout.",
    shipping_upgrade: "Select your shipping preference.",
    privacy_settings: "Configure your privacy settings.",
    decline_offer: "Respond to the special offer.",
  };
  return instructions[context] || "Complete the task below.";
};

export default function TaskScreen({
  task,
  onComplete,
  participantId,
  progress,
}) {
  // State for tracking metrics
  const [metrics, setMetrics] = useState({
    taskStartTime: Date.now(),
    firstActionTime: null,
    scrollEvents: [],
    hoverEvents: [],
    clickEvents: [],
    expandEvents: [],
    maxScrollDepth: 0,
    currentStep: 1,
    totalSteps: 1,
    backtracks: 0,
  });

  // State for timer (if time pressure)
  const [timeRemaining, setTimeRemaining] = useState(
    task?.time_limit_seconds || null,
  );
  const [timerExpired, setTimerExpired] = useState(false);

  // State for post-task questions (including qualitative feedback like phishing ActionModal)
  const [showPostTask, setShowPostTask] = useState(false);
  const [showScenario, setShowScenario] = useState(true); // Show scenario first
  const [finalAction, setFinalAction] = useState(null);
  const [postTaskResponses, setPostTaskResponses] = useState({
    perceived_difficulty: null,
    noticed_unusual: null,
    confidence_rating: null,
    // Qualitative feedback fields (like phishing ActionModal)
    details_noticed: "",
    decision_reason: "",
    what_influenced: "",
    frustration_points: "",
  });

  // Refs
  const containerRef = useRef(null);
  const clickPathRef = useRef([]);

  // Timer effect - only start timer after scenario screen is dismissed
  useEffect(() => {
    if (
      !showScenario &&
      task?.time_pressure &&
      timeRemaining !== null &&
      timeRemaining > 0
    ) {
      const timer = setInterval(() => {
        setTimeRemaining((prev) => {
          if (prev <= 1) {
            setTimerExpired(true);
            clearInterval(timer);
            return 0;
          }
          return prev - 1;
        });
      }, 1000);
      return () => clearInterval(timer);
    }
  }, [showScenario, task?.time_pressure, timeRemaining]);

  // Track scroll depth
  const handleScroll = useCallback((e) => {
    const element = e.target;
    const scrollTop = element.scrollTop;
    const scrollHeight = element.scrollHeight - element.clientHeight;
    const depth = scrollHeight > 0 ? scrollTop / scrollHeight : 0;

    setMetrics((prev) => ({
      ...prev,
      maxScrollDepth: Math.max(prev.maxScrollDepth, depth),
      scrollEvents: [...prev.scrollEvents, { depth, timestamp: Date.now() }],
    }));
  }, []);

  // Track hover events
  const trackHover = useCallback(
    (elementId, elementType, isCorrectOption = false) => {
      setMetrics((prev) => ({
        ...prev,
        hoverEvents: [
          ...prev.hoverEvents,
          {
            elementId,
            elementType,
            isCorrectOption,
            timestamp: Date.now(),
          },
        ],
      }));
    },
    [],
  );

  // Track click events
  const trackClick = useCallback(
    (elementId, elementType, isCorrectOption = false) => {
      // Record first action time
      if (!metrics.firstActionTime) {
        setMetrics((prev) => ({ ...prev, firstActionTime: Date.now() }));
      }

      // Add to click path
      clickPathRef.current.push(elementId);

      setMetrics((prev) => ({
        ...prev,
        clickEvents: [
          ...prev.clickEvents,
          {
            elementId,
            elementType,
            isCorrectOption,
            timestamp: Date.now(),
          },
        ],
      }));
    },
    [metrics.firstActionTime],
  );

  // Track fine print expansion
  const trackExpand = useCallback(() => {
    setMetrics((prev) => ({
      ...prev,
      expandEvents: [...prev.expandEvents, { timestamp: Date.now() }],
    }));
  }, []);

  // Track step changes (for multi-step flows)
  const setCurrentStep = useCallback((step, total) => {
    setMetrics((prev) => {
      const isBacktrack = step < prev.currentStep;
      return {
        ...prev,
        currentStep: step,
        totalSteps: total,
        backtracks: isBacktrack ? prev.backtracks + 1 : prev.backtracks,
      };
    });
  }, []);

  // Handle task completion
  const handleTaskAction = useCallback((action) => {
    setFinalAction(action);
    setShowPostTask(true);
  }, []);

  // Submit final response with post-task questions
  const handleSubmitPostTask = () => {
    const completionTime = Date.now() - metrics.taskStartTime;
    const timeToFirstAction = metrics.firstActionTime
      ? metrics.firstActionTime - metrics.taskStartTime
      : completionTime;

    // Calculate hover metrics
    const correctOptionHovers = metrics.hoverEvents.filter(
      (e) => e.isCorrectOption,
    );
    const correctOptionHoverTime = correctOptionHovers.length > 0 ? 500 : 0; // Simplified

    // Combine qualitative fields into structured reason (like phishing ActionModal)
    const combinedReason = [
      `DETAILS_NOTICED: ${postTaskResponses.details_noticed}`,
      `DECISION_REASON: ${postTaskResponses.decision_reason}`,
      `WHAT_INFLUENCED: ${postTaskResponses.what_influenced}`,
      `FRUSTRATION_POINTS: ${postTaskResponses.frustration_points}`,
    ].join("\n");

    // Prepare submission data
    const submissionData = {
      task_id: task.task_id,
      final_action: finalAction,
      task_completion_time_ms: completionTime,
      time_to_first_action_ms: timeToFirstAction,
      scroll_depth: metrics.maxScrollDepth,
      scroll_to_target_ms: null,
      fine_print_expand_count: metrics.expandEvents.length,
      fine_print_hover_time_ms:
        metrics.hoverEvents.filter((e) => e.elementType === "fine_print")
          .length * 200,
      option_hover_count: metrics.hoverEvents.filter((e) =>
        ["button", "option", "checkbox"].includes(e.elementType),
      ).length,
      correct_option_hover: correctOptionHovers.length > 0,
      correct_option_hover_time_ms: correctOptionHoverTime,
      backtrack_count: metrics.backtracks,
      click_path: clickPathRef.current,
      steps_completed: metrics.currentStep,
      steps_total: metrics.totalSteps,
      perceived_difficulty: postTaskResponses.perceived_difficulty,
      noticed_unusual: postTaskResponses.noticed_unusual,
      confidence_rating: postTaskResponses.confidence_rating,
      // Qualitative feedback (like phishing experiment)
      reason: combinedReason,
      details_noticed: postTaskResponses.details_noticed,
      decision_reason: postTaskResponses.decision_reason,
      what_influenced: postTaskResponses.what_influenced,
      frustration_points: postTaskResponses.frustration_points,
      scroll_events: metrics.scrollEvents,
      hover_events: metrics.hoverEvents,
      click_events: metrics.clickEvents,
    };

    onComplete(submissionData);
  };

  // Get the task component
  const TaskComponent = TASK_COMPONENTS[task?.context];

  if (!TaskComponent) {
    return (
      <div className="flex items-center justify-center h-full">
        <p className="text-red-500">Unknown task type: {task?.context}</p>
      </div>
    );
  }

  // Render post-task questions with qualitative feedback (like phishing ActionModal)
  if (showPostTask) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-lg p-8 max-w-2xl w-full max-h-[90vh] overflow-y-auto">
          <h2 className="text-xl font-bold text-gray-800 mb-2">
            Task Feedback
          </h2>
          <p className="text-gray-600 mb-6 text-sm">
            Please answer the following questions to help us understand your
            experience.
          </p>

          {/* Section 1: Your Observations (Qualitative) */}
          <div className="bg-gray-50 p-4 rounded-lg mb-6">
            <h3 className="font-semibold text-gray-800 text-sm uppercase tracking-wide mb-4">
              Your Observations
            </h3>

            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  What specific details did you notice about this interface?
                </label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-md p-3 focus:ring-purple-500 focus:border-purple-500"
                  value={postTaskResponses.details_noticed}
                  onChange={(e) =>
                    setPostTaskResponses((prev) => ({
                      ...prev,
                      details_noticed: e.target.value,
                    }))
                  }
                  placeholder="e.g., button colors, text size, layout, hidden options..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Why did you make this particular decision?
                </label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-md p-3 focus:ring-purple-500 focus:border-purple-500"
                  value={postTaskResponses.decision_reason}
                  onChange={(e) =>
                    setPostTaskResponses((prev) => ({
                      ...prev,
                      decision_reason: e.target.value,
                    }))
                  }
                  placeholder="Explain your reasoning..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  What influenced your choice the most?
                </label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-md p-3 focus:ring-purple-500 focus:border-purple-500"
                  value={postTaskResponses.what_influenced}
                  onChange={(e) =>
                    setPostTaskResponses((prev) => ({
                      ...prev,
                      what_influenced: e.target.value,
                    }))
                  }
                  placeholder="e.g., wording, visual design, time pressure, past experience..."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Was there anything frustrating or confusing about this task?
                </label>
                <input
                  type="text"
                  className="w-full border border-gray-300 rounded-md p-3 focus:ring-purple-500 focus:border-purple-500"
                  value={postTaskResponses.frustration_points}
                  onChange={(e) =>
                    setPostTaskResponses((prev) => ({
                      ...prev,
                      frustration_points: e.target.value,
                    }))
                  }
                  placeholder="e.g., nothing, or I found it hard to find the option I wanted..."
                />
              </div>
            </div>
          </div>

          {/* Section 2: Ratings */}
          <div className="bg-purple-50 p-4 rounded-lg mb-6">
            <h3 className="font-semibold text-gray-800 text-sm uppercase tracking-wide mb-4">
              Rate Your Experience
            </h3>

            {/* Perceived Difficulty */}
            <div className="mb-6">
              <p className="text-gray-700 mb-3 text-sm font-medium">
                How difficult was this task? (1-7)
              </p>
              <div className="flex gap-2">
                {[1, 2, 3, 4, 5, 6, 7].map((n) => (
                  <button
                    key={n}
                    onClick={() =>
                      setPostTaskResponses((prev) => ({
                        ...prev,
                        perceived_difficulty: n,
                      }))
                    }
                    className={`w-10 h-10 rounded-lg border-2 font-medium transition-colors ${
                      postTaskResponses.perceived_difficulty === n
                        ? "bg-purple-100 border-purple-500 text-purple-700"
                        : "border-gray-300 hover:border-gray-400"
                    }`}
                  >
                    {n}
                  </button>
                ))}
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Very Easy</span>
                <span>Very Hard</span>
              </div>
            </div>

            {/* Noticed Unusual */}
            <div className="mb-6">
              <p className="text-gray-700 mb-3 text-sm font-medium">
                Did you notice anything unusual about this interface?
              </p>
              <div className="flex gap-3">
                <button
                  onClick={() =>
                    setPostTaskResponses((prev) => ({
                      ...prev,
                      noticed_unusual: true,
                    }))
                  }
                  className={`px-6 py-2 rounded-lg border-2 font-medium transition-colors ${
                    postTaskResponses.noticed_unusual === true
                      ? "bg-purple-100 border-purple-500 text-purple-700"
                      : "border-gray-300 hover:border-gray-400"
                  }`}
                >
                  Yes
                </button>
                <button
                  onClick={() =>
                    setPostTaskResponses((prev) => ({
                      ...prev,
                      noticed_unusual: false,
                    }))
                  }
                  className={`px-6 py-2 rounded-lg border-2 font-medium transition-colors ${
                    postTaskResponses.noticed_unusual === false
                      ? "bg-purple-100 border-purple-500 text-purple-700"
                      : "border-gray-300 hover:border-gray-400"
                  }`}
                >
                  No
                </button>
              </div>
            </div>

            {/* Confidence Rating */}
            <div>
              <p className="text-gray-700 mb-3 text-sm font-medium">
                How confident are you in your decision? (1-7)
              </p>
              <div className="flex gap-2">
                {[1, 2, 3, 4, 5, 6, 7].map((n) => (
                  <button
                    key={n}
                    onClick={() =>
                      setPostTaskResponses((prev) => ({
                        ...prev,
                        confidence_rating: n,
                      }))
                    }
                    className={`w-10 h-10 rounded-lg border-2 font-medium transition-colors ${
                      postTaskResponses.confidence_rating === n
                        ? "bg-purple-100 border-purple-500 text-purple-700"
                        : "border-gray-300 hover:border-gray-400"
                    }`}
                  >
                    {n}
                  </button>
                ))}
              </div>
              <div className="flex justify-between text-xs text-gray-500 mt-1">
                <span>Not Confident</span>
                <span>Very Confident</span>
              </div>
            </div>
          </div>

          {/* Submit Button */}
          <button
            onClick={handleSubmitPostTask}
            disabled={false /* TODO: re-enable validation for production:
              postTaskResponses.perceived_difficulty === null ||
              postTaskResponses.noticed_unusual === null ||
              postTaskResponses.confidence_rating === null
            */}
            className="w-full bg-purple-600 text-white py-3 rounded-xl font-semibold hover:bg-purple-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
          >
            Continue to Next Task
          </button>
        </div>
      </div>
    );
  }

  // Get scenario details from task data (fetched from backend)
  // Falls back to defaults if scenario_info is not present
  const scenarioInfo = task?.scenario_info || {
    title: "Task",
    scenario: "Complete the task below.",
    goal: "Make your decision.",
    context: "",
  };

  // Render scenario introduction first
  if (showScenario) {
    return (
      <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
        <div className="bg-white rounded-2xl shadow-lg max-w-2xl w-full overflow-hidden">
          {/* Scenario Header */}
          <div className="bg-purple-600 px-6 py-4 text-white">
            <h2 className="text-xl font-bold">
              {progress
                ? `Task ${progress.current + 1}/${progress.total}: `
                : ""}
              {scenarioInfo.title}
            </h2>
          </div>

          {/* Scenario Content */}
          <div className="p-6 space-y-4">
            {/* Scenario Description */}
            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4">
              <h3 className="font-semibold text-blue-800 mb-2 flex items-center gap-2">
                <Info className="w-5 h-5" />
                Scenario
              </h3>
              <p className="text-blue-900 leading-relaxed">
                {scenarioInfo.scenario}
              </p>
            </div>

            {/* Your Goal */}
            <div className="bg-green-50 border border-green-200 rounded-xl p-4">
              <h3 className="font-semibold text-green-800 mb-2">Your Goal</h3>
              <p className="text-green-900">{scenarioInfo.goal}</p>
            </div>

            {/* Additional Context */}
            {scenarioInfo.context && (
              <div className="bg-gray-50 border border-gray-200 rounded-xl p-4">
                <h3 className="font-semibold text-gray-700 mb-2">
                  Keep in Mind
                </h3>
                <p className="text-gray-600 text-sm">{scenarioInfo.context}</p>
              </div>
            )}

            {/* Timer Warning if applicable */}
            {task.time_pressure && (
              <div className="bg-yellow-50 border border-yellow-200 rounded-xl p-4">
                <div className="flex items-center gap-2 text-yellow-800">
                  <Clock className="w-5 h-5" />
                  <span className="font-semibold">
                    Time Limit: {task.time_limit_seconds} seconds
                  </span>
                </div>
                <p className="text-yellow-700 text-sm mt-1">
                  This task has a time limit. Make your decision within the
                  given time.
                </p>
              </div>
            )}
          </div>

          {/* Start Button */}
          <div className="px-6 pb-6">
            <button
              onClick={() => setShowScenario(false)}
              className="w-full bg-purple-600 text-white py-3 rounded-xl font-semibold hover:bg-purple-700 transition-colors flex items-center justify-center gap-2"
            >
              I understand. Start Task
              <span className="text-lg">→</span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  // Render task
  return (
    <div className="min-h-screen bg-gray-100 flex flex-col">
      {/* Task Header */}
      <div className="bg-white shadow-sm px-6 py-3">
        <div className="flex items-center gap-3">
          <Info className="w-5 h-5 text-purple-600" />
          <span className="text-gray-700">
            <strong>{scenarioInfo.title}:</strong>{" "}
            {getTaskInstruction(task.context)}
          </span>
        </div>
      </div>

      {/* Prominent Timer Banner - Centered above content for better visibility */}
      {task.time_pressure && timeRemaining !== null && (
        <div
          className={`px-6 py-3 flex items-center justify-center gap-3 ${
            timeRemaining <= 10
              ? "bg-red-500 text-white animate-pulse"
              : timeRemaining <= 30
                ? "bg-orange-400 text-white"
                : "bg-purple-100 text-purple-800"
          }`}
        >
          <Clock
            className={`w-6 h-6 ${timeRemaining <= 10 ? "animate-bounce" : ""}`}
          />
          <span className="font-mono font-bold text-2xl">
            {Math.floor(timeRemaining / 60)}:
            {(timeRemaining % 60).toString().padStart(2, "0")}
          </span>
          <span className="text-sm font-medium">
            {timeRemaining <= 10
              ? "Hurry!"
              : timeRemaining <= 30
                ? "Time running low"
                : "Time Remaining"}
          </span>
        </div>
      )}

      {/* Timer Expired Warning */}
      {timerExpired && (
        <div className="bg-red-500 text-white px-6 py-4 flex items-center justify-center gap-3">
          <AlertCircle className="w-6 h-6" />
          <span className="font-semibold text-lg">
            Time has expired! Please complete your action now.
          </span>
        </div>
      )}

      {/* Task Content */}
      <div
        ref={containerRef}
        className="flex-1 overflow-auto"
        onScroll={handleScroll}
      >
        <TaskComponent
          task={task}
          uiContent={task.ui_content}
          onAction={handleTaskAction}
          trackClick={trackClick}
          trackHover={trackHover}
          trackExpand={trackExpand}
          setCurrentStep={setCurrentStep}
        />
      </div>
    </div>
  );
}
