/**
 * CYPEARL Experiment Web App - Main Application (UPDATED v5 - FULL SURVEY FLOW)
 *
 * UPDATES IN THIS VERSION:
 * 1. Added Pre-Experiment Survey (single page with all validated instruments)
 * 2. Added Post-Experiment Survey (STAI-6, PSS-4, Fatigue)
 * 3. New flow: Pre-Survey → Prolific ID → Email Experiment → Post-Survey → Complete
 *
 * FLOW:
 * 1. preExperiment: Single page pre-experiment questionnaire
 * 2. prolificId: Enter Prolific ID
 * 3. emailExperiment: 16 emails one by one
 * 4. postExperiment: Post-experiment state measures
 * 5. completed: Thank you screen
 *
 * FEATURES:
 * - Complete data collection matching phishing_study_participants.csv schema
 * - Pre-experiment questionnaires capture all individual differences
 * - Post-experiment measures capture state anxiety, stress, fatigue
 * - Alert shown after final email to proceed to post-survey
 *
 * Tracks all observational data required for phishing_study_responses.csv:
 * - response_latency_ms: Time from email open to action
 * - dwell_time_ms: Total time viewing email
 * - clicked: Whether any link was clicked (separate from action)
 * - hovered_link: Whether links were hovered
 * - inspected_sender: Whether user clicked to expand sender details
 * - confidence_rating: Self-reported confidence (1-10)
 * - suspicion_rating: Self-reported suspicion (1-10)
 */

import React, { useState, useEffect, useCallback, useRef } from "react";
import Sidebar from "./components/Sidebar";
import EmailList from "./components/EmailList";
import ReadingPane from "./components/ReadingPane";
import ActionModal from "./components/ActionModal";
import PreExperimentSurvey from "./components/PreExperimentSurvey";
import PostExperimentSurvey from "./components/PostExperimentSurvey";
import {
  Search,
  Check,
  Settings,
  HelpCircle,
  Mail,
  Calendar,
  Users,
  Paperclip,
  ArrowRight,
  CheckCircle,
} from "lucide-react";
import axios from "axios";

// =========================================================================
// HELPERS
// =========================================================================

const getParticipantId = () => localStorage.getItem("participant_id");
const getStudyStage = () =>
  localStorage.getItem("study_stage") || "preExperiment";
const getPreSurveyData = () => {
  const data = localStorage.getItem("pre_survey_data");
  return data ? JSON.parse(data) : null;
};

// Generate or retrieve session ID for linking pre-survey to participant
const getSessionId = () => {
  let sessionId = localStorage.getItem("session_id");
  if (!sessionId) {
    sessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    localStorage.setItem("session_id", sessionId);
  }
  return sessionId;
};

const getClientInfo = () => ({
  screen_width: window.screen.width,
  screen_height: window.screen.height,
  window_width: window.innerWidth,
  window_height: window.innerHeight,
  pixel_ratio: window.devicePixelRatio,
  user_agent: navigator.userAgent,
});

const API_URL = import.meta.env.VITE_API_URL || "http://localhost:8000";

// =========================================================================
// MAIN APP COMPONENT
// =========================================================================

function App() {
  // =========================================================================
  // STUDY FLOW STATE
  // Stages: preExperiment → prolificId → emailExperiment → postExperiment → completed
  // =========================================================================
  const [studyStage, setStudyStage] = useState(getStudyStage());
  const [preSurveyData, setPreSurveyData] = useState(getPreSurveyData());
  const [postSurveyData, setPostSurveyData] = useState(null);
  const [bonusTotal, setBonusTotal] = useState(null);

  // Auth state
  const [participantId, setParticipantId] = useState(getParticipantId());
  const [prolificIdInput, setProlificIdInput] = useState("");

  // Email state
  const [emailList, setEmailList] = useState([]);
  const [selectedEmailId, setSelectedEmailId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [finished, setFinished] = useState(false);
  const [activeFolder, setActiveFolder] = useState("inbox");

  // UI state
  const [modalOpen, setModalOpen] = useState(false);
  const [pendingAction, setPendingAction] = useState(null);
  const [actionsTaken, setActionsTaken] = useState(false);
  const [isTransitioning, setIsTransitioning] = useState(false);
  const [showPostSurveyAlert, setShowPostSurveyAlert] = useState(false);

  // Counts
  const [unreadCount, setUnreadCount] = useState(0);
  const [deletedCount, setDeletedCount] = useState(0);

  // Session tracking for observational data
  const [emailOpenTime, setEmailOpenTime] = useState(null);
  const [currentSessionData, setCurrentSessionData] = useState({
    linkClicked: false,
    linkHovered: false,
    senderInspected: false,
    linkHoverCount: 0,
    senderClickCount: 0,
  });

  // Refs to prevent duplicate API calls
  const openedEmailsRef = useRef(new Set());
  const markedReadRef = useRef(new Set());

  // =========================================================================
  // STAGE MANAGEMENT
  // =========================================================================

  const updateStudyStage = (newStage) => {
    localStorage.setItem("study_stage", newStage);
    setStudyStage(newStage);
  };

  // =========================================================================
  // PRE-EXPERIMENT SURVEY COMPLETION
  // =========================================================================

  const handlePreSurveyComplete = async (data) => {
    // Save to localStorage first (as backup)
    localStorage.setItem("pre_survey_data", JSON.stringify(data));
    setPreSurveyData(data);

    // Immediately save to backend - don't wait for Prolific ID step
    const sessionId = getSessionId();
    try {
      await axios.post(`${API_URL}/survey/pre`, {
        session_id: sessionId,
        user_agent: navigator.userAgent,
        screen_resolution: `${window.screen.width}x${window.screen.height}`,
        pre_survey_data: data,
      });
      console.log("Pre-survey saved to MongoDB immediately");
    } catch (error) {
      console.error("Error saving pre-survey immediately:", error);
      // Continue anyway - will try again during login
    }

    updateStudyStage("prolificId");
  };

  // =========================================================================
  // PROLIFIC ID SUBMISSION
  // =========================================================================

  const handleProlificSubmit = async (e) => {
    e.preventDefault();
    if (!prolificIdInput.trim()) return;

    // Get session ID to link pre-survey record
    const sessionId = getSessionId();

    // Read pre-survey data from localStorage as fallback
    const storedPreSurveyData = getPreSurveyData();
    const surveyDataToSend = storedPreSurveyData || preSurveyData;

    try {
      const response = await axios.post(`${API_URL}/auth/login`, {
        prolific_id: prolificIdInput.trim(),
        session_id: sessionId, // Link to pre-survey record
        user_agent: navigator.userAgent,
        screen_resolution: `${window.screen.width}x${window.screen.height}`,
        pre_survey_data: surveyDataToSend, // Fallback if no pre-survey record found
      });

      const newParticipantId = response.data.participant_id;
      localStorage.setItem("participant_id", newParticipantId);
      setParticipantId(newParticipantId);
      updateStudyStage("emailExperiment");
    } catch (error) {
      console.error("Login error:", error);
      alert("Failed to start session. Please try again.");
    }
  };

  // =========================================================================
  // POST-EXPERIMENT SURVEY COMPLETION
  // =========================================================================

  const handlePostSurveyComplete = async (data) => {
    setPostSurveyData(data);

    // Save post-survey data to backend
    try {
      await axios.post(`${API_URL}/survey/post/${participantId}`, {
        ...data,
        completed_at: new Date().toISOString(),
      });
    } catch (error) {
      console.error("Error saving post-survey:", error);
    }

    updateStudyStage("completed");
  };

  // =========================================================================
  // PROCEED TO POST-SURVEY (from finished emails)
  // =========================================================================

  const handleProceedToPostSurvey = () => {
    setShowPostSurveyAlert(false);
    updateStudyStage("postExperiment");
  };

  // =========================================================================
  // EMAIL FETCHING
  // =========================================================================

  const fetchEmails = async (
    folder = activeFolder,
    targetSelectionId = null,
  ) => {
    if (!participantId) return;

    setLoading(true);
    try {
      const response = await axios.get(
        `${API_URL}/emails/inbox/${participantId}?folder=${folder}`,
      );
      const data = response.data;
      const emails = data.emails || [];

      if (data.counts) {
        setUnreadCount(data.counts.unread);
        setDeletedCount(data.counts.deleted);
      }

      setEmailList(emails);

      if (targetSelectionId && emails.find((e) => e.id === targetSelectionId)) {
        setSelectedEmailId(targetSelectionId);
      } else if (!selectedEmailId && emails.length > 0) {
        setSelectedEmailId(emails[0].id);
      } else if (
        selectedEmailId &&
        !emails.find((e) => e.id === selectedEmailId)
      ) {
        setSelectedEmailId(emails.length > 0 ? emails[0].id : null);
      }

      if (data.is_finished !== undefined) {
        setFinished(data.is_finished);
        if (data.is_finished && studyStage === "emailExperiment") {
          // Get bonus and show alert to proceed
          try {
            const bonusResponse = await axios.get(
              `${API_URL}/bonus/${participantId}`,
            );
            setBonusTotal(bonusResponse.data.bonus_total || 0);
          } catch (e) {
            setBonusTotal(0);
          }
          setShowPostSurveyAlert(true);
        }
      } else {
        if (folder === "inbox" && emails.length === 0) {
          setFinished(true);
        } else {
          setFinished(false);
        }
      }
    } catch (error) {
      if (error.response && error.response.status === 404) {
        console.log("Participant not found, resetting session");
        localStorage.removeItem("participant_id");
        setParticipantId(null);
        updateStudyStage("prolificId");
        setEmailList([]);
        setFinished(false);
      } else {
        console.error("Error fetching emails:", error);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (participantId && studyStage === "emailExperiment") {
      fetchEmails(activeFolder);
    }
  }, [activeFolder, participantId, studyStage]);

  // =========================================================================
  // CLEAR SELECTION WHEN STUDY FINISHES
  // =========================================================================

  useEffect(() => {
    if (finished) {
      setSelectedEmailId(null);
    }
  }, [finished]);

  // =========================================================================
  // AUTO-ADVANCE FOR WELCOME EMAIL
  // =========================================================================

  useEffect(() => {
    if (
      emailList.length > 0 &&
      emailList[0].order_id === 0 &&
      !finished &&
      studyStage === "emailExperiment"
    ) {
      const timer = setTimeout(async () => {
        try {
          await axios.post(`${API_URL}/complete/${participantId}`);
          const response = await axios.get(
            `${API_URL}/emails/inbox/${participantId}?folder=${activeFolder}`,
          );
          const data = response.data;
          const emails = data.emails || [];

          if (data.counts) {
            setUnreadCount(data.counts.unread);
            setDeletedCount(data.counts.deleted);
          }

          setEmailList(emails);

          if (data.is_finished !== undefined) {
            setFinished(data.is_finished);
          } else if (activeFolder === "inbox" && emails.length === 0) {
            setFinished(true);
          }
        } catch (error) {
          console.error("Error auto-advancing:", error);
        }
      }, 8000);
      return () => clearTimeout(timer);
    }
  }, [emailList, finished, studyStage]);

  // =========================================================================
  // SESSION TRACKING - OPEN EMAIL SESSION
  // =========================================================================

  useEffect(() => {
    if (
      selectedEmailId &&
      participantId &&
      emailList.length > 0 &&
      studyStage === "emailExperiment"
    ) {
      const isLatestEmail = selectedEmailId === emailList[0].id;

      if (isLatestEmail && activeFolder === "inbox" && !finished) {
        if (!openedEmailsRef.current.has(selectedEmailId)) {
          setEmailOpenTime(Date.now());
          setActionsTaken(false);

          setCurrentSessionData({
            linkClicked: false,
            linkHovered: false,
            senderInspected: false,
            linkHoverCount: 0,
            senderClickCount: 0,
          });

          openedEmailsRef.current.add(selectedEmailId);
          openEmailSession(selectedEmailId);
        }
      }
    }
  }, [
    selectedEmailId,
    emailList,
    participantId,
    activeFolder,
    finished,
    studyStage,
  ]);

  const openEmailSession = async (emailId) => {
    try {
      await axios.post(`${API_URL}/session/open/${participantId}`, {
        email_id: emailId,
      });
    } catch (error) {
      console.log("Session open endpoint not available, using fallback");
    }
  };

  // =========================================================================
  // MARK EMAIL AS READ
  // =========================================================================

  useEffect(() => {
    if (
      selectedEmailId &&
      activeFolder === "inbox" &&
      !finished &&
      studyStage === "emailExperiment"
    ) {
      const email = emailList.find((e) => e.id === selectedEmailId);

      if (
        email &&
        !email.is_read &&
        !markedReadRef.current.has(selectedEmailId)
      ) {
        markedReadRef.current.add(selectedEmailId);

        axios
          .post(`${API_URL}/action/${participantId}`, {
            email_id: selectedEmailId,
            action_type: "mark_read",
          })
          .then(() => {
            setEmailList((prev) =>
              prev.map((e) =>
                e.id === selectedEmailId ? { ...e, is_read: true } : e,
              ),
            );
            setUnreadCount((prev) => Math.max(0, prev - 1));
          })
          .catch((err) => {
            markedReadRef.current.delete(selectedEmailId);
          });
      }
    }
  }, [selectedEmailId, finished, studyStage]);

  // =========================================================================
  // ACTION HANDLING
  // =========================================================================

  const handleAction = async (actionType, data = {}) => {
    if (["link_hover", "link_click", "sender_click"].includes(actionType)) {
      setCurrentSessionData((prev) => {
        const updated = { ...prev };

        if (actionType === "link_hover") {
          updated.linkHovered = true;
          updated.linkHoverCount = (prev.linkHoverCount || 0) + 1;
        } else if (actionType === "link_click") {
          if (!prev.linkClicked) {
            updated.linkClicked = true;
            calculateBonusSilently(data.link);
          }
        } else if (actionType === "sender_click") {
          updated.senderInspected = true;
          updated.senderClickCount = (prev.senderClickCount || 0) + 1;
        }

        return updated;
      });

      return;
    }

    if (["safe", "report", "delete", "ignore"].includes(actionType)) {
      setPendingAction(actionType);
      setModalOpen(true);
    }
  };

  // =========================================================================
  // SILENT BONUS CALCULATION
  // =========================================================================

  const calculateBonusSilently = async (link) => {
    if (!participantId || !link) return;

    try {
      await axios.post(`${API_URL}/bonus/calculate/${participantId}`, {
        link: link,
        email_id: selectedEmailId,
        timestamp: new Date().toISOString(),
      });
    } catch (error) {
      console.log("Bonus calculation sent to backend");
    }
  };

  // =========================================================================
  // SUBMIT FINAL ACTION
  // =========================================================================

  const handleSubmitAction = async ({ reason, confidence, suspicion }) => {
    if (!selectedEmailId || !pendingAction) return;

    const dwellTime = emailOpenTime ? Date.now() - emailOpenTime : 0;
    const latency = emailOpenTime ? Date.now() - emailOpenTime : 0;
    const isDeleteAction = pendingAction === "delete";

    try {
      await axios.post(`${API_URL}/action/${participantId}`, {
        email_id: selectedEmailId,
        action_type: pendingAction,
        reason: reason,
        confidence: confidence,
        suspicion: suspicion,
        latency_ms: latency,
        dwell_time_ms: dwellTime,
        clicked_link: currentSessionData.linkClicked,
        hovered_link: currentSessionData.linkHovered,
        inspected_sender: currentSessionData.senderInspected,
        link_hover_count: currentSessionData.linkHoverCount,
        sender_click_count: currentSessionData.senderClickCount,
        client_info: getClientInfo(),
      });

      setModalOpen(false);
      setPendingAction(null);

      if (isDeleteAction) {
        setDeletedCount((prev) => prev + 1);

        const deletedIndex = emailList.findIndex(
          (e) => e.id === selectedEmailId,
        );
        const nextIndex =
          deletedIndex < emailList.length - 1
            ? deletedIndex + 1
            : Math.max(0, deletedIndex - 1);
        const emailToSelectAfterDelete = emailList[nextIndex];
        const targetEmailId = emailToSelectAfterDelete?.id;

        setEmailList((prev) => prev.filter((e) => e.id !== selectedEmailId));

        try {
          await axios.post(`${API_URL}/complete/${participantId}`);

          openedEmailsRef.current.clear();
          markedReadRef.current.clear();

          await fetchEmails(activeFolder, targetEmailId);

          setActionsTaken(false);
          setEmailOpenTime(null);
          setCurrentSessionData({
            linkClicked: false,
            linkHovered: false,
            senderInspected: false,
            linkHoverCount: 0,
            senderClickCount: 0,
          });
        } catch (error) {
          console.error("Error after delete:", error);
        }

        return;
      }

      setActionsTaken(true);
    } catch (error) {
      console.error("Error submitting action:", error);
      alert("Failed to submit action. Please try again.");
    }
  };

  // =========================================================================
  // NAVIGATION
  // =========================================================================

  const handleDone = async () => {
    if (!selectedEmailId && !finished) return;

    setIsTransitioning(true);

    try {
      await axios.post(`${API_URL}/complete/${participantId}`);

      openedEmailsRef.current.clear();
      markedReadRef.current.clear();

      await fetchEmails(activeFolder);

      setActionsTaken(false);
      setEmailOpenTime(null);
      setCurrentSessionData({
        linkClicked: false,
        linkHovered: false,
        senderInspected: false,
        linkHoverCount: 0,
        senderClickCount: 0,
      });
    } catch (error) {
      console.error("Error advancing to next email:", error);
    } finally {
      setIsTransitioning(false);
    }
  };

  // =========================================================================
  // RESET HANDLER
  // =========================================================================

  const handleReset = async () => {
    if (
      confirm(
        "Reset your session? This will clear all progress including your survey responses.",
      )
    ) {
      localStorage.removeItem("participant_id");
      localStorage.removeItem("study_stage");
      localStorage.removeItem("pre_survey_data");
      localStorage.removeItem("session_id");
      localStorage.removeItem("pre_survey_responses_draft");
      localStorage.removeItem("post_survey_responses_draft");
      openedEmailsRef.current.clear();
      markedReadRef.current.clear();
      setParticipantId(null);
      setStudyStage("preExperiment");
      setPreSurveyData(null);
      setPostSurveyData(null);
      setEmailList([]);
      setFinished(false);
      setSelectedEmailId(null);
      setActionsTaken(false);
      setShowPostSurveyAlert(false);
      setBonusTotal(null);
    }
  };

  // =========================================================================
  // DERIVED STATE
  // =========================================================================

  const selectedEmail = emailList.find((e) => e.id === selectedEmailId);
  const isLatest = emailList.length > 0 && selectedEmailId === emailList[0].id;

  // =========================================================================
  // RENDER: PRE-EXPERIMENT SURVEY
  // =========================================================================

  if (studyStage === "preExperiment") {
    return (
      <PreExperimentSurvey
        onComplete={handlePreSurveyComplete}
        onGoToEmail={() => updateStudyStage("emailExperiment")}
      />
    );
  }

  // =========================================================================
  // RENDER: PROLIFIC ID PROMPT
  // =========================================================================

  if (studyStage === "prolificId") {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-100 font-sans">
        <div className="bg-white p-8 rounded-lg shadow-md max-w-md w-full">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle className="w-6 h-6 text-green-600" />
            <span className="text-green-700 font-medium">
              Pre-experiment survey completed!
            </span>
          </div>
          <h1 className="text-2xl font-bold mb-6 text-center text-[#0078d4]">
            Enter Your ID
          </h1>
          <form onSubmit={handleProlificSubmit} className="space-y-4">
            <div>
              <label
                htmlFor="prolificId"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Please enter your Prolific ID to continue...
              </label>
              <input
                type="text"
                id="prolificId"
                value={prolificIdInput}
                onChange={(e) => setProlificIdInput(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                placeholder="Enter ID here"
                required
              />
            </div>
            <button
              type="submit"
              className="w-full bg-[#0078d4] text-white py-2 px-4 rounded-md hover:bg-[#106ebe] transition-colors font-semibold flex items-center justify-center gap-2"
            >
              Start Email Evaluation
              <ArrowRight size={18} />
            </button>
          </form>
        </div>
      </div>
    );
  }

  // =========================================================================
  // RENDER: POST-EXPERIMENT SURVEY
  // =========================================================================

  if (studyStage === "postExperiment") {
    return (
      <PostExperimentSurvey
        onComplete={handlePostSurveyComplete}
        participantId={participantId}
        bonusTotal={bonusTotal}
        onGoToEmail={() => updateStudyStage("emailExperiment")}
      />
    );
  }

  // =========================================================================
  // RENDER: COMPLETED
  // =========================================================================

  if (studyStage === "completed") {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-100 font-sans">
        <div className="bg-white p-8 rounded-lg shadow-md max-w-lg w-full text-center">
          <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <CheckCircle className="w-10 h-10 text-green-600" />
          </div>
          <h1 className="text-2xl font-bold mb-4 text-green-700">
            Study Complete!
          </h1>
          <p className="text-gray-600 mb-6">
            Thank you for participating in this research study. Your responses
            have been recorded.
          </p>
          {bonusTotal !== null && (
            <div className="p-4 bg-green-50 rounded-lg border border-green-200 mb-6">
              <p className="text-green-800 font-medium">
                Your final performance bonus:{" "}
                <span className="text-2xl">
                  {bonusTotal >= 0 ? "+" : ""}
                  {bonusTotal} points
                </span>
              </p>
            </div>
          )}
          <p className="text-sm text-gray-500 mb-4">
            Participant ID:{" "}
            <code className="bg-gray-100 px-2 py-1 rounded">
              {participantId}
            </code>
          </p>
          <p className="text-sm text-gray-500">
            You may now close this window or return to Prolific.
          </p>
        </div>
      </div>
    );
  }

  // =========================================================================
  // RENDER: LOADING
  // =========================================================================

  if (
    loading &&
    emailList.length === 0 &&
    !finished &&
    studyStage === "emailExperiment"
  ) {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-100">
        <div className="flex flex-col items-center">
          <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
          <div className="text-blue-600 font-semibold">
            Loading ProMail Suite...
          </div>
        </div>
      </div>
    );
  }

  // =========================================================================
  // RENDER: POST-SURVEY ALERT (After finishing all emails)
  // =========================================================================

  if (showPostSurveyAlert && studyStage === "emailExperiment") {
    return (
      <div className="flex h-screen items-center justify-center bg-gray-100 font-sans">
        <div className="bg-white p-8 rounded-lg shadow-md max-w-lg w-full text-center">
          <div className="w-16 h-16 bg-blue-100 rounded-full flex items-center justify-center mx-auto mb-4">
            <CheckCircle className="w-10 h-10 text-blue-600" />
          </div>
          <h1 className="text-2xl font-bold mb-4 text-[#0078d4]">
            Email Evaluation Complete!
          </h1>
          <p className="text-gray-600 mb-6">
            You have reviewed all 16 emails. Please proceed to complete a brief
            final questionnaire.
          </p>
          {bonusTotal !== null && (
            <div className="p-4 bg-green-50 rounded-lg border border-green-200 mb-6">
              <p className="text-green-800 font-medium">
                Your performance bonus so far:{" "}
                <span className="text-xl">
                  {bonusTotal >= 0 ? "+" : ""}
                  {bonusTotal} points
                </span>
              </p>
            </div>
          )}
          <button
            onClick={handleProceedToPostSurvey}
            className="w-full bg-[#0078d4] text-white py-3 px-4 rounded-md hover:bg-[#106ebe] transition-colors font-semibold flex items-center justify-center gap-2"
          >
            Continue to Final Questionnaire
            <ArrowRight size={18} />
          </button>
        </div>
      </div>
    );
  }

  // =========================================================================
  // RENDER: MAIN EMAIL APP
  // =========================================================================

  return (
    <div className="flex h-screen bg-white overflow-hidden font-sans">
      {/* OWA Header */}
      <div className="absolute top-0 left-0 w-full h-[48px] bg-[#0078d4] flex items-center justify-between px-2 z-50 text-white select-none">
        <div className="flex items-center">
          <div className="p-3 hover:bg-[#106ebe] cursor-pointer transition-colors rounded-sm mr-1">
            <div className="grid grid-cols-3 gap-[2px]">
              {[...Array(9)].map((_, i) => (
                <div
                  key={i}
                  className="w-[3px] h-[3px] bg-white rounded-full"
                ></div>
              ))}
            </div>
          </div>
          <span className="font-semibold text-base tracking-wide ml-1">
            ProMail Suite
          </span>
        </div>

        {/* Search Bar */}
        <div className="flex-1 max-w-2xl mx-4">
          <div className="relative group">
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <Search size={16} className="text-[#0078d4]" />
            </div>
            <input
              type="text"
              placeholder="Search"
              className="block w-full pl-10 pr-3 py-1.5 border-none rounded-[4px] leading-5 bg-[#c3e0fa] text-black placeholder-gray-600 focus:outline-none focus:bg-white focus:ring-0 transition-colors h-[32px] text-sm"
            />
          </div>
        </div>

        {/* Right Actions */}
        <div className="flex items-center space-x-1">
          <div
            className="w-8 h-8 flex items-center justify-center hover:bg-[#106ebe] rounded-sm cursor-pointer"
            onClick={handleReset}
            title="Reset Simulation"
          >
            <Settings size={20} strokeWidth={1.5} />
          </div>
          <div className="w-8 h-8 flex items-center justify-center hover:bg-[#106ebe] rounded-sm cursor-pointer">
            <HelpCircle size={20} strokeWidth={1.5} />
          </div>
          <div className="ml-2 w-8 h-8 rounded-full bg-[#004578] flex items-center justify-center text-xs font-bold border-2 border-white/20 cursor-pointer hover:opacity-90">
            JD
          </div>
        </div>
      </div>

      {/* Main Content Area */}
      <div className="flex w-full h-full pt-[48px]">
        {/* Left Rail */}
        <div className="w-[48px] bg-[#f3f2f1] flex flex-col items-center py-2 space-y-4 border-r border-gray-200 z-40">
          <div className="p-2 text-[#0078d4] bg-white shadow-sm rounded cursor-pointer">
            <Mail size={20} strokeWidth={1.5} />
          </div>
          <div className="p-2 text-[#605e5c] hover:bg-gray-200 rounded cursor-pointer">
            <Calendar size={20} strokeWidth={1.5} />
          </div>
          <div className="p-2 text-[#605e5c] hover:bg-gray-200 rounded cursor-pointer">
            <Users size={20} strokeWidth={1.5} />
          </div>
          <div className="p-2 text-[#605e5c] hover:bg-gray-200 rounded cursor-pointer">
            <Paperclip size={20} strokeWidth={1.5} />
          </div>
        </div>

        <Sidebar
          activeFolder={activeFolder}
          onFolderSelect={setActiveFolder}
          unreadCount={unreadCount}
          deletedCount={deletedCount}
        />

        {finished ? (
          <>
            <EmailList
              emails={emailList}
              selectedId={selectedEmailId}
              onSelect={setSelectedEmailId}
              onDone={() => {}}
              actionsTaken={true}
            />
            <ReadingPane
              email={selectedEmail}
              onAction={() => {}}
              isLatest={false}
              actionsTaken={true}
              onDone={() => {}}
              isFinished={true}
              participantId={participantId}
            />
          </>
        ) : activeFolder === "inbox" || activeFolder === "deleted" ? (
          <>
            <EmailList
              emails={emailList}
              selectedId={selectedEmailId}
              onSelect={setSelectedEmailId}
              onDone={handleDone}
              actionsTaken={actionsTaken}
            />
            <ReadingPane
              email={selectedEmail}
              onAction={handleAction}
              isLatest={isLatest && !isTransitioning}
              actionsTaken={actionsTaken}
              onDone={handleDone}
              isFinished={false}
              participantId={participantId}
            />
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center bg-[#f3f2f1] text-[#605e5c]">
            <div className="text-center">
              <div className="text-lg font-semibold mb-1">Nothing here</div>
              <div className="text-sm">This folder is empty</div>
            </div>
          </div>
        )}
      </div>

      <ActionModal
        isOpen={modalOpen}
        onClose={() => setModalOpen(false)}
        onSubmit={handleSubmitAction}
        actionType={pendingAction}
      />

      {/* TESTING ONLY: Navigation buttons for quick access to surveys */}
      <div className="fixed bottom-0 left-0 right-0 px-4 py-2 flex justify-between items-center z-50 pointer-events-none">
        <button
          onClick={() => updateStudyStage("preExperiment")}
          className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium pointer-events-auto"
        >
          ← Go to Pre-Experiment Survey
        </button>
        <button
          onClick={() => {
            // Fetch bonus before showing post-survey
            if (participantId) {
              axios
                .get(`${API_URL}/bonus/${participantId}`)
                .then((res) => setBonusTotal(res.data.bonus_total || 0))
                .catch(() => setBonusTotal(0));
            }
            updateStudyStage("postExperiment");
          }}
          className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium pointer-events-auto"
        >
          Go to Post-Experiment Survey →
        </button>
      </div>
    </div>
  );
}

export default App;
