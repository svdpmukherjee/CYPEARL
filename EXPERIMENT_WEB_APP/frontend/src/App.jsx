/**
 * CYPEARL Experiment Web App - Main Application (UPDATED v8 - JOB ROLE SELECTION)
 *
 * UPDATES IN THIS VERSION:
 * 1. Added scenario selector landing page (Phishing, Dark Patterns, Fake News)
 * 2. Scenario-aware routing and database selection
 * 3. Added job role selection stage for phishing experiment
 * 4. Added comprehension check stage (if endpoint available)
 * 5. Added perceived_relevance to action submissions
 *
 * SCENARIOS:
 * - phishing: Email security experiment (existing)
 * - dark-patterns: Deceptive UI/UX experiment (new)
 * - fake-news: Misinformation detection experiment (new)
 *
 * FLOW (Phishing):
 * 1. scenarioSelection: Choose experiment type
 * 2. prolificId: Enter Prolific ID (creates participant)
 * 3. jobRoleSelection: Select closest job cluster (screen-out if "Other")
 * 4. preExperiment: Single page pre-experiment questionnaire
 * 5. comprehensionCheck: Answer comprehension questions (if endpoint exists)
 * 6. instructions: Dedicated page showing study instructions (welcome email content)
 * 7. emailExperiment: Scenario-specific tasks (16 items)
 * 8. postExperiment: Post-experiment state measures
 * 9. completed: Thank you screen
 *
 * FEATURES:
 * - Complete data collection matching study schema
 * - Pre-experiment questionnaires capture all individual differences
 * - Job role clustering for personalized email assignment
 * - Post-experiment measures capture state anxiety, stress, fatigue
 * - Alert shown after final task to proceed to post-survey
 *
 * Tracks all observational data required for study responses
 */

import React, { useState, useEffect, useCallback, useRef } from "react";
// Phishing experiment components
import Sidebar from "./components/phishing/Sidebar";
import EmailList from "./components/phishing/EmailList";
import ReadingPane from "./components/phishing/ReadingPane";
import ActionModal from "./components/phishing/ActionModal";
import PreExperimentSurvey from "./components/phishing/PreExperimentSurvey";
import PostExperimentSurvey from "./components/phishing/PostExperimentSurvey";
// Shared components
import ScenarioSelector from "./components/ScenarioSelector";
// Dark patterns experiment
import DarkPatternsExperiment from "./components/darkpatterns/DarkPatternsExperiment";
// Fake news experiment
import FakeNewsExperiment from "./components/fakenews/FakeNewsExperiment";
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
const getScenario = () => localStorage.getItem("scenario") || null;
const getStudyStage = () =>
  localStorage.getItem("study_stage") || "scenarioSelection";
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

const API_URL = import.meta.env.VITE_API_URL || "/api";

// =========================================================================
// JOB CLUSTER CONSTANTS
// =========================================================================

const JOB_CLUSTERS = [
  "Finance/Accounts Payable",
  "IT Support/Helpdesk",
  "HR/People Operations",
  "Sales/Business Development",
  "Operations/Logistics",
  "Customer Service/Client Support",
  "Marketing/Communications",
  "Procurement/Purchasing",
  "Administrative/Executive Support",
  "Compliance/Risk/Audit",
];

const INDUSTRY_TO_CLUSTER = {
  finance: "Finance/Accounts Payable",
  banking: "Finance/Accounts Payable",
  accounting: "Finance/Accounts Payable",
  accounts_payable: "Finance/Accounts Payable",
  it: "IT Support/Helpdesk",
  technology: "IT Support/Helpdesk",
  software: "IT Support/Helpdesk",
  helpdesk: "IT Support/Helpdesk",
  human_resources: "HR/People Operations",
  hr: "HR/People Operations",
  people_operations: "HR/People Operations",
  sales: "Sales/Business Development",
  business_development: "Sales/Business Development",
  consulting: "Sales/Business Development",
  operations: "Operations/Logistics",
  logistics: "Operations/Logistics",
  engineering: "Operations/Logistics",
  manufacturing: "Operations/Logistics",
  customer_service: "Customer Service/Client Support",
  client_support: "Customer Service/Client Support",
  support: "Customer Service/Client Support",
  marketing: "Marketing/Communications",
  communications: "Marketing/Communications",
  public_relations: "Marketing/Communications",
  procurement: "Procurement/Purchasing",
  purchasing: "Procurement/Purchasing",
  supply_chain: "Procurement/Purchasing",
  administration: "Administrative/Executive Support",
  executive_assistant: "Administrative/Executive Support",
  office_management: "Administrative/Executive Support",
  compliance: "Compliance/Risk/Audit",
  risk: "Compliance/Risk/Audit",
  audit: "Compliance/Risk/Audit",
  legal: "Compliance/Risk/Audit",
};

const JOB_CLUSTER_DESCRIPTIONS = {
  "Finance/Accounts Payable": "Financial services, accounting, auditing, banking, invoice processing, payroll",
  "IT Support/Helpdesk": "IT support, helpdesk, system administration, technical troubleshooting",
  "HR/People Operations": "Human resources, recruitment, onboarding, employee relations, people ops",
  "Sales/Business Development": "Sales, business development, account management, client acquisition",
  "Operations/Logistics": "Operations management, supply chain, logistics, warehouse, manufacturing",
  "Customer Service/Client Support": "Customer service, client support, call centre, complaint resolution",
  "Marketing/Communications": "Marketing, communications, PR, content creation, brand management",
  "Procurement/Purchasing": "Procurement, purchasing, vendor management, sourcing, contract negotiation",
  "Administrative/Executive Support": "Office administration, executive assistant, scheduling, office management",
  "Compliance/Risk/Audit": "Compliance, risk management, internal audit, regulatory affairs, legal",
};

// Scenario metadata for tab title and favicon
const SCENARIO_META = {
  phishing: {
    title: "ProMail Suite: Smart. Fast. Reliable.",
    favicon: "/icon.png",
  },
  "dark-patterns": {
    title: "Dark Patterns Study",
    favicon: "/favicon-darkpatterns.svg",
  },
  "fake-news": {
    title: "News Evaluation Study",
    favicon: "/favicon-fakenews.svg",
  },
};

// =========================================================================
// MAIN APP COMPONENT
// =========================================================================

function App() {
  // =========================================================================
  // STUDY FLOW STATE
  // Stages: scenarioSelection → prolificId → jobRoleSelection → preExperiment → comprehensionCheck → instructions → emailExperiment → postExperiment → completed
  // =========================================================================
  const [studyStage, setStudyStage] = useState(getStudyStage());
  const [scenario, setScenario] = useState(getScenario());
  const [preSurveyData, setPreSurveyData] = useState(getPreSurveyData());
  const [postSurveyData, setPostSurveyData] = useState(null);
  const [bonusTotal, setBonusTotal] = useState(null);
  const [jobCluster, setJobCluster] = useState(localStorage.getItem("job_cluster") || null);
  const [jobNatureDescription, setJobNatureDescription] = useState("");
  const [instructionEmail, setInstructionEmail] = useState(null);

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
  const welcomeAutoAdvancedRef = useRef(false);

  // =========================================================================
  // STAGE MANAGEMENT
  // =========================================================================

  const updateStudyStage = (newStage) => {
    localStorage.setItem("study_stage", newStage);
    setStudyStage(newStage);
  };

  // =========================================================================
  // DYNAMIC TAB TITLE AND FAVICON BASED ON SCENARIO
  // =========================================================================

  useEffect(() => {
    const currentScenario = scenario || getScenario();

    // Default meta for scenario selector
    const defaultMeta = {
      title: "CYPEARL Data Collection Platform",
      favicon: "/screenicon.png",
    };

    const meta =
      currentScenario && SCENARIO_META[currentScenario]
        ? SCENARIO_META[currentScenario]
        : defaultMeta;

    // Update document title
    document.title = meta.title;

    // Update favicon
    let link = document.querySelector("link[rel~='icon']");
    if (!link) {
      link = document.createElement("link");
      link.rel = "icon";
      document.head.appendChild(link);
    }
    link.href = meta.favicon;
    link.type = meta.favicon.endsWith(".svg") ? "image/svg+xml" : "image/png";
  }, [scenario]);

  // =========================================================================
  // SCENARIO SELECTION
  // =========================================================================

  const handleScenarioSelect = (selectedScenario) => {
    localStorage.setItem("scenario", selectedScenario);
    setScenario(selectedScenario);
    updateStudyStage("prolificId");
  };

  const handleGoToScenarioSelector = () => {
    localStorage.removeItem("scenario");
    setScenario(null);
    updateStudyStage("scenarioSelection");
  };

  // =========================================================================
  // PRE-EXPERIMENT SURVEY COMPLETION
  // =========================================================================

  const handlePreSurveyComplete = async (data) => {
    // Save to localStorage first (as backup)
    localStorage.setItem("pre_survey_data", JSON.stringify(data));
    setPreSurveyData(data);

    // Save to backend with participant_id (login already happened)
    const currentParticipantId = participantId || getParticipantId();
    try {
      await axios.post(`${API_URL}/survey/pre`, {
        participant_id: currentParticipantId,
        session_id: getSessionId(),
        user_agent: navigator.userAgent,
        screen_resolution: `${window.screen.width}x${window.screen.height}`,
        pre_survey_data: data,
      });
      console.log("Pre-survey saved to MongoDB with participant_id");
    } catch (error) {
      console.error("Error saving pre-survey:", error);
      // Continue anyway - data is in localStorage
    }

    // Try comprehension check; if it doesn't exist, skip to instructions
    try {
      await axios.get(`${API_URL}/comprehension-check/${currentParticipantId}`);
      updateStudyStage("comprehensionCheck");
    } catch {
      updateStudyStage("instructions");
    }
  };

  // =========================================================================
  // PROLIFIC ID SUBMISSION
  // =========================================================================

  const handleProlificSubmit = async (e) => {
    e.preventDefault();
    if (!prolificIdInput.trim()) return;

    // Get session ID for linking
    const sessionId = getSessionId();
    const currentScenario = scenario || getScenario() || "phishing";

    // Use the correct auth endpoint based on scenario
    let authEndpoint = `${API_URL}/auth/login`;
    if (currentScenario === "dark-patterns") {
      authEndpoint = `${API_URL}/dark-patterns/auth/login`;
    } else if (currentScenario === "fake-news") {
      authEndpoint = `${API_URL}/fake-news/auth/login`;
    }

    try {
      const response = await axios.post(authEndpoint, {
        prolific_id: prolificIdInput.trim(),
        session_id: sessionId,
        scenario: currentScenario,
        user_agent: navigator.userAgent,
        screen_resolution: `${window.screen.width}x${window.screen.height}`,
      });

      const newParticipantId = response.data.participant_id;
      localStorage.setItem("participant_id", newParticipantId);
      setParticipantId(newParticipantId);
      updateStudyStage("jobRoleSelection"); // Now go to job role selection
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
    forceSelectTop = false,
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

      // Filter out the welcome/instruction email (order_id=0) — it is shown
      // on the dedicated instructions page before the email experiment starts.
      const nonWelcome = emails.filter((e) => e.order_id !== 0);

      // Sort newest-first using delivery timestamp, with order_id as a stable fallback.
      const sorted = [...nonWelcome].sort((a, b) => {
        const aTime = new Date(a.timestamp || 0).getTime();
        const bTime = new Date(b.timestamp || 0).getTime();

        if (bTime !== aTime) {
          return bTime - aTime;
        }

        const aOrder = typeof a.order_id === "number" ? a.order_id : -1;
        const bOrder = typeof b.order_id === "number" ? b.order_id : -1;
        return bOrder - aOrder;
      });
      setEmailList(sorted);

      if (targetSelectionId && sorted.find((e) => e.id === targetSelectionId)) {
        setSelectedEmailId(targetSelectionId);
      } else if (forceSelectTop && sorted.length > 0) {
        setSelectedEmailId(sorted[0].id);
      } else if (!selectedEmailId && sorted.length > 0) {
        setSelectedEmailId(sorted[0].id);
      } else if (
        selectedEmailId &&
        !sorted.find((e) => e.id === selectedEmailId)
      ) {
        setSelectedEmailId(sorted.length > 0 ? sorted[0].id : null);
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
    // Auto-advance only when the welcome email is the sole email and hasn't been auto-advanced yet
    const hasOnlyWelcome =
      emailList.length > 0 &&
      emailList.every((e) => e.order_id === 0);

    if (
      hasOnlyWelcome &&
      !finished &&
      studyStage === "emailExperiment" &&
      !welcomeAutoAdvancedRef.current
    ) {
      const timer = setTimeout(async () => {
        welcomeAutoAdvancedRef.current = true;
        try {
          await axios.post(`${API_URL}/complete/${participantId}`);
          // Re-use fetchEmails with current selection as target to preserve focus
          await fetchEmails(activeFolder, selectedEmailId);
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

    if (["safe", "report"].includes(actionType)) {
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

  const handleSubmitAction = async ({ reason, confidence, suspicion, perceivedRelevance }) => {
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
        perceived_relevance: perceivedRelevance,
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

      // Brief pause before refreshing inbox so next email appears with a delay.
      await new Promise((resolve) => setTimeout(resolve, 2000));

      // Preserve current selection after advancing so newly arrived emails stay unread.
      await fetchEmails(activeFolder, selectedEmailId);

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
      localStorage.removeItem("scenario");
      localStorage.removeItem("pre_survey_data");
      localStorage.removeItem("session_id");
      localStorage.removeItem("pre_survey_responses_draft");
      localStorage.removeItem("post_survey_responses_draft");
      localStorage.removeItem("job_cluster");
      openedEmailsRef.current.clear();
      markedReadRef.current.clear();
      welcomeAutoAdvancedRef.current = false;
      setParticipantId(null);
      setScenario(null);
      setStudyStage("scenarioSelection");
      setPreSurveyData(null);
      setPostSurveyData(null);
      setJobCluster(null);
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
  // RENDER: SCENARIO SELECTION
  // =========================================================================

  if (studyStage === "scenarioSelection") {
    return <ScenarioSelector onSelectScenario={handleScenarioSelect} />;
  }

  // =========================================================================
  // RENDER: DARK PATTERNS EXPERIMENT (Full flow)
  // =========================================================================

  if (
    scenario === "dark-patterns" &&
    studyStage !== "scenarioSelection" &&
    studyStage !== "prolificId"
  ) {
    return (
      <DarkPatternsExperiment
        participantId={participantId}
        onComplete={() => updateStudyStage("completed")}
        onGoToScenarioSelector={handleGoToScenarioSelector}
      />
    );
  }

  // =========================================================================
  // RENDER: FAKE NEWS EXPERIMENT (Full flow)
  // =========================================================================

  if (
    scenario === "fake-news" &&
    studyStage !== "scenarioSelection" &&
    studyStage !== "prolificId"
  ) {
    return (
      <FakeNewsExperiment
        participantId={participantId}
        onComplete={() => updateStudyStage("completed")}
        onGoToScenarioSelector={handleGoToScenarioSelector}
      />
    );
  }

  // =========================================================================
  // RENDER: PRE-EXPERIMENT SURVEY (Phishing)
  // =========================================================================

  if (studyStage === "preExperiment") {
    // Phishing pre-survey
    return (
      <>
        <PreExperimentSurvey
          onComplete={handlePreSurveyComplete}
          onGoToEmail={() => updateStudyStage("emailExperiment")}
        />
        <div className="fixed bottom-0 left-0 px-4 py-2 z-50">
          <button
            onClick={handleGoToScenarioSelector}
            className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium"
          >
            ← Go to Scenario Selector
          </button>
        </div>
      </>
    );
  }

  // =========================================================================
  // RENDER: JOB ROLE SELECTION (Phishing)
  // =========================================================================

  if (studyStage === "jobRoleSelection") {
    const [selectedCluster, setSelectedCluster] = [
      jobCluster,
      (val) => setJobCluster(val),
    ];

    const handleClusterSubmit = async () => {
      if (!selectedCluster || selectedCluster === "Other") return;
      if (!jobNatureDescription.trim()) {
        alert("Please describe your specific job role before continuing.");
        return;
      }
      const currentParticipantId = participantId || getParticipantId();
      try {
        const response = await axios.post(
          `${API_URL}/assign-cluster/${currentParticipantId}`,
          {
            job_cluster: selectedCluster,
            job_nature_description: jobNatureDescription.trim(),
          },
        );
        localStorage.setItem("job_cluster", selectedCluster);
        setJobCluster(selectedCluster);
        // Store counterbalance_version if returned
        if (response.data?.counterbalance_version) {
          localStorage.setItem(
            "counterbalance_version",
            response.data.counterbalance_version,
          );
        }
        // Proceed to pre-experiment survey
        updateStudyStage("preExperiment");
      } catch (error) {
        console.error("Error assigning cluster:", error);
        alert("Failed to save job category. Please try again.");
      }
    };

    // Split clusters into two columns (alternating for proximity)
    const leftColumn = JOB_CLUSTERS.filter((_, i) => i % 2 === 0);
    const rightColumn = JOB_CLUSTERS.filter((_, i) => i % 2 === 1);

    return (
      <div className="flex h-screen items-center justify-center bg-gray-100 font-sans">
        <div className="bg-white p-8 rounded-lg shadow-md max-w-3xl w-full">
          <h1 className="text-2xl font-bold mb-2 text-center text-slate-800">
            What best describes your job?
          </h1>
          <p className="text-gray-500 mb-1 text-center text-sm">
            To personalise the study emails to your professional background, please select the
            category that is closest to your current role or daily responsibilities.
          </p>
          <p className="text-gray-400 mb-6 text-center text-xs">
            Select one option below, then briefly describe what you do.
          </p>

          {/* Two-column grid */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-4">
            {/* Left column */}
            <div className="space-y-2">
              {leftColumn.map((cluster) => (
                <label
                  key={cluster}
                  className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                    selectedCluster === cluster
                      ? "border-blue-500 bg-blue-50"
                      : "border-gray-200 hover:bg-gray-50"
                  }`}
                >
                  <input
                    type="radio"
                    name="jobCluster"
                    value={cluster}
                    checked={selectedCluster === cluster}
                    onChange={() => setJobCluster(cluster)}
                    className="mt-1 accent-blue-600"
                  />
                  <div>
                    <div className="font-medium text-slate-800 text-sm">
                      {cluster}
                    </div>
                    <div className="text-xs text-gray-500">
                      {JOB_CLUSTER_DESCRIPTIONS[cluster]}
                    </div>
                  </div>
                </label>
              ))}
            </div>
            {/* Right column */}
            <div className="space-y-2">
              {rightColumn.map((cluster) => (
                <label
                  key={cluster}
                  className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                    selectedCluster === cluster
                      ? "border-blue-500 bg-blue-50"
                      : "border-gray-200 hover:bg-gray-50"
                  }`}
                >
                  <input
                    type="radio"
                    name="jobCluster"
                    value={cluster}
                    checked={selectedCluster === cluster}
                    onChange={() => setJobCluster(cluster)}
                    className="mt-1 accent-blue-600"
                  />
                  <div>
                    <div className="font-medium text-slate-800 text-sm">
                      {cluster}
                    </div>
                    <div className="text-xs text-gray-500">
                      {JOB_CLUSTER_DESCRIPTIONS[cluster]}
                    </div>
                  </div>
                </label>
              ))}
              {/* Other option — screen-out */}
              <label
                className={`flex items-start gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                  selectedCluster === "Other"
                    ? "border-orange-500 bg-orange-50"
                    : "border-gray-200 hover:bg-gray-50"
                }`}
              >
                <input
                  type="radio"
                  name="jobCluster"
                  value="Other"
                  checked={selectedCluster === "Other"}
                  onChange={() => { setJobCluster("Other"); setJobNatureDescription(""); }}
                  className="mt-1 accent-orange-600"
                />
                <div>
                  <div className="font-medium text-slate-800 text-sm">
                    Other (none of the above)
                  </div>
                  <div className="text-xs text-gray-500">
                    My job does not fit any of the categories listed
                  </div>
                </div>
              </label>
            </div>
          </div>

          {/* Job nature text field — shown when a valid cluster is selected */}
          {selectedCluster && selectedCluster !== "Other" && (
            <div className="mb-6">
              <label className="block text-sm font-medium text-slate-700 mb-1">
                Please describe your specific job role / daily responsibilities:
              </label>
              <textarea
                value={jobNatureDescription}
                onChange={(e) => setJobNatureDescription(e.target.value)}
                placeholder="e.g. I am a payroll accountant processing invoices and managing vendor payments..."
                rows={3}
                className="w-full border border-gray-300 rounded-md p-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none"
              />
            </div>
          )}

          {/* Screen-out message for "Other" */}
          {selectedCluster === "Other" && (
            <div className="mb-6 p-4 bg-orange-50 border border-orange-300 rounded-lg text-sm text-orange-900">
              <p className="font-semibold mb-2">Thank you for your interest!</p>
              <p>
                Unfortunately, this study requires participants whose job falls within one of
                the listed categories. If your role does not match any category, you are not
                eligible to participate in this study.
              </p>
              <p className="mt-2">
                Please return your submission on Prolific. We appreciate your time!
              </p>
            </div>
          )}

          <button
            onClick={handleClusterSubmit}
            disabled={!selectedCluster || selectedCluster === "Other" || !jobNatureDescription.trim()}
            className={`w-full py-2 px-4 rounded-md font-semibold flex items-center justify-center gap-2 transition-colors ${
              selectedCluster && selectedCluster !== "Other" && jobNatureDescription.trim()
                ? "bg-slate-800 text-white hover:bg-slate-700"
                : "bg-gray-300 text-gray-500 cursor-not-allowed"
            }`}
          >
            Continue
            <ArrowRight size={18} />
          </button>

          <button
            onClick={handleGoToScenarioSelector}
            className="w-full mt-4 text-slate-500 hover:text-slate-700 text-sm"
          >
            &larr; Back to Scenario Selector
          </button>
        </div>
      </div>
    );
  }

  // =========================================================================
  // RENDER: COMPREHENSION CHECK (Phishing)
  // =========================================================================

  if (studyStage === "comprehensionCheck") {
    const ComprehensionCheck = () => {
      const [questions, setQuestions] = React.useState([]);
      const [answers, setAnswers] = React.useState({});
      const [attempts, setAttempts] = React.useState(0);
      const [feedback, setFeedback] = React.useState(null);
      const [failed, setFailed] = React.useState(false);
      const [loadingQuestions, setLoadingQuestions] = React.useState(true);
      const maxAttempts = 3;
      const currentParticipantId = participantId || getParticipantId();

      React.useEffect(() => {
        const fetchQuestions = async () => {
          try {
            const response = await axios.get(
              `${API_URL}/comprehension-check/${currentParticipantId}`,
            );
            setQuestions(response.data.questions || []);
          } catch (error) {
            console.error("Error fetching comprehension questions:", error);
            // If endpoint doesn't exist, skip to instructions
            updateStudyStage("instructions");
          } finally {
            setLoadingQuestions(false);
          }
        };
        fetchQuestions();
      }, []);

      const handleSubmitAnswers = async () => {
        try {
          const response = await axios.post(
            `${API_URL}/comprehension-check/${currentParticipantId}`,
            { answers },
          );

          if (response.data.passed) {
            updateStudyStage("instructions");
          } else {
            const newAttempts = attempts + 1;
            setAttempts(newAttempts);
            setFeedback(response.data);

            if (newAttempts >= maxAttempts) {
              setFailed(true);
            }
          }
        } catch (error) {
          console.error("Error submitting comprehension check:", error);
          // If endpoint fails, proceed anyway
          updateStudyStage("instructions");
        }
      };

      const allAnswered =
        questions.length > 0 &&
        Object.keys(answers).length === questions.length;

      if (loadingQuestions) {
        return (
          <div className="flex h-screen items-center justify-center bg-gray-100">
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
              <div className="text-blue-600 font-semibold">
                Loading comprehension check...
              </div>
            </div>
          </div>
        );
      }

      if (failed) {
        return (
          <div className="flex h-screen items-center justify-center bg-gray-100 font-sans">
            <div className="bg-white p-8 rounded-lg shadow-md max-w-md w-full text-center">
              <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <span className="text-red-600 text-2xl font-bold">X</span>
              </div>
              <h1 className="text-2xl font-bold mb-4 text-red-700">
                Unable to Continue
              </h1>
              <p className="text-gray-600 mb-4">
                You have used all {maxAttempts} attempts for the comprehension
                check. Unfortunately, you cannot proceed with the experiment.
              </p>
              <p className="text-sm text-gray-500">
                Please return to Prolific and indicate that you were unable to
                complete the study.
              </p>
            </div>
          </div>
        );
      }

      return (
        <div className="flex h-screen items-center justify-center bg-gray-100 font-sans">
          <div className="bg-white p-8 rounded-lg shadow-md max-w-lg w-full">
            <h1 className="text-2xl font-bold mb-2 text-center text-slate-800">
              Comprehension Check
            </h1>
            <p className="text-gray-600 mb-6 text-center text-sm">
              Please answer the following questions to confirm you understand the
              task. Attempt {attempts + 1} of {maxAttempts}.
            </p>

            {feedback && !feedback.passed && (
              <div className="mb-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg text-sm text-yellow-800">
                Some answers were incorrect. Please review and try again.
                {feedback.explanations &&
                  Object.entries(feedback.explanations).map(([qId, explanation]) => (
                    <div key={qId} className="mt-2 text-xs">
                      {explanation}
                    </div>
                  ))}
              </div>
            )}

            <div className="space-y-6 mb-6">
              {questions.map((q, qIndex) => (
                <div key={q.id || qIndex}>
                  <p className="font-medium text-slate-800 text-sm mb-2">
                    {qIndex + 1}. {q.question}
                  </p>
                  <div className="space-y-1">
                    {q.options.map((option, oIndex) => (
                      <label
                        key={oIndex}
                        className={`flex items-center gap-2 p-2 rounded border cursor-pointer transition-colors text-sm ${
                          answers[q.id || qIndex] === option
                            ? "border-blue-500 bg-blue-50"
                            : "border-gray-200 hover:bg-gray-50"
                        }`}
                      >
                        <input
                          type="radio"
                          name={`q-${q.id || qIndex}`}
                          value={option}
                          checked={answers[q.id || qIndex] === option}
                          onChange={() =>
                            setAnswers((prev) => ({
                              ...prev,
                              [q.id || qIndex]: option,
                            }))
                          }
                          className="accent-blue-600"
                        />
                        {option}
                      </label>
                    ))}
                  </div>
                </div>
              ))}
            </div>

            <button
              onClick={handleSubmitAnswers}
              disabled={!allAnswered}
              className={`w-full py-2 px-4 rounded-md font-semibold flex items-center justify-center gap-2 transition-colors ${
                allAnswered
                  ? "bg-slate-800 text-white hover:bg-slate-700"
                  : "bg-gray-300 text-gray-500 cursor-not-allowed"
              }`}
            >
              Submit Answers
              <ArrowRight size={18} />
            </button>
          </div>
        </div>
      );
    };

    return <ComprehensionCheck />;
  }

  // =========================================================================
  // RENDER: INSTRUCTIONS PAGE (shows welcome email content as a dedicated page)
  // =========================================================================

  if (studyStage === "instructions") {
    const InstructionsPage = () => {
      const [emailContent, setEmailContent] = React.useState(instructionEmail);
      const [loadingInstructions, setLoadingInstructions] = React.useState(!instructionEmail);
      const [starting, setStarting] = React.useState(false);
      const currentParticipantId = participantId || getParticipantId();

      React.useEffect(() => {
        if (emailContent) return;
        const fetchWelcomeEmail = async () => {
          try {
            const response = await axios.get(
              `${API_URL}/emails/inbox/${currentParticipantId}`,
            );
            const emails = response.data.emails || response.data || [];
            const welcome = emails.find((e) => e.order_id === 0);
            if (welcome) {
              setEmailContent(welcome);
              setInstructionEmail(welcome);
            }
          } catch (error) {
            console.error("Error fetching instruction email:", error);
          } finally {
            setLoadingInstructions(false);
          }
        };
        fetchWelcomeEmail();
      }, []);

      const handleStartStudy = async () => {
        setStarting(true);
        try {
          // Complete the welcome email so real emails load next
          await axios.post(`${API_URL}/complete/${currentParticipantId}`);
        } catch (error) {
          console.error("Error completing welcome email:", error);
        }
        updateStudyStage("emailExperiment");
      };

      if (loadingInstructions) {
        return (
          <div className="flex h-screen items-center justify-center bg-gray-100">
            <div className="flex flex-col items-center">
              <div className="w-12 h-12 border-4 border-blue-600 border-t-transparent rounded-full animate-spin mb-4"></div>
              <div className="text-blue-600 font-semibold">Loading instructions...</div>
            </div>
          </div>
        );
      }

      return (
        <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 font-sans py-8 px-4">
          <div className="max-w-3xl mx-auto">
            {/* Header card */}
            <div className="bg-white rounded-t-xl shadow-sm border border-gray-200 overflow-hidden">
              {/* Blue header bar */}
              <div className="bg-[#0078d4] px-6 py-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-white/20 flex items-center justify-center">
                    <Mail size={20} className="text-white" />
                  </div>
                  <div>
                    <h1 className="text-xl font-bold text-white">Study Instructions</h1>
                    <p className="text-blue-100 text-sm">Please read carefully before starting</p>
                  </div>
                </div>
              </div>

              {/* Email-style header */}
              {emailContent && (
                <div className="px-6 py-4 border-b border-gray-200 bg-gray-50">
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-full bg-[#0078d4] flex items-center justify-center text-white font-bold text-lg flex-shrink-0">
                      I
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold text-[#252423] text-base">
                        {emailContent.sender_name}
                      </div>
                      <div className="text-xs text-[#605e5c] mt-0.5">
                        {emailContent.sender_email}
                      </div>
                      <div className="text-sm text-[#252423] font-medium mt-2">
                        {emailContent.subject}
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>

            {/* Email body content */}
            <div className="bg-white border-x border-gray-200 shadow-sm">
              <div className="px-6 py-6">
                {emailContent ? (
                  <div
                    className="prose max-w-none text-[#252423] leading-relaxed"
                    dangerouslySetInnerHTML={{ __html: emailContent.body }}
                  />
                ) : (
                  <div className="text-center text-gray-500 py-12">
                    <p>Instructions could not be loaded. You may proceed to the study.</p>
                  </div>
                )}
              </div>
            </div>

            {/* Start Study button */}
            <div className="bg-white rounded-b-xl shadow-sm border border-gray-200 border-t-0 px-6 py-6">
              <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
                <p className="text-sm text-blue-800 font-medium">
                  Once you click "Start Study" below, you will enter the email client and begin
                  reviewing emails. Make sure you have read and understood all the instructions above.
                </p>
              </div>
              <button
                onClick={handleStartStudy}
                disabled={starting}
                className={`w-full py-3 px-4 rounded-lg font-semibold flex items-center justify-center gap-2 transition-colors text-lg ${
                  starting
                    ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                    : "bg-[#0078d4] text-white hover:bg-[#106ebe] shadow-md hover:shadow-lg"
                }`}
              >
                {starting ? "Starting..." : "Start Study"}
                {!starting && <ArrowRight size={20} />}
              </button>
            </div>
          </div>
        </div>
      );
    };

    return <InstructionsPage />;
  }

  // =========================================================================
  // RENDER: PROLIFIC ID PROMPT
  // =========================================================================

  if (studyStage === "prolificId") {
    const scenarioLabels = {
      phishing: {
        name: "Phishing Emails",
        color: "text-gray-800",
        bg: "bg-gray-50",
      },
      "dark-patterns": {
        name: "Dark Patterns",
        color: "text-gray-800",
        bg: "bg-gray-50",
      },
      "fake-news": {
        name: "Fake News",
        color: "text-orange-800",
        bg: "bg-gray-50",
      },
    };
    const currentScenario = scenario || "phishing";
    const scenarioInfo =
      scenarioLabels[currentScenario] || scenarioLabels["phishing"];

    return (
      <div className="flex h-screen items-center justify-center bg-gray-100 font-sans">
        <div className="bg-white p-8 rounded-lg shadow-md max-w-md w-full">
          {/* Scenario Badge */}
          <div className="flex justify-center mb-4">
            <span
              className={`${scenarioInfo.bg} ${scenarioInfo.color} px-3 py-1 rounded-full text-sm font-medium`}
            >
              {scenarioInfo.name} Study
            </span>
          </div>

          <h1 className="text-2xl font-bold mb-6 text-center text-slate-800">
            Welcome to the Research Study
          </h1>
          <p className="text-gray-600 mb-6 text-center">
            Thank you for participating in this research study.
          </p>
          <form onSubmit={handleProlificSubmit} className="space-y-4">
            <div>
              <label
                htmlFor="prolificId"
                className="block text-sm font-medium text-gray-700 mb-1"
              >
                Please enter your Prolific ID to begin...
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
              className="w-full bg-slate-800 text-white py-2 px-4 rounded-md hover:bg-slate-700 transition-colors font-semibold flex items-center justify-center gap-2"
            >
              Continue
              <ArrowRight size={18} />
            </button>
          </form>

          {/* Back to scenario selection */}
          <button
            onClick={() => {
              localStorage.removeItem("scenario");
              setScenario(null);
              updateStudyStage("scenarioSelection");
            }}
            className="w-full mt-4 text-slate-500 hover:text-slate-700 text-sm"
          >
            ← Choose a different study
          </button>
        </div>
      </div>
    );
  }

  // =========================================================================
  // RENDER: POST-EXPERIMENT SURVEY
  // =========================================================================

  if (studyStage === "postExperiment") {
    return (
      <>
        <PostExperimentSurvey
          onComplete={handlePostSurveyComplete}
          participantId={participantId}
          bonusTotal={bonusTotal}
          onGoToEmail={() => updateStudyStage("emailExperiment")}
        />
        <div className="fixed bottom-0 right-0 px-4 py-2 z-50">
          <button
            onClick={handleGoToScenarioSelector}
            className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium"
          >
            Go to Scenario Selector →
          </button>
        </div>
      </>
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
          <button
            onClick={handleGoToScenarioSelector}
            className="mt-4 text-slate-500 hover:text-slate-700 text-sm underline"
          >
            ← Back to Scenario Selector
          </button>
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
          onClick={handleGoToScenarioSelector}
          className="px-4 py-2 text-black underline rounded hover:bg-gray-200 hover:rounded-lg text-sm font-medium pointer-events-auto"
        >
          Go to Scenario Selector
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
