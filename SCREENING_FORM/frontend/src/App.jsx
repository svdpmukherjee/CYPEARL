import { useState, useRef, useEffect } from "react";
import sampleEmailImg from "./assets/sample_email.jpg";
import {
  ArrowRight,
  ArrowLeft,
  CheckCircle,
  Briefcase,
  Mail,
  Plus,
  Inbox,
  User,
  ShieldAlert,
  Trash2,
  Monitor,
  Gift,
  Copy,
} from "lucide-react";

import {
  FREQUENCY_OPTIONS,
  JOB_CLUSTERS,
  MANDATORY_EMAILS,
  REQUIRED_GENERIC,
  REQUIRED_SUSPICIOUS,
  BONUS_PER_EMAIL_PENCE,
  MAX_BONUS_EMAILS,
} from "./constants";
import { isEmailFilled, isSenderComplete, formatBonus } from "./lib/helpers";
import { registerUser, loadDraft, submitFinal } from "./lib/api";
import { loadLocal, clearLocal } from "./lib/persistence";
import useDraftSync from "./hooks/useDraftSync";
import EmailCard from "./components/EmailCard";
import RewardPanel from "./components/RewardPanel";
import CompletedPartSummary from "./components/CompletedPartSummary";

// ═════════════════════════════════════════════════════════════════════════
// MAIN APP
// ═════════════════════════════════════════════════════════════════════════

export default function App() {
  const COMPLETION_CODE = "C1X95X6D";
  const [step, setStep] = useState(0);
  const [landingView, setLandingView] = useState("overview");
  const [consentChecked, setConsentChecked] = useState(false);

  // Attention check (between landing details and Step 1)
  const [attentionPrivacy, setAttentionPrivacy] = useState(""); // "right" | "wrong"
  const [attentionBonusIdx, setAttentionBonusIdx] = useState(-1); // selected option index
  const [attentionSubmitted, setAttentionSubmitted] = useState(false);
  const [partAView, setPartAView] = useState("intro");
  const [partBView, setPartBView] = useState("intro");
  const [partCView, setPartCView] = useState("intro");
  const [senderPageIdx, setSenderPageIdx] = useState(0);
  const [genericPageIdx, setGenericPageIdx] = useState(0);
  const [suspiciousPageIdx, setSuspiciousPageIdx] = useState(0);
  const [showValidation, setShowValidation] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [codeCopied, setCodeCopied] = useState(false);
  const [error, setError] = useState("");

  // Step 0
  const [prolificId, setProlificId] = useState("");

  // Step 1
  const [jobCluster, setJobCluster] = useState("");
  const [jobTitle, setJobTitle] = useState("");
  const [dailyTasks, setDailyTasks] = useState("");

  // Step 2 — sender pages (start with 5 mandatory)
  const EMPTY_SENDER = () => ({
    role: "",
    type: "",
    emails: [{ subject: "", content: "", frequency: "" }],
  });

  const [senders, setSenders] = useState(
    Array.from({ length: MANDATORY_EMAILS }, () => EMPTY_SENDER()),
  );

  // Step 3 — generic (non-job-specific) workplace emails
  const EMPTY_GENERIC = () => ({ sender: "", description: "" });
  const [genericEmails, setGenericEmails] = useState([EMPTY_GENERIC()]);

  // Step 4 — hard-to-judge emails
  const EMPTY_SUSPICIOUS = () => ({ description: "" });
  const [suspiciousEmails, setSuspiciousEmails] = useState([
    EMPTY_SUSPICIOUS(),
  ]);

  // Ref for the most recently added email card — used to scroll it into view
  // after the user clicks "Add another email type…" so the subject field is
  // immediately focused, without them having to scroll manually.
  const lastEmailCardRef = useRef(null);
  const [shouldScrollToNewEmail, setShouldScrollToNewEmail] = useState(false);

  useEffect(() => {
    if (shouldScrollToNewEmail && lastEmailCardRef.current) {
      lastEmailCardRef.current.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
      const subjectInput =
        lastEmailCardRef.current.querySelector("input[type='text']");
      if (subjectInput) subjectInput.focus({ preventScroll: true });
      setShouldScrollToNewEmail(false);
    }
  }, [shouldScrollToNewEmail]);

  // ── Persistence: restore on mount ────────────────────────────────────
  // localStorage is the fast path; if Prolific ID is present we also try the
  // server draft and prefer whichever has more recent content. This survives
  // accidental refreshes and (via server draft) browser/device changes.
  const [restored, setRestored] = useState(false);
  useEffect(() => {
    const local = loadLocal();
    if (local) applyDraft(local);

    const candidatePid = local?.prolificId?.trim();
    if (candidatePid) {
      loadDraft(candidatePid)
        .then((server) => {
          if (server?.draft) applyDraft(server.draft);
        })
        .catch(() => {})
        .finally(() => setRestored(true));
    } else {
      setRestored(true);
    }
  }, []);

  function applyDraft(d) {
    if (!d || typeof d !== "object") return;
    if (typeof d.step === "number") setStep(d.step);
    if (typeof d.landingView === "string") setLandingView(d.landingView);
    if (typeof d.consentChecked === "boolean")
      setConsentChecked(d.consentChecked);
    if (typeof d.prolificId === "string") setProlificId(d.prolificId);
    if (typeof d.jobCluster === "string") setJobCluster(d.jobCluster);
    if (typeof d.jobTitle === "string") setJobTitle(d.jobTitle);
    if (typeof d.dailyTasks === "string") setDailyTasks(d.dailyTasks);
    if (Array.isArray(d.senders) && d.senders.length > 0) setSenders(d.senders);
    if (Array.isArray(d.genericEmails) && d.genericEmails.length > 0)
      setGenericEmails(d.genericEmails);
    if (Array.isArray(d.suspiciousEmails) && d.suspiciousEmails.length > 0)
      setSuspiciousEmails(d.suspiciousEmails);
    if (typeof d.partAView === "string") setPartAView(d.partAView);
    if (typeof d.partBView === "string") setPartBView(d.partBView);
    if (typeof d.partCView === "string") setPartCView(d.partCView);
    if (typeof d.senderPageIdx === "number") setSenderPageIdx(d.senderPageIdx);
    if (typeof d.genericPageIdx === "number")
      setGenericPageIdx(d.genericPageIdx);
    if (typeof d.suspiciousPageIdx === "number")
      setSuspiciousPageIdx(d.suspiciousPageIdx);
  }

  // Register the participant on the backend when they first consent + give an
  // ID. The backend rejects (409) if they have already submitted; we surface
  // that as a top-level error.
  const registeredRef = useRef(false);
  useEffect(() => {
    if (!restored) return;
    const pid = prolificId.trim();
    if (!pid || !consentChecked || registeredRef.current) return;
    registeredRef.current = true;
    registerUser(pid).catch((err) => {
      if (err.response?.status === 409) {
        setError(
          "A response with this Prolific ID has already been submitted.",
        );
      }
    });
  }, [restored, prolificId, consentChecked]);

  // Bundle form state for persistence.
  const draft = {
    step,
    landingView,
    consentChecked,
    prolificId,
    jobCluster,
    jobTitle,
    dailyTasks,
    senders,
    genericEmails,
    suspiciousEmails,
    partAView,
    partBView,
    partCView,
    senderPageIdx,
    genericPageIdx,
    suspiciousPageIdx,
  };
  useDraftSync({
    prolificId: restored ? prolificId.trim() : "",
    draft,
  });

  // ── Computed counts ───────────────────────────────────────────────────
  // Derive from the actual data so that removing an email via the X button
  // immediately decreases the count (and the bonus). Using live counters
  // that only increment on commit would overpay users who delete entries.
  const totalEmailCount = senders.reduce(
    (sum, s) => sum + s.emails.filter(isEmailFilled).length,
    0,
  );
  const committedGeneric = genericEmails.filter(
    (g) => g.sender.trim() && g.description.trim(),
  ).length;
  const committedSuspicious = suspiciousEmails.filter((s) =>
    s.description.trim(),
  ).length;

  const bonusEmails = Math.max(0, totalEmailCount - MANDATORY_EMAILS);
  const maxBonusPence = Math.round(MAX_BONUS_EMAILS * 100);
  const bonusPenceRaw = bonusEmails * BONUS_PER_EMAIL_PENCE;
  const bonusPence = Math.min(bonusPenceRaw, maxBonusPence);
  const bonusMaxReached = bonusPenceRaw >= maxBonusPence;

  const handleCopyCompletionCode = async () => {
    try {
      await navigator.clipboard.writeText(COMPLETION_CODE);
      clearLocal();
      setCodeCopied(true);
      window.setTimeout(() => setCodeCopied(false), 2000);
    } catch (err) {
      setCodeCopied(false);
    }
  };

  // ── Current sender helpers ────────────────────────────────────────────

  const currentSender = senders[senderPageIdx] || EMPTY_SENDER();
  const isLastSenderPage = senderPageIdx === senders.length - 1;

  const updateCurrentSender = (field, value) => {
    const updated = [...senders];
    updated[senderPageIdx] = { ...updated[senderPageIdx], [field]: value };
    setSenders(updated);
  };

  // Validate the most recently filled-in email on the current sender page.
  // Used before committing an additional email or moving to the next sender.
  const isLastEmailValid = (sender) => {
    const last = sender.emails[sender.emails.length - 1];
    return (
      sender.role.trim() &&
      sender.type &&
      last &&
      last.subject.trim() &&
      last.content.trim() &&
      last.frequency
    );
  };

  const addEmailToCurrentSender = () => {
    if (!isLastEmailValid(currentSender)) {
      setShowValidation(true);
      return;
    }
    setShowValidation(false);
    const updated = [...senders];
    updated[senderPageIdx] = {
      ...updated[senderPageIdx],
      emails: [
        ...updated[senderPageIdx].emails,
        { subject: "", content: "", frequency: "" },
      ],
    };
    setSenders(updated);
    setShouldScrollToNewEmail(true);
  };

  const updateEmail = (emailIdx, updatedEmail) => {
    const updated = [...senders];
    const emails = updated[senderPageIdx].emails.map((e, i) =>
      i === emailIdx ? updatedEmail : e,
    );
    updated[senderPageIdx] = { ...updated[senderPageIdx], emails };
    setSenders(updated);
  };

  const removeEmail = (emailIdx) => {
    if (currentSender.emails.length <= 1) return;
    const updated = [...senders];
    updated[senderPageIdx] = {
      ...updated[senderPageIdx],
      emails: updated[senderPageIdx].emails.filter((_, i) => i !== emailIdx),
    };
    setSenders(updated);
  };

  const addNewSenderPage = () => {
    if (!isLastEmailValid(currentSender)) {
      setShowValidation(true);
      return;
    }
    setShowValidation(false);
    const newSenders = [...senders, EMPTY_SENDER()];
    setSenders(newSenders);
    setSenderPageIdx(newSenders.length - 1);
  };

  // ── Step 2 navigation ────────────────────────────────────────────────

  // To leave Part A, the user must have described at least MANDATORY_EMAILS
  // complete emails. Sender count is tracked for analysis and used for a soft
  // nudge below, but is no longer a hard gate — some participants legitimately
  // receive work email from only 1–2 distinct senders.
  const uniqueCompletedSenderRoles = new Set(
    senders.filter(isSenderComplete).map((s) => s.role.trim().toLowerCase()),
  );
  const completedSendersCount = uniqueCompletedSenderRoles.size;
  const canLeaveStep2 = totalEmailCount >= MANDATORY_EMAILS;
  const showLowSenderDiversityNudge =
    canLeaveStep2 && completedSendersCount < 3;

  // ── Page-level validation (all fields required) ───────────────────────

  const getPageErrors = (sender) => {
    const errors = [];
    if (!sender.role.trim()) errors.push("Job role or title of the sender");
    if (!sender.type) errors.push("Inside or outside your organisation");
    sender.emails.forEach((email, i) => {
      const emailLabel =
        sender.emails.length === 1
          ? `Email ${senderPageIdx + 1}`
          : `Email ${senderPageIdx + 1}.${i + 1}`;
      const tag = sender.emails.length > 1 ? ` (${emailLabel})` : "";
      if (!email.subject.trim()) errors.push(`Subject line${tag}`);
      if (!email.content.trim()) errors.push(`What the email is about${tag}`);
      if (!email.frequency) errors.push(`Email frequency${tag}`);
    });
    return errors;
  };

  const tryNavigateForward = () => {
    const errors = getPageErrors(currentSender);
    if (errors.length > 0) {
      setShowValidation(true);
      return;
    }
    setShowValidation(false);
    if (isLastSenderPage) {
      if (canLeaveStep2) setStep(3);
    } else {
      setSenderPageIdx(senderPageIdx + 1);
    }
  };

  // ── Validation ────────────────────────────────────────────────────────

  const canAdvance = () => {
    if (step === 0) {
      if (landingView === "overview")
        return consentChecked && prolificId.trim().length > 0;
      return true;
    }
    if (step === 1) return jobCluster && jobTitle.trim() && dailyTasks.trim();
    if (step === 3) return committedGeneric >= REQUIRED_GENERIC;
    if (step === 4) return committedSuspicious >= REQUIRED_SUSPICIOUS;
    return true;
  };

  // ── Submit ────────────────────────────────────────────────────────────

  const handleSubmit = async () => {
    setSubmitting(true);
    setError("");
    try {
      const cleanedSenders = senders.filter(isSenderComplete).map((s) => ({
        role: s.role.trim(),
        type: s.type,
        emails: s.emails.filter(isEmailFilled).map((e) => ({
          subject: e.subject.trim(),
          content: e.content.trim(),
          frequency: e.frequency || "",
        })),
      }));

      const cleanedGeneric = genericEmails
        .filter((g) => g.sender.trim() && g.description.trim())
        .map((g) => ({
          sender: g.sender.trim(),
          description: g.description.trim(),
        }));

      await submitFinal({
        prolific_id: prolificId.trim(),
        job_cluster: jobCluster,
        job_title: jobTitle.trim(),
        daily_tasks: dailyTasks.trim(),
        email_senders: cleanedSenders,
        generic_emails: cleanedGeneric,
        suspicious_emails: suspiciousEmails
          .map((s) => s.description.trim())
          .filter((s) => s.length > 0),
      });
      clearLocal();
      setStep(5);
    } catch (err) {
      if (err.response?.status === 409) {
        setError(
          "A response with this Prolific ID has already been submitted.",
        );
      } else {
        setError("Something went wrong. Please try again.");
      }
    } finally {
      setSubmitting(false);
    }
  };

  // Two-column layout for job clusters
  const leftClusters = JOB_CLUSTERS.filter((_, i) => i % 2 === 0);
  const rightClusters = JOB_CLUSTERS.filter((_, i) => i % 2 === 1);

  // ═══════════════════════════════════════════════════════════════════
  // STEP 0 — Landing / Prolific ID
  // ═══════════════════════════════════════════════════════════════════

  const renderStep0 = () => {
    if (landingView === "overview") {
      return (
        <div>
          <div className="flex items-center gap-3 mb-2">
            {/* <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center">
              <ClipboardList size={20} className="text-white" />
            </div> */}
            {/* <h2 className="text-xl font-bold text-slate-800">
              Welcome to Workplace Email Patterns Survey
            </h2> */}
          </div>

          <div className="bg-amber-50 border-2 border-amber-400 rounded-lg p-4 mb-5 flex items-start gap-3 shadow-sm">
            <Monitor className="w-6 h-6 text-amber-700 flex-shrink-0 mt-0.5" />
            <div className="text-sm text-amber-900 leading-relaxed">
              <p className="font-bold uppercase tracking-wide mb-1">
                Please use a laptop or desktop computer
              </p>
              <p>
                This survey is designed for a wider screen and is{" "}
                <strong>not supported on mobile phones or tablets</strong>.
                <br />
                Completing this survey on a mobile device or tablet may lead to{" "}
                <span className="font-semibold">
                  cancellation of your submission and loss of compensation.
                </span>
              </p>
            </div>
          </div>

          <p className=" text-sm mb-10 leading-relaxed">
            {/* Thank you for participating in this research study.
            <br /> */}
            We are creating a collection of{" "}
            <strong>realistic workplace email types</strong> for research on{" "}
            <span className="font-semibold">how people judge </span>
            whether an email seems legitimate or trustworthy.
            <br />
            Your task is to describe the{" "}
            <span className="font-semibold">types</span> of emails you usually
            receive at work so we can make these email examples feel as
            realistic as possible.
          </p>

          <div className="rounded-lg p-4 mb-10 shadow-[0_0_24px_rgba(255,0.2,0,0.2)]">
            <div className="inline-flex items-center px-4 py-2 mb-4 rounded-xl border-2 border-violet-300 bg-violet-100 shadow-[4px_4px_0_rgba(124,58,237,0.35)]">
              <p className="text-sm font-semibold text-violet-900">
                This survey has 3 parts:
              </p>
            </div>

            <div className="space-y-6">
              <p className="text-xs text-slate-500 italic -mt-2 mb-1">
                The examples below are for a school teacher named Claudia.
              </p>

              <div className="mt-4">
                <p className="text-sm text-black font-semibold mt-1">
                  Part A: Job-Specific Emails
                </p>
                <p className="text-sm text-black-700 ml-4 mt-2">
                  List <strong>at least five</strong> types of emails that you
                  typically receive that are{" "}
                  <span className="font-semibold">
                    directly related to your job role.
                  </span>
                </p>
                <p className="text-xs text-slate-500 italic ml-4 mt-1">
                  e.g., Claudia receives emails from parents asking about their
                  children’s progress, or from the principal about the next
                  term’s curriculum.
                </p>
              </div>

              <div className="mt-4">
                <p className="text-sm text-black font-semibold">
                  Part B: General Workplace Emails
                </p>
                <p className="text-sm text-black-700 ml-4 mt-2">
                  List <strong>at least three</strong> types of emails that you
                  typically receive but that are{" "}
                  <span className="font-semibold">
                    not specific to your job role.
                  </span>
                </p>
                <p className="text-xs text-slate-500 italic ml-4 mt-1">
                  e.g., Claudia receives a staff-wide notice about a fire drill
                  in her school.
                </p>
              </div>

              <div className="mt-4">
                <p className="text-sm text-black font-semibold">
                  Part C: Suspicious / Hard-to-Judge Emails
                </p>
                <p className="text-sm text-black-700 ml-4 mt-2">
                  List <strong>at least five</strong> types of work emails where
                  you are unsure whether they are legitimate or a scam.
                </p>
                <p className="text-xs text-slate-500 italic ml-4 mt-1">
                  e.g., Claudia receives a message claiming to be from the
                  Ministry of Education asking her to verify her login details.
                </p>
              </div>
            </div>

            {/* <div className="mt-4 pt-3 border-t border-blue-200">
              <p className="text-sm text-blue-700 mb-2">
                For each email, think of it as a <strong>skeleton</strong>:
              </p>
              <ul className="list-disc list-inside text-sm text-blue-700 space-y-1 ml-2">
                <li>Who (not names, only job roles) sent it?</li>
                <li>What the subject line said?</li>
                <li>A one-line summary of what it was about.</li>
              </ul>
            </div> */}
          </div>

          <div className=" border-slate-200 rounded-lg p-4 mb-5">
            {/* <div className="inline-flex items-center px-4 py-2 mb-3 rounded-xl border-2 border-violet-300 bg-violet-100 shadow-[4px_4px_0_rgba(124,58,237,0.35)]"> */}
            <p className="text-md font-semibold text-black mb-4">
              Informed consent &amp; privacy
            </p>
            {/* </div> */}
            <ul className="text-sm leading-relaxed list-disc list-outside ml-5 space-y-1.5 mb-3">
              <li>
                Participation is <strong>voluntary</strong>. You may withdraw at
                any time by closing the tab — your responses will not be
                submitted.
              </li>
              <li>
                Your <strong>Prolific ID</strong> is collected only to pay you.
                After the study it is{" "}
                <strong>detached from your responses</strong>, so the data is
                made anonymous and used solely for academic research.
              </li>
              <li>
                You will type free text.{" "}
                <strong>
                  Do not include any personal names (yours, colleagues', or a
                  manager's) or company names
                </strong>{" "}
                — we only need job roles and general email patterns.
              </li>
            </ul>
            <label className="flex items-start gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={consentChecked}
                onChange={(e) => setConsentChecked(e.target.checked)}
                className="mt-0.5 cursor-pointer"
              />
              <span className="text-sm text-slate-800">
                I have read the information above and I <strong>consent</strong>{" "}
                to participate in this study.
              </span>
            </label>
          </div>

          <label className="block text-sm font-semibold text-slate-700 mb-2">
            Please enter your Prolific ID
          </label>
          <input
            type="text"
            value={prolificId}
            onChange={(e) => setProlificId(e.target.value)}
            placeholder="Please double-check your Prolific ID so we can verify your submission"
            className="w-full border border-gray-300 rounded-md p-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 mb-5"
          />

          <div className="flex justify-end">
            <button
              type="button"
              onClick={() => setLandingView("details")}
              disabled={!canAdvance()}
              className={`flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold transition-colors ${
                canAdvance()
                  ? "bg-slate-800 text-white hover:bg-slate-700 cursor-pointer"
                  : "bg-gray-200 text-gray-400 cursor-not-allowed"
              }`}
            >
              Next <ArrowRight size={16} />
            </button>
          </div>
        </div>
      );
    }

    if (landingView === "details") {
      return (
        <div>
          <p className="text-sm font-semibold text-slate-800 mb-4">
            Study Information
          </p>
          {/* Stats — flattened, no nested box */}
          <div className="bg-slate-100 rounded-lg p-4 mb-4 text-black shadow-sm">
            <div className="flex items-center justify-around">
              <div className="text-center">
                <p className="text-xs uppercase tracking-wider text-slate-800 mb-0.5">
                  Estimated total time
                </p>
                <p className="text-lg font-bold">~7 mins</p>
              </div>
              <div className="w-px h-8 bg-slate-500" />
              <div className="text-center">
                <p className="text-xs uppercase tracking-wider text-slate-800 mb-0.5">
                  Base compensation
                </p>
                <p className="text-lg font-bold">£1.00</p>
              </div>
              <div className="w-px h-8 bg-slate-500" />
              <div className="text-center">
                <p className="text-xs uppercase tracking-wider text-slate-800 mb-0.5">
                  Bonus
                </p>
                <p className="text-lg font-bold text-blue-700">
                  Max. {MAX_BONUS_EMAILS * 100} pence (
                  {formatBonus(BONUS_PER_EMAIL_PENCE * 10)})
                </p>
              </div>
            </div>
          </div>
          {/* Bonus details — separate, not nested */}
          <div className="bg-white border border-slate-200 rounded-lg p-3 mb-6 flex items-start gap-2 ">
            <ShieldAlert size={28} className="text-blue-800 mt-0.5 shrink-0" />{" "}
            <div className="text-sm text-slate-800 leading-7">
              <p className="font-bold">
                Make Sure You Understand the Bonus Criteria.
              </p>
              <ul className="mt-3 text-sm leading-relaxed list-disc list-outside ml-5 space-y-1.5">
                <li>
                  In Part A, you need to describe{" "}
                  <span className="font-bold">
                    more than {MANDATORY_EMAILS} emails
                  </span>{" "}
                  to be eligible for a bonus.
                </li>
                <li>
                  You can keep adding as many emails as you wish — each
                  additional email beyond the required {MANDATORY_EMAILS} earns
                  you{" "}
                  <strong className="text-blue-800">
                    {BONUS_PER_EMAIL_PENCE} pence
                  </strong>{" "}
                  via Prolific.
                </li>
                <li>
                  <span className="font-semibold">
                    However, there is a maximum of {MAX_BONUS_EMAILS * 100}{" "}
                    pence bonus.
                  </span>
                </li>
                <li>
                  If you typically receive emails from{" "}
                  <span className="font-bold">several different senders</span>
                  {""}, please try to cover{" "}
                  <span className="font-semibold">a few different senders</span>{" "}
                  (this helps us understand the variety in your inbox). If you
                  mostly hear from one or two people, that is fine too.
                </li>
                <li>
                  Parts B and C are required but{" "}
                  <span className="font-bold">do NOT earn bonus</span>.
                </li>
              </ul>
            </div>
          </div>
          <div className="flex items-center justify-between mt-8">
            <button
              type="button"
              onClick={() => setLandingView("overview")}
              className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-slate-800 transition-colors"
            >
              <ArrowLeft size={16} /> Back
            </button>

            <button
              type="button"
              onClick={() => {
                setAttentionSubmitted(false);
                setLandingView("attention");
              }}
              disabled={!canAdvance()}
              className={`flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold transition-colors cursor-pointer ${
                canAdvance()
                  ? "bg-slate-800 text-white hover:bg-slate-700"
                  : "bg-gray-200 text-gray-400 cursor-not-allowed"
              }`}
            >
              Next <ArrowRight size={16} />
            </button>
          </div>
        </div>
      );
    }

    // landingView === "attention" — quick comprehension check before Step 1
    const CORRECT_BONUS_IDX = 1;
    const privacyCorrect = attentionPrivacy === "right";
    const bonusCorrect = attentionBonusIdx === CORRECT_BONUS_IDX;
    const allCorrect = privacyCorrect && bonusCorrect;
    return (
      <div>
        <div className=" border-blue-200 rounded-lg p-4 mb-5">
          <p className="text-sm text-black font-semibold mb-1">Quick check</p>
          <p className="text-sm text-black">
            Before you start, please answer the two questions below to confirm
            you have understood the instructions.
          </p>
        </div>

        {/* Question 1 — privacy */}
        <div className="border border-slate-200 rounded-lg p-4 mb-4">
          <p className="text-sm font-semibold text-slate-800 mb-3">
            1. When describing emails in this study, I should:
          </p>
          <div className="space-y-2">
            {[
              {
                value: "wrong",
                label:
                  "Describe emails using personal names (mine, colleagues', or my company's).",
              },
              {
                value: "right",
                label:
                  "Describe the overall content using only job roles (e.g. CEO) — no personal or company names.",
              },
            ].map((opt) => (
              <label
                key={opt.value}
                className={`flex items-start gap-2 p-2.5 rounded-md border cursor-pointer transition-colors ${
                  attentionPrivacy === opt.value
                    ? attentionSubmitted
                      ? opt.value === "right"
                        ? "border-emerald-400 bg-emerald-50"
                        : "border-red-400 bg-red-50"
                      : "border-blue-400 bg-blue-50"
                    : "border-slate-200 hover:bg-slate-50"
                }`}
              >
                <input
                  type="radio"
                  name="attentionPrivacy"
                  checked={attentionPrivacy === opt.value}
                  onChange={() => setAttentionPrivacy(opt.value)}
                  className="mt-0.5 cursor-pointer"
                />
                <span className="text-sm text-slate-700">{opt.label}</span>
              </label>
            ))}
          </div>
          {attentionSubmitted && !privacyCorrect && (
            <p className="text-xs text-red-600 mt-2">
              Please go back and re-read the consent &amp; privacy notice on the
              first page.
            </p>
          )}
        </div>

        {/* Question 2 — bonus */}
        <div className="border border-slate-200 rounded-lg p-4 mb-5">
          <p className="text-sm font-semibold text-slate-800 mb-3">
            2. Which part of the survey earns a bonus per extra email?
          </p>
          <div className="space-y-2">
            {[
              "All three parts earn a bonus per extra email.",
              "Only Part A earns a bonus — extra emails beyond the required minimum (5 emails). Parts B and C are required but do not earn a bonus.",
              "Only Parts B and C earn a bonus.",
            ].map((label, i) => {
              const selected = attentionBonusIdx === i;
              const isCorrect = i === CORRECT_BONUS_IDX;
              const borderClass = selected
                ? attentionSubmitted
                  ? isCorrect
                    ? "border-emerald-400 bg-emerald-50"
                    : "border-red-400 bg-red-50"
                  : "border-blue-400 bg-blue-50"
                : "border-slate-200 hover:bg-slate-50";
              return (
                <label
                  key={i}
                  className={`flex items-start gap-2 p-2.5 rounded-md border cursor-pointer transition-colors ${borderClass}`}
                >
                  <input
                    type="radio"
                    name="attentionBonus"
                    checked={selected}
                    onChange={() => setAttentionBonusIdx(i)}
                    className="mt-0.5 cursor-pointer"
                  />
                  <span className="text-sm text-slate-700">{label}</span>
                </label>
              );
            })}
          </div>
          {attentionSubmitted && !bonusCorrect && (
            <p className="text-xs text-red-600 mt-2">
              Please go back and re-read the bonus details on the previous page.
            </p>
          )}
        </div>

        <div className="flex items-center justify-between">
          <button
            type="button"
            onClick={() => {
              setAttentionSubmitted(false);
              setLandingView("details");
            }}
            className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-slate-800 transition-colors"
          >
            <ArrowLeft size={16} /> Back
          </button>
          <button
            type="button"
            onClick={() => {
              if (allCorrect) {
                setAttentionSubmitted(false);
                setStep(1);
              } else {
                setAttentionSubmitted(true);
              }
            }}
            disabled={!attentionPrivacy || attentionBonusIdx < 0}
            className={`flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold transition-colors cursor-pointer ${
              !attentionPrivacy || attentionBonusIdx < 0
                ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                : "bg-slate-800 text-white hover:bg-slate-700"
            }`}
          >
            Next <ArrowRight size={16} />
          </button>
        </div>
      </div>
    );
  };

  // ═══════════════════════════════════════════════════════════════════
  // STEP 1 — Job role
  // ═══════════════════════════════════════════════════════════════════

  const renderStep1 = () => (
    <div>
      <div className="flex items-center gap-3 mb-2">
        <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center">
          <Briefcase size={20} className="text-white" />
        </div>
        <h2 className="text-xl font-bold text-slate-800">
          Your Professional Role
        </h2>
      </div>
      <p className="text-gray-600 text-sm mb-6">
        Select the category closest to your current job and tell us briefly what
        you do day-to-day.
      </p>

      <label className="block text-sm font-semibold text-slate-700 mb-2">
        Job Category
      </label>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-5">
        <div className="space-y-2">
          {leftClusters.map((c) => (
            <label
              key={c}
              className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                jobCluster === c
                  ? "border-blue-500 bg-blue-50"
                  : "border-gray-200 hover:bg-gray-50"
              }`}
            >
              <input
                type="radio"
                name="cluster"
                checked={jobCluster === c}
                onChange={() => setJobCluster(c)}
                className="accent-blue-600"
              />
              <span className="text-sm text-slate-700 font-medium">{c}</span>
            </label>
          ))}
        </div>
        <div className="space-y-2">
          {rightClusters.map((c) => (
            <label
              key={c}
              className={`flex items-center gap-3 p-3 rounded-lg border cursor-pointer transition-colors ${
                jobCluster === c
                  ? "border-blue-500 bg-blue-50"
                  : "border-gray-200 hover:bg-gray-50"
              }`}
            >
              <input
                type="radio"
                name="cluster"
                checked={jobCluster === c}
                onChange={() => setJobCluster(c)}
                className="accent-blue-600"
              />
              <span className="text-sm text-slate-700 font-medium">{c}</span>
            </label>
          ))}
        </div>
      </div>

      <label className="block text-sm font-semibold text-slate-700 mb-1">
        Your Job Title
      </label>
      <input
        type="text"
        value={jobTitle}
        onChange={(e) => setJobTitle(e.target.value)}
        placeholder="Type here..."
        className="w-full border border-gray-300 rounded-md p-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 mb-5"
      />

      <label className="block text-sm font-semibold text-slate-700 mb-1">
        Describe your main daily tasks in 1-2 sentences
      </label>
      <textarea
        value={dailyTasks}
        onChange={(e) => setDailyTasks(e.target.value)}
        placeholder="Type here..."
        rows={3}
        className="w-full border border-gray-300 rounded-md p-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 resize-none"
      />
    </div>
  );

  // ═══════════════════════════════════════════════════════════════════
  // STEP 2 — Email Landscape (paginated per sender)
  // ═══════════════════════════════════════════════════════════════════

  const renderStep2 = () => {
    if (partAView === "intro") {
      return (
        <div className="flex flex-col justify-center py-4">
          <div className="max-w-2xl mx-auto text-center">
            <div className="w-14 h-14 rounded-full bg-slate-800 flex items-center justify-center mx-auto mb-4">
              <Inbox size={26} className="text-white" />
            </div>

            <h2 className="text-2xl font-bold text-slate-800 mb-3">
              Part A: Job-Specific Email Types
            </h2>

            <p className="text-slate-600 text-sm leading-relaxed mb-5">
              In this part, you will describe work-related email types you
              receive one by one. We only need the structure of each email, not
              real content.
            </p>

            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mb-4">
              <p className="text-sm text-amber-800 text-left">
                <strong>What does each email look like?</strong>
                <br /> A brief description: who sends it (job role only), what
                the subject line looks like, and what the email is about.
              </p>
            </div>

            <div className="p-3 mb-5">
              <p className="text-xs text-slate-500 mb-2 uppercase tracking-wider font-semibold">
                Example
              </p>
              <img
                src={sampleEmailImg}
                alt="Example of a filled-in email type"
                className="w-7/8 rounded-lg  mx-auto"
              />
              <ol className="text-left text-sm text-slate-700 mt-3 space-y-1 list-decimal list-inside">
                <li>
                  <strong>Sender's job role</strong> — who typically sends this
                  email (no personal names).
                </li>
                <li>
                  <strong>Inside / outside</strong> your organisation.
                </li>
                <li>
                  <strong>Subject line</strong> — a short, realistic example.
                </li>
                <li>
                  <strong>About</strong> — a one-line summary of what the email
                  is for.
                </li>
                <li>
                  <strong>How often</strong> you receive this type of email.
                </li>
                <li>
                  <strong>Add another email</strong> — describe one more email
                  type from the same sender.
                </li>
                <li>
                  <strong>Move to next sender</strong> — start describing emails
                  from a different job role.
                </li>
              </ol>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-5 text-left">
              <p className="text-sm text-blue-900 font-semibold mb-2">
                Before you start, please note:
              </p>
              <ul className="list-disc list-inside text-sm text-blue-900 space-y-1.5">
                <li>
                  You need to describe{" "}
                  <strong>at least {MANDATORY_EMAILS} emails</strong> in total.
                </li>
                <li>
                  For each sender, you can add{" "}
                  <strong>as many emails as you want</strong>.
                </li>
                <li>
                  Where possible, try to describe emails from{" "}
                  <strong>different senders</strong> (e.g., CEO, external
                  clients).
                  {/* If you mainly hear from one or two people, that's
                  okay — just describe the types of emails you typically
                  receive. */}
                </li>

                <li>
                  Once you have described the required {MANDATORY_EMAILS}{" "}
                  emails, every <strong>additional email</strong> earns a bonus
                  — whether it's from a new sender or another type from a sender
                  you have already added.
                </li>
                <li>
                  Once you move forward, you{" "}
                  <strong>cannot go back to edit earlier senders</strong>, so
                  please complete each sender carefully before moving on.
                </li>
              </ul>
            </div>

            <div className="flex items-center justify-end">
              {/* <button
                type="button"
                onClick={() => {
                  setShowValidation(false);
                  setStep(1);
                }}
                className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-slate-800 transition-colors"
              >
                <ArrowLeft size={16} /> Back to Job Role
              </button> */}

              <button
                type="button"
                onClick={() => {
                  setShowValidation(false);
                  setPartAView("emails");
                }}
                className="flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold bg-slate-800 text-white hover:bg-slate-700 transition-colors cursor-pointer"
              >
                I am ready <ArrowRight size={16} />
              </button>
            </div>
          </div>
        </div>
      );
    }

    const sender = currentSender;
    const pageLabel = `Sender ${senderPageIdx + 1}`;

    const pageErrors = showValidation ? getPageErrors(sender) : [];
    const roleMissing = showValidation && !sender.role.trim();
    const typeMissing = showValidation && !sender.type;

    return (
      <div>
        {/* Validation error alert */}
        {showValidation && pageErrors.length > 0 && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
            <p className="text-sm font-semibold text-red-700 mb-1">
              Please complete the following before continuing:
            </p>
            <ul className="list-disc list-inside text-xs text-red-600 space-y-0.5">
              {pageErrors.map((err, i) => (
                <li key={i}>{err}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="mb-5">
          <div className="inline-flex items-center rounded-lg border border-slate-300 bg-slate-50 px-3 py-1.5 text-sm font-semibold text-slate-700">
            {pageLabel}
          </div>
        </div>

        {/* ── Sender info ─────────────────────────────────────────────── */}
        <div className="border border-slate-300 rounded-xl overflow-hidden mb-4">
          <div className=" p-4">
            <div className="space-y-2.5">
              {/* Sender role */}
              <div>
                <p
                  className={`text-sm font-medium mb-1 ${
                    roleMissing ? "text-red-500" : "text-black"
                  }`}
                >
                  What is the job role or title of someone who regularly emails
                  you at work?
                  {roleMissing && " *"}
                </p>
                <div className="relative">
                  <User
                    size={16}
                    className="absolute left-2.5 top-1/2 -translate-y-1/2 text-slate-400 pointer-events-none"
                  />
                  <input
                    type="text"
                    value={sender.role}
                    onChange={(e) =>
                      updateCurrentSender("role", e.target.value)
                    }
                    placeholder="Type in here..."
                    className={`w-full border rounded-md p-2 pl-9 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white ${
                      roleMissing ? "border-red-400" : "border-slate-300"
                    }`}
                  />
                </div>
              </div>

              {/* Internal / External */}
              <div>
                <p
                  className={`text-sm font-medium mb-1.5 mt-5 ${
                    typeMissing ? "text-red-500" : "text-black"
                  }`}
                >
                  Is this role inside or outside your organisation?
                  {typeMissing && " *"}
                </p>
                <div className="flex gap-2">
                  <button
                    type="button"
                    onClick={() => updateCurrentSender("type", "internal")}
                    className={`flex-1 py-2 rounded-lg text-xs font-semibold border-2 transition-all text-center ${
                      sender.type === "internal"
                        ? "bg-indigo-600 text-white border-indigo-600 shadow-sm"
                        : typeMissing
                          ? "bg-white text-red-400 border-red-300 hover:border-indigo-300 hover:text-indigo-600"
                          : "bg-white text-slate-500 border-slate-200 hover:border-indigo-300 hover:text-indigo-600"
                    }`}
                  >
                    Inside my organisation
                  </button>
                  <button
                    type="button"
                    onClick={() => updateCurrentSender("type", "external")}
                    className={`flex-1 py-2 rounded-lg text-xs font-semibold border-2 transition-all text-center ${
                      sender.type === "external"
                        ? "bg-indigo-600 text-white border-indigo-600 shadow-sm"
                        : typeMissing
                          ? "bg-white text-red-400 border-red-300 hover:border-indigo-300 hover:text-indigo-600"
                          : "bg-white text-slate-500 border-slate-200 hover:border-indigo-300 hover:text-indigo-600"
                    }`}
                  >
                    Outside my organisation
                  </button>
                </div>
              </div>
            </div>
          </div>

          {/* Email entries */}
          <div className="p-4 pt-3 space-y-3 ">
            <p className="text-sm font-medium text-black">
              What kinds of emails does{" "}
              <strong className="text-emerald-600">
                {`"${sender.role || "this role"}"`}
              </strong>{" "}
              typically send you?
            </p>
            {sender.emails.map((email, emailIdx) => {
              const isLast = emailIdx === sender.emails.length - 1;
              const hasMultiple = sender.emails.length > 1;
              return (
                <div
                  key={emailIdx}
                  ref={isLast ? lastEmailCardRef : null}
                  className="space-y-2 scroll-mt-4"
                >
                  {hasMultiple && (
                    <div className="flex items-center gap-1.5 text-xs font-semibold text-slate-600">
                      <Mail size={14} className="text-slate-500" />
                      <span>
                        Email {senderPageIdx + 1}.{emailIdx + 1}
                      </span>
                    </div>
                  )}
                  <EmailCard
                    email={email}
                    senderRole={sender.role}
                    senderType={sender.type}
                    onChange={(updated) => updateEmail(emailIdx, updated)}
                    onRemove={() => removeEmail(emailIdx)}
                    canRemove={sender.emails.length > 1}
                    showValidation={showValidation}
                  />
                </div>
              );
            })}
          </div>
        </div>

        {/* Show the "add another email" + "move to different sender" options
            only once the current email is fully filled (sender role/type set
            and the last email has subject, content, and frequency). */}
        {isLastEmailValid(sender) && (
          <div className="mt-8 pt-4 border-t border-dashed border-slate-200 space-y-3">
            {senderPageIdx === 0 && sender.emails.length === 1 && (
              <p className="text-xs text-slate-500 mb-2 text-center">
                Employees often receive <strong>2-3 different types</strong> of
                emails from the same sender
              </p>
            )}
            <button
              type="button"
              onClick={addEmailToCurrentSender}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl border-2 border-dashed border-emerald-400 bg-emerald-50 hover:bg-emerald-100 hover:border-emerald-500 text-emerald-700 hover:text-emerald-800 font-semibold text-sm transition-all group cursor-pointer shadow-sm hover:shadow"
            >
              <div className="w-6 h-6 rounded-full bg-emerald-500 group-hover:bg-emerald-600 flex items-center justify-center transition-colors">
                <Plus size={14} className="text-white" />
              </div>
              <span>
                Add another email from the same sender
                {/* ("<strong>{sender.role || "this sender"}</strong>") */}
              </span>
            </button>

            <button
              type="button"
              onClick={isLastSenderPage ? addNewSenderPage : tryNavigateForward}
              className="w-full flex items-center justify-center gap-2 px-4 py-3 rounded-xl border-2 border-dashed border-indigo-400 bg-indigo-50 hover:bg-indigo-100 hover:border-indigo-500 text-indigo-700 hover:text-indigo-800 font-semibold text-sm transition-all group cursor-pointer shadow-sm hover:shadow"
            >
              <div className="w-6 h-6 rounded-full bg-indigo-500 group-hover:bg-indigo-600 flex items-center justify-center transition-colors">
                <ArrowRight size={14} className="text-white" />
              </div>
              <span>
                {/* That's it for "<strong>{sender.role || "this sender"}</strong>" */}{" "}
                Move on to a different sender
              </span>
              {/* {isLastSenderPage && totalEmailCount >= MANDATORY_EMAILS && (
                <span className="text-indigo-600 font-semibold">
                  (+{formatBonus(BONUS_PER_EMAIL_PENCE)})
                </span>
              )} */}
            </button>
          </div>
        )}

        {/* ── Page navigation ─────────────────────────────────────────── */}
        <div className="flex items-center justify-end mt-6">
          {/* Back */}
          {/* <button
            type="button"
            onClick={() => {
              setShowValidation(false);
              setStep(1);
            }}
            className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-slate-800 transition-colors"
          >
            <ArrowLeft size={16} /> Back to Job Role
          </button>*/}

          <div className="flex flex-col items-stretch gap-3 w-full">
            {/* Soft nudge — non-blocking prompt to add more sender variety */}
            {showLowSenderDiversityNudge && (
              <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-900">
                Most participants describe emails from{" "}
                <strong>at least 3 different senders</strong>. So far you have
                described emails from{" "}
                <strong>
                  {completedSendersCount} sender
                  {completedSendersCount === 1 ? "" : "s"}
                </strong>
                . If anyone else emails you for work — even occasionally —
                please add them above. If not, you can move to next part.
              </div>
            )}

            {canLeaveStep2 && (
              <div className="flex justify-end">
                <button
                  type="button"
                  onClick={() => {
                    setShowValidation(false);
                    setStep(3);
                  }}
                  className="flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold transition-colors bg-slate-800 text-white hover:bg-slate-700 cursor-pointer"
                >
                  I am done! Move on to Part B <ArrowRight size={16} />
                </button>
              </div>
            )}
          </div>
        </div>

        {/* {!canLeaveStep2 && isLastSenderPage && (
          <p className="text-xs text-amber-600 mt-3 text-right">
            Describe at least {MANDATORY_EMAILS} emails in total to continue.
            You have {totalEmailCount} so far.
          </p>
        )} */}
      </div>
    );
  };

  // ═══════════════════════════════════════════════════════════════════
  // STEP 3 — General (non-job-specific) workplace emails (paginated)
  // ═══════════════════════════════════════════════════════════════════

  const currentGeneric = genericEmails[genericPageIdx] || EMPTY_GENERIC();
  const isLastGenericPage = genericPageIdx === genericEmails.length - 1;

  const updateCurrentGeneric = (field, value) => {
    const updated = [...genericEmails];
    updated[genericPageIdx] = { ...updated[genericPageIdx], [field]: value };
    setGenericEmails(updated);
  };

  const isCurrentGenericValid = () =>
    currentGeneric.sender.trim() && currentGeneric.description.trim();

  const getGenericErrors = () => {
    const errors = [];
    if (!currentGeneric.sender.trim()) errors.push("Who sends this email");
    if (!currentGeneric.description.trim())
      errors.push("What this email is typically about");
    return errors;
  };

  const tryNavigateGenericForward = () => {
    if (!isCurrentGenericValid()) {
      setShowValidation(true);
      return;
    }
    setShowValidation(false);
    if (isLastGenericPage) {
      if (committedGeneric >= REQUIRED_GENERIC) setStep(4);
      else {
        // Need more; auto-advance by appending a fresh page.
        setGenericEmails([...genericEmails, EMPTY_GENERIC()]);
        setGenericPageIdx(genericPageIdx + 1);
      }
    } else {
      setGenericPageIdx(genericPageIdx + 1);
    }
  };

  const addAnotherGenericPage = () => {
    if (!isCurrentGenericValid()) {
      setShowValidation(true);
      return;
    }
    setShowValidation(false);
    const updated = [...genericEmails, EMPTY_GENERIC()];
    setGenericEmails(updated);
    setGenericPageIdx(updated.length - 1);
  };

  const removeCurrentGeneric = () => {
    if (genericEmails.length <= 1) {
      // Last remaining entry: clear it instead of removing.
      setGenericEmails([EMPTY_GENERIC()]);
      setGenericPageIdx(0);
      setShowValidation(false);
      return;
    }
    const updated = genericEmails.filter((_, i) => i !== genericPageIdx);
    setGenericEmails(updated);
    setGenericPageIdx(
      Math.max(0, Math.min(genericPageIdx, updated.length - 1)),
    );
    setShowValidation(false);
  };

  const renderStep3Generic = () => {
    if (partBView === "intro") {
      return (
        <div className="flex flex-col justify-center py-4">
          <div className="max-w-2xl mx-auto text-center">
            <div className="w-14 h-14 rounded-full bg-slate-800 flex items-center justify-center mx-auto mb-4">
              <Inbox size={26} className="text-white" />
            </div>

            <h2 className="text-2xl font-bold text-slate-800 mb-3">
              Part B: General Workplace Email Types
            </h2>

            <p className="text-slate-600 text-sm leading-relaxed mb-5">
              Now think about emails that{" "}
              <strong>everyone in your company</strong> receives, regardless of
              job role — e.g., any alerts, announcements, newsletters etc.
            </p>

            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mb-4">
              <p className="text-sm text-amber-800 text-left">
                <strong>What does each email look like?</strong> A brief
                description: who sends it (job role only) and what the email is
                about.
              </p>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-5 text-left">
              <p className="text-sm text-blue-900 font-semibold mb-2">
                Before you start, please note:
              </p>
              <ul className="list-disc list-inside text-sm text-blue-900 space-y-1.5">
                <li>
                  Describe <strong>at least {REQUIRED_GENERIC}</strong> such
                  emails.
                </li>
                <li>
                  These should <strong>NOT</strong> be related to your specific
                  job duties — they apply to everyone in the company.
                </li>
                <li>
                  You can <strong>add or remove</strong> entries as you go, and
                  your progress is tracked on the right.
                </li>
              </ul>
            </div>

            {/* <div className="bg-slate-50 border-2 border-dashed border-slate-300 rounded-xl p-8 mb-6">
              <p className="text-sm font-semibold text-slate-700 mb-1">
                GIF placeholder
              </p>
              <p className="text-xs text-slate-500">
                We will put the explanatory GIF here for the participants.
              </p>
            </div> */}

            <div className="flex items-center justify-end">
              {/* <button
                type="button"
                onClick={() => {
                  setShowValidation(false);
                  setStep(2);
                  setPartAView("emails");
                  setSenderPageIdx(senders.length - 1);
                }}
                className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-slate-800 transition-colors"
              >
                <ArrowLeft size={16} /> Back to Part A
              </button> */}

              <button
                type="button"
                onClick={() => {
                  setShowValidation(false);
                  setPartBView("emails");
                }}
                className="flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold bg-slate-800 text-white hover:bg-slate-700 transition-colors cursor-pointer"
              >
                I am ready <ArrowRight size={16} />
              </button>
            </div>
          </div>
        </div>
      );
    }

    const pageErrors = showValidation ? getGenericErrors() : [];
    const senderMissing = showValidation && !currentGeneric.sender.trim();
    const descMissing = showValidation && !currentGeneric.description.trim();
    const canLeaveStep3 = committedGeneric >= REQUIRED_GENERIC;

    return (
      <div>
        {showValidation && pageErrors.length > 0 && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
            <p className="text-sm font-semibold text-red-700 mb-1">
              Please complete the following before continuing:
            </p>
            <ul className="list-disc list-inside text-xs text-red-600 space-y-0.5">
              {pageErrors.map((err, i) => (
                <li key={i}>{err}</li>
              ))}
            </ul>
          </div>
        )}

        <div className="rounded-lg p-3 mb-4 text-sm text-black">
          Describe emails you receive that are{" "}
          <strong>not specific to your job role</strong> — e.g., all-staff
          announcements. <br />
          This section should <strong>NOT include</strong> your day-to-day work
          related emails (those went in Part A).
        </div>

        <div className="mb-5 flex items-center justify-between">
          <div className="inline-flex items-center rounded-lg border border-slate-300 bg-slate-50 px-3 py-1.5 text-sm font-semibold text-slate-700">
            General email {genericPageIdx + 1}
          </div>
          <button
            type="button"
            onClick={removeCurrentGeneric}
            title="Remove this entry"
            className="flex items-center gap-1 text-xs text-slate-500 hover:text-red-600 transition-colors cursor-pointer"
          >
            <Trash2 size={14} /> Remove
          </button>
        </div>

        <div className="border border-slate-300 rounded-xl overflow-hidden mb-4">
          <div className="bg-slate-50 p-5 space-y-3">
            <div>
              <p
                className={`text-sm font-medium mb-1 ${
                  senderMissing ? "text-red-500" : "text-black"
                }`}
              >
                Who sends this email?
                {senderMissing && " *"}
              </p>
              <input
                type="text"
                value={currentGeneric.sender}
                onChange={(e) => updateCurrentGeneric("sender", e.target.value)}
                placeholder="Type in here..."
                className={`w-full rounded-md p-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white ${
                  senderMissing
                    ? "border border-red-400"
                    : "border border-slate-300"
                }`}
              />
            </div>
            <div>
              <p
                className={`text-sm font-medium mb-1 ${
                  descMissing ? "text-red-500" : "text-black"
                }`}
              >
                What is this email typically about?
                {descMissing && " *"}
              </p>
              <textarea
                value={currentGeneric.description}
                onChange={(e) =>
                  updateCurrentGeneric("description", e.target.value)
                }
                placeholder="Type in here..."
                rows={3}
                className={`w-full rounded-md p-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white resize-none ${
                  descMissing
                    ? "border border-red-400"
                    : "border border-slate-300"
                }`}
              />
            </div>
          </div>
        </div>

        <div className="flex items-center justify-end mt-6">
          {/* <button
            type="button"
            onClick={() => {
              setShowValidation(false);
              setPartBView("intro");
            }}
            className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-slate-800 transition-colors"
          >
            <ArrowLeft size={16} /> Back
          </button> */}

          <div className="flex items-center gap-3">
            {canLeaveStep3 ? (
              <>
                {isLastGenericPage && (
                  <button
                    type="button"
                    onClick={addAnotherGenericPage}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold bg-white text-slate-700 border border-slate-300 hover:bg-slate-50 transition-colors cursor-pointer"
                  >
                    <Plus size={14} />
                    Add more
                  </button>
                )}
                <button
                  type="button"
                  onClick={() => {
                    setShowValidation(false);
                    setStep(4);
                  }}
                  className="flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold bg-slate-800 text-white hover:bg-slate-700 transition-colors cursor-pointer"
                >
                  I am done! Move on to Part C <ArrowRight size={16} />
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={tryNavigateGenericForward}
                className="flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold bg-slate-800 text-white hover:bg-slate-700 transition-colors cursor-pointer"
              >
                Add next email <ArrowRight size={16} />
              </button>
            )}
          </div>
        </div>

        {!canLeaveStep3 && (
          <p className="text-xs text-amber-600 mt-3 text-right">
            Describe at least {REQUIRED_GENERIC} general emails to continue. You
            have {committedGeneric} so far.
          </p>
        )}
      </div>
    );
  };

  // ═══════════════════════════════════════════════════════════════════
  // STEP 4 — Hard-to-judge emails
  // ═══════════════════════════════════════════════════════════════════

  const currentSuspicious =
    suspiciousEmails[suspiciousPageIdx] || EMPTY_SUSPICIOUS();
  const isLastSuspiciousPage =
    suspiciousPageIdx === suspiciousEmails.length - 1;

  const updateCurrentSuspicious = (value) => {
    const updated = [...suspiciousEmails];
    updated[suspiciousPageIdx] = { description: value };
    setSuspiciousEmails(updated);
  };

  const isCurrentSuspiciousValid = () => !!currentSuspicious.description.trim();

  const tryNavigateSuspiciousForward = () => {
    if (!isCurrentSuspiciousValid()) {
      setShowValidation(true);
      return;
    }
    setShowValidation(false);
    if (isLastSuspiciousPage) {
      if (committedSuspicious >= REQUIRED_SUSPICIOUS) {
        // Stay on this page; submit is handled by the bottom nav.
      } else {
        setSuspiciousEmails([...suspiciousEmails, EMPTY_SUSPICIOUS()]);
        setSuspiciousPageIdx(suspiciousPageIdx + 1);
      }
    } else {
      setSuspiciousPageIdx(suspiciousPageIdx + 1);
    }
  };

  const addAnotherSuspiciousPage = () => {
    if (!isCurrentSuspiciousValid()) {
      setShowValidation(true);
      return;
    }
    setShowValidation(false);
    const updated = [...suspiciousEmails, EMPTY_SUSPICIOUS()];
    setSuspiciousEmails(updated);
    setSuspiciousPageIdx(updated.length - 1);
  };

  const removeCurrentSuspicious = () => {
    if (suspiciousEmails.length <= 1) {
      setSuspiciousEmails([EMPTY_SUSPICIOUS()]);
      setSuspiciousPageIdx(0);
      setShowValidation(false);
      return;
    }
    const updated = suspiciousEmails.filter((_, i) => i !== suspiciousPageIdx);
    setSuspiciousEmails(updated);
    setSuspiciousPageIdx(
      Math.max(0, Math.min(suspiciousPageIdx, updated.length - 1)),
    );
    setShowValidation(false);
  };

  const renderStep4 = () => {
    if (partCView === "intro") {
      return (
        <div className="flex flex-col justify-center py-4">
          <div className="max-w-2xl mx-auto text-center">
            <div className="w-14 h-14 rounded-full bg-slate-800 flex items-center justify-center mx-auto mb-4">
              <Mail size={26} className="text-white" />
            </div>

            <h2 className="text-2xl font-bold text-slate-800 mb-3">
              Part C: Suspicious / Hard-to-Judge Email Types
            </h2>

            <p className="text-slate-600 text-sm leading-relaxed mb-5">
              Finally, think of work emails where you were{" "}
              <strong>unsure whether they were legitimate or a scam</strong>.
              <br />
              These help us understand which email scenarios are hardest for
              employees to judge.
            </p>

            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mb-4">
              <p className="text-sm text-amber-800 text-left">
                <strong>What does each email look like?</strong> <br />A brief
                description of the <strong>type</strong> of email — not the
                actual content.
              </p>
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-5 text-left">
              <p className="text-sm text-blue-900 font-semibold mb-2">
                Before you start, please note:
              </p>
              <ul className="list-disc list-inside text-sm text-blue-900 space-y-1.5">
                <li>
                  Describe <strong>at least {REQUIRED_SUSPICIOUS}</strong>{" "}
                  suspicious or hard-to-judge emails.
                </li>
                <li>
                  Do not mention any <strong>personal or company names</strong>{" "}
                  in your answers.
                </li>
                <li>
                  You can <strong>add or remove</strong> entries as you go, and
                  your progress is tracked on the right.
                </li>
              </ul>
            </div>

            {/* <div className="bg-slate-50 border-2 border-dashed border-slate-300 rounded-xl p-8 mb-6">
              <p className="text-sm font-semibold text-slate-700 mb-1">
                GIF placeholder
              </p>
              <p className="text-xs text-slate-500">
                We will put the explanatory GIF here for the participants.
              </p>
            </div> */}

            <div className="flex items-center justify-end">
              {/* <button
                type="button"
                onClick={() => {
                  setShowValidation(false);
                  setStep(3);
                  setPartBView("emails");
                  setGenericPageIdx(genericEmails.length - 1);
                }}
                className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-slate-800 transition-colors"
              >
                <ArrowLeft size={16} /> Back to Part B
              </button> */}

              <button
                type="button"
                onClick={() => {
                  setShowValidation(false);
                  setPartCView("emails");
                }}
                className="flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold bg-slate-800 text-white hover:bg-slate-700 transition-colors cursor-pointer"
              >
                I am ready <ArrowRight size={16} />
              </button>
            </div>
          </div>
        </div>
      );
    }

    const descMissing = showValidation && !currentSuspicious.description.trim();
    const canLeaveStep4 = committedSuspicious >= REQUIRED_SUSPICIOUS;

    return (
      <div>
        {showValidation && descMissing && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-3 mb-4">
            <p className="text-sm font-semibold text-red-700">
              Please complete the following before continuing.
            </p>
          </div>
        )}

        <div className="rounded-lg p-3 mb-4 text-sm text-black">
          Describe work emails where you are{" "}
          <strong>unsure whether they are legitimate or a scam</strong> —
          messages that look plausible but make you hesitate (e.g., unexpected
          requests).
        </div>

        <div className="mb-5 flex items-center justify-between">
          <div className="inline-flex items-center rounded-lg border border-slate-300 bg-slate-50 px-3 py-1.5 text-sm font-semibold text-slate-700">
            Suspicious email {suspiciousPageIdx + 1}
          </div>
          <button
            type="button"
            onClick={removeCurrentSuspicious}
            title="Remove this entry"
            className="flex items-center gap-1 text-xs text-slate-500 hover:text-red-600 transition-colors cursor-pointer"
          >
            <Trash2 size={14} /> Remove
          </button>
        </div>

        <div className="border border-slate-300 rounded-xl overflow-hidden mb-4">
          <div className="bg-slate-50 p-5">
            <p
              className={`text-sm font-medium mb-1 ${
                descMissing ? "text-red-500" : "text-black"
              }`}
            >
              Describe the type of email you usually find suspicious or hard to
              judge
              {descMissing && " *"}
            </p>
            <textarea
              value={currentSuspicious.description}
              onChange={(e) => updateCurrentSuspicious(e.target.value)}
              placeholder="Type in here..."
              rows={4}
              className={`w-full rounded-md p-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white resize-none ${
                descMissing
                  ? "border border-red-400"
                  : "border border-slate-300"
              }`}
            />
          </div>
        </div>

        <div className="flex items-center justify-end mt-6">
          {/* <button
            type="button"
            onClick={() => {
              setShowValidation(false);
              setPartCView("intro");
            }}
            className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-slate-800 transition-colors"
          >
            <ArrowLeft size={16} /> Back
          </button> */}

          <div className="flex items-center gap-3">
            {canLeaveStep4 ? (
              <>
                {isLastSuspiciousPage && (
                  <button
                    type="button"
                    onClick={addAnotherSuspiciousPage}
                    disabled={submitting}
                    className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold bg-white text-slate-700 border border-slate-300 hover:bg-slate-50 transition-colors cursor-pointer"
                  >
                    <Plus size={14} />
                    Add more
                  </button>
                )}
                <button
                  type="button"
                  onClick={handleSubmit}
                  disabled={submitting}
                  className={`flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold transition-colors cursor-pointer ${
                    submitting
                      ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                      : "bg-emerald-600 text-white hover:bg-emerald-700"
                  }`}
                >
                  {submitting ? "Submitting..." : "Submit"}{" "}
                  {!submitting && <CheckCircle size={16} />}
                </button>
              </>
            ) : (
              <button
                type="button"
                onClick={tryNavigateSuspiciousForward}
                className="flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold bg-slate-800 text-white hover:bg-slate-700 transition-colors cursor-pointer"
              >
                Add next email <ArrowRight size={16} />
              </button>
            )}
          </div>
        </div>

        {!canLeaveStep4 && (
          <p className="text-xs text-amber-600 mt-3 text-right">
            Describe at least {REQUIRED_SUSPICIOUS} suspicious emails to submit.
            You have {committedSuspicious} so far.
          </p>
        )}
      </div>
    );
  };

  // ═══════════════════════════════════════════════════════════════════
  // STEP 4 — Thank you
  // ═══════════════════════════════════════════════════════════════════

  const renderThankYou = () => (
    <div className="text-center py-8">
      <div className="w-16 h-16 bg-emerald-100 rounded-full flex items-center justify-center mx-auto mb-4">
        <CheckCircle className="w-10 h-10 text-emerald-600" />
      </div>
      <h2 className="text-2xl font-bold text-slate-800 mb-3">
        Your response has been recorded successfully.
      </h2>
      {/* <p className="text-gray-600 mb-4 max-w-md mx-auto">
        Your response has been recorded successfully.
      </p> */}

      {bonusPence > 0 && (
        <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4 max-w-md mx-auto mb-4">
          <p className="text-sm text-emerald-800 flex items-center text-left gap-2">
            <Gift size={18} />
            <span>
              You described <strong>{totalEmailCount}</strong> emails in Part A
              ({bonusEmails} additional emails). <br />
              {bonusMaxReached ? (
                <>
                  You earned a <strong>£0.50 bonus</strong> (the maximum set for
                  this study)!
                </>
              ) : (
                <>
                  You earned a <strong>{formatBonus(bonusPence)} bonus</strong>!
                </>
              )}{" "}
              <br />
              The bonus will be paid via Prolific.
            </span>
          </p>
        </div>
      )}

      <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 max-w-md mx-auto">
        <p className="text-sm text-slate-700">
          Your insights into workplace email types across different professions
          are invaluable to our research on email security. Thank you for
          contributing to this study.
        </p>
      </div>

      <div className="bg-blue-50 border-2 border-blue-400 rounded-lg p-5 max-w-md mx-auto mt-6 shadow-sm">
        <p className="text-sm font-semibold text-blue-900 uppercase tracking-wide mb-2">
          Your Prolific completion code
        </p>
        <div className="bg-white border border-blue-300 rounded-md py-2 px-3 mb-3 flex items-center justify-between gap-3">
          <p className="text-lg font-mono font-bold text-blue-900 tracking-widest">
            {COMPLETION_CODE}
          </p>
          <button
            type="button"
            onClick={handleCopyCompletionCode}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-semibold border border-blue-200 text-blue-700 hover:bg-blue-50 transition-colors"
          >
            <Copy size={14} />
            {codeCopied ? "Copied" : "Copy code"}
          </button>
        </div>
        <p className="text-sm text-blue-900 mb-3 leading-relaxed">
          Click the link below to return to Prolific and confirm your submission
          so your payment can be released.
        </p>
        <a
          href={`https://app.prolific.com/submissions/complete?cc=${COMPLETION_CODE}`}
          onClick={clearLocal}
          className="inline-flex items-center gap-2 bg-blue-600 text-white px-4 py-2 rounded-lg text-sm font-semibold hover:bg-blue-700 transition-colors"
        >
          Submit on Prolific <ArrowRight size={16} />
        </a>
      </div>
    </div>
  );

  // ═══════════════════════════════════════════════════════════════════
  // RENDER
  // ═══════════════════════════════════════════════════════════════════

  const showSideRailA = step === 2 && partAView === "emails";
  const showSideRailB = step === 3 && partBView === "emails";
  const showSideRailC = step === 4 && partCView === "emails";
  const showSideRail = showSideRailA || showSideRailB || showSideRailC;
  const containerWidth = showSideRail ? "max-w-6xl" : "max-w-4xl";

  // Part A summary (always derivable; shown in side rail once we leave Part A)
  const partASummary = (() => {
    const byRole = new Map();
    senders.filter(isSenderComplete).forEach((s) => {
      const key = s.role.trim().toLowerCase();
      const emailCount = s.emails.filter(isEmailFilled).length;
      const existing = byRole.get(key);
      if (existing) {
        existing.emailCount += emailCount;
      } else {
        byRole.set(key, { role: s.role.trim(), emailCount });
      }
    });
    const entries = Array.from(byRole.values());
    const totalA = entries.reduce((sum, r) => sum + r.emailCount, 0);
    const bonusA = Math.max(0, totalA - MANDATORY_EMAILS);
    const bonusPenceA = Math.min(bonusA * BONUS_PER_EMAIL_PENCE, maxBonusPence);
    return {
      uniqueSenders: entries.length,
      totalEmails: totalA,
      bonusEmails: bonusA,
      bonusPence: bonusPenceA,
      items: entries.map((r) => ({
        label: r.role,
        suffix: `${r.emailCount} email${r.emailCount === 1 ? "" : "s"}`,
      })),
    };
  })();

  // Part B summary
  const partBSummary = (() => {
    const items = genericEmails
      .filter((g) => g.sender.trim() && g.description.trim())
      .map((g) => ({ label: g.sender.trim() }));
    return { count: items.length, items };
  })();

  let railTotal = 0;
  let railMandatory = MANDATORY_EMAILS;
  let railShowBonus = true;
  let railSenderBreakdown = null;
  let railShowSenderCount = true;
  let railEntriesLabel = "Entries";
  if (showSideRailA) {
    railTotal = totalEmailCount;
    // Reuse the same grouped breakdown that powers the post-Part-A summary,
    // so the active rail and the cumulative summary stay in sync.
    railSenderBreakdown = partASummary.items.map((it) => ({
      role: it.label,
      emailCount: parseInt(it.suffix, 10) || 0,
    }));
  } else if (showSideRailB) {
    railTotal = committedGeneric;
    railMandatory = REQUIRED_GENERIC;
    railShowBonus = false;
    railShowSenderCount = false;
    railEntriesLabel = "Email types added";
    railSenderBreakdown = genericEmails
      .filter((g) => g.sender.trim() && g.description.trim())
      .map((g) => ({ role: g.sender.trim(), emailCount: 1 }));
  } else if (showSideRailC) {
    railTotal = committedSuspicious;
    railMandatory = REQUIRED_SUSPICIOUS;
    railShowBonus = false;
    railShowSenderCount = false;
    railEntriesLabel = "Email types added";
    railSenderBreakdown = suspiciousEmails
      .filter((s) => s.description.trim())
      .map((s, i) => ({
        role:
          s.description.trim().length > 40
            ? s.description.trim().slice(0, 40) + "…"
            : s.description.trim() || `Entry ${i + 1}`,
        emailCount: 1,
      }));
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 py-8 px-4">
      <div className={`${containerWidth} mx-auto`}>
        {step === 0 && (
          <div className="text-center mb-6">
            <h1 className="text-2xl font-bold text-slate-800">
              Welcome to Workplace Email Types Survey
            </h1>
          </div>
        )}

        <div
          className={
            showSideRail
              ? "grid grid-cols-1 lg:grid-cols-[1fr_280px] gap-5"
              : ""
          }
        >
          <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
            {step === 0 && renderStep0()}
            {step === 1 && renderStep1()}
            {step === 2 && renderStep2()}
            {step === 3 && renderStep3Generic()}
            {step === 4 && renderStep4()}
            {step === 5 && renderThankYou()}

            {error && (
              <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
                {error}
              </div>
            )}

            {/* Navigation only for step 1 — steps 0, 2, 3, 4 have their own nav */}
            {step === 1 && (
              <div className="flex justify-between mt-8">
                <button
                  onClick={() => setStep(step - 1)}
                  className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-slate-800 transition-colors"
                >
                  <ArrowLeft size={16} /> Back
                </button>

                <button
                  onClick={() => {
                    setStep(2);
                    setPartAView("intro");
                  }}
                  disabled={!canAdvance()}
                  className={`flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold transition-colors cursor-pointer ${
                    canAdvance()
                      ? "bg-slate-800 text-white hover:bg-slate-700"
                      : "bg-gray-200 text-gray-400 cursor-not-allowed"
                  }`}
                >
                  Next <ArrowRight size={16} />
                </button>
              </div>
            )}
          </div>

          {showSideRail && (
            <aside className="hidden lg:block self-start sticky top-6">
              <RewardPanel
                totalEmails={railTotal}
                mandatory={railMandatory}
                showBonus={railShowBonus}
                senderBreakdown={railSenderBreakdown}
                showSenderCount={railShowSenderCount}
                entriesLabel={railEntriesLabel}
              />
              {/* Cumulative summaries of earlier parts — collapsed by default */}
              {(showSideRailB || showSideRailC) && (
                <CompletedPartSummary
                  title="Part A — done"
                  headline={
                    `${partASummary.uniqueSenders} sender${
                      partASummary.uniqueSenders === 1 ? "" : "s"
                    } · ${partASummary.totalEmails} email${
                      partASummary.totalEmails === 1 ? "" : "s"
                    }` +
                    (partASummary.bonusEmails > 0
                      ? ` · ${formatBonus(partASummary.bonusPence)} bonus`
                      : "")
                  }
                  items={partASummary.items}
                />
              )}
              {showSideRailC && (
                <CompletedPartSummary
                  title="Part B — done"
                  headline={`${partBSummary.count} email type${
                    partBSummary.count === 1 ? "" : "s"
                  } added`}
                  items={partBSummary.items}
                />
              )}
            </aside>
          )}
        </div>

        <p className="text-center text-xs text-gray-400 mt-6">
          {/* IRiSC Lab · University of Luxembourg <br />·{" "} */}
          <span className="font-bold text-gray-500">
            All responses are strictly anonymous and used solely for research
            purposes.
          </span>
        </p>
      </div>
    </div>
  );
}
