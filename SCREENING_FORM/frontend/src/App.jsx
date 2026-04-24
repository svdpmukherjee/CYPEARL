import { useState, useRef, useEffect } from "react";
import axios from "axios";
import sampleEmailImg from "./assets/sample_email.jpg";
import {
  ArrowRight,
  ArrowLeft,
  CheckCircle,
  Briefcase,
  Mail,
  ClipboardList,
  Plus,
  X,
  Inbox,
  User,
  Gift,
  ShieldAlert,
} from "lucide-react";

const API_URL = (
  import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD
    ? "https://prolific-screening-form.onrender.com"
    : "/api")
).replace(/\/$/, "");

const FREQUENCY_OPTIONS = ["Daily", "Weekly", "Monthly", "Rarely"];

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

const MANDATORY_EMAILS = 5;
const MANDATORY_SENDERS = 5;
const REQUIRED_GENERIC = 5;
const REQUIRED_SUSPICIOUS = 5;
const BONUS_PER_EMAIL_PENCE = 2; // £0.02 per additional email beyond mandatory

// ── Helpers ─────────────────────────────────────────────────────────────

function isEmailFilled(e) {
  return !!(e.subject?.trim() || e.content?.trim());
}

function isSenderComplete(sender) {
  return sender.role.trim() && sender.type && sender.emails.some(isEmailFilled);
}

function formatBonus(pence) {
  if (pence <= 0) return "£0.00";
  return `£${(pence / 100).toFixed(2)}`;
}

// ── Dynamic text-box list ───────────────────────────────────────────────

function TextBoxList({
  items,
  onChange,
  placeholder,
  minBoxes = 1,
  labelPrefix = "",
}) {
  const updateItem = (index, value) => {
    const updated = [...items];
    updated[index] = value;
    onChange(updated);
  };
  const addBox = () => onChange([...items, ""]);
  const removeBox = (index) => {
    if (items.length <= minBoxes) return;
    onChange(items.filter((_, i) => i !== index));
  };

  return (
    <div className="space-y-2">
      {items.map((val, i) => (
        <div key={i} className="flex items-center gap-2">
          <span className="text-xs text-slate-500 shrink-0 font-medium w-16 text-right">
            {labelPrefix ? `${labelPrefix} ${i + 1}` : `${i + 1}.`}
          </span>
          <input
            type="text"
            value={val}
            onChange={(e) => updateItem(i, e.target.value)}
            placeholder={placeholder}
            className="flex-1 border border-slate-300 rounded-md p-2 text-sm bg-white focus:outline-none focus:ring-2 focus:ring-blue-400 placeholder:text-slate-400"
          />
          {items.length > minBoxes && (
            <button
              type="button"
              onClick={() => removeBox(i)}
              className="p-1 text-slate-400 hover:text-red-400 transition-colors"
              title="Remove"
            >
              <X size={16} />
            </button>
          )}
        </div>
      ))}
      <button
        type="button"
        onClick={addBox}
        className="flex items-center gap-1.5 text-sm text-blue-600 hover:text-blue-800 transition-colors mt-1"
      >
        <Plus size={14} /> Add more
      </button>
    </div>
  );
}

// ── Email Card ──────────────────────────────────────────────────────────

function EmailCard({
  email,
  senderRole,
  senderType,
  onChange,
  onRemove,
  canRemove,
  showValidation,
}) {
  const hasContent = email.subject.trim() || email.content.trim();
  const subjectMissing = showValidation && !email.subject.trim();
  const contentMissing = showValidation && !email.content.trim();
  const frequencyMissing = showValidation && !email.frequency;

  return (
    <div
      className={`rounded-lg border overflow-hidden transition-all ${
        hasContent
          ? "border-slate-300 shadow-sm"
          : "border-dashed border-gray-300"
      }`}
    >
      {/* Email toolbar */}
      <div className="bg-slate-200 px-3 py-2 flex items-center justify-between border-b border-slate-300">
        <div className="flex items-center gap-2 text-xs text-slate-600 min-w-0">
          <strong>From</strong>
          {/* <div
            className={`w-5 h-5 rounded-full flex items-center justify-center shrink-0 ${
              senderType === "internal"
                ? "bg-indigo-100 text-indigo-700"
                : senderType === "external"
                  ? "bg-indigo-100 text-indigo-700"
                  : "bg-slate-300 text-slate-600"
            }`}
          >
            {" "}
            <User size={16} />
          </div> */}
          <span className="truncate font-medium text-black">
            "{senderRole || "Sender"}"
          </span>
          {/* {senderType && (
            <span
              className={`px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide ${
                senderType === "internal"
                  ? "bg-indigo-100 text-indigo-700"
                  : "bg-indigo-100 text-indigo-700"
              }`}
            >
              {senderType}
            </span>
          )} */}
        </div>
        {canRemove && (
          <button
            type="button"
            onClick={onRemove}
            className="p-0.5 text-slate-500 hover:text-red-500 transition-colors"
            title="Remove this email"
          >
            <X size={16} />
          </button>
        )}
      </div>

      {/* Email body */}
      <div className="bg-slate-50 p-5 space-y-2">
        <div className="flex items-center gap-2 p-2">
          <span
            className={`text-xs font-semibold uppercase tracking-wider w-14 shrink-0 ${
              subjectMissing ? "text-red-500" : "text-slate-600"
            }`}
          >
            Subject
          </span>
          <input
            type="text"
            value={email.subject}
            onChange={(e) => onChange({ ...email, subject: e.target.value })}
            placeholder="Type in here..."
            className={`flex-1 rounded-md p-1.5 text-sm font-medium text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white placeholder:text-slate-400 ${
              subjectMissing ? "ring-2 ring-red-400" : "border border-slate-300"
            }`}
          />
        </div>
        <div className="flex items-start gap-2 p-2">
          <span
            className={`text-xs font-semibold uppercase tracking-wider w-14 shrink--0 mt-1.5 ${
              contentMissing ? "text-red-500" : "text-slate-600"
            }`}
          >
            About
          </span>
          <textarea
            value={email.content}
            onChange={(e) => onChange({ ...email, content: e.target.value })}
            placeholder="Type in here..."
            rows={2}
            className={`flex-1 rounded-md p-1.5 text-sm text-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white resize-none placeholder:text-slate-400 ${
              contentMissing ? "ring-2 ring-red-400" : "border border-slate-300"
            }`}
          />
        </div>
      </div>

      {/* Frequency is outside the email skeleton content area */}
      <div className="bg-slate-50 border-t border-slate-200 px-3 py-2.5">
        <div>
          <p
            className={`text-sm font-medium mb-1.5 ${
              frequencyMissing ? "text-red-500" : "text-slate-600"
            }`}
          >
            How often do you receive this type of email?
            {frequencyMissing && " *"}
          </p>
          <div className="flex gap-1.5">
            {FREQUENCY_OPTIONS.map((freq) => (
              <button
                key={freq}
                type="button"
                onClick={() => onChange({ ...email, frequency: freq })}
                className={`px-3 py-1.5 rounded-lg text-xs font-semibold border-2 transition-all ${
                  email.frequency === freq
                    ? "bg-indigo-500 text-white border-indigo-500 shadow-sm"
                    : frequencyMissing
                      ? "bg-white text-red-400 border-red-300 hover:border-red-400"
                      : "bg-white text-slate-500 border-slate-200 hover:border-slate-400 hover:text-slate-700"
                }`}
              >
                {freq}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ── Gamified Reward Panel (side rail) ───────────────────────────────────

function RewardPanel({
  totalEmails,
  mandatory = MANDATORY_EMAILS,
  showBonus = true,
  senderBreakdown = null,
  mandatorySenders = MANDATORY_SENDERS,
}) {
  const mandatoryDone = Math.min(totalEmails, mandatory);
  const remaining = Math.max(0, mandatory - totalEmails);
  const bonusEmails = Math.max(0, totalEmails - mandatory);
  const bonusPence = bonusEmails * BONUS_PER_EMAIL_PENCE;
  const mandatoryComplete = totalEmails >= mandatory;
  const sendersDone = senderBreakdown ? senderBreakdown.length : 0;
  const sendersRemaining = Math.max(0, mandatorySenders - sendersDone);

  // When bonus is hidden, the required portion fills the entire bar.
  const requiredWidth = showBonus ? 60 : 100;
  const mandatoryPct = Math.round((mandatoryDone / mandatory) * requiredWidth);
  const bonusPct = showBonus ? Math.min(40, bonusEmails * 4) : 0;

  return (
    <div className="bg-gray-800 rounded-lg p-4 shadow-md">
      <div className="flex items-center gap-2 mb-3">
        <Mail size={16} className="text-white" />
        <h3 className="text-white text-sm font-semibold uppercase tracking-wider">
          Your progress
        </h3>
      </div>

      {/* Stats grid */}
      <div className="space-y-2.5 mb-4">
        {senderBreakdown && (
          <div className="bg-gray-700/60 rounded-md px-3 py-2">
            <div className="flex items-baseline justify-between">
              <span className="text-sm text-slate-300">Senders added</span>
              <span className="text-white font-bold text-sm">
                {sendersDone}
                <span className="text-slate-400 font-normal">
                  {" "}
                  / {mandatorySenders}
                </span>
              </span>
            </div>
            {sendersRemaining > 0 && (
              <p className="text-[11px] text-amber-300 mt-0.5">
                {sendersRemaining} more sender
                {sendersRemaining === 1 ? "" : "s"} needed
              </p>
            )}
            {senderBreakdown.length > 0 && (
              <ul className="mt-2 space-y-1 max-h-40 overflow-y-auto pr-1">
                {senderBreakdown.map((s, i) => (
                  <li
                    key={i}
                    className="flex items-baseline justify-between text-[11px] text-slate-300"
                  >
                    <span className="truncate mr-2" title={s.role}>
                      {i + 1}. {s.role}
                    </span>
                    <span className="text-slate-400 shrink-0">
                      {s.emailCount} email{s.emailCount === 1 ? "" : "s"}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
        <div className="bg-gray-700/60 rounded-md px-3 py-2">
          <div className="flex items-baseline justify-between">
            <span className="text-sm text-slate-300">
              {senderBreakdown ? "Total emails" : "Required completed"}
            </span>
            <span className="text-white font-bold text-sm">
              <span className={bonusEmails > 0 ? "text-emerald-300" : ""}>
                {totalEmails}
              </span>
              <span className="text-slate-400 font-normal"> / {mandatory}</span>
            </span>
          </div>
          {!mandatoryComplete && (
            <p className="text-[11px] text-amber-300 mt-0.5">
              {remaining} more required
              {showBonus ? " to unlock bonus" : ""}
            </p>
          )}
          {showBonus && bonusEmails > 0 && (
            <p className="text-[11px] text-emerald-300 mt-0.5">
              +{bonusEmails} beyond the {mandatory} required
            </p>
          )}
        </div>

        {showBonus && (
          <>
            <div
              className={`rounded-md px-3 py-2 ${
                bonusPence > 0
                  ? "bg-emerald-500/20 border border-emerald-400/40"
                  : "bg-gray-700/60"
              }`}
            >
              <div className="flex items-baseline justify-between">
                <span className="text-sm text-slate-300 flex items-center gap-1">
                  <Gift size={12} /> Bonus earned
                </span>
                <span
                  className={`font-bold text-sm ${
                    bonusPence > 0 ? "text-emerald-300" : "text-slate-400"
                  }`}
                >
                  {formatBonus(bonusPence)}
                </span>
              </div>
              {mandatoryComplete && (
                <p className="text-[11px] text-emerald-300 mt-0.5">
                  +{formatBonus(BONUS_PER_EMAIL_PENCE)} per extra email
                </p>
              )}
            </div>
          </>
        )}
      </div>

      {/* Progress bar */}
      <div className="w-full bg-gray-100 rounded-full h-2.5 overflow-hidden relative mb-2">
        <div
          className="absolute top-0 left-0 h-full rounded-full transition-all duration-500 ease-out bg-gradient-to-r from-blue-400 to-blue-500"
          style={{ width: `${mandatoryPct}%` }}
        />
        {showBonus && bonusEmails > 0 && (
          <div
            className="absolute top-0 h-full rounded-r-full transition-all duration-500 ease-out bg-gradient-to-r from-emerald-400 to-emerald-500"
            style={{ left: `${mandatoryPct}%`, width: `${bonusPct}%` }}
          />
        )}
        {showBonus && (
          <div
            className="absolute top-0 h-full w-0.5 bg-white"
            style={{ left: "60%" }}
          />
        )}
      </div>

      {showBonus && (
        <div className="flex justify-between text-xs text-slate-400">
          <span className="flex items-center gap-1">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-blue-500" />
            Required
          </span>
          <span className="flex items-center gap-1">
            <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500" />
            Bonus
          </span>
        </div>
      )}
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════
// MAIN APP
// ═════════════════════════════════════════════════════════════════════════

export default function App() {
  const [step, setStep] = useState(0);
  const [landingView, setLandingView] = useState("overview");
  const [partAView, setPartAView] = useState("intro");
  const [partBView, setPartBView] = useState("intro");
  const [partCView, setPartCView] = useState("intro");
  const [senderPageIdx, setSenderPageIdx] = useState(0);
  const [genericPageIdx, setGenericPageIdx] = useState(0);
  const [suspiciousPageIdx, setSuspiciousPageIdx] = useState(0);
  const [showValidation, setShowValidation] = useState(false);
  const [submitting, setSubmitting] = useState(false);
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
  const bonusPence = bonusEmails * BONUS_PER_EMAIL_PENCE;

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

  // To leave Part A, the user must have described at least MANDATORY_SENDERS
  // distinct senders (each with ≥1 complete email).
  const completedSendersCount = senders.filter(isSenderComplete).length;
  const canLeaveStep2 = completedSendersCount >= MANDATORY_SENDERS;

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
      if (landingView === "overview") return true;
      return prolificId.trim().length > 0;
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

      await axios.post(`${API_URL}/submit`, {
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
          <p className="text-gray-600 text-md mb-4 leading-relaxed">
            Thank you for participating in this research study. We are building
            a library of <strong>realistic workplace email templates</strong> to
            study how employees evaluate email legitimacy and make trust
            decisions. <br />
            Your task is to describe the{" "}
            <span className="font-semibold">types</span> of emails you typically
            receive at work so we can make those templates as authentic as
            possible.
          </p>

          <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
            <p className="text-sm text-blue-800 font-semibold mb-4">
              This survey has 3 parts:
            </p>

            <div className="space-y-4">
              <div>
                <p className="text-sm text-blue-800 font-semibold">Part A</p>
                <p className="text-sm text-black-700 ml-4 mt-0.5">
                  <strong>[Job-Specific Emails]</strong>: You will List as many
                  types of emails as possible (not verbatim, but types) that you
                  typically receive and that are{" "}
                  <span className="font-semibold">
                    directly related to your job role
                  </span>
                  .
                </p>
              </div>

              <div>
                <p className="text-sm text-blue-800 font-semibold">Part B</p>
                <p className="text-sm text-black-700 ml-4 mt-0.5">
                  <strong>[General Workplace Emails]</strong>: You will list as
                  many types of emails as possible that you typically receive
                  but that are{" "}
                  <span className="font-semibold">
                    not specific to your job role
                  </span>
                  .
                </p>
              </div>

              <div>
                <p className="text-sm text-blue-800 font-semibold">Part C</p>
                <p className="text-sm text-black-700 ml-4 mt-0.5">
                  <strong>[Suspicious / Hard-to-Judge Emails]</strong>: You will
                  list types of work emails where you are unsure whether they
                  are legitimate or a scam.
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

          <div className="flex justify-end">
            <button
              type="button"
              onClick={() => setLandingView("details")}
              className="flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold bg-slate-800 text-white hover:bg-slate-700 transition-colors cursor-pointer"
            >
              Next <ArrowRight size={16} />
            </button>
          </div>
        </div>
      );
    }

    return (
      <div>
        {/* Anonymity notice */}
        <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 mb-4">
          <div className="text-sm text-slate-700 leading-8">
            <div className="text-slate-800 uppercase tracking-wider font-semibold">
              Your privacy is protected.
            </div>
            <div className="mt-2">
              Your identity will be <strong>strictly anonymised</strong>; there
              is no way for anyone to identify you from your responses. Please{" "}
              <strong>do not mention any personal names</strong> (yours or
              anyone else&apos;s) in any of your answers. We are only interested
              in job roles, email types, and general workplace patterns.
            </div>
          </div>
        </div>

        <div className="bg-gradient-to-r from-slate-200 to-slate-200 rounded-lg p-4 mb-6 text-black shadow-md">
          <div className="flex items-center justify-around mb-6">
            <div className="text-center">
              <p className="text-[10px] uppercase tracking-wider text-slate-800 mb-0.5">
                Estimated time
              </p>
              <p className="text-lg font-bold">6 mins</p>
            </div>
            <div className="w-px h-8 bg-slate-500" />
            <div className="text-center">
              <p className="text-[10px] uppercase tracking-wider text-slate-800 mb-0.5">
                Base pay
              </p>
              <p className="text-lg font-bold">£1.00</p>
            </div>
            <div className="w-px h-8 bg-slate-500" />
            <div className="text-center">
              <p className="text-[10px] uppercase tracking-wider text-slate-800 mb-0.5">
                Bonus
              </p>
              <p className="text-lg font-bold text-blue-400">+ extra</p>
            </div>
          </div>
          <div className="bg-white rounded-md p-3 flex items-start gap-2">
            <ShieldAlert size={36} className="text-blue-800 mt-0.5 shrink-0" />
            <p className="text-sm leading-10 text-slate-800">
              In Part A, you need to describe emails from{" "}
              <span className="font-bold">
                at least {MANDATORY_SENDERS} distinct senders
              </span>{" "}
              (so <strong>at least {MANDATORY_EMAILS} emails</strong>). <br />
              Each additional email you create beyond the required{" "}
              {MANDATORY_EMAILS} earns you{" "}
              <strong className="text-blue-800">
                {formatBonus(BONUS_PER_EMAIL_PENCE)} bonus
              </strong>{" "}
              via Prolific. <br />
              Parts B and C are required but{" "}
              <span className="font-bold"> do not earn bonus</span>.
            </p>
          </div>
        </div>

        <label className="block text-sm font-semibold text-slate-700 mt-10">
          Please Enter Your Prolific ID
        </label>
        <input
          type="text"
          value={prolificId}
          onChange={(e) => setProlificId(e.target.value)}
          placeholder="Please double-check your Prolific ID so we can verify your submission"
          className="w-full border border-gray-300 rounded-md p-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 mb-2"
        />
        {/* <p className="text-xs text-gray-400">
          Please double-check your Prolific ID so we can verify your submission.
        </p> */}

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
            onClick={() => setStep(1)}
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
              Part A: Job-Specific Email Skeletons
            </h2>

            <p className="text-slate-600 text-sm leading-relaxed mb-5">
              In this part, you will describe work-related email types you
              receive one by one. We only need the structure of each email, not
              real content.
            </p>

            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mb-4">
              <p className="text-sm text-amber-800">
                <strong>What is an email skeleton?</strong> A brief template:
                who sends it (job role only), what the subject line looks like,
                and what the email is about.
              </p>
            </div>

            <div className="p-3 mb-5">
              <p className="text-xs text-slate-500 mb-2 uppercase tracking-wider font-semibold">
                Example
              </p>
              <img
                src={sampleEmailImg}
                alt="Example of a filled-in email skeleton"
                className="w-7/8 rounded-lg border border-slate-200 mx-auto"
              />
            </div>

            <div className="bg-blue-50 border border-blue-200 rounded-xl p-4 mb-5 text-left">
              <p className="text-sm text-blue-900 font-semibold mb-2">
                Before you start, please note:
              </p>
              <ul className="list-disc list-inside text-sm text-blue-900 space-y-1.5">
                <li>
                  You need to describe emails from{" "}
                  <strong>at least {MANDATORY_SENDERS} distinct senders</strong>{" "}
                  (i.e. {MANDATORY_SENDERS} different job roles). This means at
                  least {MANDATORY_SENDERS} emails in total.
                </li>
                <li>
                  For each sender, you can add{" "}
                  <strong>as many email types as you want</strong>, or move on
                  to a different sender — whichever reflects your inbox better.
                </li>
                <li>
                  You can keep adding{" "}
                  <strong>more senders and more emails</strong> beyond the
                  required {MANDATORY_SENDERS} — each extra email earns a bonus.
                </li>
                <li>
                  Once you move forward, you{" "}
                  <strong>cannot go back to edit earlier senders</strong>, so
                  please complete each sender carefully before moving on.
                </li>
              </ul>
            </div>

            <div className="flex items-center justify-between">
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
    const isMandatoryPage = senderPageIdx < MANDATORY_EMAILS;
    const pageLabel = isMandatoryPage
      ? `Email ${senderPageIdx + 1}`
      : `Email ${senderPageIdx + 1} (bonus)`;

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
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-slate-300 flex items-center justify-center shrink-0">
                <User size={20} className="text-slate-600" />
              </div>
              <div className="flex-1 space-y-2.5">
                {/* Sender role */}
                <div>
                  <p
                    className={`text-sm font-medium mb-1 ${
                      roleMissing ? "text-red-500" : "text-black"
                    }`}
                  >
                    What is the job role or title of someone who regularly
                    emails you at work?
                    {roleMissing && " *"}
                  </p>
                  <input
                    type="text"
                    value={sender.role}
                    onChange={(e) =>
                      updateCurrentSender("role", e.target.value)
                    }
                    placeholder="Type in here..."
                    className={`w-full border rounded-md p-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white ${
                      roleMissing ? "border-red-400" : "border-slate-300"
                    }`}
                  />
                </div>

                {/* Internal / External */}
                <div>
                  <p
                    className={`text-sm font-medium mb-1.5 ${
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
          </div>

          {/* Email entries */}
          <div className="p-4 pt-3 space-y-3 ">
            <p className="text-sm font-medium text-black">
              What kinds of emails does{" "}
              <strong className="text-slate-600">
                {sender.role || "this role"}
              </strong>{" "}
              typically send you?
            </p>
            {sender.emails.map((email, emailIdx) => {
              const isLast = emailIdx === sender.emails.length - 1;
              return (
                <div
                  key={emailIdx}
                  ref={isLast ? lastEmailCardRef : null}
                  className="space-y-2 scroll-mt-4"
                >
                  <div className="inline-flex items-center rounded-md px-2.5 py-1 text-xs font-semibold bg-slate-700 text-white">
                    {sender.emails.length === 1
                      ? `Email ${senderPageIdx + 1}`
                      : `Email ${senderPageIdx + 1}.${emailIdx + 1}`}
                  </div>
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
                Add another email type from same sender ("
                <strong>{sender.role || "this sender"}</strong>")
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
                That's it for "<strong>{sender.role || "this sender"}</strong>"
                — move on to a different sender
              </span>
              {isLastSenderPage && totalEmailCount >= MANDATORY_EMAILS && (
                <span className="text-indigo-600 font-semibold">
                  (+{formatBonus(BONUS_PER_EMAIL_PENCE)})
                </span>
              )}
            </button>
          </div>
        )}

        {/* ── Page navigation ─────────────────────────────────────────── */}
        <div className="flex items-center justify-between mt-6">
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

          <div className="flex items-center gap-3">
            {/* Proceed to Part B — only on last sender page */}
            {isLastSenderPage && (
              <button
                type="button"
                onClick={tryNavigateForward}
                disabled={!canLeaveStep2}
                className={`flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold transition-colors ${
                  !canLeaveStep2
                    ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                    : "bg-slate-800 text-white hover:bg-slate-700 cursor-pointer"
                }`}
              >
                I am done! Move on to Part B <ArrowRight size={16} />
              </button>
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

  const renderStep3Generic = () => {
    if (partBView === "intro") {
      return (
        <div className="flex flex-col justify-center py-4">
          <div className="max-w-2xl mx-auto text-center">
            <div className="w-14 h-14 rounded-full bg-slate-800 flex items-center justify-center mx-auto mb-4">
              <Inbox size={26} className="text-white" />
            </div>

            <h2 className="text-2xl font-bold text-slate-800 mb-3">
              Part B: General Workplace Email Skeletons
            </h2>

            <p className="text-slate-600 text-sm leading-relaxed mb-5">
              Now think about emails that{" "}
              <strong>everyone in your company</strong> receives, regardless of
              job role — e.g. any alerts, announcements, newsletters, notices,
              vendor promotions, etc.
            </p>

            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mb-5">
              <p className="text-sm text-amber-800 leading-10">
                <strong>Reminder:</strong> An email skeleton = who sends it +
                what it is about. <br />
                The emails should <strong>NOT</strong> be related to your
                specific job duties. Please describe at least {REQUIRED_GENERIC}
                .
              </p>
            </div>

            {/* <div className="bg-slate-50 border-2 border-dashed border-slate-300 rounded-xl p-8 mb-6">
              <p className="text-sm font-semibold text-slate-700 mb-1">
                GIF placeholder
              </p>
              <p className="text-xs text-slate-500">
                We will put the explanatory GIF here for the participants.
              </p>
            </div> */}

            <div className="flex items-center justify-between">
              <button
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
              </button>

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

        <div className="mb-5">
          <div className="inline-flex items-center rounded-lg border border-slate-300 bg-slate-50 px-3 py-1.5 text-sm font-semibold text-slate-700">
            General email {genericPageIdx + 1}
          </div>
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

        <div className="flex items-center justify-between mt-6">
          <button
            type="button"
            onClick={() => {
              setShowValidation(false);
              setPartBView("intro");
            }}
            className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-slate-800 transition-colors"
          >
            <ArrowLeft size={16} /> Back
          </button>

          <div className="flex items-center gap-3">
            {isLastGenericPage && canLeaveStep3 ? (
              <>
                <button
                  type="button"
                  onClick={addAnotherGenericPage}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold bg-white text-slate-700 border border-slate-300 hover:bg-slate-50 transition-colors cursor-pointer"
                >
                  <Plus size={14} />
                  Add more
                </button>
                <button
                  type="button"
                  onClick={tryNavigateGenericForward}
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

  const renderStep4 = () => {
    if (partCView === "intro") {
      return (
        <div className="flex flex-col justify-center py-4">
          <div className="max-w-2xl mx-auto text-center">
            <div className="w-14 h-14 rounded-full bg-slate-800 flex items-center justify-center mx-auto mb-4">
              <Mail size={26} className="text-white" />
            </div>

            <h2 className="text-2xl font-bold text-slate-800 mb-3">
              Part C: Suspicious / Hard-to-Judge Email Skeletons
            </h2>

            <p className="text-slate-600 text-sm leading-relaxed mb-5">
              Finally, think of work emails where you were{" "}
              <strong>unsure whether they were legitimate or a scam</strong>.
              <br />
              These help us understand which email scenarios are hardest for
              employees to judge.
            </p>

            <div className="bg-amber-50 border border-amber-200 rounded-xl p-4 mb-5">
              <p className="text-sm text-amber-800 leading-10">
                <strong>Reminder:</strong> Describe the <strong>type</strong> of
                email — not the actual content. <br />
                Please provide at least {REQUIRED_SUSPICIOUS}. Do not mention
                any personal names.
              </p>
            </div>

            {/* <div className="bg-slate-50 border-2 border-dashed border-slate-300 rounded-xl p-8 mb-6">
              <p className="text-sm font-semibold text-slate-700 mb-1">
                GIF placeholder
              </p>
              <p className="text-xs text-slate-500">
                We will put the explanatory GIF here for the participants.
              </p>
            </div> */}

            <div className="flex items-center justify-between">
              <button
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
              </button>

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
              Please describe this email type before continuing.
            </p>
          </div>
        )}

        <div className="mb-5">
          <div className="inline-flex items-center rounded-lg border border-slate-300 bg-slate-50 px-3 py-1.5 text-sm font-semibold text-slate-700">
            Suspicious email {suspiciousPageIdx + 1}
          </div>
        </div>

        <div className="border border-slate-300 rounded-xl overflow-hidden mb-4">
          <div className="bg-slate-50 p-5">
            <p
              className={`text-sm font-medium mb-1 ${
                descMissing ? "text-red-500" : "text-black"
              }`}
            >
              Describe the type of email you found suspicious or hard to judge
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

        <div className="flex items-center justify-between mt-6">
          <button
            type="button"
            onClick={() => {
              setShowValidation(false);
              setPartCView("intro");
            }}
            className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-slate-800 transition-colors"
          >
            <ArrowLeft size={16} /> Back
          </button>

          <div className="flex items-center gap-3">
            {isLastSuspiciousPage && canLeaveStep4 ? (
              <>
                <button
                  type="button"
                  onClick={addAnotherSuspiciousPage}
                  disabled={submitting}
                  className="flex items-center gap-1.5 px-4 py-2 rounded-lg text-sm font-semibold bg-white text-slate-700 border border-slate-300 hover:bg-slate-50 transition-colors cursor-pointer"
                >
                  <Plus size={14} />
                  Add more
                </button>
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
                Add more <ArrowRight size={16} />
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
      <h2 className="text-2xl font-bold text-slate-800 mb-3">Thank You!</h2>
      <p className="text-gray-600 mb-4 max-w-md mx-auto">
        Your response has been recorded successfully.
      </p>

      {bonusPence > 0 && (
        <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-4 max-w-md mx-auto mb-4">
          <p className="text-sm text-emerald-800 flex items-center justify-center gap-2">
            <Gift size={16} />
            <span>
              You described <strong>{totalEmailCount}</strong> emails (
              {bonusEmails} bonus). You earned a{" "}
              <strong>{formatBonus(bonusPence)} bonus</strong>! The bonus will
              be paid via Prolific.
            </span>
          </p>
        </div>
      )}

      <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 max-w-md mx-auto">
        <p className="text-sm text-slate-700">
          Your insights into workplace email patterns across different
          professions are invaluable to our research on email security. Thank
          you for contributing to this study.
        </p>
      </div>
      <p className="text-sm text-gray-400 mt-6">
        You may now close this window or return to Prolific.
      </p>
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

  let railTotal = 0;
  let railMandatory = MANDATORY_EMAILS;
  let railShowBonus = true;
  let railSenderBreakdown = null;
  if (showSideRailA) {
    railTotal = totalEmailCount;
    railSenderBreakdown = senders.filter(isSenderComplete).map((s) => ({
      role: s.role.trim(),
      emailCount: s.emails.filter(isEmailFilled).length,
    }));
  } else if (showSideRailB) {
    railTotal = committedGeneric;
    railMandatory = REQUIRED_GENERIC;
    railShowBonus = false;
  } else if (showSideRailC) {
    railTotal = committedSuspicious;
    railMandatory = REQUIRED_SUSPICIOUS;
    railShowBonus = false;
  }

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 py-8 px-4">
      <div className={`${containerWidth} mx-auto`}>
        {step === 0 && (
          <div className="text-center mb-6">
            <h1 className="text-2xl font-bold text-slate-800">
              {/* Research Study on Workplace Email Patterns */}
              Welcome to Workplace Email Patterns Survey
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
              />
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
