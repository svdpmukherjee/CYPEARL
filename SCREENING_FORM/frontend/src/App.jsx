import { useState, useMemo } from "react";
import axios from "axios";
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
const BONUS_PER_EMAIL_PENCE = 2; // £0.02 per additional email beyond mandatory

// ── Helpers ─────────────────────────────────────────────────────────────

function isEmailFilled(e) {
  return !!(e.subject?.trim() || e.content?.trim());
}

function countFilledEmails(senders) {
  return senders.reduce(
    (sum, s) => sum + s.emails.filter(isEmailFilled).length,
    0,
  );
}

function isSenderComplete(sender) {
  return (
    sender.role.trim() && sender.type && sender.emails.some(isEmailFilled)
  );
}

function formatBonus(pence) {
  if (pence <= 0) return "£0.00";
  return `£${(pence / 100).toFixed(2)}`;
}

// ── Step Indicator ──────────────────────────────────────────────────────

function StepIndicator({ current, total }) {
  return (
    <div className="flex items-center justify-center gap-2 mb-8">
      {Array.from({ length: total }, (_, i) => (
        <div key={i} className="flex items-center gap-2">
          <div
            className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-semibold transition-colors ${
              i < current
                ? "bg-emerald-500 text-white"
                : i === current
                  ? "bg-slate-800 text-white"
                  : "bg-gray-200 text-gray-500"
            }`}
          >
            {i < current ? <CheckCircle size={16} /> : i + 1}
          </div>
          {i < total - 1 && (
            <div
              className={`w-8 h-0.5 ${i < current ? "bg-emerald-400" : "bg-gray-200"}`}
            />
          )}
        </div>
      ))}
    </div>
  );
}

// ── Dynamic text-box list ───────────────────────────────────────────────

function TextBoxList({ items, onChange, placeholder, minBoxes = 1 }) {
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
          <span className="text-xs text-gray-400 w-5 text-right shrink-0">
            {i + 1}.
          </span>
          <input
            type="text"
            value={val}
            onChange={(e) => updateItem(i, e.target.value)}
            placeholder={placeholder}
            className="flex-1 border border-gray-300 rounded-md p-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400"
          />
          {items.length > minBoxes && (
            <button
              type="button"
              onClick={() => removeBox(i)}
              className="p-1 text-gray-400 hover:text-red-500 transition-colors"
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
  index,
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
      <div className="bg-slate-50 px-3 py-2 flex items-center justify-between border-b border-slate-200">
        <div className="flex items-center gap-2 text-xs text-slate-500 min-w-0">
          <div
            className={`w-5 h-5 rounded-full flex items-center justify-center shrink-0 ${
              senderType === "internal"
                ? "bg-indigo-100 text-indigo-600"
                : senderType === "external"
                  ? "bg-amber-100 text-amber-600"
                  : "bg-gray-100 text-gray-400"
            }`}
          >
            <User size={10} />
          </div>
          <span className="truncate font-medium text-slate-600">
            {senderRole || "Sender"} — Email {index + 1}
          </span>
          {senderType && (
            <span
              className={`px-1.5 py-0.5 rounded text-[10px] font-medium uppercase tracking-wide ${
                senderType === "internal"
                  ? "bg-indigo-100 text-indigo-600"
                  : "bg-amber-100 text-amber-600"
              }`}
            >
              {senderType}
            </span>
          )}
        </div>
        {canRemove && (
          <button
            type="button"
            onClick={onRemove}
            className="p-0.5 text-gray-400 hover:text-red-500 transition-colors"
            title="Remove this email"
          >
            <X size={12} />
          </button>
        )}
      </div>

      {/* Email body */}
      <div className="bg-white px-3 py-2.5 space-y-2">
        <div className="flex items-center gap-2">
          <span
            className={`text-[10px] font-semibold uppercase tracking-wider w-14 shrink-0 ${
              subjectMissing ? "text-red-500" : "text-slate-400"
            }`}
          >
            Subject
          </span>
          <input
            type="text"
            value={email.subject}
            onChange={(e) => onChange({ ...email, subject: e.target.value })}
            placeholder="What does the subject line usually say?"
            className={`flex-1 border-0 border-b p-1 text-sm font-medium text-slate-700 focus:outline-none focus:border-blue-400 bg-transparent placeholder:text-slate-300 ${
              subjectMissing ? "border-red-400" : "border-slate-200"
            }`}
          />
        </div>
        <div className="flex items-start gap-2">
          <span
            className={`text-[10px] font-semibold uppercase tracking-wider w-14 shrink-0 mt-1.5 ${
              contentMissing ? "text-red-500" : "text-slate-400"
            }`}
          >
            About
          </span>
          <textarea
            value={email.content}
            onChange={(e) => onChange({ ...email, content: e.target.value })}
            placeholder="In a sentence, what is this email typically about?"
            rows={2}
            className={`flex-1 border-0 border-b p-1 text-sm text-slate-600 focus:outline-none focus:border-blue-400 bg-transparent resize-none placeholder:text-slate-300 ${
              contentMissing ? "border-red-400" : "border-slate-200"
            }`}
          />
        </div>

        {/* Frequency */}
        <div className="pt-1">
          <p
            className={`text-[11px] font-medium mb-1.5 ${
              frequencyMissing ? "text-red-500" : "text-slate-500"
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
                    ? "bg-slate-800 text-white border-slate-800 shadow-sm"
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

// ── Gamified Reward Bar (sticky) ────────────────────────────────────────

function RewardBar({ totalEmails }) {
  const mandatoryDone = Math.min(totalEmails, MANDATORY_EMAILS);
  const bonusEmails = Math.max(0, totalEmails - MANDATORY_EMAILS);
  const bonusPence = bonusEmails * BONUS_PER_EMAIL_PENCE;
  const mandatoryComplete = totalEmails >= MANDATORY_EMAILS;

  // Progress as percentage — mandatory fills 0-60%, bonus extends beyond
  const mandatoryPct = Math.round((mandatoryDone / MANDATORY_EMAILS) * 60);
  const bonusPct = Math.min(40, bonusEmails * 4); // each bonus email adds 4%, cap at 100%
  const totalPct = Math.min(100, mandatoryPct + bonusPct);

  return (
    <div className="sticky top-0 z-10 bg-white border border-slate-200 rounded-lg p-3 mb-5 shadow-sm">
      {/* Top row: counts and bonus */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-1.5 text-sm">
          <Mail size={14} className="text-slate-500" />
          <span className="text-slate-700 font-semibold">
            {totalEmails}
          </span>
          <span className="text-slate-500 text-xs">
            {totalEmails === 1 ? "email" : "emails"} described
          </span>
          {!mandatoryComplete && (
            <span className="text-xs text-slate-400 ml-1">
              ({MANDATORY_EMAILS - mandatoryDone} more required)
            </span>
          )}
        </div>

        {mandatoryComplete && bonusEmails > 0 && (
          <div className="flex items-center gap-1.5 text-sm font-bold text-emerald-600 animate-pulse">
            <Gift size={14} />
            <span>Bonus: {formatBonus(bonusPence)}</span>
          </div>
        )}
      </div>

      {/* Progress bar — two-tone: blue for mandatory, emerald for bonus */}
      <div className="w-full bg-gray-100 rounded-full h-3 overflow-hidden relative">
        {/* Mandatory segment */}
        <div
          className="absolute top-0 left-0 h-full rounded-full transition-all duration-500 ease-out bg-gradient-to-r from-blue-400 to-blue-500"
          style={{ width: `${mandatoryPct}%` }}
        />
        {/* Bonus segment */}
        {bonusEmails > 0 && (
          <div
            className="absolute top-0 h-full rounded-r-full transition-all duration-500 ease-out bg-gradient-to-r from-emerald-400 to-emerald-500"
            style={{ left: `${mandatoryPct}%`, width: `${bonusPct}%` }}
          />
        )}
        {/* Mandatory/bonus boundary marker */}
        <div
          className="absolute top-0 h-full w-0.5 bg-white"
          style={{ left: "60%" }}
        />
      </div>

      {/* Bottom row: motivational text */}
      <div className="flex items-center justify-between mt-1.5">
        <div className="flex gap-3 text-[10px] text-slate-400">
          <span>
            <span className="inline-block w-2 h-2 rounded-full bg-blue-500 mr-1" />
            Required ({mandatoryDone}/{MANDATORY_EMAILS})
          </span>
          <span>
            <span className="inline-block w-2 h-2 rounded-full bg-emerald-500 mr-1" />
            Bonus ({bonusEmails})
          </span>
        </div>
        {mandatoryComplete ? (
          <p className="text-[10px] text-emerald-600 font-medium">
            Each additional email earns you{" "}
            {formatBonus(BONUS_PER_EMAIL_PENCE)} bonus!
          </p>
        ) : (
          <p className="text-[10px] text-slate-400">
            Complete {MANDATORY_EMAILS} emails to unlock bonus rewards
          </p>
        )}
      </div>
    </div>
  );
}

// ═════════════════════════════════════════════════════════════════════════
// MAIN APP
// ═════════════════════════════════════════════════════════════════════════

export default function App() {
  const [step, setStep] = useState(0);
  const [senderPageIdx, setSenderPageIdx] = useState(0);
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

  // Step 3 — hard-to-judge emails
  const [suspiciousEmails, setSuspiciousEmails] = useState(
    Array(3).fill(""),
  );

  // ── Computed ──────────────────────────────────────────────────────────

  const totalEmailCount = useMemo(
    () => countFilledEmails(senders),
    [senders],
  );

  const bonusEmails = Math.max(0, totalEmailCount - MANDATORY_EMAILS);
  const bonusPence = bonusEmails * BONUS_PER_EMAIL_PENCE;

  const filledEntries = (arr) => arr.filter((s) => s.trim().length > 0);

  const completedSenders = useMemo(
    () => senders.filter(isSenderComplete),
    [senders],
  );

  // ── Current sender helpers ────────────────────────────────────────────

  const currentSender = senders[senderPageIdx] || EMPTY_SENDER();
  const isLastSenderPage = senderPageIdx === senders.length - 1;

  const updateCurrentSender = (field, value) => {
    const updated = [...senders];
    updated[senderPageIdx] = { ...updated[senderPageIdx], [field]: value };
    setSenders(updated);
  };

  const addEmailToCurrentSender = () => {
    const updated = [...senders];
    updated[senderPageIdx] = {
      ...updated[senderPageIdx],
      emails: [
        ...updated[senderPageIdx].emails,
        { subject: "", content: "", frequency: "" },
      ],
    };
    setSenders(updated);
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
    const newSenders = [...senders, EMPTY_SENDER()];
    setSenders(newSenders);
    setSenderPageIdx(newSenders.length - 1);
  };

  // ── Step 2 navigation ────────────────────────────────────────────────

  const goNextFromStep2 = () => {
    if (isLastSenderPage) {
      setStep(3);
    } else {
      setSenderPageIdx(senderPageIdx + 1);
    }
  };

  const goPrevFromStep2 = () => {
    setShowValidation(false);
    if (senderPageIdx > 0) {
      setSenderPageIdx(senderPageIdx - 1);
    } else {
      setStep(1);
    }
  };

  const canLeaveStep2 = totalEmailCount >= MANDATORY_EMAILS;

  // ── Page-level validation (all fields required) ───────────────────────

  const getPageErrors = (sender) => {
    const errors = [];
    if (!sender.role.trim()) errors.push("Job role or title of the sender");
    if (!sender.type) errors.push("Inside or outside your organisation");
    sender.emails.forEach((email, i) => {
      const tag = sender.emails.length > 1 ? ` (Email ${i + 1})` : "";
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
    if (step === 0) return prolificId.trim().length > 0;
    if (step === 1) return jobCluster && jobTitle.trim() && dailyTasks.trim();
    if (step === 3) return filledEntries(suspiciousEmails).length >= 1;
    return true;
  };

  // ── Submit ────────────────────────────────────────────────────────────

  const handleSubmit = async () => {
    setSubmitting(true);
    setError("");
    try {
      const cleanedSenders = senders
        .filter(isSenderComplete)
        .map((s) => ({
          role: s.role.trim(),
          type: s.type,
          emails: s.emails
            .filter(isEmailFilled)
            .map((e) => ({
              subject: e.subject.trim(),
              content: e.content.trim(),
              frequency: e.frequency || "",
            })),
        }));

      await axios.post(`${API_URL}/submit`, {
        prolific_id: prolificId.trim(),
        job_cluster: jobCluster,
        job_title: jobTitle.trim(),
        daily_tasks: dailyTasks.trim(),
        email_senders: cleanedSenders,
        suspicious_emails: filledEntries(suspiciousEmails),
      });
      setStep(4);
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

  const renderStep0 = () => (
    <div>
      <div className="flex items-center gap-3 mb-2">
        <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center">
          <ClipboardList size={20} className="text-white" />
        </div>
        <h2 className="text-xl font-bold text-slate-800">
          Welcome — Workplace Email Patterns Survey
        </h2>
      </div>
      <p className="text-gray-600 text-sm mb-6 leading-relaxed">
        Thank you for participating in this research study on{" "}
        <strong>workplace email patterns</strong>. This short survey
        (approx. 3–4 minutes) helps us understand the types of emails
        professionals in different roles receive at work.
      </p>

      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
        <p className="text-sm text-blue-800">
          <strong>Why does this matter?</strong> Your answers help researchers
          understand how workplace email landscapes differ across professions.
          The more accurately you describe the emails you receive, the more
          valuable your contribution will be. Please draw from your actual
          workplace experience.
        </p>
      </div>

      {/* Anonymity notice */}
      <div className="bg-slate-50 border border-slate-200 rounded-lg p-4 mb-4">
        <p className="text-sm text-slate-700">
          <strong>Your privacy is protected.</strong> Your identity will be{" "}
          <strong>strictly anonymised</strong> — there is no way for anyone to
          identify you from your responses. Please{" "}
          <strong>do not mention any personal names</strong> (yours or anyone
          else&apos;s) in any of your answers. We are only interested in job
          roles, email types, and general workplace patterns.
        </p>
      </div>

      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 mb-6">
        <p className="text-sm text-gray-700 mb-2">
          <strong>Estimated time:</strong> 3–4 minutes &nbsp;|&nbsp;{" "}
          <strong>Base compensation:</strong> £0.50
        </p>
        <div className="flex items-start gap-2 mt-2 pt-2 border-t border-gray-200">
          <Gift size={16} className="text-emerald-600 mt-0.5 shrink-0" />
          <div className="text-sm text-gray-700">
            <strong>Earn a bonus</strong> for sharing more detail about your
            inbox. You will describe at least {MANDATORY_EMAILS} emails
            (required). Each additional email beyond {MANDATORY_EMAILS} earns
            you{" "}
            <strong className="text-emerald-600">
              {formatBonus(BONUS_PER_EMAIL_PENCE)}
            </strong>{" "}
            bonus (paid via Prolific).
          </div>
        </div>
      </div>

      <label className="block text-sm font-semibold text-slate-700 mb-1">
        Prolific ID
      </label>
      <input
        type="text"
        value={prolificId}
        onChange={(e) => setProlificId(e.target.value)}
        placeholder="Enter your Prolific ID"
        className="w-full border border-gray-300 rounded-md p-2.5 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 mb-2"
      />
      <p className="text-xs text-gray-400">
        Please double-check your Prolific ID so we can verify your submission.
      </p>
    </div>
  );

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
    const sender = currentSender;
    const isMandatoryPage = senderPageIdx < MANDATORY_EMAILS;
    const pageLabel = isMandatoryPage
      ? `Email ${senderPageIdx + 1} of ${MANDATORY_EMAILS} (required)`
      : `Bonus email ${senderPageIdx + 1}`;

    const pageErrors = showValidation ? getPageErrors(sender) : [];
    const roleMissing = showValidation && !sender.role.trim();
    const typeMissing = showValidation && !sender.type;

    return (
      <div>
        <div className="flex items-center gap-3 mb-2">
          <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center">
            <Inbox size={20} className="text-white" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-slate-800">
              Your Email Landscape
            </h2>
            <p className="text-xs text-slate-500">{pageLabel}</p>
          </div>
        </div>

        <p className="text-gray-600 text-sm mb-4">
          Think about your <strong>workplace inbox</strong> over a typical week.
          Describe the emails you regularly receive — the job role of the sender,
          whether they are inside or outside your organisation, and what
          the email is typically about.
        </p>

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

        {/* Gamified reward bar */}
        <RewardBar totalEmails={totalEmailCount} />

        {/* Page dots */}
        <div className="flex items-center gap-1.5 mb-5">
          {senders.map((s, i) => {
            const filled = isSenderComplete(s);
            const isCurrent = i === senderPageIdx;
            return (
              <button
                key={i}
                type="button"
                onClick={() => {
                  setShowValidation(false);
                  setSenderPageIdx(i);
                }}
                className={`w-7 h-7 rounded-full text-[10px] font-bold transition-all border-2 ${
                  isCurrent
                    ? "border-slate-800 bg-slate-800 text-white scale-110"
                    : filled
                      ? "border-emerald-500 bg-emerald-50 text-emerald-700"
                      : "border-gray-200 bg-white text-gray-400 hover:border-gray-400"
                }`}
                title={`Email page ${i + 1}`}
              >
                {filled ? <CheckCircle size={12} className="mx-auto" /> : i + 1}
              </button>
            );
          })}
        </div>

        {/* ── Sender info ─────────────────────────────────────────────── */}
        <div className="border border-gray-200 rounded-xl overflow-hidden mb-4">
          <div className="bg-gray-50 p-4">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-full bg-slate-200 flex items-center justify-center shrink-0">
                <User size={14} className="text-slate-500" />
              </div>
              <div className="flex-1 space-y-2.5">
                {/* Sender role */}
                <div>
                  <p
                    className={`text-[11px] font-medium mb-1 ${
                      roleMissing ? "text-red-500" : "text-slate-500"
                    }`}
                  >
                    What is the job role or title of someone who regularly emails
                    you at work?
                    {roleMissing && " *"}
                  </p>
                  <input
                    type="text"
                    value={sender.role}
                    onChange={(e) => updateCurrentSender("role", e.target.value)}
                    placeholder="Enter a job role or title..."
                    className={`w-full border rounded-md p-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-400 ${
                      roleMissing
                        ? "border-red-400 bg-red-50/50"
                        : "border-gray-300"
                    }`}
                  />
                </div>

                {/* Internal / External */}
                <div>
                  <p
                    className={`text-[11px] font-medium mb-1.5 ${
                      typeMissing ? "text-red-500" : "text-slate-500"
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
                          ? "bg-amber-600 text-white border-amber-600 shadow-sm"
                          : typeMissing
                            ? "bg-white text-red-400 border-red-300 hover:border-amber-300 hover:text-amber-600"
                            : "bg-white text-slate-500 border-slate-200 hover:border-amber-300 hover:text-amber-600"
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
          <div className="p-4 pt-3 space-y-3 bg-slate-50/50">
            <p className="text-xs text-gray-500">
              What kinds of emails does{" "}
              <strong className="text-slate-600">
                {sender.role || "this role"}
              </strong>{" "}
              typically send you?
            </p>
            {sender.emails.map((email, emailIdx) => (
              <EmailCard
                key={emailIdx}
                email={email}
                senderRole={sender.role}
                senderType={sender.type}
                onChange={(updated) => updateEmail(emailIdx, updated)}
                onRemove={() => removeEmail(emailIdx)}
                canRemove={sender.emails.length > 1}
                index={emailIdx}
                showValidation={showValidation}
              />
            ))}
            <button
              type="button"
              onClick={addEmailToCurrentSender}
              className="flex items-center gap-1.5 text-xs text-blue-600 hover:text-blue-800 transition-colors group"
            >
              <div className="w-5 h-5 rounded-full border border-blue-300 group-hover:border-blue-500 flex items-center justify-center transition-colors">
                <Plus size={10} />
              </div>
              This role sends me other types of emails too
              {totalEmailCount >= MANDATORY_EMAILS && (
                <span className="text-emerald-600 font-semibold ml-1">
                  (+{formatBonus(BONUS_PER_EMAIL_PENCE)} bonus)
                </span>
              )}
            </button>
          </div>
        </div>

        {/* ── Page navigation ─────────────────────────────────────────── */}
        <div className="flex items-center justify-between mt-6">
          {/* Back */}
          <button
            type="button"
            onClick={goPrevFromStep2}
            className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-slate-800 transition-colors"
          >
            <ArrowLeft size={16} />{" "}
            {senderPageIdx === 0 ? "Back to Job Role" : "Previous email"}
          </button>

          <div className="flex items-center gap-3">
            {/* Add another sender — visible on last page only */}
            {isLastSenderPage && (
              <button
                type="button"
                onClick={addNewSenderPage}
                className="flex items-center gap-1.5 text-xs text-blue-600 hover:text-blue-800 border border-blue-200 hover:border-blue-400 rounded-lg px-3 py-2 transition-all"
              >
                <Plus size={14} />
                Someone else emails me regularly too
                {totalEmailCount >= MANDATORY_EMAILS && (
                  <span className="text-emerald-600 font-semibold">
                    (+{formatBonus(BONUS_PER_EMAIL_PENCE)})
                  </span>
                )}
              </button>
            )}

            {/* Next / Proceed — both use tryNavigateForward */}
            <button
              type="button"
              onClick={tryNavigateForward}
              disabled={isLastSenderPage && !canLeaveStep2}
              className={`flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold transition-colors ${
                isLastSenderPage && !canLeaveStep2
                  ? "bg-gray-200 text-gray-400 cursor-not-allowed"
                  : "bg-slate-800 text-white hover:bg-slate-700 cursor-pointer"
              }`}
            >
              {isLastSenderPage
                ? "Next: Hard-to-Judge Emails"
                : "Next email"}{" "}
              <ArrowRight size={16} />
            </button>
          </div>
        </div>

        {!canLeaveStep2 && isLastSenderPage && (
          <p className="text-xs text-amber-600 mt-3 text-right">
            Describe at least {MANDATORY_EMAILS} emails in total to continue.
            You have {totalEmailCount} so far.
          </p>
        )}
      </div>
    );
  };

  // ═══════════════════════════════════════════════════════════════════
  // STEP 3 — Hard-to-judge emails
  // ═══════════════════════════════════════════════════════════════════

  const renderStep3 = () => (
    <div>
      <div className="flex items-center gap-3 mb-2">
        <div className="w-10 h-10 rounded-full bg-slate-800 flex items-center justify-center">
          <Mail size={20} className="text-white" />
        </div>
        <h2 className="text-xl font-bold text-slate-800">
          Hard-to-Judge Emails
        </h2>
      </div>
      <p className="text-gray-600 text-sm mb-6">
        Finally, think of work emails where you were{" "}
        <strong>unsure whether they were legitimate or suspicious</strong>.
        Describe the types of emails that were hardest to judge.
      </p>

      <label className="block text-md font-semibold text-slate-700 mb-2">
        Types of work emails you find hardest to judge as safe or risky
      </label>
      <p className="text-xs text-gray-500 mb-3">
        Think of specific <strong>types</strong> of emails you have received in
        your workplace inbox where you were unsure whether they were legitimate
        or suspicious. One per box. Do not mention any personal names.
      </p>
      <TextBoxList
        items={suspiciousEmails}
        onChange={setSuspiciousEmails}
        placeholder="Describe one type of email..."
        minBoxes={1}
      />
    </div>
  );

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

  const totalSteps = 4;

  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-gray-100 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        <div className="text-center mb-6">
          <h1 className="text-2xl font-bold text-slate-800">
            Research Study on Workplace Email Patterns
          </h1>
          <p className="text-gray-500 text-sm">
            Workplace Email Patterns Survey
          </p>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 md:p-8">
          {step < totalSteps && (
            <StepIndicator current={step} total={totalSteps} />
          )}

          {step === 0 && renderStep0()}
          {step === 1 && renderStep1()}
          {step === 2 && renderStep2()}
          {step === 3 && renderStep3()}
          {step === 4 && renderThankYou()}

          {error && (
            <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
              {error}
            </div>
          )}

          {/* Navigation for steps 0, 1, 3 — step 2 has its own nav */}
          {step < totalSteps && step !== 2 && (
            <div className="flex justify-between mt-8">
              {step > 0 ? (
                <button
                  onClick={() => {
                    if (step === 3) {
                      // Go back to last sender page
                      setStep(2);
                      setSenderPageIdx(senders.length - 1);
                    } else {
                      setStep(step - 1);
                    }
                  }}
                  className="flex items-center gap-1.5 text-sm text-slate-600 hover:text-slate-800 transition-colors"
                >
                  <ArrowLeft size={16} /> Back
                </button>
              ) : (
                <div />
              )}

              {step < 3 ? (
                <button
                  onClick={() => setStep(step + 1)}
                  disabled={!canAdvance()}
                  className={`flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold transition-colors cursor-pointer ${
                    canAdvance()
                      ? "bg-slate-800 text-white hover:bg-slate-700"
                      : "bg-gray-200 text-gray-400 cursor-not-allowed"
                  }`}
                >
                  Next <ArrowRight size={16} />
                </button>
              ) : (
                <button
                  onClick={handleSubmit}
                  disabled={!canAdvance() || submitting}
                  className={`flex items-center gap-1.5 px-5 py-2 rounded-lg text-sm font-semibold transition-colors cursor-pointer ${
                    canAdvance() && !submitting
                      ? "bg-emerald-600 text-white hover:bg-emerald-700"
                      : "bg-gray-200 text-gray-400 cursor-not-allowed"
                  }`}
                >
                  {submitting ? "Submitting..." : "Submit"}{" "}
                  {!submitting && <CheckCircle size={16} />}
                </button>
              )}
            </div>
          )}
        </div>

        <p className="text-center text-xs text-gray-400 mt-6">
          IRiSC Lab · University of Luxembourg <br />·{" "}
          <span className="font-bold text-gray-500">
            All responses are strictly anonymous and used solely for research
            purposes.
          </span>
        </p>
      </div>
    </div>
  );
}
