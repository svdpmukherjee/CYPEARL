import { Trash2 } from "lucide-react";
import { FREQUENCY_OPTIONS } from "../constants";

export default function EmailCard({
  email,
  senderRole,
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
      <div className="bg-slate-200 px-3 py-2 flex items-center justify-between border-b border-slate-300">
        <div className="flex items-center gap-2 text-xs text-slate-600 min-w-0">
          <strong>From</strong>
          <span className="truncate font-medium text-black">
            "{senderRole || "Sender"}"
          </span>
        </div>
        {canRemove && (
          <span className="relative group">
            <button
              type="button"
              onClick={onRemove}
              aria-label="Remove this email"
              className="p-0.5 text-slate-500 hover:text-red-500 transition-colors cursor-pointer"
            >
              <Trash2 size={16} />
            </button>
            <span className="pointer-events-none absolute right-0 top-full mt-1 z-10 whitespace-nowrap rounded bg-slate-800 px-2 py-1 text-[11px] font-medium text-white opacity-0 group-hover:opacity-100 transition-opacity duration-0">
              Remove this email
            </span>
          </span>
        )}
      </div>

      <div className="bg-slate-50 p-5 space-y-2">
        <div className="flex items-center gap-2 p-2">
          <span
            className={`text-xs font-medium w-16 shrink-0 text-right ${
              subjectMissing ? "text-red-500" : "text-slate-600"
            }`}
          >
            subject:
          </span>
          <input
            type="text"
            value={email.subject}
            onChange={(e) => onChange({ ...email, subject: e.target.value })}
            placeholder='e.g., "Q3 budget review — please confirm"'
            className={`flex-1 rounded-md p-1.5 text-sm font-medium text-slate-700 focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white placeholder:text-slate-400 ${
              subjectMissing ? "ring-2 ring-red-400" : "border border-slate-300"
            }`}
          />
        </div>
        <div className="flex items-start gap-2 p-2">
          <span
            className={`text-xs font-medium w-16 shrink-0 text-right mt-1.5 ${
              contentMissing ? "text-red-500" : "text-slate-600"
            }`}
          >
            about:
          </span>
          <textarea
            value={email.content}
            onChange={(e) => onChange({ ...email, content: e.target.value })}
            placeholder="One line on what this email is for — e.g., 'asks me to review and approve the quarterly budget'"
            rows={2}
            className={`flex-1 rounded-md p-1.5 text-sm text-slate-600 focus:outline-none focus:ring-2 focus:ring-blue-400 bg-white resize-none placeholder:text-slate-400 ${
              contentMissing ? "ring-2 ring-red-400" : "border border-slate-300"
            }`}
          />
        </div>
      </div>

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
