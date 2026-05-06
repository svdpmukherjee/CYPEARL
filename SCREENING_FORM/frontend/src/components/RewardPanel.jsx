import { Mail, Gift } from "lucide-react";
import {
  MANDATORY_EMAILS,
  BONUS_PER_EMAIL_PENCE,
  MAX_BONUS_EMAILS,
} from "../constants";
import { formatBonus } from "../lib/helpers";

export default function RewardPanel({
  totalEmails,
  mandatory = MANDATORY_EMAILS,
  showBonus = true,
  senderBreakdown = null,
  showSenderCount = true,
  entriesLabel = "Entries",
}) {
  const mandatoryDone = Math.min(totalEmails, mandatory);
  const remaining = Math.max(0, mandatory - totalEmails);
  const bonusEmails = Math.max(0, totalEmails - mandatory);
  const maxBonusPence = Math.round(MAX_BONUS_EMAILS * 100);
  const bonusPence = Math.min(
    bonusEmails * BONUS_PER_EMAIL_PENCE,
    maxBonusPence,
  );
  const mandatoryComplete = totalEmails >= mandatory;
  const sendersDone = senderBreakdown ? senderBreakdown.length : 0;

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

      <div className="space-y-2.5 mb-4">
        {senderBreakdown && (
          <div className="bg-gray-700/60 rounded-md px-3 py-2">
            {showSenderCount ? (
              <div className="flex items-baseline justify-between">
                <span className="text-sm text-slate-300">Senders added</span>
                <span className="text-white font-bold text-sm">
                  {sendersDone}
                </span>
              </div>
            ) : (
              <div className="flex items-baseline justify-between">
                <span className="text-sm text-slate-300">{entriesLabel}</span>
                <span className="text-white font-bold text-sm">
                  {senderBreakdown.length}
                </span>
              </div>
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
                    {showSenderCount && (
                      <span className="text-slate-400 shrink-0">
                        {s.emailCount} email{s.emailCount === 1 ? "" : "s"}
                      </span>
                    )}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}
        <div className="bg-gray-700/60 rounded-md px-3 py-2">
          <div className="flex items-baseline justify-between">
            <span className="text-sm text-slate-300">
              {showSenderCount && senderBreakdown
                ? "Total emails"
                : "Required completed"}
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
        )}
      </div>

      <div className="w-full bg-gray-100 rounded-full h-2.5 overflow-hidden relative mb-2">
        <div
          className="absolute top-0 left-0 h-full transition-all duration-500 ease-out bg-gradient-to-r from-blue-400 to-blue-500"
          style={{ width: `${mandatoryPct}%` }}
        />
        {showBonus && bonusEmails > 0 && (
          <div
            className="absolute top-0 h-full transition-all duration-500 ease-out bg-gradient-to-r from-emerald-400 to-emerald-500"
            style={{ left: `${mandatoryPct}%`, width: `${bonusPct}%` }}
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
