import { useState } from "react";
import { CheckCircle } from "lucide-react";

export default function CompletedPartSummary({
  title,
  headline,
  items,
  defaultOpen = false,
}) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="bg-gray-800 rounded-lg p-3 shadow-sm mt-3">
      <button
        type="button"
        onClick={() => setOpen(!open)}
        className="w-full flex items-center justify-between gap-2 text-left cursor-pointer"
      >
        <div className="flex items-center gap-2 min-w-0">
          <CheckCircle size={14} className="text-emerald-400 shrink-0" />
          <div className="min-w-0">
            <p className="text-[11px] uppercase tracking-wider text-slate-400">
              {title}
            </p>
            <p className="text-xs text-slate-200 truncate">{headline}</p>
          </div>
        </div>
        <span
          className={`text-slate-400 text-xs shrink-0 transition-transform ${
            open ? "rotate-90" : ""
          }`}
        >
          ▶
        </span>
      </button>
      {open && items.length > 0 && (
        <ul className="mt-2 space-y-1 max-h-40 overflow-y-auto pr-1 border-t border-gray-700 pt-2">
          {items.map((it, i) => (
            <li
              key={i}
              className="flex items-baseline justify-between text-[11px] text-slate-300 gap-2"
            >
              <span className="truncate" title={it.label}>
                {i + 1}. {it.label}
              </span>
              {it.suffix && (
                <span className="text-slate-400 shrink-0">{it.suffix}</span>
              )}
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
