import React, { useState } from "react";
import { rich } from "../content.jsx";

// The first page of the study. It is only reached by participants who arrived
// without the ?PROLIFIC_PID= parameter Prolific normally appends, since App
// resolves that automatically and skips straight past this page.
//
// Submitting looks the ID up in the invitation roster, which is a network call,
// so the button reports that it is working and refuses a second click.
//
// Forward only: the study has no Back control on any page, so an answer cannot
// be revisited once the participant has moved past it.
export default function ProlificId({ content, value, onNext }) {
  const t = content.prolific;
  const [pid, setPid] = useState(value || "");
  const [busy, setBusy] = useState(false);
  const clean = pid.trim();
  // Prolific IDs are 24-character alphanumeric strings; we accept a little
  // slack but require a plausible length so typos are caught early.
  const valid = /^[A-Za-z0-9]{6,40}$/.test(clean);

  return (
    <div className="card wide">
      <h1>{t.title}</h1>
      <p className="lead">{rich(t.lead)}</p>

      <div className="field">
        <label htmlFor="pid">{t.fieldLabel}</label>
        <input
          id="pid"
          type="text"
          autoComplete="off"
          spellCheck={false}
          placeholder={t.placeholder}
          value={pid}
          onChange={(e) => setPid(e.target.value)}
        />
        {!valid && clean.length > 0 && (
          <span className="hint error">{t.invalidHint}</span>
        )}
      </div>

      <div className="navbar">
        {/* empty span so the single control keeps its place on the right */}
        <span />
        <button
          className="btn primary"
          disabled={!valid || busy}
          onClick={async () => {
            setBusy(true);
            try {
              await onNext(clean);
            } finally {
              setBusy(false);
            }
          }}
        >
          {busy ? t.checkingButton || t.continueButton : t.continueButton}
        </button>
      </div>
    </div>
  );
}
