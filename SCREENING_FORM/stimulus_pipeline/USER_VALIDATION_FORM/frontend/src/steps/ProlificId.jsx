import React, { useState } from "react";
import { rich } from "../content.jsx";

export default function ProlificId({ content, value, onBack, onNext }) {
  const t = content.prolific;
  const [pid, setPid] = useState(value || "");
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
        <button className="btn" onClick={onBack}>
          {t.backButton}
        </button>
        <button
          className="btn primary"
          disabled={!valid}
          onClick={() => onNext(clean)}
        >
          {t.continueButton}
        </button>
      </div>
    </div>
  );
}
