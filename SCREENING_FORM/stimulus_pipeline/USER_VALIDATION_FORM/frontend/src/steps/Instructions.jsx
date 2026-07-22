import React from "react";
import { rich } from "../content.jsx";

export default function Instructions({ content, consent, onConsent, onBack, onNext }) {
  const t = content.instructions;

  return (
    <div className="card wide">
      <h1>{t.title}</h1>
      <p className="lead">{rich(t.lead)}</p>

      <h2>{t.whatYouDoHeading}</h2>
      <ol className="steps">
        {t.whatYouDo.map((item, i) => (
          <li key={i}>{rich(item)}</li>
        ))}
      </ol>
      <div className="notice">{rich(t.compensation)}</div>

      <h2>{t.consentHeading}</h2>
      <div className="consentbox">
        <p>{rich(t.consentText)}</p>
        <label className="check">
          <input
            type="checkbox"
            checked={consent}
            onChange={(e) => onConsent(e.target.checked)}
          />
          <span>{rich(t.consentCheckbox)}</span>
        </label>
      </div>

      <div className="navbar">
        <button className="btn" onClick={onBack}>
          {t.backButton}
        </button>
        <button className="btn primary" disabled={!consent} onClick={onNext}>
          {t.consentButton}
        </button>
      </div>
    </div>
  );
}
