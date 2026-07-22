import React, { useState } from "react";
import { rich } from "../content.jsx";

export default function RecapName({ content, recipientRole, cluster, initialName, onBack, onNext }) {
  const t = content.recap;
  const roleInfo = (content.roles && content.roles[cluster]) || {};
  const role = roleInfo.title || recipientRole;

  const [name, setName] = useState(initialName || "");
  const [notOwn, setNotOwn] = useState(false);
  const clean = name.trim();
  // Names should be a single given name, letters only, not the participant's own.
  const looksLikeName = /^[A-Za-z][A-Za-z '-]{0,29}$/.test(clean);
  const ready = looksLikeName && notOwn;

  return (
    <div className="card wide">
      <h1>{t.pageTitle}</h1>

      <div className="recap">
        <h2>{t.recapHeading}</h2>
        <ul>
          {t.recapPoints.map((point, i) => (
            <li key={i}>{rich(point, { role })}</li>
          ))}
        </ul>
      </div>

      <h2>{t.nameHeading}</h2>
      <p className="lead">{rich(t.nameLead)}</p>

      <div className="warn">{rich(t.nameWarning)}</div>

      <div className="field">
        <label htmlFor="pname">{t.nameFieldLabel}</label>
        <input
          id="pname"
          type="text"
          autoComplete="off"
          spellCheck={false}
          placeholder={t.namePlaceholder}
          value={name}
          onChange={(e) => setName(e.target.value)}
        />
        {!looksLikeName && clean.length > 0 && (
          <span className="hint error">{t.nameInvalidHint}</span>
        )}
      </div>

      <label className="check">
        <input
          type="checkbox"
          checked={notOwn}
          onChange={(e) => setNotOwn(e.target.checked)}
        />
        <span>{rich(t.nameConfirmCheckbox)}</span>
      </label>

      <div className="navbar">
        <button className="btn" onClick={onBack}>
          {t.backButton}
        </button>
        <button
          className="btn primary"
          disabled={!ready}
          onClick={() => onNext(clean)}
        >
          {t.startButton}
        </button>
      </div>
    </div>
  );
}
