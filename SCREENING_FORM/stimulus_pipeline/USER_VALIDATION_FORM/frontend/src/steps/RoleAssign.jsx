import React, { useState } from "react";
import { rich } from "../content.jsx";

export default function RoleAssign({
  content,
  cluster,
  recipientRole,
  note,
  onBack,
  onNext,
}) {
  const t = content.role;
  // Role title and description come from the editable content file, keyed by
  // cluster. Fall back to the values passed from the backend if a cluster is
  // missing from the file.
  const roleInfo = (content.roles && content.roles[cluster]) || {};
  const title = roleInfo.title || recipientRole;
  const description = roleInfo.description || note;

  // "reading" shows the role plus the familiarity (fit) gate. "exit" shows the
  // polite way out for participants who do not know this kind of role.
  const [phase, setPhase] = useState("reading");

  if (phase === "exit") {
    return (
      <div className="card wide">
        <h1>{t.exitTitle}</h1>
        <p className="lead">{rich(t.exitText)}</p>
        <div className="notice">{rich(t.exitProlificNote)}</div>
        <div className="navbar">
          <a
            className="btn primary"
            href={t.exitProlificUrl}
            rel="noopener noreferrer"
          >
            {t.exitProlificButton}
          </a>
          <span />
        </div>
      </div>
    );
  }

  return (
    <div className="card wide">
      <h1>{t.title}</h1>
      <p className="lead">{rich(t.lead)}</p>

      <div className="rolepanel">
        <div className="roletitle">{title}</div>
        {description && <p className="rolenote">{rich(description)}</p>}
      </div>

      <p className="muted">{rich(t.guidance)}</p>

      <h2>{t.fitHeading}</h2>
      <p className="lead">{rich(t.fitText, { role: title })}</p>
      <p><b>{rich(t.fitQuestion)}</b></p>

      <div className="fitchoice">
        <button className="btn primary" onClick={onNext}>
          {t.fitConfirmButton}
        </button>
        <button className="btn subtle" onClick={() => setPhase("exit")}>
          {t.fitDeclineButton}
        </button>
      </div>

      <div className="navbar">
        <button className="btn" onClick={onBack}>
          {t.backButton}
        </button>
        <span />
      </div>
    </div>
  );
}
