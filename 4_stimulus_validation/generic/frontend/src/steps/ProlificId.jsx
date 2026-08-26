import React, { useState } from "react";
import { rich, titleCase, clusterLabel } from "../content.jsx";

// The first page of the study, and the one that opens it. It runs in two
// phases on a single page:
//
//   before the ID is matched  the returning-participant note, then the ID field
//   after                     the same note, the ID we matched, and the job area
//                             and job title they gave us in the screening survey
//
// Everybody sees it, including the participants who arrive from Prolific with
// ?PROLIFIC_PID= in the URL and have the ID filled in and checked for them. It
// would be easy to skip the page for them, and wrong: the returning-participant
// note is the reason this study looks different from the last one, and nearly
// everybody arrives with the parameter, so skipping would mean nearly nobody
// read it.
//
// Submitting looks the ID up in the invitation roster, which is a network call,
// so the button reports that it is working and refuses a second click.
//
// Forward only: the study has no Back control on any page, so an answer cannot
// be revisited once the participant has moved past it.
export default function ProlificId({
  content,
  value,
  cluster,
  ownRole,
  onNext,
  onConfirm,
}) {
  const t = content.prolific;
  const [pid, setPid] = useState(value || "");
  const [busy, setBusy] = useState(false);
  const clean = pid.trim();
  // Prolific IDs are 24-character alphanumeric strings; we accept a little
  // slack but require a plausible length so typos are caught early.
  const valid = /^[A-Za-z0-9]{6,40}$/.test(clean);

  // The job area only arrives once the ID has been matched against the roster,
  // so it is the signal that we are in the second phase of the page.
  const matched = Boolean(cluster);

  return (
    <div className="card wide">
      <h1>{t.title}</h1>

      {/* First, before anything is asked of them: they have already done a
          study for us, this one is separate, and taking it is their choice. */}
      <div className="notice">{rich(t.returningNote)}</div>

      {matched ? (
        <>
          <div className="field">
            <label htmlFor="pid">{t.idLabel || t.fieldLabel}</label>
            <input id="pid" type="text" value={value || clean} readOnly />
          </div>

          {/* Read only: the answers this participant gave in the screening
              survey, shown back so they can see we have the right person. */}
          <h2>{t.confirmHeading}</h2>
          <dl className="screenerecap">
            <div className="screenerow">
              <dt>{t.areaLabel}</dt>
              <dd>{clusterLabel(content, cluster)}</dd>
            </div>
            <div className="screenerow">
              <dt>{t.titleLabel}</dt>
              <dd>{titleCase(String(ownRole || "").trim())}</dd>
            </div>
          </dl>
          {t.confirmNote && <p className="muted">{rich(t.confirmNote)}</p>}

          <div className="navbar">
            <span />
            <button className="btn primary" onClick={onConfirm}>
              {t.verifyButton || t.continueButton}
            </button>
          </div>
        </>
      ) : (
        <>
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
        </>
      )}
    </div>
  );
}
