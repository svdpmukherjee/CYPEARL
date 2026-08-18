import React, { useEffect, useRef } from "react";
import { rich } from "../content.jsx";
import { useReadingFocus } from "../readingFocus.js";

// Page 1 of 5: what this survey is, what it pays, and who they are.
//
// It says NOTHING about what the follow-up study asks. That is page 3, and it
// sits there on purpose: the job areas on page 2 have to be claimed by somebody
// who does not yet know what we are recruiting for, or the claim stops being a
// report of their working life and becomes a guess at what gets them into a
// £4.00 study.
//
// This page carries the ONLY .payhighlight strip in the survey, and it shows
// what THIS screening survey pays. The follow-up study's own time and pay are
// stated in words on page 5. Two strips of tiles and nobody could tell which
// figure was theirs.
//
// The Prolific ID is TYPED, always. The study link carries no parameters at all
// (no PROLIFIC_PID, no STUDY_ID, no SESSION_ID), so nothing about a participant
// reaches this survey except what they enter on these five pages. A Prolific
// profile records job titles people set once and may never have updated, and in
// the first pilot recruiting against those stale titles is what let a
// Sales-screened participant fill a Customer Service place.
//
// The ID is still the one thing a typo can cost us a participant on, so it is
// checked against Prolific's format before Continue unlocks, and App.jsx looks
// it up on the way out of this page to catch somebody who has already answered.
export default function Intro({ content, draft, patch, checking, onNext }) {
  const t = content.intro;
  const tp = content.prolific;
  const tj = content.jobTitle;

  const pid = String(draft.prolificId || "").trim();
  // Prolific IDs are 24-character alphanumeric strings; we accept a little slack
  // but require a plausible length so typos are caught here rather than by a
  // failed allowlist upload weeks later.
  const pidOk = /^[A-Za-z0-9]{6,40}$/.test(pid);
  const trimmedTitle = draft.jobTitle.trim();
  const ready = pidOk && trimmedTitle.length > 0 && !checking;

  const pidRef = useRef(null);
  useEffect(() => {
    // Only reach for the cursor when the field is still empty, so returning to
    // page 1 with Back does not yank a reader past the intro they came back for.
    if (!draft.prolificId) pidRef.current?.focus({ preventScroll: true });
  }, []);

  const pageRef = useRef(null);
  useReadingFocus(pageRef, []);

  return (
    <div className="card wide" ref={pageRef}>
      <div className="focusblock">
        <h1>{t.title}</h1>
        <p className="lead">{rich(t.lead)}</p>

        <div className="payhighlight">
          <p className="payline">{rich(t.payNote)}</p>
          <div className="paystats">
            <div className="paystat">
              <span className="paystat-value">{t.timeValue}</span>
              <span className="paystat-label">{t.timeLabel}</span>
            </div>
            <div className="paystat">
              <span className="paystat-value">{t.payValue}</span>
              <span className="paystat-label">{t.payLabel}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="rolestep focusblock">
        <h2>{tp.heading}</h2>
        {tp.help && <p className="muted">{rich(tp.help)}</p>}
        <div className="field">
          {tp.label && <label htmlFor="pid">{tp.label}</label>}
          <input
            id="pid"
            ref={pidRef}
            type="text"
            autoComplete="off"
            spellCheck={false}
            placeholder={tp.placeholder}
            value={draft.prolificId}
            onChange={(e) => patch({ prolificId: e.target.value })}
          />
          {!pidOk && pid.length > 0 && (
            <span className="hint error">{tp.invalid}</span>
          )}
        </div>
      </div>

      <div className="rolestep focusblock">
        <h2>{tj.heading}</h2>
        {tj.help && <p className="muted">{rich(tj.help)}</p>}
        <div className="field">
          {tj.label && <label htmlFor="jobtitle">{tj.label}</label>}
          <input
            id="jobtitle"
            type="text"
            value={draft.jobTitle}
            placeholder={tj.placeholder}
            onChange={(e) => patch({ jobTitle: e.target.value })}
            onKeyDown={(e) => {
              if (e.key === "Enter" && ready) onNext();
            }}
          />
        </div>
      </div>

      <div className="navbar">
        {/* empty span so the single control keeps its place on the right */}
        <span />
        <button className="btn primary" disabled={!ready} onClick={onNext}>
          {t.continueButton}
        </button>
      </div>
    </div>
  );
}
