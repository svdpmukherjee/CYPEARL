import React, { useRef, useState } from "react";
import { rich, articleFor } from "../content.jsx";
import { useReadingFocus } from "../readingFocus.js";

// The second page: what this study is, and who they are in it. The returning
// participant note and the screening-survey recap are both on the page before
// this one, so this page opens with the study's title and its objective.
//
// It is the counterpart to the job-specific app's role page, and it is
// deliberately much shorter. These are the same people, and they read the long
// version there; repeating it spends attention they need for the emails. Three
// things that page has to do are not needed here:
//
//   1. The role is not assigned, it is carried over. They keep the role the
//      job-specific study gave them, so it is shown here rather than chosen.
//      Holding the recipient's standing constant across the two studies is what
//      makes the two sets of realism ratings comparable: "could a sender at
//      that level send me this" has to mean the same thing in both. The line
//      under the role is that role's own description, taken word for word from
//      the job-specific study, so they read exactly what they read there.
//   2. No "how does your own role compare to the assigned role" question. They
//      answered it in the job-specific study, and the answer has not changed.
//   3. No familiarity gate. That gate turned away anyone who could not judge a
//      role they had not held; these participants already passed it.
//
// This is also where the participant row is created, on Continue, so anyone who
// reads this page and decides against it leaves no trace.
export default function Landing({ content, role, roleDescription, onNext }) {
  const t = content.landing;
  const [busy, setBusy] = useState(false);
  const article = articleFor(role);

  const pageRef = useRef(null);
  useReadingFocus(pageRef, [busy]);

  return (
    <div className="card wide" ref={pageRef}>
      <div className="focusblock">
        <h1>{t.title}</h1>
        {/* vars passed even though the lead sits above the role panel and so names
            the role as "the role below": every other rich() call on this page takes
            them, and a copy edit that used {role} here would otherwise print it raw. */}
        <p className="lead">{rich(t.lead, { role })}</p>

        {/* The role they keep, carried over rather than assigned again. Only
            shown when we actually resolved one, so a participant whose role we
            cannot determine is never told to imagine being nobody. */}
        {role && (
          <>
            <h2>{t.roleHeading}</h2>
            <div className="rolepanel">
              <div className="roletitle">
                {t.rolePrefix} {article} {role}
              </div>
              {/* The role's own description from the job-specific study, and
                  nothing else. Do not add copy here. */}
              {roleDescription && (
                <p className="rolenote">{rich(roleDescription)}</p>
              )}
            </div>
          </>
        )}
      </div>

      <div className="focusblock">
        <section className="objective">
          <div className="objectiveband">{t.explainer.eyebrow}</div>
          <div className="objectivebody">
            {t.explainer.heading && (
              <h3 className="objectiveheading">{rich(t.explainer.heading)}</h3>
            )}
            <p className="objectivelead">{rich(t.explainer.lead, { role })}</p>
            {Array.isArray(t.explainer.contrast) && t.explainer.contrast.length > 0 && (
              <div className="contrast">
                {t.explainer.contrast.map((row, i) => (
                  <div key={i} className="contrastrow">
                    <span className="contrastlabel">{row.label}</span>
                    <p className="contrasttext">{rich(row.text, { role })}</p>
                  </div>
                ))}
              </div>
            )}
            {t.explainer.closing && (
              <p className="objectiveclosing">{rich(t.explainer.closing, { role })}</p>
            )}
            {t.explainer.note && (
              <p className="objectivenote">{rich(t.explainer.note, { role })}</p>
            )}
          </div>
        </section>

        <div className="navbar">
          <span />
          <button
            className="btn primary"
            disabled={busy}
            onClick={() => {
              setBusy(true);
              Promise.resolve(onNext()).finally(() => setBusy(false));
            }}
          >
            {busy ? t.continuingButton : t.continueButton}
          </button>
        </div>
      </div>
    </div>
  );
}
