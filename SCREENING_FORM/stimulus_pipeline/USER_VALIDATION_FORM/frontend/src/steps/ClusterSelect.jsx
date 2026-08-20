import React, { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { rich, fmt, titleCase, clusterLabel } from "../content.jsx";
import { useReadingFocus } from "../readingFocus.js";

// Landing page, shown once the Prolific ID has been matched to the roster.
//
// The study is invitation-only, and every invited participant already told us
// their job area and their job title in the screener app. So this page no
// longer ASKS for either: it shows both back to them as a confirmation, then
// reveals the role assigned for that area (framed against their own role so the
// two are not confused), asks how their own role compares to it, and runs the
// familiarity (fit) gate. "Not familiar" shows a polite exit, and because the
// participant row is only created when they confirm, declining leaves no data.
//
// Nothing on this page is editable. A participant whose title has drifted since
// the screener is caught by the fit gate below, which is the question that
// actually matters for the task.
export default function ClusterSelect({
  content,
  cluster,
  ownRole,
  initialRelation,
  onNext,
}) {
  const t = content.clusterSelect;
  const rt = content.role;
  // "What we mean by a realistic email": the objective explainer shown just
  // above the fit question. Optional, so removing it from content.json simply
  // hides it rather than breaking the page.
  const explainer = rt.taskExplainer;

  const [clusterInfo, setClusterInfo] = useState(null);
  // The confirmation panel is acknowledged before the assigned role is
  // revealed, so the participant reads their real role and the assigned one as
  // two separate things rather than everything appearing at once.
  const [confirmed, setConfirmed] = useState(Boolean(initialRelation));
  const [relation, setRelation] = useState(initialRelation || "");
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");
  const [phase, setPhase] = useState("select"); // "select" | "exit"

  // The assigned role for this job area. Only the participant's own cluster is
  // needed, so we pick it out of the list the backend already serves.
  useEffect(() => {
    api
      .clusters()
      .then((rows) => setClusterInfo(rows.find((c) => c.cluster === cluster) || null))
      .catch((e) => setErr(e.message))
      .finally(() => setLoading(false));
  }, [cluster]);

  // Role title and description come from the editable content file, keyed by the
  // job area. Fall back to the values the backend sends if a key is missing.
  const roleInfo = (content.roles && content.roles[cluster]) || {};
  const roleTitle = roleInfo.title || clusterInfo?.recipientRole || "";
  const roleDesc = roleInfo.description || clusterInfo?.note || "";
  // "You are a/an {role}". Pick the article from the role title's first letter.
  const article = /^[aeiou]/i.test(roleTitle) ? "an" : "a";
  // Shown with each word capitalised so it reads like the assigned-role titles;
  // the raw screener entry is what stays in the database.
  const displayOwn = titleCase(String(ownRole || "").trim());
  const areaLabel = clusterLabel(content, cluster);

  // --- focus & scroll guidance ---------------------------------------------
  // As each section is revealed, bring it into view and move the cursor to the
  // next thing to do, so the participant never has to hunt or scroll manually.
  const assignedHeadingRef = useRef(null);
  const relationRef = useRef(null);
  const fitHeadingRef = useRef(null);
  const mounted = useRef(false);

  const revealTo = (el, block, focusEl) => {
    if (!el) return;
    // wait one frame so the just-revealed section is laid out before we scroll
    requestAnimationFrame(() => {
      el.scrollIntoView({ behavior: "smooth", block });
      if (focusEl) focusEl.focus({ preventScroll: true });
    });
  };

  // The confirmation was acknowledged: bring the assigned-role heading to the
  // top and activate the comparison dropdown (it takes focus, showing its blue
  // border).
  useEffect(() => {
    if (!mounted.current) return;
    if (confirmed) revealTo(assignedHeadingRef.current, "start", relationRef.current);
  }, [confirmed]);

  // A comparison was chosen: reveal the "Before you continue" fit gate. It is
  // brought to the TOP of the viewport (not centred) because it now carries the
  // objective explainer, which must be read whole before the fit question.
  useEffect(() => {
    if (!mounted.current) return;
    if (relation) revealTo(fitHeadingRef.current, "start");
  }, [relation]);

  // Mark mounted last so none of the effects above fire on the first render.
  useEffect(() => {
    mounted.current = true;
  }, []);

  // Fade the steps the participant is not on, so the eye lands on the current
  // one after each auto-scroll. Re-run as steps appear, since each reveal adds
  // another .focusblock to grade.
  const pageRef = useRef(null);
  useReadingFocus(pageRef, [confirmed, relation, phase, Boolean(clusterInfo)]);

  // Polite exit for participants who do not know the assigned role well. No data
  // saved, no consent given, and no time spent on the instructions.
  if (phase === "exit") {
    return (
      <div className="card wide">
        <h1>{rt.exitTitle}</h1>
        <p className="lead">{rich(rt.exitText)}</p>
        <div className="notice">{rich(rt.exitProlificNote)}</div>
        <div className="navbar">
          <a
            className="btn primary"
            href={rt.exitProlificUrl}
            rel="noopener noreferrer"
          >
            {rt.exitProlificButton}
          </a>
          <span />
        </div>
      </div>
    );
  }

  return (
    // Each ".focusblock" below is one step of the page. They are siblings, never
    // nested, so the reading-focus fade grades them independently.
    <div className="card wide" ref={pageRef}>
      <div className="focusblock">
        <h1>{t.title}</h1>
        <p className="lead">{rich(t.lead)}</p>

        <h2>{t.confirmHeading}</h2>
        {t.confirmHelp && <p className="muted">{rich(t.confirmHelp)}</p>}

        {/* Read only. These are the answers this participant gave in the
            screening survey, and they are what the study was sent to them on
            the strength of, so the study shows them rather than asking again. */}
        <dl className="screenerecap">
          <div className="screenerow">
            <dt>{t.areaLabel}</dt>
            <dd>{areaLabel}</dd>
          </div>
          <div className="screenerow">
            <dt>{t.titleLabel}</dt>
            <dd>{displayOwn}</dd>
          </div>
        </dl>

        {t.confirmNote && <p className="muted">{rich(t.confirmNote)}</p>}

        {loading && <p className="muted">{t.loading}</p>}
        {err && <p className="error">{err}</p>}

        {!confirmed && (
          <div className="navbar">
            <span />
            <button
              className="btn primary"
              disabled={loading || !roleTitle}
              onClick={() => setConfirmed(true)}
            >
              {t.confirmButton}
            </button>
          </div>
        )}
      </div>

      {/* clusterInfo, not just roleTitle: the role title can come from
          content.json alone, but onNext below reads clusterInfo.recipientRole,
          so the gate must not render until the lookup has landed. */}
      {confirmed && clusterInfo && roleTitle && (
        <div className="rolereveal">
          {/* The assigned role, framed explicitly against their own role so the
              two do not get confused. This block does not carry .rolestep: the
              divider above it is drawn by .rolereveal, and both would stack two
              rules on top of each other. */}
          <div className="focusblock">
            <h2 ref={assignedHeadingRef}>{rt.title}</h2>
            <p className="lead">{rich(rt.lead, { ownRole: displayOwn })}</p>

            <div className="rolepanel">
              <div className="roletitle">
                {rt.rolePrefix || "Imagine you are"} {article} {roleTitle}
              </div>
              {roleDesc && <p className="rolenote">{rich(roleDesc)}</p>}
            </div>

            {/* How their own role sits relative to the assigned one. */}
            <h2>{rich(rt.relHeading, { role: roleTitle, ownRole: displayOwn })}</h2>
            {rt.relText && (
              <p className="lead">
                {rich(rt.relText, { role: roleTitle, ownRole: displayOwn })}
              </p>
            )}
            <select
              ref={relationRef}
              className="roleselect"
              value={relation}
              onChange={(e) => setRelation(e.target.value)}
            >
              <option value="" disabled>
                {rt.relPlaceholder}
              </option>
              {(rt.relOptions || []).map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {fmt(opt.label, { role: roleTitle })}
                </option>
              ))}
            </select>
          </div>

          {/* The familiarity (fit) gate, revealed once they have placed their
              own role relative to the assigned one. */}
          {relation && (
            // The scroll target is the step itself, not its heading: this step
            // usually opens straight into the objective callout below, with no
            // heading of its own to aim at.
            <div className="rolestep focusblock" ref={fitHeadingRef}>
              {rt.fitHeading && <h2>{rt.fitHeading}</h2>}
              {rt.fitText && (
                <p className="lead">{rich(rt.fitText, { role: roleTitle })}</p>
              )}

              {/* The single most important block on this page. A pilot showed
                  participants judged the email's content rather than whether
                  that sender, pressing that hard, with that reward or
                  consequence, is in a position to send such an email to the
                  role. The real task is spelled out here, BEFORE consent, so
                  anyone who does not want to make that judgment can leave at no
                  cost.
                  Structure: the contrast rows name the wrong question and then
                  the right one, and the three factors put the right one in
                  terms of the sender's standing. Both are phrased as questions
                  on purpose: a worked example ("a junior would not warn you
                  about a consequence") would hand participants a rule and stop
                  them judging the combination, which is the measurement. Do not
                  add examples here. */}
              {explainer && (
                <section className="objective">
                  <div className="objectiveband">{explainer.eyebrow}</div>
                  <div className="objectivebody">
                    {explainer.heading && (
                      <h3 className="objectiveheading">
                        {rich(explainer.heading, { role: roleTitle })}
                      </h3>
                    )}
                    <p className="objectivelead">
                      {rich(explainer.lead, { role: roleTitle })}
                    </p>
                    {/* The wrong question named first, the right one second.
                        Pilot participants arrived already reading for content,
                        so the misreading has to be interrupted before the three
                        factors below make sense as questions about the
                        sender. */}
                    {Array.isArray(explainer.contrast) &&
                      explainer.contrast.length > 0 && (
                        <div className="contrast">
                          {explainer.contrast.map((row, i) => (
                            <div key={i} className="contrastrow">
                              <span className="contrastlabel">{row.label}</span>
                              <p className="contrasttext">
                                {rich(row.text, { role: roleTitle })}
                              </p>
                            </div>
                          ))}
                        </div>
                      )}
                    <div className="weighgrid">
                      {(explainer.factors || []).map((f, i) => (
                        <div key={i} className="weighcard">
                          <span className="weighlabel">
                            {fmt(f.label, { role: roleTitle })}
                          </span>
                          <p className="weighq">
                            {rich(f.text, { role: roleTitle })}
                          </p>
                        </div>
                      ))}
                    </div>
                    {explainer.closing && (
                      <p className="objectiveclosing">
                        {rich(explainer.closing, { role: roleTitle })}
                      </p>
                    )}
                    {explainer.note && (
                      <p className="objectivenote">
                        {rich(explainer.note, { role: roleTitle })}
                      </p>
                    )}
                  </div>
                </section>
              )}

              {rt.fitTail && (
                <p className="lead">{rich(rt.fitTail, { role: roleTitle })}</p>
              )}
              <p>
                <b>{rich(rt.fitQuestion)}</b>
              </p>

              <div className="fitchoice">
                <button
                  className="btn primary"
                  onClick={() =>
                    onNext({
                      recipientRole: clusterInfo.recipientRole,
                      note: clusterInfo.note,
                      roleRelation: relation,
                    })
                  }
                >
                  {rt.fitConfirmButton}
                </button>
                <button className="btn subtle" onClick={() => setPhase("exit")}>
                  {rt.fitDeclineButton}
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
