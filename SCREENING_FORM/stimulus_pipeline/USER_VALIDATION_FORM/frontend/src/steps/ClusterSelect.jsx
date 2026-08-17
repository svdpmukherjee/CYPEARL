import React, { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { rich, fmt, titleCase, clusterLabel } from "../content.jsx";
import { useReadingFocus } from "../readingFocus.js";

// Landing page. Job area selection AND the assigned-role familiarity (fit) gate
// now live here, on one page, so a poor-fit participant leaves before reading
// the instructions or consenting. Picking a job area first asks for the
// participant's OWN job role, then reveals the role assigned for that area
// (framed against their own role so the two are not confused), then asks how
// their own role compares to the assigned one, then the fit question. "Not
// familiar" shows a polite exit.
export default function ClusterSelect({
  content,
  selected,
  initialOwnRole,
  initialRelation,
  onNext,
}) {
  const t = content.clusterSelect;
  const rt = content.role;
  // "What we mean by a realistic email": the objective explainer shown just
  // above the fit question. Optional, so removing it from content.json simply
  // hides it rather than breaking the page.
  const explainer = rt.taskExplainer;

  const [clusters, setClusters] = useState([]);
  const [pick, setPick] = useState(selected);
  const [ownRole, setOwnRole] = useState(initialOwnRole || "");
  // the own-role field is confirmed before the assigned role is revealed, so
  // the participant reads their real role and the assigned one as two separate
  // things rather than everything appearing at once
  const [roleConfirmed, setRoleConfirmed] = useState(Boolean(initialOwnRole));
  const [relation, setRelation] = useState(initialRelation || "");
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState("");
  const [phase, setPhase] = useState("select"); // "select" | "exit"

  useEffect(() => {
    api
      .clusters()
      .then(setClusters)
      .catch((e) => setErr(e.message))
      .finally(() => setLoading(false));
  }, []);

  // Changing the chosen job area changes the assigned role, so the
  // relative-position answer (which is about the assigned role) must reset.
  const choose = (cluster) => {
    setPick(cluster);
    setRelation("");
  };

  const chosen = clusters.find((c) => c.cluster === pick);
  // Role title and description come from the editable content file, keyed by the
  // job area. Fall back to the values the backend sends if a key is missing.
  const roleInfo = (chosen && content.roles && content.roles[chosen.cluster]) || {};
  const roleTitle = roleInfo.title || (chosen && chosen.recipientRole) || "";
  const roleDesc = roleInfo.description || (chosen && chosen.note) || "";
  // "You are a/an {role}". Pick the article from the role title's first letter.
  const article = /^[aeiou]/i.test(roleTitle) ? "an" : "a";
  const trimmedOwn = ownRole.trim();
  // shown with each word capitalised so it reads like the assigned-role titles;
  // the raw entry (trimmedOwn) is what we store
  const displayOwn = titleCase(trimmedOwn);

  // --- focus & scroll guidance ---------------------------------------------
  // As each section is revealed, bring it into view and move the cursor to the
  // next thing to do, so the participant never has to hunt or scroll manually.
  const ownRoleHeadingRef = useRef(null);
  const ownRoleInputRef = useRef(null);
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

  // A job area was picked: centre the "What is your job title?" field and put
  // the cursor in it.
  useEffect(() => {
    if (!mounted.current) return;
    if (pick) revealTo(ownRoleHeadingRef.current, "center", ownRoleInputRef.current);
  }, [pick]);

  // The own role was confirmed: bring the assigned-role heading to the top and
  // activate the comparison dropdown (it takes focus, showing its blue border).
  useEffect(() => {
    if (!mounted.current) return;
    if (roleConfirmed && trimmedOwn) {
      revealTo(assignedHeadingRef.current, "start", relationRef.current);
    }
  }, [roleConfirmed]);

  // A comparison was chosen: reveal the "Before you continue" fit gate. It is
  // brought to the TOP of the viewport (not centred) because it now carries the
  // objective explainer, which must be read whole before the fit question.
  useEffect(() => {
    if (!mounted.current) return;
    if (relation) revealTo(fitHeadingRef.current, "start");
  }, [relation]);

  // Mark mounted last so none of the effects above fire on the first render
  // (for example when the participant returns to this page via Back).
  useEffect(() => {
    mounted.current = true;
  }, []);

  // Fade the steps the participant is not on, so the eye lands on the current
  // one after each auto-scroll. Re-run as steps appear, since each reveal adds
  // another .focusblock to grade.
  const pageRef = useRef(null);
  useReadingFocus(pageRef, [pick, roleConfirmed, relation, phase, clusters.length]);

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

        <h2>{t.selectPrompt}</h2>
        <p className="muted">{rich(t.selectHelp)}</p>

        {loading && <p className="muted">{t.loading}</p>}
        {err && <p className="error">{err}</p>}

        <div className="clustergrid">
          {clusters.map((c) => (
            <button
              key={c.cluster}
              className={"clustercard" + (pick === c.cluster ? " on" : "")}
              onClick={() => choose(c.cluster)}
              type="button"
            >
              {/* Show only the job area. The specific role is revealed below,
                  once an area is chosen. */}
              <span className="cname">{clusterLabel(content, c.cluster)}</span>
            </button>
          ))}
        </div>
      </div>

      {chosen && (
        <div className="rolereveal">
          {/* Step 1: the participant's OWN job role, captured before the
              assigned role appears. */}
          <div className="focusblock">
            <h2 ref={ownRoleHeadingRef}>{t.ownRoleHeading}</h2>
            {/* Optional lines are rendered only when they hold text. An empty
                string in content.json would otherwise leave an empty paragraph
                or label behind, which reads as a gap in the spacing. */}
            {t.ownRoleHelp && <p className="muted">{rich(t.ownRoleHelp)}</p>}
            <div className="field">
              {t.ownRoleLabel && <label>{t.ownRoleLabel}</label>}
              <input
                ref={ownRoleInputRef}
                type="text"
                value={ownRole}
                placeholder={t.ownRolePlaceholder}
                onChange={(e) => setOwnRole(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && trimmedOwn) setRoleConfirmed(true);
                }}
              />
            </div>
            {!roleConfirmed && (
              <div className="navbar">
                <span />
                <button
                  className="btn primary"
                  disabled={!trimmedOwn}
                  onClick={() => setRoleConfirmed(true)}
                >
                  {t.ownRoleContinue}
                </button>
              </div>
            )}
          </div>

          {roleConfirmed && trimmedOwn && (
            <>
              {/* Step 2: the assigned role, framed explicitly against their own
                  role so the two do not get confused. */}
              <div className="rolestep focusblock">
                <h2 ref={assignedHeadingRef}>{rt.title}</h2>
                <p className="lead">{rich(rt.lead, { ownRole: displayOwn })}</p>

                <div className="rolepanel">
                  <div className="roletitle">
                    {rt.rolePrefix || "Imagine you are"} {article} {roleTitle}
                  </div>
                  {roleDesc && <p className="rolenote">{rich(roleDesc)}</p>}
                </div>

                {/* Step 3: how their own role sits relative to the assigned one. */}
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

              {/* Step 4: the familiarity (fit) gate, revealed once they have
                  placed their own role relative to the assigned one. */}
              {relation && (
                // The scroll target is the step itself, not its heading: this
                // step usually opens straight into the objective callout below,
                // with no heading of its own to aim at.
                <div className="rolestep focusblock" ref={fitHeadingRef}>
                  {rt.fitHeading && <h2>{rt.fitHeading}</h2>}
                  {rt.fitText && (
                    <p className="lead">{rich(rt.fitText, { role: roleTitle })}</p>
                  )}

                  {/* The single most important block on this page. A pilot
                      showed participants judged the email's content rather
                      than whether that sender, pressing that hard, with that
                      reward or consequence, is in a position to send such an
                      email to the role. The real task is spelled out here,
                      BEFORE consent, so anyone who does not want to make that
                      judgment can leave at no cost.
                      Structure: the contrast rows name the wrong question and
                      then the right one, and the three factors put the right
                      one in terms of the sender's standing. Both are phrased
                      as questions on purpose: a worked example ("a junior
                      would not warn you about a consequence") would hand
                      participants a rule and stop them judging the
                      combination, which is the measurement. Do not add
                      examples here. */}
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
                        {/* The wrong question named first, the right one
                            second. Pilot participants arrived already reading
                            for content, so the misreading has to be
                            interrupted before the three factors below make
                            sense as questions about the sender. */}
                        {Array.isArray(explainer.contrast) &&
                          explainer.contrast.length > 0 && (
                            <div className="contrast">
                              {explainer.contrast.map((row, i) => (
                                <div key={i} className="contrastrow">
                                  <span className="contrastlabel">
                                    {row.label}
                                  </span>
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
                          cluster: chosen.cluster,
                          recipientRole: chosen.recipientRole,
                          note: chosen.note,
                          ownRole: trimmedOwn,
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
            </>
          )}
        </div>
      )}
    </div>
  );
}
