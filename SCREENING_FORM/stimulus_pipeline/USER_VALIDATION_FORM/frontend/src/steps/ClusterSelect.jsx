import React, { useEffect, useRef, useState } from "react";
import { api } from "../api.js";
import { rich, fmt, titleCase, clusterLabel } from "../content.jsx";

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

  // A comparison was chosen: reveal the "Before you continue" fit gate.
  useEffect(() => {
    if (!mounted.current) return;
    if (relation) revealTo(fitHeadingRef.current, "center");
  }, [relation]);

  // Mark mounted last so none of the effects above fire on the first render
  // (for example when the participant returns to this page via Back).
  useEffect(() => {
    mounted.current = true;
  }, []);

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
    <div className="card wide">
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

      {chosen && (
        <div className="rolereveal">
          {/* Step 1: the participant's OWN job role, captured before the
              assigned role appears. */}
          <h2 ref={ownRoleHeadingRef}>{t.ownRoleHeading}</h2>
          <p className="muted">{rich(t.ownRoleHelp)}</p>
          <div className="field">
            <label>{t.ownRoleLabel}</label>
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

          {roleConfirmed && trimmedOwn && (
            <>
              {/* Step 2: the assigned role, framed explicitly against their own
                  role so the two do not get confused. */}
              <div className="rolestep">
                <h2 ref={assignedHeadingRef}>{rt.title}</h2>
                <p className="lead">{rich(rt.lead, { ownRole: displayOwn })}</p>

                <div className="rolepanel">
                  <div className="roletitle">
                    {rt.rolePrefix || "You are"} {article} {roleTitle}
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
                <div className="rolestep">
                  <h2 ref={fitHeadingRef}>{rt.fitHeading}</h2>
                  <p className="lead">{rich(rt.fitText, { role: roleTitle })}</p>
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
