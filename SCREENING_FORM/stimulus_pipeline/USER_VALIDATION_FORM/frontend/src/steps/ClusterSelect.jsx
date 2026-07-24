import React, { useEffect, useState } from "react";
import { api } from "../api.js";
import { rich, clusterLabel } from "../content.jsx";

// Landing page. Job area selection AND the assigned-role familiarity (fit) gate
// now live here, on one page, so a poor-fit participant leaves before reading
// the instructions or consenting. Picking a job area reveals the role assigned
// for that area, then the fit question. "Not familiar" shows a polite exit.
export default function ClusterSelect({ content, selected, onNext }) {
  const t = content.clusterSelect;
  const rt = content.role;

  const [clusters, setClusters] = useState([]);
  const [pick, setPick] = useState(selected);
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

  const chosen = clusters.find((c) => c.cluster === pick);
  // Role title and description come from the editable content file, keyed by the
  // job area. Fall back to the values the backend sends if a key is missing.
  const roleInfo = (chosen && content.roles && content.roles[chosen.cluster]) || {};
  const roleTitle = roleInfo.title || (chosen && chosen.recipientRole) || "";
  const roleDesc = roleInfo.description || (chosen && chosen.note) || "";
  // "You are a/an {role}". Pick the article from the role title's first letter.
  const article = /^[aeiou]/i.test(roleTitle) ? "an" : "a";

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
            onClick={() => setPick(c.cluster)}
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
          <h2>{rt.title}</h2>
          <p className="lead">{rich(rt.lead)}</p>

          <div className="rolepanel">
            <div className="roletitle">
              {rt.rolePrefix || "You are"} {article} {roleTitle}
            </div>
            {roleDesc && <p className="rolenote">{rich(roleDesc)}</p>}
          </div>

          <h2>{rt.fitHeading}</h2>
          <p className="lead">{rich(rt.fitText, { role: roleTitle })}</p>
          <p>
            <b>{rich(rt.fitQuestion)}</b>
          </p>

          <div className="fitchoice">
            <button
              className="btn primary"
              onClick={() => onNext(chosen.cluster, chosen.recipientRole, chosen.note)}
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
  );
}
