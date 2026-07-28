import React from "react";
import { rich, clusterLabel } from "../content.jsx";

// Shown when a returning Prolific ID is refused: either it has already finished
// the study (one submission per ID, a hard dead-end) or it is locked to a
// different job area than the one just chosen. In the cluster-locked case we
// offer a "Start over" button back to the landing page, so the participant can
// re-pick their registered area and resume; a completed ID has nowhere to go.
export default function Blocked({ content, code, cluster, onRestart }) {
  const t = content.blocked || {};
  const label = clusterLabel(content, cluster);
  const completed = code === "ALREADY_COMPLETED";
  const title = completed ? t.completedTitle : t.clusterTitle;
  const body = completed ? t.completedBody : t.clusterBody;

  return (
    <div className="card wide">
      <h1>{title}</h1>
      <p className="lead">{rich(body, { cluster: label })}</p>
      {!completed && onRestart && (
        <div className="navbar">
          <button className="btn primary" onClick={onRestart}>
            {t.startOverButton || "Start over"}
          </button>
        </div>
      )}
    </div>
  );
}
