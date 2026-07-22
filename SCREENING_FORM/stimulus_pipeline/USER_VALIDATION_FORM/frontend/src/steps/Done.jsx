import React from "react";
import { rich } from "../content.jsx";

export default function Done({ content, prolificId, onRestart }) {
  const t = content.done;
  return (
    <div className="card wide center">
      <h1>{t.title}</h1>
      <p className="lead">{rich(t.lead)}</p>
      <p className="muted">{t.prolificLabel} <b>{prolificId}</b></p>
      <div className="navbar center">
        <button className="btn" onClick={onRestart}>
          {t.restartButton}
        </button>
      </div>
    </div>
  );
}
