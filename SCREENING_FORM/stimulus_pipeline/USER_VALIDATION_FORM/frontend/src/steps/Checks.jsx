import React, { useState } from "react";
import { rich, fmt } from "../content.jsx";

// Two comprehension checks that confirm the participant has internalised the
// role before they start rating emails. Soft gate: they retry until correct,
// and we record how many attempts it took (roleCheckAttempts).
export default function Checks({ content, recipientRole, cluster, onBack, onNext }) {
  const t = content.checks;
  const roleInfo = (content.roles && content.roles[cluster]) || {};
  const role = roleInfo.title || recipientRole;

  const questions = t.questions || [];
  const [picks, setPicks] = useState({}); // question index -> chosen option (1-based)
  const [attempts, setAttempts] = useState(0);
  const [wrong, setWrong] = useState(false);

  const allAnswered = questions.every((_, i) => picks[i] != null);

  const submit = () => {
    const attemptNo = attempts + 1;
    setAttempts(attemptNo);
    const allCorrect = questions.every((q, i) => picks[i] === q.answer);
    if (allCorrect) onNext(attemptNo);
    else setWrong(true);
  };

  return (
    <div className="card wide">
      <h1>{t.heading}</h1>
      <p className="lead">{rich(t.intro)}</p>

      {questions.map((q, qi) => (
        <div className="checkq" key={qi}>
          <div className="qh">{rich(q.prompt, { role })}</div>
          <div className="optlist">
            {q.options.map((opt, oi) => {
              const value = oi + 1; // options numbered from 1 in content.json
              const on = picks[qi] === value;
              return (
                <button
                  key={oi}
                  type="button"
                  className={"optbtn" + (on ? " on" : "")}
                  onClick={() => {
                    setPicks((p) => ({ ...p, [qi]: value }));
                    setWrong(false);
                  }}
                >
                  {fmt(opt, { role })}
                </button>
              );
            })}
          </div>
        </div>
      ))}

      {wrong && <div className="warn">{rich(t.error)}</div>}

      <div className="navbar">
        <button className="btn" onClick={onBack}>
          {t.backButton}
        </button>
        <button className="btn primary" disabled={!allAnswered} onClick={submit}>
          {t.submit}
        </button>
      </div>
    </div>
  );
}
