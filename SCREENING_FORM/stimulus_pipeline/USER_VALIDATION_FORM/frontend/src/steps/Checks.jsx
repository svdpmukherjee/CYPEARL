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
  // Result of the last attempt, per question: index -> true (right) / false
  // (wrong). A question missing from here has not been graded yet. Marking each
  // question rather than the whole page means a retry only costs the
  // participant the questions they actually got wrong.
  const [graded, setGraded] = useState({});

  const allAnswered = questions.every((_, i) => picks[i] != null);
  const anyWrong = questions.some((_, i) => graded[i] === false);

  const pick = (qi, value) => {
    setPicks((p) => ({ ...p, [qi]: value }));
    // Clear the mark on THIS question only, so the results of the others stay
    // on screen while they rework it.
    setGraded((g) => {
      if (g[qi] == null) return g;
      const next = { ...g };
      delete next[qi];
      return next;
    });
  };

  const submit = () => {
    const attemptNo = attempts + 1;
    setAttempts(attemptNo);
    const result = {};
    questions.forEach((q, i) => {
      result[i] = picks[i] === q.answer;
    });
    if (questions.every((_, i) => result[i])) return onNext(attemptNo);
    setGraded(result);
  };

  return (
    <div className="card wide">
      <h1>{t.heading}</h1>
      <p className="lead">{rich(t.intro)}</p>

      {questions.map((q, qi) => {
        const mark = graded[qi]; // true = right, false = wrong, undefined = ungraded
        return (
          <div className="checkq" key={qi}>
            <div className="qh">{rich(q.prompt, { role })}</div>
            <div className="optlist">
              {q.options.map((opt, oi) => {
                const value = oi + 1; // options numbered from 1 in content.json
                const on = picks[qi] === value;
                // Only the option they actually chose is marked. The correct
                // one is never revealed on a wrong answer, or the retry would
                // stop being a comprehension check.
                const verdict = on && mark != null ? (mark ? " ok" : " bad") : "";
                return (
                  <button
                    key={oi}
                    type="button"
                    className={"optbtn" + (on ? " on" : "") + verdict}
                    onClick={() => pick(qi, value)}
                  >
                    {fmt(opt, { role })}
                    {/* the tick and cross are drawn in CSS, so the verdict is
                        spelled out here for screen readers */}
                    {verdict && (
                      <span className="visually-hidden">
                        {mark ? t.correctLabel : t.incorrectLabel}
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}

      {anyWrong && <p className="error checkerror">{rich(t.error)}</p>}

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
