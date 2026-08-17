import React from "react";
import { rich } from "../content.jsx";

export default function Instructions({ content, recipientRole, cluster, consent, onConsent, onBack, onNext }) {
  const t = content.instructions;
  const roleInfo = (content.roles && content.roles[cluster]) || {};
  const role = roleInfo.title || recipientRole;

  return (
    <div className="card wide">
      <h1>{t.title}</h1>
      {/* the lead is optional: an empty string in content.json should leave no
          gap behind, so the heading sits straight above the first section */}
      {t.lead && <p className="lead">{rich(t.lead, { role })}</p>}

      <h2>{t.whatYouDoHeading}</h2>
      <ol className="steps stages">
        {t.whatYouDo.map((item, i) => {
          const main = typeof item === "string" ? item : item.main;
          const sub = typeof item === "string" ? null : item.sub;
          // Optional line closing a step, after its sub-list. It lets one step
          // say "here are the three parts... and this is what you do with
          // them", instead of a second step repeating the same three parts.
          const tail = typeof item === "string" ? null : item.tail;
          return (
            <li key={i}>
              {rich(main, { role })}
              {Array.isArray(sub) && sub.length > 0 && (
                <ul className="substeps">
                  {sub.map((s, j) => (
                    <li key={j}>{rich(s, { role })}</li>
                  ))}
                </ul>
              )}
              {tail && <p className="steptail">{rich(tail, { role })}</p>}
            </li>
          );
        })}
      </ol>
      {typeof t.compensation === "string" ? (
        <div className="notice">{rich(t.compensation)}</div>
      ) : (
        <div className="payhighlight">
          <p className="payline">{rich(t.compensation.note)}</p>
          <div className="paystats">
            <div className="paystat">
              <span className="paystat-value">{t.compensation.timeValue}</span>
              <span className="paystat-label">{t.compensation.timeLabel}</span>
            </div>
            <div className="paystat">
              <span className="paystat-value">{t.compensation.payValue}</span>
              <span className="paystat-label">{t.compensation.payLabel}</span>
            </div>
          </div>
        </div>
      )}

      <h2>{t.consentHeading}</h2>
      <div className="consentbox">
        {Array.isArray(t.consentText) ? (
          <ul className="bullets">
            {t.consentText.map((point, i) => (
              <li key={i}>{rich(point, { role })}</li>
            ))}
          </ul>
        ) : (
          <p>{rich(t.consentText, { role })}</p>
        )}
        <label className="check">
          <input
            type="checkbox"
            checked={consent}
            onChange={(e) => onConsent(e.target.checked)}
          />
          <span>{rich(t.consentCheckbox)}</span>
        </label>
      </div>

      <div className="navbar">
        <button className="btn" onClick={onBack}>
          {t.backButton}
        </button>
        <button className="btn primary" disabled={!consent} onClick={onNext}>
          {t.consentButton}
        </button>
      </div>
    </div>
  );
}
