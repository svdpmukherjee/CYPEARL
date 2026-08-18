import React from "react";
import { rich } from "../content.jsx";

// Page 5 of 5: the follow-up study's terms, then the decision.
//
// Its time and pay are stated in the sentences of the points, so "interested"
// means something we can plan against, and the last point rules out any
// advantage in answering one way rather than another. Deliberately NOT a second
// .payhighlight strip: the only strip of tiles in the survey is on page 1,
// showing what this survey itself pays. Two strips and nobody could tell which
// figure was theirs.
//
// Either button submits. There is no screen-out path: the participant who says
// this is not a good fit for them is paid exactly like the one who says yes,
// because that answer is the data this survey exists to buy.
export default function Terms({ content, submitting, onSubmit }) {
  const tt = content.terms;
  const td = content.decision;

  return (
    <div className="card wide">
      <div className="focusblock">
        <h2>{tt.heading}</h2>

        {tt.note && <p className="muted">{rich(tt.note)}</p>}

        <ul className="steps">
          {(tt.points || []).map((p, i) => (
            <li key={i}>{rich(p)}</li>
          ))}
        </ul>

        <p className="decisionq">
          <b>{rich(td.question)}</b>
        </p>

        <div className="fitchoice">
          <button
            className="btn primary"
            disabled={submitting}
            onClick={() => onSubmit(true)}
          >
            {td.yesButton}
          </button>
          <button
            className="btn subtle"
            disabled={submitting}
            onClick={() => onSubmit(false)}
          >
            {td.noButton}
          </button>
        </div>
        {submitting && <p className="hint muted">{td.submitting}</p>}
      </div>
    </div>
  );
}
