import React from "react";
import { rich } from "../content.jsx";

// Shown when a Prolific ID is refused at the door. Two reasons, in the order
// they can happen:
//
//   NOT_INVITED        the ID is not in the invitation roster. The study was
//                      sent to a hand-picked set of participants we screened,
//                      so this is somebody who followed a shared link.
//   ALREADY_COMPLETED  the ID has already finished. One submission per person,
//                      a hard dead end.
//
// The job-specific study has a third, CLUSTER_LOCKED, which keeps a participant
// with the job area they started in. There is no equivalent here: everybody
// rates the same 16 generic emails, so the job area locks nothing.
//
// Only NOT_INVITED offers a way forward, and it is there for one reason: a
// participant who reached the manual entry page and mistyped their ID lands
// here, and without a retry they would be stuck on a dead end that is their
// typo rather than their eligibility. ALREADY_COMPLETED is a genuine dead end,
// and restarting cannot change it.
export default function Blocked({ content, code, onRestart }) {
  const t = content.blocked || {};

  const copy = {
    NOT_INVITED: { title: t.notInvitedTitle, body: t.notInvitedBody },
    ALREADY_COMPLETED: { title: t.completedTitle, body: t.completedBody },
  };
  // App only ever sets the two codes above, so the fallback is for a stale or
  // hand-edited saved session that reached this page without a usable one.
  // It has to name a key that still exists: the earlier fallback pointed at
  // CLUSTER_LOCKED, which was dropped along with the cluster lock, so an
  // unknown code destructured undefined and blanked the page. The invitation
  // copy is the safe landing, and the retry below is the one way to clear it.
  const known = copy[code];
  const { title, body } = known || copy.NOT_INVITED;
  const canRestart = !known || code === "NOT_INVITED";

  return (
    <div className="card wide">
      <h1>{title}</h1>
      <p className="lead">{rich(body)}</p>
      {canRestart && onRestart && (
        <div className="navbar">
          <button className="btn primary" onClick={onRestart}>
            {t.retryIdButton || "Try a different ID"}
          </button>
        </div>
      )}
    </div>
  );
}
