import React from "react";
import { rich, clusterLabel } from "../content.jsx";

// Shown when a Prolific ID is refused at the door. Three reasons, in the order
// they can happen:
//
//   NOT_INVITED        the ID is not in the invitation roster. The study was
//                      sent to a hand-picked set of participants we screened,
//                      so this is somebody who followed a shared link.
//   ALREADY_COMPLETED  the ID has already finished. One submission per person,
//                      a hard dead end.
//   CLUSTER_LOCKED     the ID is registered for a different job area than the
//                      roster now assigns it. Unreachable in normal use, since
//                      the area is no longer something a participant picks; it
//                      shows only if the roster was regenerated mid-study.
//
// Only NOT_INVITED offers a way forward, and it is there for one reason: a
// participant who reached the manual entry page and mistyped their ID lands
// here, and without a retry they would be stuck on a dead end that is their
// typo rather than their eligibility. The other two are genuine dead ends, and
// restarting cannot change either: the job area is assigned from the roster, so
// coming round again produces the same answer.
export default function Blocked({ content, code, cluster, onRestart }) {
  const t = content.blocked || {};
  const label = clusterLabel(content, cluster);

  const copy = {
    NOT_INVITED: { title: t.notInvitedTitle, body: t.notInvitedBody },
    ALREADY_COMPLETED: { title: t.completedTitle, body: t.completedBody },
    CLUSTER_LOCKED: { title: t.clusterTitle, body: t.clusterBody },
  };
  const { title, body } = copy[code] || copy.CLUSTER_LOCKED;
  const canRestart = code === "NOT_INVITED";

  return (
    <div className="card wide">
      <h1>{title}</h1>
      <p className="lead">{rich(body, { cluster: label })}</p>
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
