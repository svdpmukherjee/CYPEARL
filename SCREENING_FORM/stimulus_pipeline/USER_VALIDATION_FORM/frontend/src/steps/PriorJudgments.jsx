import React, { useMemo, useState } from "react";
import { rich } from "../content.jsx";

// A tiny deterministic PRNG + Fisher-Yates shuffle. Seeded from the Prolific ID
// so each participant sees the eight scenarios in a stable but individual order:
// stable across a refresh (same seed), yet varied across the sample so item
// order effects wash out. Ratings are keyed by the item's own key, not by
// position, so the order can differ freely without affecting the stored data.
function seededShuffle(arr, seedStr) {
  let h = 2166136261;
  for (let i = 0; i < seedStr.length; i++) {
    h ^= seedStr.charCodeAt(i);
    h = Math.imul(h, 16777619);
  }
  const rand = () => {
    h += 0x6d2b79f5;
    let t = h;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  const out = arr.slice();
  for (let i = out.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

// One-off page shown after the recap and before the 16 emails. It records the
// participant's general, un-primed expectation for each of the eight everyday
// email situations (the 2x2x2 sender / urgency / framing cells) BEFORE they see
// any of our crafted examples. The three factors are deliberately never named
// on screen: each cell is phrased as a plain-language situation.
export default function PriorJudgments({
  content,
  recipientRole,
  cluster,
  prolificId,
  initial,
  onBack,
  onNext,
}) {
  const t = content.priorJudgments;
  const roleInfo = (content.roles && content.roles[cluster]) || {};
  const role = roleInfo.title || recipientRole;

  const items = t.items || [];
  const order = useMemo(
    () => seededShuffle(items, prolificId || "seed"),
    [items, prolificId],
  );

  const [ratings, setRatings] = useState(initial || {});
  const [tried, setTried] = useState(false);

  const allRated = items.every((it) => ratings[it.key] != null);

  const setRating = (key, v) => {
    setRatings((prev) => ({ ...prev, [key]: v }));
    setTried(false);
  };

  const next = () => {
    if (!allRated) {
      setTried(true);
      return;
    }
    onNext(ratings);
  };

  return (
    <div className="card wide">
      <h1>{t.pageTitle}</h1>
      <p className="lead">{rich(t.lead, { role })}</p>

      {order.map((it, idx) => (
        <div
          className={"judgeitem" + (ratings[it.key] != null ? " done" : "")}
          key={it.key}
        >
          <div className="judgeprompt">
            <span className="jnum">{idx + 1}.</span> {rich(it.text, { role })}
          </div>
          <div className="likert">
            {t.scaleLabels.map((lbl, i) => {
              const v = i + 1; // stored value stays 1..5 for analysis
              return (
                <button
                  key={v}
                  type="button"
                  className={"likertopt" + (ratings[it.key] === v ? " on" : "")}
                  onClick={() => setRating(it.key, v)}
                >
                  {lbl}
                </button>
              );
            })}
          </div>
        </div>
      ))}

      {tried && !allRated && <div className="warn">{rich(t.requiredHint)}</div>}

      <div className="navbar">
        <button className="btn" onClick={onBack}>
          {t.backButton}
        </button>
        <button className="btn primary" onClick={next}>
          {t.startButton}
        </button>
      </div>
    </div>
  );
}
