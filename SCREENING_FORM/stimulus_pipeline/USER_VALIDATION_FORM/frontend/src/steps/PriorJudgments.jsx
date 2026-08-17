import React, { useMemo, useRef, useState } from "react";
import { rich } from "../content.jsx";
import RatingBoxes from "../RatingBoxes.jsx";

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
// Forward only: the study has no Back control on any page, so an answer cannot
// be revisited once the participant has moved past it.
export default function PriorJudgments({
  content,
  recipientRole,
  cluster,
  prolificId,
  initial,
  onNext,
}) {
  const t = content.priorJudgments;
  const roleInfo = (content.roles && content.roles[cluster]) || {};
  const role = roleInfo.title || recipientRole;

  const items = t.items || [];
  const labels = t.scaleLabels || [];
  const minLabel = labels[0];
  const maxLabel = labels[labels.length - 1];
  const order = useMemo(
    () => seededShuffle(items, prolificId || "seed"),
    [items, prolificId],
  );

  // Nothing is pre-selected. An earlier version seeded every situation to the
  // midpoint so an untouched slider still recorded a value, which meant a
  // skipped item was stored as a genuine 5 and the "rate all eight" check below
  // could never fire. Starting empty keeps "not answered" and "answered
  // neutral" distinct, and makes every stored rating a real choice.
  const [ratings, setRatings] = useState(() => ({ ...(initial || {}) }));
  const [tried, setTried] = useState(false);
  const itemRefs = useRef({});

  const allRated = items.every((it) => ratings[it.key] != null);

  const setRating = (key, v) => {
    setRatings((prev) => ({ ...prev, [key]: v }));
    setTried(false);
  };

  const next = () => {
    if (!allRated) {
      setTried(true);
      // Eight items is enough that a missed one is easy to lose on the page, so
      // take them to the first unanswered situation rather than just warning.
      const missed = order.find((it) => ratings[it.key] == null);
      const el = missed && itemRefs.current[missed.key];
      if (el) el.scrollIntoView({ behavior: "smooth", block: "center" });
      return;
    }
    onNext(ratings);
  };

  return (
    <div className="card wide">
      <h1>{t.pageTitle}</h1>
      <p className="lead">{rich(t.lead, { role })}</p>

      <div className="judgelist">
        {order.map((it, idx) => {
          const rated = ratings[it.key] != null;
          return (
            <div
              className={
                "judgeitem" +
                (rated ? " done" : "") +
                (tried && !rated ? " missing" : "")
              }
              key={it.key}
              ref={(el) => {
                itemRefs.current[it.key] = el;
              }}
            >
              <div className="judgeprompt">
                <span className="jnum">{idx + 1}.</span>
                <span className="jtext">{rich(it.text, { role })}</span>
              </div>
              {/* Stored value stays 1..10, as it was with the slider, so
                  ratings collected before this change remain comparable. */}
              <RatingBoxes
                name={"prior_" + it.key}
                value={ratings[it.key] ?? null}
                onChange={(v) => setRating(it.key, v)}
                min={1}
                max={10}
                minLabel={minLabel}
                maxLabel={maxLabel}
                ariaLabel={"Believability rating, situation " + (idx + 1)}
              />
            </div>
          );
        })}
      </div>

      {t.closingLead && (
        <div className="nextsteps">
          {t.closingHeading && <h2>{t.closingHeading}</h2>}
          <p className="lead">{rich(t.closingLead, { role })}</p>
          {Array.isArray(t.closingPoints) && (
            <ol className="steps">
              {t.closingPoints.map((p, i) => (
                <li key={i}>{rich(p, { role })}</li>
              ))}
            </ol>
          )}
        </div>
      )}

      {tried && !allRated && <div className="warn">{rich(t.requiredHint)}</div>}

      <div className="navbar">
        {/* empty span so the single control keeps its place on the right */}
        <span />
        <button className="btn primary" onClick={next}>
          {t.startButton}
        </button>
      </div>
    </div>
  );
}
