import React from "react";

// A labelled 1..10 slider used for the believability (prior judgments) and the
// realism (email) ratings. It starts at the midpoint (5) and the participant
// drags left or right; the chosen number rides in a bubble directly above the
// thumb so it is always read where the eye already is. The bubble's left
// position corrects for the thumb width so it tracks the thumb centre, not the
// raw track percentage, right to both ends.
export default function RatingSlider({
  value,
  onChange,
  min = 1,
  max = 10,
  minLabel,
  maxLabel,
  ariaLabel,
}) {
  const mid = Math.round((min + max) / 2);
  const v = value == null ? mid : value;
  const frac = (v - min) / (max - min); // 0 at the left end, 1 at the right
  const pct = frac * 100;

  return (
    <div className="ratingslider">
      <div className="rstrack">
        <div
          className="rsbubble"
          style={{ left: `calc(${pct}% - (${frac} - 0.5) * 22px)` }}
          aria-hidden="true"
        >
          {v}
        </div>
        <input
          type="range"
          className="rsrange"
          min={min}
          max={max}
          step={1}
          value={v}
          aria-label={ariaLabel}
          aria-valuetext={String(v)}
          style={{ "--rs-pct": pct + "%" }}
          onChange={(e) => onChange(Number(e.target.value))}
        />
      </div>
      <div className="rsends">
        <span>{minLabel}</span>
        <span>{maxLabel}</span>
      </div>
    </div>
  );
}
