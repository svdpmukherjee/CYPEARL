import React from "react";

// A 1..10 rating as ten boxes, with the two end labels underneath.
//
// It replaces the slider for the believability and realism ratings. The slider
// had to start somewhere, and starting at the midpoint meant two things we did
// not want: an untouched item was stored as a real 5 (so "did not answer" and
// "answered neutral" were the same value, and the required-answer checks could
// never fire), and a thumb parked in the middle pulls answers toward the middle.
// Boxes start with nothing chosen, so a rating is always a decision.
//
// The stored value is still 1..10, so ratings collected with the slider stay
// directly comparable with these.
//
// Built on real radio inputs rather than buttons: that gives grouping, arrow-key
// navigation and screen-reader semantics for free. The input is stretched
// invisibly over its box, so the whole box is the hit target.
export default function RatingBoxes({
  name,
  value,
  onChange,
  min = 1,
  max = 10,
  minLabel,
  maxLabel,
  ariaLabel,
}) {
  const points = [];
  for (let n = min; n <= max; n++) points.push(n);

  // The end labels flank the row rather than sitting under it. Underneath, a
  // label is wider than the single box it belongs to, so it always looked like
  // it covered boxes 1 and 2 (or 9 and 10). Beside the row there is no such
  // ambiguity: each label is simply at that end of the scale.
  return (
    <div className="rbscale">
      <span className="rbend">{minLabel}</span>
      <div className="rbrow" role="radiogroup" aria-label={ariaLabel}>
        {points.map((n) => (
          <label
            key={n}
            className={"rbbox" + (value === n ? " on" : "")}
            title={
              n === min ? minLabel : n === max ? maxLabel : undefined
            }
          >
            <input
              type="radio"
              name={name}
              value={n}
              checked={value === n}
              onChange={() => onChange(n)}
            />
            <span aria-hidden="true">{n}</span>
            {/* the number alone means nothing read aloud, so each box carries
                its position and, at the ends, what that end stands for */}
            <span className="visually-hidden">
              {n === min
                ? `${n}, ${minLabel}`
                : n === max
                  ? `${n}, ${maxLabel}`
                  : String(n)}
            </span>
          </label>
        ))}
      </div>
      <span className="rbend">{maxLabel}</span>
    </div>
  );
}
