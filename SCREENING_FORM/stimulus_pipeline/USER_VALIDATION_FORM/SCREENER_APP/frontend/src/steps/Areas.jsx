import React, { useEffect, useRef } from "react";
import { rich } from "../content.jsx";
import { useReadingFocus } from "../readingFocus.js";

// Page 3 of 5: which job areas they have worked in, and which one they are in
// now.
//
// MULTI-select, up to maxAreas, on purpose. Asking only for the single closest
// area throws away the overlap between areas, and that overlap is the whole
// reason the assignment step can fill a scarce cluster: someone who qualifies
// for both Procurement and Customer Service should always be spent on
// Procurement. On the 400-person simulation the overlap is worth 11 extra
// filled slots out of 200. Do not reduce this to a single choice.
//
// The areas are shown in the participant's own shuffled order (drawn once in
// App.jsx and kept in the saved draft), so no area collects picks just by
// sitting at the top of the list.
export default function Areas({
  content,
  clusters,
  maxAreas,
  draft,
  patch,
  onNext,
}) {
  const ta = content.areas;
  const tp = content.primary;
  const tn = content.areasNote || {};

  const byName = new Map(clusters.map((c) => [c.cluster, c]));
  const ordered = draft.order.length
    ? draft.order.map((n) => byName.get(n)).filter(Boolean)
    : clusters;

  const atLimit = draft.areas.length >= maxAreas;
  // With a single area there is nothing to choose between, so the "which is
  // current" question is skipped and that area is the current one by definition
  // (App.jsx sends it as primaryArea on submit).
  const needsPrimary = draft.areas.length > 1;
  const ready =
    draft.areas.length > 0 && (!needsPrimary || Boolean(draft.primary));

  const toggleArea = (area) => {
    const has = draft.areas.includes(area);
    if (!has && atLimit) return;
    const areas = has
      ? draft.areas.filter((a) => a !== area)
      : [...draft.areas, area];
    // Dropping an area must drop everything downstream of it, or a stale page-4
    // answer for an unpicked area could still reach the server.
    const det = { ...draft.det };
    if (has) delete det[area];
    patch({
      areas,
      det,
      primary: areas.includes(draft.primary) ? draft.primary : "",
    });
  };

  // Bring the "which is current" question into view once the participant has
  // picked as many areas as they are allowed, which is the first moment we know
  // they have finished choosing.
  //
  // It used to fire on the SECOND tick, when the question first appears, and
  // that interrupted people mid-choice: someone with three areas to claim had
  // the page pulled away from the grid after the second, while they were still
  // reading the remaining cards. Below the limit the question can look after
  // itself, because it renders between the grid and the Continue button and so
  // cannot be scrolled past on the way to it.
  //
  // Not fired when they already answered it, so re-picking a third area after
  // choosing a primary does not drag them back down the page.
  const primaryHeadingRef = useRef(null);
  const primarySelectRef = useRef(null);
  const mounted = useRef(false);
  useEffect(() => {
    if (!mounted.current) return;
    if (!atLimit || !needsPrimary || draft.primary) return;
    requestAnimationFrame(() => {
      primaryHeadingRef.current?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
      primarySelectRef.current?.focus({ preventScroll: true });
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [atLimit, needsPrimary]);
  useEffect(() => {
    mounted.current = true;
  }, []);

  const pageRef = useRef(null);
  useReadingFocus(pageRef, [clusters.length, draft.areas.length]);

  return (
    <div className="card wide" ref={pageRef}>
      <div className="focusblock">
        <h2>{ta.heading}</h2>
        <p className="muted">{rich(ta.help, { max: maxAreas })}</p>

        {!clusters.length && <p className="muted">{ta.loading}</p>}

        <div className="clustergrid">
          {ordered.map((c) => {
            const on = draft.areas.includes(c.cluster);
            return (
              <button
                key={c.cluster}
                type="button"
                className={"clustercard pickable" + (on ? " on" : "")}
                aria-pressed={on}
                disabled={!on && atLimit}
                onClick={() => toggleArea(c.cluster)}
              >
                <span className="cpick" aria-hidden="true" />
                <span className="cname">{c.cluster}</span>
                {/* The qualifying job titles, so the choice is made against
                    concrete titles rather than an abstract area label. */}
                <span className="ctitles">{c.titles.join(", ")}</span>
              </button>
            );
          })}
        </div>

        {atLimit && (
          <p className="hint muted">{rich(ta.limitNote, { max: maxAreas })}</p>
        )}
      </div>

      {needsPrimary && (
        <div className="rolestep focusblock">
          <h2 ref={primaryHeadingRef}>{tp.heading}</h2>
          {tp.help && <p className="muted">{rich(tp.help)}</p>}
          <select
            ref={primarySelectRef}
            className="roleselect"
            value={draft.primary}
            onChange={(e) => patch({ primary: e.target.value })}
          >
            <option value="" disabled>
              {tp.placeholder}
            </option>
            {draft.areas.map((a) => (
              <option key={a} value={a}>
                {a}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Optional, and never gates Continue. The ten areas are fixed and the
          survey forces a pick, so this is the only way somebody can say the
          pick is wrong for them: their title is filed under an area they would
          not have chosen, or none of the ten fits their work at all. Without
          it a bad fit is recorded as a good one and we never hear about it.
          Answers are for fixing clusters.json, not for the assignment, which
          never reads this field. */}
      {tn.heading && (
        <div className="rolestep focusblock">
          <h2>{tn.heading}</h2>
          {tn.help && <p className="muted">{rich(tn.help)}</p>}
          <textarea
            className="commentarea"
            placeholder={tn.placeholder}
            value={draft.areasNote || ""}
            onChange={(e) => patch({ areasNote: e.target.value })}
          />
        </div>
      )}

      <div className="navbar end">
        <button className="btn primary" disabled={!ready} onClick={onNext}>
          {ta.continueButton}
        </button>
      </div>
    </div>
  );
}
