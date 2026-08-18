import React, { useRef } from "react";
import { rich, fmt, articleFor } from "../content.jsx";
import { useReadingFocus } from "../readingFocus.js";

// Page 4 of 5: one block per job area the participant picked, in the same
// shuffled order as page 3.
//
// Three questions each: how recently they worked in that area, how long in
// total, and whether they know what that area's role does day to day and what
// kinds of requests reach it. The first two are what
// scripts/assign_clusters.py ranks candidates by; the third is the eligibility
// gate (`qualified`).
//
// The fit question is answerable here only because page 1 already said what the
// follow-up study would ask of them. If the task explainer is ever moved after
// this page, this question stops meaning anything.
export default function AreaDetail({
  content,
  clusters,
  draft,
  patch,
  onNext,
}) {
  const tb = content.areaBlock;

  const byName = new Map(clusters.map((c) => [c.cluster, c]));

  // Role title and description come from clusters.json, with an optional
  // display-only override in content.json.
  const roleOf = (area) => {
    const c = byName.get(area) || {};
    const o = (content.roles && content.roles[area]) || {};
    return {
      title: o.title || c.role || area,
      description: o.description || c.roleDescription || "",
    };
  };

  const setDet = (area, key, value) =>
    patch({
      det: { ...draft.det, [area]: { ...(draft.det[area] || {}), [key]: value } },
    });

  const blockDone = (area) => {
    const d = draft.det[area] || {};
    return Boolean(d.recency && d.tenure && d.fit);
  };
  const ready = draft.areas.length > 0 && draft.areas.every(blockDone);

  const areas = draft.order.filter((a) => draft.areas.includes(a));

  const pageRef = useRef(null);
  useReadingFocus(pageRef, [areas.length]);

  return (
    <div className="card wide" ref={pageRef}>
      {/* The page title, not a lead paragraph. Every other page opens with a
          heading and this one opened with body copy, so it read as a stray
          sentence above the first area block rather than as the top of a page. */}
      {tb.pageLead && (
        <div className="focusblock">
          <h1>{rich(tb.pageLead)}</h1>
        </div>
      )}

      {areas.map((area) => {
        const role = roleOf(area);
        const vars = { area, role: role.title, article: articleFor(role.title) };
        const d = draft.det[area] || {};
        return (
          <div className="rolestep focusblock" key={area}>
            <h2>{fmt(tb.heading, vars)}</h2>

            {/* Hidden while `rolePrefix` is empty, which it is: the fit question
                below names the role anyway, so the panel was saying the same
                thing twice on every picked area. */}
            {tb.rolePrefix && (
              <div className="rolepanel">
                <div className="roletitle">
                  {fmt(tb.rolePrefix, vars)} {vars.article} {role.title}
                </div>
                {role.description && (
                  <p className="rolenote">{rich(role.description)}</p>
                )}
              </div>
            )}

            <p className="qlabel">{rich(tb.recencyLabel, vars)}</p>
            <select
              className="roleselect"
              value={d.recency || ""}
              onChange={(e) => setDet(area, "recency", e.target.value)}
            >
              <option value="" disabled>
                {tb.recencyPlaceholder}
              </option>
              {(tb.recencyOptions || []).map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>

            <p className="qlabel spaced">{rich(tb.tenureLabel, vars)}</p>
            <select
              className="roleselect"
              value={d.tenure || ""}
              onChange={(e) => setDet(area, "tenure", e.target.value)}
            >
              <option value="" disabled>
                {tb.tenurePlaceholder}
              </option>
              {(tb.tenureOptions || []).map((o) => (
                <option key={o.value} value={o.value}>
                  {o.label}
                </option>
              ))}
            </select>

            {/* Asked last, once they have said how close they are to the area.
                Both answers are useful data, so neither button is styled as the
                preferred one. */}
            <p className="qlabel spaced">{rich(tb.fitQuestion, vars)}</p>
            <div className="optlist row">
              <button
                type="button"
                className={"optbtn" + (d.fit === "yes" ? " on" : "")}
                onClick={() => setDet(area, "fit", "yes")}
              >
                {tb.fitYes}
              </button>
              <button
                type="button"
                className={"optbtn" + (d.fit === "no" ? " on" : "")}
                onClick={() => setDet(area, "fit", "no")}
              >
                {tb.fitNo}
              </button>
            </div>
          </div>
        );
      })}

      <div className="navbar end">
        <button className="btn primary" disabled={!ready} onClick={onNext}>
          {tb.continueButton}
        </button>
      </div>
    </div>
  );
}
