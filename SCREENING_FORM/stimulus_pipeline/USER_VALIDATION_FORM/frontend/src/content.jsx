import React from "react";

// Loads the editable copy deck (public/content.json). Because it lives in the
// public folder it is a plain static file: a researcher can edit the words and
// simply refresh the browser, no rebuild needed. cache: no-cache so an edited
// file is picked up on refresh rather than served stale.
export async function loadContent() {
  const res = await fetch(`${import.meta.env.BASE_URL}content.json`, {
    cache: "no-cache",
  });
  if (!res.ok) throw new Error("Could not load content.json");
  return res.json();
}

// Choose "a" or "an" for a word, by its first letter. Spelling-based, which
// covers every current role title (e.g. "an Executive Assistant", "an IT
// Support Analyst", "a Sales Executive"). It does not catch silent-h or
// long-u exceptions ("an hour", "a university"), which none of the roles hit.
export function articleFor(word) {
  return /^[aeiou]/i.test(String(word ?? "").trim()) ? "an" : "a";
}

// Fill in {placeholders} like {role}, {name}, {total} from the given values.
// Whenever a {role} is supplied, {article} and {Article} (the "a"/"an" before
// it, lower- and capitalised) are derived automatically, so any content string
// can write "{article} {role}" and stay grammatical for every assigned role
// without each page having to wire the article up itself.
export function fmt(text, vars = {}) {
  const v = { ...vars };
  if (v.role != null) {
    const a = articleFor(v.role);
    if (v.article == null) v.article = a;
    if (v.Article == null) v.Article = a.charAt(0).toUpperCase() + a.slice(1);
  }
  return String(text ?? "").replace(/\{(\w+)\}/g, (m, k) =>
    v[k] != null ? v[k] : m
  );
}

// Render a string that may contain **bold** or *italic* markers as React nodes,
// after filling in any {placeholders}. This lets non-programmers add light
// emphasis in content.json without writing HTML.
export function rich(text, vars = {}) {
  const filled = fmt(text, vars);
  const parts = filled.split(/(\*\*[^*]+\*\*|\*[^*]+\*)/g);
  return parts.map((p, i) => {
    if (/^\*\*[^*]+\*\*$/.test(p)) return <b key={i}>{p.slice(2, -2)}</b>;
    if (/^\*[^*]+\*$/.test(p)) return <i key={i}>{p.slice(1, -1)}</i>;
    return <React.Fragment key={i}>{p}</React.Fragment>;
  });
}

// Words that should stay fully upper-case in a job role, even after
// title-casing. Add to this set if participants use other acronyms.
const ACRONYMS = new Set([
  "hr", "it", "pa", "ea", "qa", "pr", "ux", "ui",
  "ceo", "cfo", "cto", "coo", "cio", "vp",
  "kyc", "cdd", "aml", "seo", "erp", "crm",
]);

// Title-case a free-text entry for display, so a role someone typed in all
// lower or all upper case (for example "hr officer" or "HR OFFICER") is shown
// with each word capitalised, matching the assigned-role titles ("Recruitment
// Coordinator"). Words in ACRONYMS (HR, IT, CEO...) are kept fully upper-case.
export function titleCase(text) {
  return String(text ?? "")
    .toLowerCase()
    .split(/(\s+)/) // keep the whitespace runs so spacing is preserved
    .map((w) => {
      if (!w.trim()) return w;
      if (ACRONYMS.has(w)) return w.toUpperCase();
      return w.charAt(0).toUpperCase() + w.slice(1);
    })
    .join("");
}

// Display name for a stored cluster value, using clusterLabels overrides.
export function clusterLabel(content, cluster) {
  const map = content?.clusterLabels || {};
  return map[cluster] || cluster;
}
