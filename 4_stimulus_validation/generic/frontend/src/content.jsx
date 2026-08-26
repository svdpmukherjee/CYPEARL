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

// The role this participant keeps for this study.
//
// They are not assigned a role here: they carry over the one the job-specific
// study gave them, so the recipient's standing is identical across the two sets
// and "could a sender at that level send me this" means the same thing in both.
//
// `recipientRole` is what the backend read from their job-specific participant
// row, which is the authoritative record of what they were actually shown. The
// roles map in content.json, keyed by job area, is the fallback for the title
// and the only source of the description, which is copied word for word from
// the job-specific study so the participant reads the role exactly as they read
// it there. Both can be absent, so callers must cope with "".
//
// The description is matched on the stored title first, and only then on the
// job area, so a participant whose row disagrees with the map is never shown
// one role's name above another role's description.
export function roleInfoFor(content, cluster, recipientRole) {
  const roles = content?.roles || {};
  const stored = String(recipientRole ?? "").trim();
  const byArea = roles[cluster];

  if (stored) {
    const entry =
      byArea?.title === stored
        ? byArea
        : Object.entries(roles).find(
            ([k, v]) => !k.startsWith("_") && v?.title === stored
          )?.[1];
    return { title: stored, description: entry?.description || "" };
  }
  return { title: byArea?.title || "", description: byArea?.description || "" };
}
