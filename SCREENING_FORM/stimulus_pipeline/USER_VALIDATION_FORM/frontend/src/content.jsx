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

// Fill in {placeholders} like {role}, {name}, {total} from the given values.
export function fmt(text, vars = {}) {
  return String(text ?? "").replace(/\{(\w+)\}/g, (m, k) =>
    vars[k] != null ? vars[k] : m
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

// Display name for a stored cluster value, using clusterLabels overrides.
export function clusterLabel(content, cluster) {
  const map = content?.clusterLabels || {};
  return map[cluster] || cluster;
}
