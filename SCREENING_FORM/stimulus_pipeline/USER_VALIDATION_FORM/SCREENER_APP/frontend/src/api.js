// Thin wrapper around the backend API. In dev, Vite proxies /api to Express.
const base = "";

async function req(path, options = {}) {
  const res = await fetch(base + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    let body = null;
    try {
      body = await res.json();
      if (body && body.error) msg = body.error;
    } catch (_) {}
    // Attach the status and any machine-readable code so callers can branch
    // (for example on a 409 ALREADY_COMPLETED) without string-matching.
    const err = new Error(msg);
    err.status = res.status;
    if (body && body.code) err.code = body.code;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  config: () => req("/api/config"),
  clusters: () => req("/api/clusters"),
  // 404 when this Prolific ID has not answered yet, which is the normal case.
  existing: (prolificId) => req(`/api/screener/${encodeURIComponent(prolificId)}`),
  submit: (body) =>
    req("/api/screener", { method: "POST", body: JSON.stringify(body) }),
};
