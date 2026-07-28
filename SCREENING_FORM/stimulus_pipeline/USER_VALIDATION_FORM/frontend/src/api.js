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
    // Attach the status and any machine-readable fields (code, cluster,
    // recipientRole) so callers can branch, e.g. on a 409 ALREADY_COMPLETED or
    // CLUSTER_LOCKED, without string-matching the message.
    const err = new Error(msg);
    err.status = res.status;
    if (body && body.code) err.code = body.code;
    if (body && body.cluster) err.cluster = body.cluster;
    if (body && body.recipientRole) err.recipientRole = body.recipientRole;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  config: () => req("/api/config"),
  clusters: () => req("/api/clusters"),
  emails: (cluster) => req(`/api/emails/${encodeURIComponent(cluster)}`),
  // Lock (or resume) a participant when they submit their Prolific ID. Rejects
  // with err.code ALREADY_COMPLETED / CLUSTER_LOCKED; otherwise returns
  // { resume, cluster, step, emailIdx }.
  registerParticipant: (body) =>
    req("/api/participant/register", { method: "POST", body: JSON.stringify(body) }),
  // Fetch a participant + their saved responses, used to rehydrate on resume.
  getParticipant: (prolificId) =>
    req(`/api/participant/${encodeURIComponent(prolificId)}`),
  // Persist the last page reached (step + email index) for cross-device resume.
  saveProgress: (prolificId, body) =>
    req(`/api/participant/${encodeURIComponent(prolificId)}/progress`, {
      method: "POST",
      body: JSON.stringify(body),
    }),
  saveParticipant: (body) =>
    req("/api/participant", { method: "POST", body: JSON.stringify(body) }),
  saveResponse: (body) =>
    req("/api/response", { method: "POST", body: JSON.stringify(body) }),
  complete: (prolificId) =>
    req(`/api/participant/${encodeURIComponent(prolificId)}/complete`, {
      method: "POST",
    }),
};
