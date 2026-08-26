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
    // Attach the status and any machine-readable fields (code, cluster) so
    // callers can branch, e.g. on a 409 ALREADY_COMPLETED, without
    // string-matching the message.
    const err = new Error(msg);
    err.status = res.status;
    if (body && body.code) err.code = body.code;
    if (body && body.cluster) err.cluster = body.cluster;
    throw err;
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  config: () => req("/api/config"),
  // One set of 16 generic emails for everybody, so there is no cluster in the
  // path. The job-specific study serves a different 16 per job area.
  emails: () => req("/api/emails"),
  // The first call the app makes. Looks an invited Prolific ID up in the
  // roster and returns { cluster, jobTitle, resume, step, emailIdx }: the job
  // area and job title this person gave in the screener app, which the landing
  // page shows back to them. Rejects with err.code NOT_INVITED (403) for an ID
  // that was not sent this study, or ALREADY_COMPLETED (409). Creates nothing.
  roster: (prolificId) => req(`/api/roster/${encodeURIComponent(prolificId)}`),
  // Create (or resume) the participant row, once they choose to take part.
  // Nothing but the ID is sent: the server reads the cluster and job title from
  // the roster. Rejects with err.code ALREADY_COMPLETED; otherwise returns
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
