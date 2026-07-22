// Thin wrapper around the backend API. In dev, Vite proxies /api to Express.
const base = "";

async function req(path, options = {}) {
  const res = await fetch(base + path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let msg = `Request failed (${res.status})`;
    try {
      const j = await res.json();
      if (j.error) msg = j.error;
    } catch (_) {}
    throw new Error(msg);
  }
  return res.status === 204 ? null : res.json();
}

export const api = {
  clusters: () => req("/api/clusters"),
  emails: (cluster) => req(`/api/emails/${encodeURIComponent(cluster)}`),
  saveParticipant: (body) =>
    req("/api/participant", { method: "POST", body: JSON.stringify(body) }),
  saveResponse: (body) =>
    req("/api/response", { method: "POST", body: JSON.stringify(body) }),
  complete: (prolificId) =>
    req(`/api/participant/${encodeURIComponent(prolificId)}/complete`, {
      method: "POST",
    }),
};
