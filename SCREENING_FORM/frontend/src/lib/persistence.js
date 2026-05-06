const LOCAL_KEY = "cypearl_screening_draft_v1";

export function saveLocal(draft) {
  try {
    localStorage.setItem(LOCAL_KEY, JSON.stringify(draft));
  } catch {
    // localStorage may be unavailable (private mode, quota); silently ignore.
  }
}

export function loadLocal() {
  try {
    const raw = localStorage.getItem(LOCAL_KEY);
    if (!raw) return null;
    return JSON.parse(raw);
  } catch {
    return null;
  }
}

export function clearLocal() {
  try {
    localStorage.removeItem(LOCAL_KEY);
  } catch {
    // ignore
  }
}
