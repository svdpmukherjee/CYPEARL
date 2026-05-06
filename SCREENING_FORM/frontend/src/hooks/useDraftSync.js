import { useEffect, useRef } from "react";
import { saveDraft } from "../lib/api";
import { saveLocal } from "../lib/persistence";

// Persists `draft` to localStorage immediately on every change, and to the
// server (debounced) when `prolificId` is set. Server failures are swallowed —
// localStorage is the source of truth for refresh recovery; server-side draft
// is a backup for cross-device or browser-cleared scenarios.
export default function useDraftSync({ prolificId, draft, debounceMs = 800 }) {
  const timerRef = useRef(null);
  const lastSentRef = useRef("");

  useEffect(() => {
    saveLocal(draft);

    if (!prolificId) return;
    const serialized = JSON.stringify(draft);
    if (serialized === lastSentRef.current) return;

    if (timerRef.current) clearTimeout(timerRef.current);
    timerRef.current = setTimeout(() => {
      lastSentRef.current = serialized;
      saveDraft(prolificId, draft).catch(() => {
        // Reset so the next change retries.
        lastSentRef.current = "";
      });
    }, debounceMs);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [prolificId, draft, debounceMs]);
}
