import React, { useEffect, useMemo, useState } from "react";
import { api } from "./api.js";
import { loadContent } from "./content.jsx";
import Instructions from "./steps/Instructions.jsx";
import ClusterSelect from "./steps/ClusterSelect.jsx";
import ProlificId from "./steps/ProlificId.jsx";
import Checks from "./steps/Checks.jsx";
import RecapName from "./steps/RecapName.jsx";
import PriorJudgments from "./steps/PriorJudgments.jsx";
import EmailPage from "./steps/EmailPage.jsx";
import Done from "./steps/Done.jsx";
import Blocked from "./steps/Blocked.jsx";

const STORAGE_KEY = "cypearl_user_validation_v1";

// Prolific completion code shown on the Thank-you page. Overridable at build
// time (VITE_PROLIFIC_COMPLETION_CODE) and, at runtime, by the backend
// /api/config response. This literal is the study default so the page works
// even without either. Not a secret: participants see it and it is in the URL.
const FALLBACK_COMPLETION_CODE =
  import.meta.env.VITE_PROLIFIC_COMPLETION_CODE || "C145A0QK";

// Ordered list of the fixed steps before the per-email loop. The landing page
// ("cluster") now also carries the assigned role and its familiarity gate, so a
// poor-fit participant leaves before consenting or reading the instructions.
const PRE = ["cluster", "instructions", "prolific", "checks", "recap", "judgments"];
const TOTAL_EMAILS = 16;

const emptyState = () => ({
  step: "cluster",
  emailIdx: 0,
  consent: false,
  cluster: null,
  recipientRole: null,
  note: null,
  ownRole: "", // the participant's own real job role, in their words
  roleRelation: null, // how their own role sits vs the assigned one: above|peer|below|not_sure
  prolificId: "",
  name: "",
  roleCheckAttempts: null, // attempts taken to pass the role attention check
  priorJudgments: {}, // un-primed believability of the 8 situations, keyed by combo
  responses: {}, // keyed by email src
  blocked: null, // { code, cluster, recipientRole } when re-entry is refused
});

function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const saved = { ...emptyState(), ...JSON.parse(raw) };
      // The standalone "role" step was folded into the landing page. Any saved
      // progress that paused there resumes on the landing instead.
      if (saved.step === "role") saved.step = "cluster";
      return saved;
    }
  } catch (_) {}
  return emptyState();
}

export default function App() {
  const [s, setS] = useState(load);
  const [emails, setEmails] = useState([]);
  const [loadingEmails, setLoadingEmails] = useState(false);
  const [error, setError] = useState("");
  const [content, setContent] = useState(null);
  // Prolific completion code + redirect URL. The backend (/api/config) is the
  // source of truth via a Vercel env var, but we seed a build-time fallback so
  // the Thank-you page always shows the code even if that request is
  // unavailable (backend down, env var unset). The code is not a secret: it is
  // handed to every participant and appears in the redirect URL, so shipping it
  // in the bundle is fine. The backend value overrides this when present.
  const [config, setConfig] = useState({
    completionCode: FALLBACK_COMPLETION_CODE,
    completionUrl: FALLBACK_COMPLETION_CODE
      ? `https://app.prolific.com/submissions/complete?cc=${encodeURIComponent(FALLBACK_COMPLETION_CODE)}`
      : "",
  });

  // load the editable copy deck once on startup
  useEffect(() => {
    loadContent()
      .then(setContent)
      .catch((e) => setError(e.message));
  }, []);

  // load the runtime config (Prolific completion code) once on startup
  useEffect(() => {
    api
      .config()
      // only let the backend override when it actually returns a code, so an
      // unset env var never blanks out the build-time fallback above
      .then((c) => {
        if (c && c.completionCode) setConfig(c);
      })
      .catch(() => {}); // non-fatal: the build-time fallback stays in place
  }, []);

  // persist locally so a refresh does not lose progress
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(s));
    } catch (_) {}
  }, [s]);

  // Persist the last page reached to the participant doc so an unfinished
  // participant can resume on any device. Skipped for the pre-registration
  // steps (no participant row exists yet) and the blocked screen. The server
  // ignores this for completed participants.
  useEffect(() => {
    if (!s.prolificId) return;
    if (["cluster", "instructions", "prolific", "blocked"].includes(s.step)) return;
    api.saveProgress(s.prolificId, { step: s.step, emailIdx: s.emailIdx }).catch(() => {});
  }, [s.step, s.emailIdx, s.prolificId]);

  const patch = (p) => setS((prev) => ({ ...prev, ...p }));

  // load the 16 emails whenever a cluster is chosen
  useEffect(() => {
    if (!s.cluster) return;
    let cancelled = false;
    setLoadingEmails(true);
    api
      .emails(s.cluster)
      .then((docs) => {
        if (!cancelled) setEmails(docs);
      })
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoadingEmails(false));
    return () => {
      cancelled = true;
    };
  }, [s.cluster]);

  // On every step change, and on each new email, jump to the top so the
  // participant starts reading the next email from its subject, not mid-page.
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "auto" });
  }, [s.step, s.emailIdx]);

  const setResponse = (src, resp) =>
    setS((prev) => ({
      ...prev,
      responses: { ...prev.responses, [src]: resp },
    }));

  // --- navigation --------------------------------------------------------
  // The landing page now confirms the role fit before returning here, so we go
  // straight to the instructions.
  const goCluster = ({ cluster, recipientRole, note, ownRole, roleRelation }) =>
    patch({ cluster, recipientRole, note, ownRole, roleRelation, step: "instructions" });

  // Submit the Prolific ID. This is where the one-ID-one-area rule is applied:
  // register locks (id, cluster) on first sight; a returning ID either resumes
  // (same cluster, unfinished) or is blocked (already completed, or a different
  // cluster than the one it is locked to).
  const goProlific = async (prolificId) => {
    const pid = String(prolificId).trim();
    try {
      const r = await api.registerParticipant({
        prolificId: pid,
        cluster: s.cluster,
        recipientRole: s.recipientRole,
        ownRole: s.ownRole,
        roleRelation: s.roleRelation,
      });
      if (r.resume) {
        await resumeParticipant(pid, r);
      } else {
        patch({ prolificId: pid, step: "checks" });
      }
    } catch (err) {
      if (err.code === "ALREADY_COMPLETED" || err.code === "CLUSTER_LOCKED") {
        patch({
          prolificId: pid,
          step: "blocked",
          blocked: {
            code: err.code,
            cluster: err.cluster || null,
            recipientRole: err.recipientRole || null,
          },
        });
      } else {
        setError(err.message);
      }
    }
  };

  // Rehydrate an unfinished participant from the server (works on any device,
  // not just the browser that holds the localStorage copy) and jump to the last
  // saved page.
  const resumeParticipant = async (pid, reg) => {
    try {
      const data = await api.getParticipant(pid);
      const p = data.participant || {};
      const responses = {};
      for (const row of data.responses || []) {
        responses[row.src] = {
          realism: row.realism ?? null,
          realismReason: row.realismReason || "",
          changeText: row.changeText || "",
          editedEmail: row.editedEmail || null,
        };
      }
      patch({
        prolificId: pid,
        cluster: p.cluster || reg.cluster,
        recipientRole: p.recipientRole || null,
        ownRole: p.ownRole || "",
        roleRelation: p.roleRelation || null,
        name: p.personalizationName || "",
        consent: !!p.consent,
        roleCheckAttempts: p.roleCheckAttempts ?? null,
        priorJudgments: p.priorJudgments || {},
        responses,
        emailIdx: Number.isFinite(reg.emailIdx) ? reg.emailIdx : p.emailIdx ?? 0,
        step: reg.step || p.step || "checks",
        blocked: null,
      });
    } catch (err) {
      setError(err.message);
    }
  };

  const startEmails = async (priorJudgments) => {
    // persist participant meta (now including the un-primed prior judgments)
    // before the email loop begins
    try {
      await api.saveParticipant({
        prolificId: s.prolificId,
        cluster: s.cluster,
        recipientRole: s.recipientRole,
        ownRole: s.ownRole,
        roleRelation: s.roleRelation,
        personalizationName: s.name,
        consent: s.consent,
        roleCheckAttempts: s.roleCheckAttempts,
        priorJudgments,
      });
    } catch (e) {
      setError(e.message);
      return;
    }
    patch({ priorJudgments, step: "email", emailIdx: 0 });
  };

  const saveAndNext = async (src, resp) => {
    setResponse(src, resp);
    const email = emails.find((e) => e.src === src);
    try {
      await api.saveResponse({
        prolificId: s.prolificId,
        cluster: s.cluster,
        src,
        n: email?.n,
        conditions: email?.conditions,
        realism: resp.realism,
        realismReason: resp.realismReason,
        changeText: resp.changeText,
        editedEmail: resp.editedEmail,
      });
    } catch (e) {
      setError(e.message);
      return;
    }
    if (s.emailIdx + 1 >= TOTAL_EMAILS) {
      try {
        await api.complete(s.prolificId);
      } catch (_) {}
      patch({ step: "done" });
    } else {
      patch({ emailIdx: s.emailIdx + 1 });
    }
  };

  const prevEmail = () => {
    if (s.emailIdx === 0) patch({ step: "judgments" });
    else patch({ emailIdx: s.emailIdx - 1 });
  };

  const progress = useMemo(() => {
    // One continuous scale across all steps: the pre-email steps followed by the
    // 16 emails. Only the label changes between the two phases, not the maths,
    // so the bar fills smoothly from start to finish.
    const totalSteps = PRE.length + TOTAL_EMAILS;
    if (s.step === "done") return { label: "Complete", pct: 100 };
    if (s.step === "email") {
      const done = PRE.length + s.emailIdx;
      return { label: `Email ${s.emailIdx + 1} of ${TOTAL_EMAILS}`, pct: (done / totalSteps) * 100 };
    }
    const i = PRE.indexOf(s.step);
    return { label: "Getting started", pct: i >= 0 ? (i / totalSteps) * 100 : 0 };
  }, [s.step, s.emailIdx]);

  const currentEmail = emails[s.emailIdx];

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">{content?.app?.brand || "CYPEARL"}</div>
        <div className="progresswrap">
          <div className="pbar">
            <i style={{ width: `${progress.pct}%` }} />
          </div>
          <span className="plabel">{progress.label}</span>
        </div>
      </header>

      {error && (
        <div className="errbar">
          {error} <button onClick={() => setError("")}>dismiss</button>
        </div>
      )}

      <main className="stage">
        {!content ? (
          <div className="card wide"><p>Loading...</p></div>
        ) : (
          <>
            {s.step === "cluster" && (
              <ClusterSelect
                content={content}
                selected={s.cluster}
                initialOwnRole={s.ownRole}
                initialRelation={s.roleRelation}
                onNext={goCluster}
              />
            )}

            {s.step === "instructions" && (
              <Instructions
                content={content}
                recipientRole={s.recipientRole}
                cluster={s.cluster}
                consent={s.consent}
                onConsent={(v) => patch({ consent: v })}
                onBack={() => patch({ step: "cluster" })}
                onNext={() => patch({ step: "prolific" })}
              />
            )}

            {s.step === "prolific" && (
              <ProlificId
                content={content}
                value={s.prolificId}
                cluster={s.cluster}
                onBack={() => patch({ step: "instructions" })}
                onNext={goProlific}
              />
            )}

            {s.step === "checks" && (
              <Checks
                content={content}
                recipientRole={s.recipientRole}
                cluster={s.cluster}
                onBack={() => patch({ step: "prolific" })}
                onNext={(attempts) => patch({ step: "recap", roleCheckAttempts: attempts })}
              />
            )}

            {s.step === "recap" && (
              <RecapName
                content={content}
                recipientRole={s.recipientRole}
                cluster={s.cluster}
                initialName={s.name}
                onBack={() => patch({ step: "checks" })}
                onNext={(name) => patch({ name, step: "judgments" })}
              />
            )}

            {s.step === "judgments" && (
              <PriorJudgments
                content={content}
                recipientRole={s.recipientRole}
                cluster={s.cluster}
                prolificId={s.prolificId}
                initial={s.priorJudgments}
                onBack={() => patch({ step: "recap" })}
                onNext={startEmails}
              />
            )}

            {s.step === "email" &&
              (loadingEmails || !currentEmail ? (
                <div className="card"><p>Loading emails...</p></div>
              ) : (
                <EmailPage
                  key={currentEmail.src}
                  content={content}
                  email={currentEmail}
                  index={s.emailIdx}
                  total={TOTAL_EMAILS}
                  participantName={s.name}
                  saved={s.responses[currentEmail.src]}
                  onBack={prevEmail}
                  onNext={(resp) => saveAndNext(currentEmail.src, resp)}
                />
              ))}

            {s.step === "done" && (
              <Done
                content={content}
                completionCode={config.completionCode}
                completionUrl={config.completionUrl}
              />
            )}

            {s.step === "blocked" && (
              <Blocked
                content={content}
                code={s.blocked?.code}
                cluster={s.blocked?.cluster}
                onRestart={() => setS(emptyState())}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}
