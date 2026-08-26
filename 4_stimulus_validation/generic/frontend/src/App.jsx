import React, { useEffect, useMemo, useState } from "react";
import { api } from "./api.js";
import { loadContent, fmt, roleInfoFor } from "./content.jsx";
import Instructions from "./steps/Instructions.jsx";
import Landing from "./steps/Landing.jsx";
import ProlificId from "./steps/ProlificId.jsx";
import Checks from "./steps/Checks.jsx";
import RecapName from "./steps/RecapName.jsx";
import EmailPage from "./steps/EmailPage.jsx";
import Done from "./steps/Done.jsx";
import Blocked from "./steps/Blocked.jsx";

// Its own key, separate from the job-specific study's. The same participants
// take both studies in the same browser, so sharing a key would let one
// study's saved progress resume into the other's pages.
const STORAGE_KEY = "cypearl_generic_validation_v1";

// Prolific completion code shown on the Thank-you page. Overridable at build
// time (VITE_PROLIFIC_COMPLETION_CODE) and, at runtime, by the backend
// /api/config response. This literal is the study default so the page works
// even without either. Not a secret: participants see it and it is in the URL.
const FALLBACK_COMPLETION_CODE = import.meta.env.VITE_PROLIFIC_COMPLETION_CODE;

// Ordered list of the fixed steps before the per-email loop.
//
// The Prolific ID comes FIRST, so the app knows who it is talking to before it
// shows anything. "landing" then confirms the job area and title we already
// hold and explains that this study is an optional addition to the one they
// have already done.
//
// Two steps the job-specific study has are absent. There is no role page,
// because the role is not assigned here: it is carried over from the
// job-specific study, so it is shown on "landing" rather than chosen, and the
// familiarity gate and the "how does your own role compare" question were both
// answered there. And there is no un-primed prior-judgments page, because these
// participants already gave those ratings in that study and asking again would
// prime rather than measure.
//
// The wizard runs FORWARD ONLY: no page offers a Back control, so an answer
// cannot be revisited once the participant has moved past it. The array still
// defines the order (and drives the progress bar), it is just never walked
// backwards.
const PRE = ["prolific", "landing", "instructions", "checks", "recap"];
const TOTAL_EMAILS = 16;

const emptyState = () => ({
  step: "prolific",
  emailIdx: 0,
  consent: false,
  cluster: null, // carried from the roster as metadata, not chosen, not used to pick emails
  ownRole: "", // their own job title, as they typed it in the screener app
  // The role the job-specific study assigned them, carried over rather than
  // assigned again, so the recipient's standing is the same in both studies.
  recipientRole: "",
  prolificId: "",
  name: "",
  roleCheckAttempts: null, // attempts taken to pass the task comprehension check
  responses: {}, // keyed by email src
  blocked: null, // { code } when re-entry is refused
});

function load() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const saved = { ...emptyState(), ...JSON.parse(raw) };
      // A saved session with no Prolific ID predates the invitation-only flow
      // (or stopped on the very first page). There is nothing to resume without
      // an ID, since the job area is now looked up from it.
      if (!saved.prolificId) return emptyState();
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
  // True until the ?PROLIFIC_PID= lookup on mount has settled. Without it the
  // manual ID page flashes on screen for a moment before the URL parameter
  // resolves and moves the participant on.
  const [bootstrapping, setBootstrapping] = useState(true);
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
    if (["landing", "instructions", "prolific", "blocked"].includes(s.step))
      return;
    api
      .saveProgress(s.prolificId, { step: s.step, emailIdx: s.emailIdx })
      .catch(() => {});
  }, [s.step, s.emailIdx, s.prolificId]);

  const patch = (p) => setS((prev) => ({ ...prev, ...p }));

  // Load the 16 generic emails once. There is a single set for everybody, so
  // unlike the job-specific study this does not depend on the job area.
  useEffect(() => {
    let cancelled = false;
    setLoadingEmails(true);
    api
      .emails()
      .then((docs) => {
        if (!cancelled) setEmails(docs);
      })
      .catch((e) => !cancelled && setError(e.message))
      .finally(() => !cancelled && setLoadingEmails(false));
    return () => {
      cancelled = true;
    };
  }, []);

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

  // Step 1a. Look the Prolific ID up in the roster. This is where the study
  // becomes invitation-only: an ID that was not sent this study is refused, an
  // ID that already finished is refused, and an ID with work in progress jumps
  // straight back to where it stopped.
  //
  // On success the participant STAYS on this page. We take the job area and job
  // title from their screener answers and the ID page shows them back, so the
  // first thing they confirm is that we have the right person. Nothing is
  // written to the database yet: the row is created on the landing page, once
  // they choose to take part.
  const goProlific = async (prolificId) => {
    const pid = String(prolificId).trim();
    try {
      const r = await api.roster(pid);
      if (r.resume) {
        await resumeParticipant(pid, r);
      } else {
        patch({
          prolificId: pid,
          cluster: r.cluster,
          ownRole: r.jobTitle || "",
          recipientRole: r.recipientRole || "",
          step: "prolific",
        });
      }
    } catch (err) {
      if (err.code === "NOT_INVITED" || err.code === "ALREADY_COMPLETED") {
        patch({ prolificId: pid, step: "blocked", blocked: { code: err.code } });
      } else {
        setError(err.message);
      }
    }
  };

  // Step 1b. They have seen their screener answers and confirmed we have the
  // right person. Still nothing written: the landing page is where they decide.
  const goProlificConfirm = () => patch({ step: "landing" });

  // Prolific appends the participant's ID to the study URL when the study is
  // set up with URL parameters (?PROLIFIC_PID=...). Reading it here means an
  // invited participant never types their ID, which removes the single most
  // common source of a mistyped ID and an unmatchable submission. The manual
  // entry field stays as the fallback for anyone arriving without it.
  //
  // This fills the ID in and checks it, but does NOT skip the page. That page
  // carries the returning-participant note, which is the whole reason this
  // study looks different from the last one; skipping it would mean nearly
  // everybody missed it, since nearly everybody arrives with the parameter.
  //
  // The URL always wins over saved progress: on a shared computer the leftover
  // session belongs to somebody else, so a different ID starts clean.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    const urlPid = (
      params.get("PROLIFIC_PID") ||
      params.get("prolific_pid") ||
      ""
    ).trim();
    if (!urlPid) {
      setBootstrapping(false);
      return;
    }
    if (s.prolificId && s.prolificId.toLowerCase() === urlPid.toLowerCase()) {
      // already resolved for this participant, keep their saved progress
      setBootstrapping(false);
      return;
    }
    if (s.prolificId) setS(emptyState());
    goProlific(urlPid).finally(() => setBootstrapping(false));
    // Runs once on mount: the URL cannot change without a reload.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Step 2. The participant has read that this is an optional second study and
  // chosen to go on. This is where the database row is created, so anyone who
  // closes the tab on the landing page leaves no trace. Nothing is sent but the
  // ID: the server reads the cluster and job title from the roster.
  const goLanding = async () => {
    try {
      const r = await api.registerParticipant({ prolificId: s.prolificId });
      if (r.resume) {
        await resumeParticipant(s.prolificId, r);
      } else {
        patch({ step: "instructions" });
      }
    } catch (err) {
      if (err.code === "NOT_INVITED" || err.code === "ALREADY_COMPLETED") {
        patch({ step: "blocked", blocked: { code: err.code } });
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
        // reg.cluster is the roster value; p.cluster is what was stored when
        // they registered. Metadata either way: it does not choose the emails.
        cluster: p.cluster || reg.cluster,
        ownRole: p.ownRole || "",
        recipientRole: p.recipientRole || reg.recipientRole || "",
        name: p.personalizationName || "",
        consent: !!p.consent,
        roleCheckAttempts: p.roleCheckAttempts ?? null,
        responses,
        emailIdx: Number.isFinite(reg.emailIdx)
          ? reg.emailIdx
          : (p.emailIdx ?? 0),
        step: reg.step || p.step || "checks",
        blocked: null,
      });
    } catch (err) {
      setError(err.message);
    }
  };

  // Persist participant meta before the email loop begins. In the job-specific
  // study this also carried the un-primed prior judgments; there are none here.
  const startEmails = async (name) => {
    try {
      // cluster and ownRole are not sent: the server reads both from the
      // roster, so there is one source of truth for them.
      await api.saveParticipant({
        prolificId: s.prolificId,
        personalizationName: name,
        consent: s.consent,
        roleCheckAttempts: s.roleCheckAttempts,
      });
    } catch (e) {
      setError(e.message);
      return;
    }
    patch({ name, step: "email", emailIdx: 0 });
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

  const progress = useMemo(() => {
    // One continuous scale across all steps: the pre-email steps followed by the
    // 16 emails. Only the label changes between the two phases, not the maths,
    // so the bar fills smoothly from start to finish.
    const totalSteps = PRE.length + TOTAL_EMAILS;
    const p = content?.app?.progress || {};
    if (s.step === "done") return { label: p.complete || "Complete", pct: 100 };
    if (s.step === "email") {
      const done = PRE.length + s.emailIdx;
      return {
        label: fmt(p.email || "Email {n} of {total}", {
          n: s.emailIdx + 1,
          total: TOTAL_EMAILS,
        }),
        pct: (done / totalSteps) * 100,
      };
    }
    const i = PRE.indexOf(s.step);
    return {
      label: p.screening || "Getting started",
      pct: i >= 0 ? (i / totalSteps) * 100 : 0,
    };
  }, [s.step, s.emailIdx, content]);

  const currentEmail = emails[s.emailIdx];

  // The role every page below writes into its {role} placeholders, plus the
  // description shown under it on the landing page. Resolved in one place so
  // the landing page, the checks, the recap and all 16 emails cannot disagree
  // about who the participant is being asked to be.
  const roleInfo = roleInfoFor(content, s.cluster, s.recipientRole);
  const role = roleInfo.title;

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
        {!content || bootstrapping ? (
          <div className="card wide">
            <p>Loading...</p>
          </div>
        ) : (
          <>
            {s.step === "landing" && (
              <Landing
                content={content}
                role={role}
                roleDescription={roleInfo.description}
                onNext={goLanding}
              />
            )}

            {s.step === "instructions" && (
              <Instructions
                content={content}
                role={role}
                consent={s.consent}
                onConsent={(v) => patch({ consent: v })}
                onNext={() => patch({ step: "checks" })}
              />
            )}

            {s.step === "prolific" && (
              <ProlificId
                content={content}
                value={s.prolificId}
                // set once the ID has been matched: the page then shows the
                // screener answers back and swaps to the confirm button
                cluster={s.cluster}
                ownRole={s.ownRole}
                onNext={goProlific}
                onConfirm={goProlificConfirm}
              />
            )}

            {s.step === "checks" && (
              <Checks
                content={content}
                role={role}
                onNext={(attempts) =>
                  patch({ step: "recap", roleCheckAttempts: attempts })
                }
              />
            )}

            {s.step === "recap" && (
              <RecapName
                content={content}
                role={role}
                initialName={s.name}
                onNext={startEmails}
              />
            )}

            {s.step === "email" &&
              (loadingEmails || !currentEmail ? (
                <div className="card">
                  <p>Loading emails...</p>
                </div>
              ) : (
                <EmailPage
                  key={currentEmail.src}
                  content={content}
                  email={currentEmail}
                  role={role}
                  index={s.emailIdx}
                  total={TOTAL_EMAILS}
                  participantName={s.name}
                  saved={s.responses[currentEmail.src]}
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
                onRestart={() => setS(emptyState())}
              />
            )}
          </>
        )}
      </main>
    </div>
  );
}
