import React, { useEffect, useMemo, useState } from "react";
import { api } from "./api.js";
import { loadContent, fmt } from "./content.jsx";
import Intro from "./steps/Intro.jsx";
import Areas from "./steps/Areas.jsx";
import Task from "./steps/Task.jsx";
import AreaDetail from "./steps/AreaDetail.jsx";
import Terms from "./steps/Terms.jsx";
import Done from "./steps/Done.jsx";

// Bumped from _v1 when the survey became five pages: a draft saved by the old
// one-page version holds keys this one does not understand (pidManual,
// areasConfirmed, titleConfirmed) and no page marker, so it would resume in a
// state that cannot be reasoned about.
const STORAGE_KEY = "cypearl_screener_v2";

// Prolific completion code shown on the thank-you page. Overridable at build
// time and, at runtime, by the backend /api/config response. Not a secret: every
// participant sees it and it appears in the redirect URL.
const FALLBACK_COMPLETION_CODE =
  import.meta.env.VITE_PROLIFIC_COMPLETION_CODE || "";

// The five pages, in order. The array is the single source of truth for what
// Continue does and for the progress bar. There is no Back: see the navigation
// section below for why.
//
//   intro   what this is, what it pays, Prolific ID, current job title
//   areas   which job areas they have worked in, which one is current
//   task    what the follow-up study would ask of them
//   detail  per picked area: recency, tenure, fit
//   terms   the follow-up study's terms, then the decision
//
// "task" SITS BETWEEN "areas" AND "detail" ON PURPOSE, and both sides of that
// are load bearing. Describing the task before the job areas are claimed tells
// people what we are recruiting for, and the areas stop being a report of their
// working life and become a guess at what gets them into a £4.00 study. Leaving
// it until after page 4 would make the fit question ("do you know this kind of
// role well enough") unanswerable and the page 5 opt-in uninformed.
const PAGES = ["intro", "areas", "task", "detail", "terms"];

// NOTE ON THE STUDY URL. This app reads NOTHING from the URL: no PROLIFIC_PID,
// no STUDY_ID, no SESSION_ID. The Prolific study link is the bare site address
// and nothing else, and every fact about a participant is one they typed on
// these five pages.
//
// That is the correction the first pilot forced. Prolific profiles carry job
// titles people set once and may never have revisited, and recruiting against
// them let somebody screened as Sales fill a Customer Service place. Job area
// now comes from the participant, in this survey, and is checked against the
// concrete job titles on page 3.

// Fisher-Yates. The ten job areas are shown in a random order per participant so
// the areas near the top of the list are not picked more often than the ones at
// the bottom. The order is drawn once and kept in the saved draft, so it does
// not reshuffle under someone who refreshes the page mid-answer.
function shuffled(list) {
  const a = [...list];
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const emptyDraft = () => ({
  page: "intro",
  order: [],
  prolificId: "",
  jobTitle: "",
  areas: [],
  primary: "",
  areasNote: "",
  det: {},
});

function loadDraft() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const saved = { ...emptyDraft(), ...JSON.parse(raw) };
      // A saved page name we no longer recognise resumes at the start rather
      // than rendering nothing.
      if (!PAGES.includes(saved.page)) saved.page = "intro";
      return saved;
    }
  } catch (_) {}
  return emptyDraft();
}

export default function App() {
  const [content, setContent] = useState(null);
  const [clusters, setClusters] = useState([]);
  const [draft, setDraft] = useState(loadDraft);
  const [done, setDone] = useState(false);
  const [already, setAlready] = useState(false); // submitted on an earlier visit
  const [interested, setInterested] = useState(null);
  const [checking, setChecking] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [config, setConfig] = useState({
    completionCode: FALLBACK_COMPLETION_CODE,
    completionUrl: FALLBACK_COMPLETION_CODE
      ? `https://app.prolific.com/submissions/complete?cc=${encodeURIComponent(FALLBACK_COMPLETION_CODE)}`
      : "",
    maxAreas: 3,
  });

  // load the editable copy deck and the job areas once on startup
  useEffect(() => {
    loadContent().then(setContent).catch((e) => setError(e.message));
    api.clusters().then(setClusters).catch((e) => setError(e.message));
    api
      .config()
      .then((c) => setConfig((prev) => ({ ...prev, ...c })))
      .catch(() => {}); // non-fatal: the build-time fallback stays in place
  }, []);

  // Draw the display order as soon as the job areas arrive, unless a saved draft
  // already holds one.
  useEffect(() => {
    if (!clusters.length) return;
    setDraft((d) =>
      d.order.length === clusters.length
        ? d
        : { ...d, order: shuffled(clusters.map((c) => c.cluster)) }
    );
  }, [clusters]);

  // persist locally so a refresh does not lose answers, and resumes on the same
  // page rather than at the start
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(draft));
    } catch (_) {}
  }, [draft]);

  // Every page change starts at the top of the new page.
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: "auto" });
  }, [draft.page, done]);

  const patch = (p) => setDraft((prev) => ({ ...prev, ...p }));

  // --- navigation ----------------------------------------------------------
  // FORWARD ONLY. No page offers a way back, which matches the main study.
  //
  // Page 3 is what forces it. It describes what the follow-up study would ask,
  // and anyone who could return to page 2 after reading it would be re-picking
  // job areas already knowing what we are recruiting for: the areas would stop
  // being a report of their working life and become a guess at what gets them
  // into a paid study. That is the same reason page 3 sits where it does, and
  // a Back button on page 3 or 4 quietly undid it.
  //
  // The cost is a mis-ticked area nobody can correct. It is carried by the
  // areasNote field on page 2, which is where somebody says the pick is wrong,
  // and by the fact that the areas are multi-select: a wrong tick is one area
  // among up to three, not the whole answer.
  const goTo = (page) => patch({ page });
  const next = () => {
    const i = PAGES.indexOf(draft.page);
    goTo(PAGES[Math.min(PAGES.length - 1, i + 1)]);
  };

  // Leaving page 1 is where we find out whether this person has answered before.
  // With no PROLIFIC_PID in the URL there is nothing to check on load, and the
  // ID they just typed is the first thing we can look up.
  const leaveIntro = async () => {
    let doc = null;
    setChecking(true);
    try {
      doc = await api.existing(draft.prolificId.trim());
    } catch (_) {
      // A 404 is the normal "not yet" answer. A network error is not worth
      // blocking on either: the POST at the end still refuses a duplicate.
    }
    setChecking(false);
    if (doc) {
      setAlready(true);
      setInterested(!!doc.interested);
      setDone(true);
    } else {
      goTo("areas");
    }
  };


  const submit = async (wantsIn) => {
    setSubmitting(true);
    setError("");
    try {
      await api.submit({
        prolificId: draft.prolificId.trim(),
        areas: draft.areas,
        // With one area picked there was nothing to choose between, so that
        // area is the current one by definition and page 2 never asked.
        primaryArea: draft.areas.length === 1 ? draft.areas[0] : draft.primary,
        jobTitle: draft.jobTitle.trim(),
        areasNote: draft.areasNote.trim(),
        areaDetails: draft.det,
        interested: wantsIn,
      });
      setInterested(wantsIn);
      setDone(true);
    } catch (err) {
      if (err.code === "ALREADY_COMPLETED") {
        setAlready(true);
        setInterested(wantsIn);
        setDone(true);
      } else {
        setError(err.message);
      }
    } finally {
      setSubmitting(false);
    }
  };

  // One bar across the five pages, so a participant can see how much is left.
  const progress = useMemo(() => {
    const p = content?.app?.progress || {};
    if (done) return { label: p.complete || "Complete", pct: 100 };
    const i = Math.max(0, PAGES.indexOf(draft.page));
    return {
      label: fmt(p.page || "Step {n} of {total}", {
        n: i + 1,
        total: PAGES.length,
      }),
      pct: (i / PAGES.length) * 100,
    };
  }, [done, draft.page, content]);

  const page = () => {
    switch (draft.page) {
      case "areas":
        return (
          <Areas
            content={content}
            clusters={clusters}
            maxAreas={config.maxAreas || 3}
            draft={draft}
            patch={patch}
            onNext={next}
          />
        );
      case "task":
        return (
          <Task
            content={content}
            onNext={next}
          />
        );
      case "detail":
        return (
          <AreaDetail
            content={content}
            clusters={clusters}
            draft={draft}
            patch={patch}
            onNext={next}
          />
        );
      case "terms":
        return (
          <Terms
            content={content}
            submitting={submitting}
            onSubmit={submit}
          />
        );
      default:
        return (
          <Intro
            content={content}
            draft={draft}
            patch={patch}
            checking={checking}
            onNext={leaveIntro}
          />
        );
    }
  };

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">{content?.app?.brand || "Workplace Email Study"}</div>
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
          <div className="card wide">
            <p>Loading...</p>
          </div>
        ) : done ? (
          <Done
            content={content}
            already={already}
            interested={interested}
            completionCode={config.completionCode}
            completionUrl={config.completionUrl}
          />
        ) : (
          page()
        )}
      </main>
    </div>
  );
}
