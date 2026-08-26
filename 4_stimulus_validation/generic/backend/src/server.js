import "dotenv/config";
import express from "express";
import cors from "cors";
import { connect, collections } from "./db.js";
import { lookupRoster, lookupAssignedRole } from "./roster.js";

const app = express();
app.use(express.json({ limit: "1mb" }));

const origins = (process.env.CORS_ORIGIN || "http://localhost:5173")
  .split(",")
  .map((s) => s.trim());
app.use(cors({ origin: origins }));

// Ensure the database is connected before any route runs. On Vercel each cold
// serverless start needs this; connect() caches, so it is cheap after the first
// call. Locally this simply awaits the already-open connection.
app.use(async (req, res, next) => {
  try {
    await connect();
    next();
  } catch (err) {
    console.error("DB connect failed:", err.message);
    res.status(503).json({ error: "Database unavailable" });
  }
});

// small helper so route handlers can throw and get a clean 500
const wrap = (fn) => (req, res) =>
  Promise.resolve(fn(req, res)).catch((err) => {
    console.error(err);
    res.status(500).json({ error: err.message || "Server error" });
  });

// One Prolific ID = one submission. Given the stored participant doc, return
// { status, body } to reject with, or null if they may proceed.
//
// The job-specific study also locks a participant to the job area they started
// in, because the 16 emails they rate are that area's. There is no equivalent
// here: every participant rates the same 16 generic emails, so the cluster is
// carried as metadata (which job areas the raters came from) and locks nothing.
function reentryBlock(existing) {
  if (!existing) return null;
  if (existing.status === "complete")
    return {
      status: 409,
      body: {
        error: "This Prolific ID has already completed the generic study.",
        code: "ALREADY_COMPLETED",
      },
    };
  return null;
}

// Resolve the roster entry for a Prolific ID, or the rejection to send back.
// Returns { entry } when the ID is invited, { block } when it is not. Every
// route that writes participant data goes through this, so an ID we have no
// screener answers for cannot create a row no matter which endpoint it hits.
// Async because the roster is a collection, not a bundled table; the connection
// is already open and cached by the time any route runs.
async function rosterGate(prolificId) {
  const entry = await lookupRoster(prolificId);
  if (entry) return { entry };
  return {
    block: {
      status: 403,
      body: {
        error: "This Prolific ID was not invited to this study.",
        code: "NOT_INVITED",
      },
    },
  };
}

app.get("/api/health", (req, res) => res.json({ ok: true }));

// --- runtime config for the client (Prolific completion code) ------------
// The completion code is study-specific and is set as an environment variable
// (PROLIFIC_COMPLETION_CODE) in Vercel, so it is never committed to source or
// baked into the client bundle. Read at request time, so changing the Vercel
// env var takes effect without a rebuild. The redirect URL is derived from it.
app.get("/api/config", (req, res) => {
  const code = (process.env.PROLIFIC_COMPLETION_CODE || "").trim();
  res.json({
    completionCode: code,
    completionUrl: code
      ? `https://app.prolific.com/submissions/complete?cc=${encodeURIComponent(code)}`
      : "",
  });
});

// --- the 16 generic emails: one set, the same for every participant ------
// The job-specific app serves 16 emails per job area and routes by cluster.
// A generic email is by definition one that any job area could receive, so
// there is a single set and no cluster in the path.
app.get(
  "/api/emails",
  wrap(async (req, res) => {
    const docs = await collections.emails().find({}).sort({ n: 1 }).toArray();
    if (!docs.length)
      return res.status(404).json({ error: "No generic emails seeded" });
    res.json(docs);
  })
);

// --- roster lookup: the first thing the app asks, before anything is shown --
// The study is sent to a hand-picked set of Prolific IDs who already gave us
// their job area and job title in the screener app. This endpoint answers three
// questions in one round trip, so the landing page knows what to render before
// it renders anything:
//
//   1. Is this ID invited?          no  -> 403 NOT_INVITED
//   2. Has it already finished?     yes -> 409 ALREADY_COMPLETED
//   3. Is there work in progress?   yes -> { resume: true, step, emailIdx }
//
// It deliberately creates NOTHING. This study is an optional addition to the
// one these participants already completed, so somebody who reads the landing
// page and decides not to take it leaves without a database row. The row is
// created by /register, once they choose to go on.
app.get(
  "/api/roster/:prolificId",
  wrap(async (req, res) => {
    const pid = String(req.params.prolificId).trim();
    const { entry, block } = await rosterGate(pid);
    if (block) return res.status(block.status).json(block.body);

    const existing = await collections.participants().findOne({ prolificId: pid });
    const reject = reentryBlock(existing);
    if (reject) return res.status(reject.status).json(reject.body);

    res.json({
      ok: true,
      prolificId: pid,
      // carried from the roster as metadata, never chosen by the participant
      cluster: entry.cluster,
      // the job title they typed in the screener app, shown back to them
      jobTitle: entry.jobTitle,
      // the role the job-specific study assigned them, which they keep here so
      // the recipient's standing is the same in both sets. null if that study
      // has no row for them; the client then falls back to the roles map.
      recipientRole: await lookupAssignedRole(pid),
      resume: Boolean(existing),
      step: existing?.step || null,
      emailIdx: Number.isFinite(existing?.emailIdx) ? existing.emailIdx : 0,
    });
  })
);

// --- register: called when the participant chooses to take part ------------
// The moment the participant row is created. The cluster and the job title are
// read from the roster, never from the request body, so there is one source of
// truth for both. The client sends nothing but the ID.
//
// It still handles the resume case, because a participant can reach this page
// again on a second device before /roster has told the client to jump ahead.
app.post(
  "/api/participant/register",
  wrap(async (req, res) => {
    const { prolificId } = req.body || {};
    if (!prolificId || !String(prolificId).trim())
      return res.status(400).json({ error: "prolificId is required" });

    const pid = String(prolificId).trim();
    const { entry, block } = await rosterGate(pid);
    if (block) return res.status(block.status).json(block.body);

    // The job area and the participant's own job title both come from the
    // screener answers, never from the request body.
    const cluster = entry.cluster;
    const ownRole = entry.jobTitle;
    const recipientRole = await lookupAssignedRole(pid);
    const now = new Date();
    const participants = collections.participants();

    const existing = await participants.findOne({ prolificId: pid });
    if (existing) {
      const reject = reentryBlock(existing);
      if (reject) return res.status(reject.status).json(reject.body);
      // in progress: resume from the last saved page
      return res.json({
        ok: true,
        resume: true,
        prolificId: pid,
        cluster: existing.cluster || cluster,
        recipientRole: existing.recipientRole || recipientRole,
        step: existing.step || "checks",
        emailIdx: Number.isFinite(existing.emailIdx) ? existing.emailIdx : 0,
      });
    }

    // First time we see this ID: create the locked record.
    const doc = {
      prolificId: pid,
      // metadata only: which job area this rater came from, so we can check
      // that the generic set reads as realistic across all ten of them
      cluster,
      ownRole,
      // the assigned role carried over from the job-specific study
      recipientRole,
      consent: false,
      status: "in_progress",
      step: "checks",
      emailIdx: 0,
      startedAt: now,
      updatedAt: now,
    };
    try {
      await participants.insertOne(doc);
    } catch (err) {
      // Lost a race with a concurrent register for the same ID (unique index):
      // re-read and apply the same rules instead of erroring.
      if (err && err.code === 11000) {
        const again = await participants.findOne({ prolificId: pid });
        const reject = reentryBlock(again);
        if (reject) return res.status(reject.status).json(reject.body);
        return res.json({
          ok: true,
          resume: true,
          prolificId: pid,
          cluster: again?.cluster || cluster,
          recipientRole: again?.recipientRole || recipientRole,
          step: again?.step || "checks",
          emailIdx: Number.isFinite(again?.emailIdx) ? again.emailIdx : 0,
        });
      }
      throw err;
    }
    return res.json({
      ok: true,
      resume: false,
      prolificId: pid,
      cluster,
      recipientRole,
      step: "checks",
      emailIdx: 0,
    });
  })
);

// --- persist the last page reached (server-side, cross-device resume) -----
app.post(
  "/api/participant/:prolificId/progress",
  wrap(async (req, res) => {
    const pid = String(req.params.prolificId).trim();
    const { step, emailIdx } = req.body || {};
    const set = { updatedAt: new Date() };
    if (typeof step === "string" && step) set.step = step;
    if (Number.isFinite(emailIdx)) set.emailIdx = emailIdx;
    // Never touch a completed participant, so a stale client cannot reopen a
    // finished submission by re-sending progress.
    const r = await collections.participants().updateOne(
      { prolificId: pid, status: { $ne: "complete" } },
      { $set: set }
    );
    res.json({ ok: true, updated: r.matchedCount > 0 });
  })
);

// --- participant meta: consent and the chosen personalisation name -------
// Saved once, just before the email loop begins. As with /register, the job
// area and the participant's own job title are read from the roster rather than
// trusted from the request, so the two writes cannot disagree.
app.post(
  "/api/participant",
  wrap(async (req, res) => {
    const {
      prolificId,
      personalizationName,
      consent,
      roleCheckAttempts,
    } = req.body || {};

    if (!prolificId || !String(prolificId).trim())
      return res.status(400).json({ error: "prolificId is required" });
    if (!consent)
      return res.status(400).json({ error: "consent is required" });

    const pid = String(prolificId).trim();
    const { entry, block } = await rosterGate(pid);
    if (block) return res.status(block.status).json(block.body);

    const cluster = entry.cluster;
    const recipientRole = await lookupAssignedRole(pid);
    const now = new Date();

    // Enforce the one-ID-one-area rule here too: a completed or cluster-locked
    // participant cannot be silently updated, even if the client skipped
    // /register or was tampered with.
    const existing = await collections.participants().findOne({ prolificId: pid });
    const reject = reentryBlock(existing);
    if (reject) return res.status(reject.status).json(reject.body);

    await collections.participants().updateOne(
      { prolificId: pid },
      {
        $set: {
          cluster,
          ownRole: entry.jobTitle,
          recipientRole,
          personalizationName: (personalizationName || "").trim(),
          consent: !!consent,
          roleCheckAttempts: Number.isFinite(roleCheckAttempts) ? roleCheckAttempts : null,
          updatedAt: now,
        },
        $setOnInsert: { prolificId: pid, startedAt: now, status: "in_progress" },
      },
      { upsert: true }
    );

    res.json({ ok: true, prolificId: pid });
  })
);

// --- fetch an existing participant (for resuming) ------------------------
app.get(
  "/api/participant/:prolificId",
  wrap(async (req, res) => {
    const pid = String(req.params.prolificId).trim();
    const participant = await collections
      .participants()
      .findOne({ prolificId: pid }, { projection: { _id: 0 } });
    if (!participant) return res.status(404).json({ error: "Not found" });
    const responses = await collections
      .responses()
      .find({ prolificId: pid }, { projection: { _id: 0 } })
      .toArray();
    res.json({ participant, responses });
  })
);

// --- save (upsert) a single email response -------------------------------
app.post(
  "/api/response",
  wrap(async (req, res) => {
    const {
      prolificId,
      cluster,
      src,
      n,
      conditions,
      realism,
      realismReason,
      changeText,
      editedEmail,
    } = req.body || {};

    if (!prolificId || !src)
      return res.status(400).json({ error: "prolificId and src are required" });

    const pid = String(prolificId).trim();

    // A completed (or unknown) participant may not write answers.
    const participant = await collections.participants().findOne({ prolificId: pid });
    if (!participant)
      return res
        .status(409)
        .json({ error: "Unknown participant", code: "UNKNOWN_PARTICIPANT" });
    if (participant.status === "complete")
      return res.status(409).json({
        error: "This Prolific ID has already completed the study.",
        code: "ALREADY_COMPLETED",
      });

    await collections.responses().updateOne(
      { prolificId: pid, src },
      {
        $set: {
          cluster: cluster || null,
          n: n ?? null,
          conditions: conditions || {},
          realism: realism ?? null,          // 1..10 or null (null = not answered)
          realismReason: (realismReason || "").trim(),
          changeText: (changeText || "").trim(), // free-text "what would you change"
          // the participant's in-place edit of the subject + body, or null if
          // they left the email as it was ({ subject, body: [..] })
          editedEmail:
            editedEmail && typeof editedEmail === "object" ? editedEmail : null,
          updatedAt: new Date(),
        },
        $setOnInsert: { prolificId: pid, src, createdAt: new Date() },
      },
      { upsert: true }
    );

    res.json({ ok: true });
  })
);

// --- mark the participant complete ---------------------------------------
app.post(
  "/api/participant/:prolificId/complete",
  wrap(async (req, res) => {
    const pid = String(req.params.prolificId).trim();
    const r = await collections.participants().updateOne(
      { prolificId: pid },
      { $set: { status: "complete", step: "done", completedAt: new Date() } }
    );
    if (!r.matchedCount)
      return res.status(404).json({ error: "Unknown participant" });
    res.json({ ok: true });
  })
);

// The Express app is exported as the request handler. Vercel wraps it as a
// serverless function (see ../../api/index.js). For local development,
// src/local.js imports this app and calls app.listen().
export default app;
