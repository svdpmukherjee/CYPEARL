import "dotenv/config";
import express from "express";
import cors from "cors";
import { connect, collections } from "./db.js";

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

// One Prolific ID = one person, one job area. Given the stored participant doc
// and the cluster the client is presenting, return { status, body } to reject
// with, or null if they may proceed. A completed participant is always blocked
// (one submission per ID); a participant whose locked cluster differs is blocked
// (the first chosen cluster is immutable). This is the single source of truth
// for both checks so /register, /participant, and /response agree.
function reentryBlock(existing, cluster) {
  if (!existing) return null;
  if (existing.status === "complete")
    return {
      status: 409,
      body: {
        error: "This Prolific ID has already completed the study.",
        code: "ALREADY_COMPLETED",
      },
    };
  if (existing.cluster && cluster && existing.cluster !== cluster)
    return {
      status: 409,
      body: {
        error: "This Prolific ID is already registered for a different job area.",
        code: "CLUSTER_LOCKED",
        cluster: existing.cluster,
        recipientRole: existing.recipientRole || null,
      },
    };
  return null;
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

// --- clusters: the list a participant picks from -------------------------
app.get(
  "/api/clusters",
  wrap(async (req, res) => {
    const emails = collections.emails();
    // one representative doc per cluster gives us the role + note
    const rows = await emails
      .aggregate([
        { $sort: { cluster: 1, n: 1 } },
        {
          $group: {
            _id: "$cluster",
            recipient_role: { $first: "$recipient_role" },
            recipient_note: { $first: "$recipient_note" },
            count: { $sum: 1 },
          },
        },
        { $sort: { _id: 1 } },
      ])
      .toArray();
    res.json(
      rows.map((r) => ({
        cluster: r._id,
        recipientRole: r.recipient_role,
        note: r.recipient_note,
        count: r.count,
      }))
    );
  })
);

// --- the 16 emails for a chosen cluster ----------------------------------
app.get(
  "/api/emails/:cluster",
  wrap(async (req, res) => {
    const cluster = req.params.cluster;
    const docs = await collections
      .emails()
      .find({ cluster })
      .sort({ n: 1 })
      .toArray();
    if (!docs.length) return res.status(404).json({ error: "Unknown cluster" });
    res.json(docs);
  })
);

// --- register / resume: called when the participant submits their ID -----
// The single enforcement point for the one-ID-one-area rule. On first sight it
// locks the (prolificId, cluster) pair; on any later visit it either blocks
// (already completed, or a different cluster) or signals a resume with the last
// saved page (step + emailIdx). Runs before the pre-email steps.
app.post(
  "/api/participant/register",
  wrap(async (req, res) => {
    const { prolificId, cluster, recipientRole, ownRole, roleRelation } =
      req.body || {};
    if (!prolificId || !String(prolificId).trim())
      return res.status(400).json({ error: "prolificId is required" });
    if (!cluster) return res.status(400).json({ error: "cluster is required" });

    const pid = String(prolificId).trim();
    const now = new Date();
    const participants = collections.participants();

    const existing = await participants.findOne({ prolificId: pid });
    if (existing) {
      const block = reentryBlock(existing, cluster);
      if (block) return res.status(block.status).json(block.body);
      // in progress, same cluster: resume from the last saved page
      return res.json({
        ok: true,
        resume: true,
        prolificId: pid,
        cluster: existing.cluster,
        step: existing.step || "checks",
        emailIdx: Number.isFinite(existing.emailIdx) ? existing.emailIdx : 0,
      });
    }

    // First time we see this ID: create the locked record.
    const doc = {
      prolificId: pid,
      cluster,
      recipientRole: recipientRole || null,
      ownRole: (ownRole || "").trim(),
      roleRelation: roleRelation || null,
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
        const block = reentryBlock(again, cluster);
        if (block) return res.status(block.status).json(block.body);
        return res.json({
          ok: true,
          resume: true,
          prolificId: pid,
          cluster: again?.cluster || cluster,
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

// --- participant meta: consent, cluster, chosen name ---------------------
app.post(
  "/api/participant",
  wrap(async (req, res) => {
    const {
      prolificId,
      cluster,
      recipientRole,
      ownRole,
      roleRelation,
      personalizationName,
      consent,
      roleCheckAttempts,
      priorJudgments,
    } = req.body || {};

    if (!prolificId || !String(prolificId).trim())
      return res.status(400).json({ error: "prolificId is required" });
    if (!consent)
      return res.status(400).json({ error: "consent is required" });

    const now = new Date();
    const pid = String(prolificId).trim();

    // Enforce the one-ID-one-area rule here too: a completed or cluster-locked
    // participant cannot be silently updated, even if the client skipped
    // /register or was tampered with.
    const existing = await collections.participants().findOne({ prolificId: pid });
    const block = reentryBlock(existing, cluster);
    if (block) return res.status(block.status).json(block.body);

    await collections.participants().updateOne(
      { prolificId: pid },
      {
        $set: {
          cluster: cluster || null,
          recipientRole: recipientRole || null,
          ownRole: (ownRole || "").trim(),
          roleRelation: roleRelation || null,
          personalizationName: (personalizationName || "").trim(),
          consent: !!consent,
          roleCheckAttempts: Number.isFinite(roleCheckAttempts) ? roleCheckAttempts : null,
          priorJudgments:
            priorJudgments && typeof priorJudgments === "object" ? priorJudgments : {},
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
