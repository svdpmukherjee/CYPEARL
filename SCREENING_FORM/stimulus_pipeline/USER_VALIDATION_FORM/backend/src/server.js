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

app.get("/api/health", (req, res) => res.json({ ok: true }));

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

// --- participant meta: consent, cluster, chosen name ---------------------
app.post(
  "/api/participant",
  wrap(async (req, res) => {
    const {
      prolificId,
      cluster,
      recipientRole,
      personalizationName,
      consent,
      roleCheckAttempts,
    } = req.body || {};

    if (!prolificId || !String(prolificId).trim())
      return res.status(400).json({ error: "prolificId is required" });
    if (!consent)
      return res.status(400).json({ error: "consent is required" });

    const now = new Date();
    const pid = String(prolificId).trim();

    await collections.participants().updateOne(
      { prolificId: pid },
      {
        $set: {
          cluster: cluster || null,
          recipientRole: recipientRole || null,
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
      offItems,
      sectionEdits,
      comment,
    } = req.body || {};

    if (!prolificId || !src)
      return res.status(400).json({ error: "prolificId and src are required" });

    const pid = String(prolificId).trim();
    await collections.responses().updateOne(
      { prolificId: pid, src },
      {
        $set: {
          cluster: cluster || null,
          n: n ?? null,
          conditions: conditions || {},
          realism: realism ?? null,          // 1..5 or null
          realismReason: (realismReason || "").trim(),
          offItems: Array.isArray(offItems) ? offItems : [],
          sectionEdits: sectionEdits || {},  // { sectionId: newValue }
          comment: (comment || "").trim(),
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
      { $set: { status: "complete", completedAt: new Date() } }
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
