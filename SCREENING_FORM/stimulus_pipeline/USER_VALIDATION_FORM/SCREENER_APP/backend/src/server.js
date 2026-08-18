import "dotenv/config";
import express from "express";
import cors from "cors";
import { createRequire } from "node:module";
import { connect, collections } from "./db.js";

// clusters.json is loaded through createRequire rather than a JSON import so
// Vercel's file tracing picks it up when bundling the serverless function.
const require = createRequire(import.meta.url);
const clustersFile = require("../../clusters.json");
const CLUSTERS = clustersFile.clusters;
const CLUSTER_NAMES = new Set(CLUSTERS.map((c) => c.cluster));

// A participant may claim at most this many job areas. Three keeps the survey to
// about two minutes while still capturing the overlap that makes the assignment
// step work: someone who qualifies for both Procurement and Customer Service is
// exactly the person who lets us fill the scarce cell.
const MAX_AREAS = 3;

const RECENCY = new Set(["current", "within_2y", "2_5y", "over_5y"]);
const TENURE = new Set(["lt_1y", "1_3y", "3_7y", "over_7y"]);
const FIT = new Set(["yes", "no"]);

const app = express();
app.use(express.json({ limit: "256kb" }));

const origins = (process.env.CORS_ORIGIN || "http://localhost:5174")
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

const clean = (v, max = 200) => String(v ?? "").trim().slice(0, max);

app.get("/api/health", (req, res) => res.json({ ok: true }));

// --- runtime config for the client ---------------------------------------
// The completion code is study-specific and set as an environment variable in
// Vercel, so it is never committed to source. Read at request time, so changing
// the env var takes effect without a rebuild.
//
// This survey has ONE completion code. Everybody who reaches the end is approved
// and paid, including the participant who says the study is not a good fit for
// them: their answer is the data we are buying. The screen-out code belongs to
// the main study, not here.
app.get("/api/config", (req, res) => {
  const code = (process.env.PROLIFIC_COMPLETION_CODE || "").trim();
  res.json({
    completionCode: code,
    completionUrl: code
      ? `https://app.prolific.com/submissions/complete?cc=${encodeURIComponent(code)}`
      : "",
    maxAreas: MAX_AREAS,
  });
});

// --- the ten job areas, with their qualifying job titles ------------------
app.get("/api/clusters", (req, res) => {
  res.json(
    CLUSTERS.map((c) => ({
      cluster: c.cluster,
      titles: c.titles,
      role: c.role,
      roleDescription: c.roleDescription,
    }))
  );
});

// --- has this person already answered? -----------------------------------
// Called when the participant leaves page 2, the moment their Prolific ID is
// first known, so somebody who answered on an earlier visit sees the thank-you
// page again rather than a blank survey they could fill twice.
app.get(
  "/api/screener/:prolificId",
  wrap(async (req, res) => {
    const pid = clean(req.params.prolificId, 64);
    const doc = await collections
      .screener()
      .findOne({ prolificId: pid }, { projection: { _id: 0 } });
    if (!doc) return res.status(404).json({ error: "Not found" });
    res.json(doc);
  })
);

// --- submit the screener --------------------------------------------------
// One row per Prolific ID, written once. Everything the assignment script needs
// is in this document: which areas the person claims, which is current, how
// recently and for how long they held each, whether they judge themselves able
// to rate that role, and whether they want to be invited.
//
// Nothing here comes from Prolific. The study link carries no parameters (no
// PROLIFIC_PID, no STUDY_ID, no SESSION_ID), so every field below is one the
// participant typed or picked in the survey. The Prolific ID is the single
// exception, and it is typed by hand too, which is why it is format-checked
// before anything else is looked at.
app.post(
  "/api/screener",
  wrap(async (req, res) => {
    const body = req.body || {};
    const prolificId = clean(body.prolificId, 64);
    if (!/^[A-Za-z0-9]{6,40}$/.test(prolificId))
      return res.status(400).json({ error: "A valid Prolific ID is required" });

    // Areas: known clusters only, no duplicates, at least one, at most MAX_AREAS.
    const areas = Array.isArray(body.areas) ? [...new Set(body.areas)] : [];
    if (!areas.length)
      return res.status(400).json({ error: "Select at least one job area" });
    if (areas.length > MAX_AREAS)
      return res.status(400).json({ error: `Select at most ${MAX_AREAS} job areas` });
    if (areas.some((a) => !CLUSTER_NAMES.has(a)))
      return res.status(400).json({ error: "Unknown job area" });

    const primaryArea = clean(body.primaryArea, 64);
    if (!areas.includes(primaryArea))
      return res
        .status(400)
        .json({ error: "The current job area must be one of the selected areas" });

    const jobTitle = clean(body.jobTitle, 200);
    if (!jobTitle) return res.status(400).json({ error: "Job title is required" });

    // Optional free text from page 2: the participant's chance to say the ten
    // fixed areas map their job badly. Never required, never used by the
    // assignment, and capped so a paste cannot bloat the row.
    const areasNote = clean(body.areasNote, 1000);

    // Per-area answers. Every selected area must carry all three, so a partial
    // client state can never produce a half-filled row in the export.
    const raw = body.areaDetails && typeof body.areaDetails === "object" ? body.areaDetails : {};
    const areaDetails = {};
    for (const area of areas) {
      const d = raw[area] || {};
      const recency = clean(d.recency, 32);
      const tenure = clean(d.tenure, 32);
      const fit = clean(d.fit, 8);
      if (!RECENCY.has(recency) || !TENURE.has(tenure) || !FIT.has(fit))
        return res
          .status(400)
          .json({ error: `Incomplete answers for ${area}` });
      areaDetails[area] = { recency, tenure, fit };
    }

    if (typeof body.interested !== "boolean")
      return res.status(400).json({ error: "An interest answer is required" });

    const now = new Date();
    const doc = {
      prolificId,
      areas,
      primaryArea,
      jobTitle,
      areasNote: areasNote || null,
      areaDetails,
      // Convenience field for the assignment script and for quick counts: the
      // areas this person both claims AND says they know well enough to judge.
      qualified: areas.filter((a) => areaDetails[a].fit === "yes"),
      interested: body.interested,
      status: "complete",
      submittedAt: now,
    };

    try {
      await collections.screener().insertOne(doc);
    } catch (err) {
      // Unique index on prolificId: a second submission is refused rather than
      // overwriting the first, so nobody can be counted twice in an allowlist.
      if (err && err.code === 11000)
        return res.status(409).json({
          error: "This Prolific ID has already completed the screening survey.",
          code: "ALREADY_COMPLETED",
        });
      throw err;
    }

    res.json({ ok: true });
  })
);

// --- live yield summary (private) ----------------------------------------
// Answers the question the recruitment plan depends on: how many usable people
// has each cluster produced so far. Enabled only when ADMIN_KEY is set, and the
// caller must pass it as ?key=. This is what you watch during the pilot batch
// before sizing the full screener.
app.get(
  "/api/summary",
  wrap(async (req, res) => {
    const key = (process.env.ADMIN_KEY || "").trim();
    if (!key) return res.status(404).json({ error: "Not enabled" });
    if (clean(req.query.key, 128) !== key)
      return res.status(401).json({ error: "Unauthorized" });

    const docs = await collections
      .screener()
      .find({}, { projection: { _id: 0 } })
      .toArray();

    const rows = CLUSTERS.map((c) => {
      const name = c.cluster;
      const claimed = docs.filter((d) => d.areas.includes(name));
      const primary = docs.filter((d) => d.primaryArea === name);
      const usable = docs.filter(
        (d) => d.interested && (d.qualified || []).includes(name)
      );
      const usableCurrent = usable.filter(
        (d) => (d.areaDetails?.[name] || {}).recency === "current"
      );
      return {
        cluster: name,
        target: c.target,
        claimed: claimed.length,
        primary: primary.length,
        usable: usable.length,
        usableCurrent: usableCurrent.length,
      };
    });

    res.json({
      total: docs.length,
      interested: docs.filter((d) => d.interested).length,
      clusters: rows,
      // What participants said about the mapping on page 2. Listed here because
      // a note nobody reads is worse than no note: a job title two or three
      // people file under the wrong area is a defect in clusters.json, and it
      // is cheapest to fix in the first batch, before the allowlists are built.
      notes: docs
        .filter((d) => d.areasNote)
        .map((d) => ({
          prolificId: d.prolificId,
          jobTitle: d.jobTitle,
          areas: d.areas,
          note: d.areasNote,
        })),
    });
  })
);

// The Express app is exported as the request handler. Vercel wraps it as a
// serverless function (see ../../api/index.js). For local development,
// src/local.js imports this app and calls app.listen().
export default app;
