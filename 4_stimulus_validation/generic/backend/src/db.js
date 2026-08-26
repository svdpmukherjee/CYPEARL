import { MongoClient } from "mongodb";

const uri = process.env.MONGODB_URI || "mongodb://localhost:27017";
const dbName = process.env.DB_NAME || "user_validation_emails";

// The generic study shares a database with the job-specific one and keeps its
// own collections. The two studies go to the SAME participants: they rated the
// 16 emails written for their job area first, and this is a second, optional
// pass over the 16 generic emails. So they must not share `participants` or
// `responses` (finishing one would lock the other out, and the responses would
// collide on the same prolificId+src key), but they DO share `roster`, because
// the invited set is identical and there is no second recruitment.
const C = {
  emails: process.env.EMAILS_COLLECTION || "generic_all_emails",
  participants: process.env.PARTICIPANTS_COLLECTION || "generic_participants",
  responses: process.env.RESPONSES_COLLECTION || "generic_responses",
  roster: process.env.ROSTER_COLLECTION || "roster",
  // The job-specific study's participant rows. Read-only here, and read for one
  // field: the role that study assigned this person (`recipientRole`). They
  // judge the generic emails as that same role, so that the standing of the
  // recipient is held constant across the two sets and the realism ratings stay
  // comparable. Never written.
  jobSpecificParticipants:
    process.env.JOBSPEC_PARTICIPANTS_COLLECTION || "participants",
};

// On Vercel every request may hit a cold serverless instance, and several can
// run at once. Cache a single connecting client on globalThis so we reuse one
// connection pool instead of opening a new one per invocation (which exhausts
// the Atlas connection limit). This is a no-op harmless cache locally too.
let clientPromise = globalThis.__cypearlMongo;
if (!clientPromise) {
  const client = new MongoClient(uri, { serverSelectionTimeoutMS: 8000 });
  clientPromise = client.connect();
  globalThis.__cypearlMongo = clientPromise;
}

let db = null;
let indexesReady = null;

export async function connect() {
  if (db) return db;
  const client = await clientPromise;
  db = client.db(dbName);

  // Create the study indexes once per process, best effort. We do not block or
  // fail a request if index creation races or was already done. The roster
  // index is left to the job-specific app, which owns that collection.
  if (!indexesReady) {
    indexesReady = Promise.all([
      db.collection(C.responses).createIndex({ prolificId: 1, src: 1 }, { unique: true }),
      db.collection(C.participants).createIndex({ prolificId: 1 }, { unique: true }),
      db.collection(C.emails).createIndex({ n: 1 }),
    ]).catch((err) => console.error("Index setup:", err.message));
  }

  return db;
}

export function getDb() {
  if (!db) throw new Error("Database not connected yet. Call connect() first.");
  return db;
}

export const collections = {
  emails: () => getDb().collection(C.emails),
  participants: () => getDb().collection(C.participants),
  responses: () => getDb().collection(C.responses),
  // Shared, read-only here. Seeded by the job-specific app's
  // scripts/seed_roster.py; this study never writes to it.
  roster: () => getDb().collection(C.roster),
  // The job-specific study's own participant rows. Read-only here, for the
  // assigned role alone; that study owns them and this one never writes.
  jobSpecificParticipants: () => getDb().collection(C.jobSpecificParticipants),
};

export const collectionNames = C;
