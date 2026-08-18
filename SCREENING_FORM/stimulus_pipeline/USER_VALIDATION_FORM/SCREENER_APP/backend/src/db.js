import { MongoClient } from "mongodb";

const uri = process.env.MONGODB_URI || "mongodb://localhost:27017";
// A separate database from the main study (user_validation_emails), so a reseed
// or a drop_collections run on that side can never touch the screening data the
// allowlists are built from.
const dbName = process.env.DB_NAME || "cypearl_screener";

// On Vercel every request may hit a cold serverless instance, and several can
// run at once. Cache a single connecting client on globalThis so we reuse one
// connection pool instead of opening a new one per invocation (which exhausts
// the Atlas connection limit). This is a no-op harmless cache locally too.
let clientPromise = globalThis.__cypearlScreenerMongo;
if (!clientPromise) {
  const client = new MongoClient(uri, { serverSelectionTimeoutMS: 8000 });
  clientPromise = client.connect();
  globalThis.__cypearlScreenerMongo = clientPromise;
}

let db = null;
let indexesReady = null;

export async function connect() {
  if (db) return db;
  const client = await clientPromise;
  db = client.db(dbName);

  // Created once per process, best effort. One row per Prolific ID: the unique
  // index is what makes a second submission from the same person impossible
  // even if two tabs race.
  if (!indexesReady) {
    indexesReady = Promise.all([
      db
        .collection("screener_responses")
        .createIndex({ prolificId: 1 }, { unique: true }),
      db.collection("screener_responses").createIndex({ primaryArea: 1 }),
    ]).catch((err) => console.error("Index setup:", err.message));
  }

  return db;
}

export function getDb() {
  if (!db) throw new Error("Database not connected yet. Call connect() first.");
  return db;
}

export const collections = {
  screener: () => getDb().collection("screener_responses"),
};
