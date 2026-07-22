import { MongoClient } from "mongodb";

const uri = process.env.MONGODB_URI || "mongodb://localhost:27017";
const dbName = process.env.DB_NAME || "user_validation_emails";

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
  // fail a request if index creation races or was already done.
  if (!indexesReady) {
    indexesReady = Promise.all([
      db
        .collection("responses")
        .createIndex({ prolificId: 1, src: 1 }, { unique: true }),
      db
        .collection("participants")
        .createIndex({ prolificId: 1 }, { unique: true }),
      db.collection("all_emails").createIndex({ cluster: 1, n: 1 }),
    ]).catch((err) => console.error("Index setup:", err.message));
  }

  return db;
}

export function getDb() {
  if (!db) throw new Error("Database not connected yet. Call connect() first.");
  return db;
}

export const collections = {
  emails: () => getDb().collection("all_emails"),
  participants: () => getDb().collection("participants"),
  responses: () => getDb().collection("responses"),
};
