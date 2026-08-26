// Vercel serverless entrypoint. All /api/* requests are rewritten here (see
// vercel.json) and handled by the Express app, which already defines routes
// under /api. The connect-guard middleware in server.js opens the (cached)
// MongoDB connection on the first request of a cold start.
import app from "../backend/src/server.js";

export default app;
