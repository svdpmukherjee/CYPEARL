// Local development entrypoint. Vercel does not use this file; it imports the
// Express app directly (see ../../api/index.js). Here we connect and listen on a
// port so `npm run dev` works the same as before.
import "dotenv/config";
import app from "./server.js";
import { connect } from "./db.js";

const PORT = process.env.PORT || 4000;

connect()
  .then(() => {
    app.listen(PORT, () => console.log(`API listening on http://localhost:${PORT}`));
  })
  .catch((err) => {
    console.error("Failed to connect to MongoDB:", err.message);
    process.exit(1);
  });
