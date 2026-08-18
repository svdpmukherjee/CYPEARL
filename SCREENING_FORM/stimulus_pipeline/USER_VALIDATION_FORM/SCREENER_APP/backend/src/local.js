import "dotenv/config";
import app from "./server.js";
import { connect } from "./db.js";

// Local development entry point. On Vercel the app is exported as a serverless
// function instead (see ../../api/index.js), which is why server.js never calls
// app.listen() itself.
const port = process.env.PORT || 4100;

connect()
  .then(() => {
    app.listen(port, () => {
      console.log(`Screener API listening on http://localhost:${port}`);
    });
  })
  .catch((err) => {
    console.error("Could not connect to MongoDB:", err.message);
    process.exit(1);
  });
