import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The frontend talks to the backend at /api. In dev we proxy that to the
// Express server so there are no CORS surprises.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://localhost:4000",
        changeOrigin: true,
      },
    },
  },
});
