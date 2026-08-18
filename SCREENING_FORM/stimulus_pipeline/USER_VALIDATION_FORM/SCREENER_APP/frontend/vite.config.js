import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// The frontend talks to the backend at /api. In dev we proxy that to the
// Express server so there are no CORS surprises. Ports are one above the main
// validation app (5173 / 4000) so both can run side by side.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5174,
    proxy: {
      "/api": {
        target: process.env.VITE_API_TARGET || "http://localhost:4100",
        changeOrigin: true,
      },
    },
  },
});
