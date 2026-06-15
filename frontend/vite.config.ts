import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/analyze": "http://localhost:8000",
      "/health": "http://localhost:8000",
      "/steer": "http://localhost:8000",
      "/generate-image": "http://localhost:8000",
      "/generate-music": "http://localhost:8000",
    },
  },
});
