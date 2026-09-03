import path from "node:path"

import tailwindcss from "@tailwindcss/vite"
import react from "@vitejs/plugin-react"
import { defineConfig } from "vite"

// Build into the Python package (served by `elbi-app`, shipped in the wheel).
// Absolute asset URLs (base "/") so nested client routes like /derivations/<name>
// still resolve /assets/... correctly; in dev the API is proxied to the local backend.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  base: "/",
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  build: {
    outDir: path.resolve(__dirname, "../src/elbi/web/dist"),
    emptyOutDir: true,
  },
  server: {
    proxy: {
      "/api": { target: "http://127.0.0.1:7700", changeOrigin: true },
      "/mcp": { target: "http://127.0.0.1:7700", changeOrigin: true },
      "/health": { target: "http://127.0.0.1:7700", changeOrigin: true },
    },
  },
})
