import { defineConfig, devices } from "@playwright/test"

// E2E smoke against the dev server. The frontend is exercised on its own (the /api
// backend is proxied but not required for these shell-level checks), so the suite runs
// without standing up the Python service or an LLM.
// The port is overridable because 5173 is Vite's default and is often already taken by
// another checkout's dev server. Reusing that one silently tests somebody else's code.
const port = process.env.E2E_PORT ?? "5173"
const url = `http://localhost:${port}`

export default defineConfig({
  testDir: "./e2e",
  fullyParallel: true,
  use: { baseURL: url, trace: "on-first-retry" },
  webServer: {
    command: `npm run dev -- --port ${port} --strictPort`,
    url,
    reuseExistingServer: !process.env.CI,
    timeout: 120_000,
  },
  projects: [{ name: "chromium", use: { ...devices["Desktop Chrome"] } }],
})
