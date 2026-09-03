import { defineConfig, mergeConfig } from "vitest/config"

import viteConfig from "./vite.config"

// Test config layered over the build config (shares the "@/" alias and plugins).
// jsdom for component rendering; test files live next to their source.
export default mergeConfig(
  viteConfig,
  defineConfig({
    test: {
      environment: "jsdom",
      setupFiles: ["./src/test/setup.ts"],
      include: ["src/**/*.test.{ts,tsx}"],
      css: false,
      coverage: {
        provider: "v8",
        // `all` is the whole point: without it the denominator is only the files a test
        // already imports, so an untested file cannot lower the number and the figure
        // flatters the suite. Counting every source file reports 7% where importing-only
        // reports 30%.
        all: true,
        include: ["src/**/*.{ts,tsx}"],
        exclude: ["src/**/*.test.{ts,tsx}", "src/test/**", "src/**/*.stories.tsx"],
        reporter: ["text-summary", "lcov"],
        // A ratchet at what is measured today, not a target. It stops the number
        // sliding; raising it is the work of writing tests, and the app is thinly
        // covered compared with the Python packages.
        thresholds: {
          statements: 7,
          branches: 6,
          functions: 5,
          lines: 7,
        },
      },
    },
  }),
)
