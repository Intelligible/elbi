import comments from "@eslint-community/eslint-plugin-eslint-comments"

import ds from "./eslint/design-system/index.js"
import { languageOptions } from "./eslint/design-system/language-options.js"

// ESLint runs only the design-system contract; Biome owns formatting and general linting.
export default [
  {
    ignores: [
      "src/components/ui/**",
      "src/components/ai-elements/**",
      "**/*.test.{ts,tsx}",
      "**/*.stories.tsx",
      "eslint/**",
      "dist/**",
    ],
  },
  {
    files: ["src/**/*.{ts,tsx}"],
    languageOptions,
    linterOptions: { reportUnusedDisableDirectives: "error" },
    plugins: { ds, "eslint-comments": comments },
    rules: {
      "eslint-comments/require-description": ["error", { ignore: [] }],
      "ds/no-raw-element": "error",
      "ds/no-arbitrary-value": "error",
      "ds/no-color-literal": "error",
      "ds/no-legacy-css": "error",
      "ds/allowed-shadow": "error",
      "ds/known-color-token": "error",
      "no-restricted-globals": [
        "error",
        { name: "confirm", message: "Use useFeedback().confirm (components/ui/feedback)." },
        { name: "alert", message: "Use useFeedback().toast (components/ui/feedback)." },
        { name: "prompt", message: "Use a Dialog with an Input." },
      ],
      "no-restricted-properties": [
        "error",
        { object: "window", property: "confirm", message: "Use useFeedback().confirm." },
        { object: "window", property: "alert", message: "Use useFeedback().toast." },
        { object: "window", property: "prompt", message: "Use a Dialog with an Input." },
      ],
      "no-restricted-imports": [
        "error",
        {
          paths: [{ name: "radix-ui", message: "Import from components/ui instead." }],
          patterns: [{ group: ["@radix-ui/*"], message: "Import from components/ui instead." }],
        },
      ],
    },
  },
  {
    // Charts resolve palette tokens at runtime; brand icons carry third-party colours.
    files: ["src/lib/chart-theme.ts", "src/components/viz/**", "src/components/warehouse/SourceIcon.tsx"],
    rules: { "ds/no-color-literal": "off" },
  },
]
