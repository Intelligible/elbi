// @vitest-environment node
import { ruleTester } from "../rule-tester.js"
import rule from "./no-color-literal.js"

ruleTester.run("no-color-literal", rule, {
  valid: [
    `<a href="#section" />`,
    `const n = "#1"`,
    `import x from "./a#b"`,
    `<p style={{ width: 40 }} />`,
    `<p className="bg-card" />`,
  ],
  invalid: [
    { code: `<p style={{ color: "#b42318" }} />`, errors: [{ messageId: "literal" }] },
    { code: `const css = \`.a { color: rgb(17 17 17 / 65%) }\``, errors: [{ messageId: "literal" }] },
    { code: `const c = "oklch(0.7 0.1 30)"`, errors: [{ messageId: "literal" }] },
    { code: `const c = { color: "hsl(78deg 13% 85%)" }`, errors: [{ messageId: "literal" }] },
    { code: `<p className="[--x:#fff]" />`, errors: [{ messageId: "literal" }] },
  ],
})
