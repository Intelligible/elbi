// @vitest-environment node
import { ruleTester } from "../rule-tester.js"
import rule from "./no-legacy-css.js"

ruleTester.run("no-legacy-css", rule, {
  valid: [`<p className="btn-ghost-like chrome" />`, `const s = "phc"`],
  invalid: [
    { code: `<button className="btn btn--secondary btn--sm" />`, errors: 3 },
    { code: `<div className="title-row" />`, errors: 1 },
    { code: `<div className="phc phc-muted" />`, errors: 2 },
    { code: `<span className="btn-frame" />`, errors: 1 },
  ],
})
