// @vitest-environment node
import { ruleTester } from "../rule-tester.js"
import rule from "./no-raw-element.js"

ruleTester.run("no-raw-element", rule, {
  valid: [
    `<Button>Save</Button>`,
    `<Select value="a" />`,
    `<input type="file" hidden />`,
    `<input type="hidden" name="x" />`,
    `<Table />`,
    `<div role="button" />`,
  ],
  invalid: [
    { code: `<button type="button">x</button>`, errors: [{ messageId: "raw" }] },
    { code: `<select />`, errors: [{ messageId: "raw" }] },
    { code: `<textarea />`, errors: [{ messageId: "raw" }] },
    { code: `<table />`, errors: [{ messageId: "raw" }] },
    { code: `<input />`, errors: [{ messageId: "raw" }] },
    { code: `<input type="number" />`, errors: [{ messageId: "raw" }] },
    { code: `<input type="checkbox" />`, errors: [{ messageId: "raw" }] },
  ],
})
