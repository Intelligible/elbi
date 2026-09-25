// @vitest-environment node
import { ruleTester } from "../rule-tester.js"
import rule from "./no-native-dialog.js"

ruleTester.run("no-native-dialog", rule, {
  valid: [
    `type E = { event: "input_request"; prompt: string; password: boolean }`,
    `interface P { alert: boolean }`,
    `const o = { confirm: true }`,
    `const { prompt } = props; prompt.trim()`,
    `function f(confirm: () => void) { confirm() }`,
    `import { confirm } from "./x"; confirm()`,
    `fb.confirm({ title: "a", body: "b" })`,
    `window.confirm("x")`,
    `const self = { confirm() {} }; self.confirm()`,
    `function f(globalThis) { globalThis.alert("x") }`,
  ],
  invalid: [
    { code: `confirm("x")`, errors: [{ messageId: "confirm" }] },
    { code: `alert("x")`, errors: [{ messageId: "alert" }] },
    { code: `prompt("x")`, errors: [{ messageId: "prompt" }] },
    { code: `globalThis.confirm("x")`, errors: [{ messageId: "confirm" }] },
  ],
})
