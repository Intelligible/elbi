// @vitest-environment node
import { ruleTester } from "../rule-tester.js"
import rule from "./allowed-shadow.js"

ruleTester.run("allowed-shadow", rule, {
  valid: [
    `<p className="shadow-panel hover:shadow-elevation shadow-modal shadow-none drop-shadow-sm" />`,
    // cva variant keys are variant names (e.g. "shadow"), not class strings.
    `cva("", { variants: { size: { shadow: "h-8" } } })`,
    `<X label="shadow-md" />`, // not a class attribute
  ],
  invalid: [
    { code: `<p className="shadow" />`, errors: [{ messageId: "shadow" }] },
    { code: `<p className="shadow-sm" />`, errors: [{ messageId: "shadow" }] },
    { code: `<p className="hover:shadow-md" />`, errors: [{ messageId: "shadow" }] },
    { code: `<p className="shadow-black/5" />`, errors: [{ messageId: "shadow" }] },
    { code: `<p className={cn({ "shadow-sm": on })} />`, errors: [{ messageId: "shadow" }] },
    { code: `<X listClassName="shadow-md" />`, errors: [{ messageId: "shadow" }] },
    { code: `const DISCLOSURE = "shadow-sm"`, errors: [{ messageId: "shadow" }] },
  ],
})
