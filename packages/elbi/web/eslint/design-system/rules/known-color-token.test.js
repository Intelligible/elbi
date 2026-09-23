// @vitest-environment node
import path from "node:path"

import { ruleTester } from "../rule-tester.js"
import rule from "./known-color-token.js"

const options = [{ css: path.join(import.meta.dirname, "../fixtures/theme.css") }]

ruleTester.run("known-color-token", rule, {
  valid: [
    { code: `<p className="text-text-tertiary hover:text-foreground text-2xs" />`, options },
    { code: `const x = "text-red-500"`, options },
  ],
  invalid: [
    { code: `<p className="dark:text-red-500" />`, options, errors: [{ messageId: "color" }] },
    { code: `<p className={cn("text-muted-foreground")} />`, options, errors: [{ messageId: "color" }] },
    { code: `<p className="text-tertiary" />`, options, errors: [{ messageId: "color" }] },
  ],
})
