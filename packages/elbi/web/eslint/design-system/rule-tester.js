import { RuleTester } from "eslint"
import { describe, it } from "vitest"

import { languageOptions } from "./language-options.js"

RuleTester.describe = describe
RuleTester.it = it
RuleTester.itOnly = it.only

export const ruleTester = new RuleTester({ languageOptions })
