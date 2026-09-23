import path from "node:path"

import { baseUtility, classStringVisitor, splitClasses } from "../class-strings.js"
import { classifyColorUtility } from "../color-utility.js"
import { readTheme } from "../theme.js"

export default {
  meta: {
    type: "problem",
    docs: { description: "Colour utilities must name a semantic token defined in src/index.css" },
    messages: { color: "{{message}}" },
    schema: [{ type: "object", properties: { css: { type: "string" } }, additionalProperties: false }],
  },
  create(context) {
    const css = path.resolve(context.cwd, context.options[0]?.css ?? "src/index.css")
    const theme = readTheme(css)
    return classStringVisitor((node, text) => {
      for (const cls of splitClasses(text)) {
        const message = classifyColorUtility(baseUtility(cls), theme)
        if (message) context.report({ node, messageId: "color", data: { message } })
      }
    })
  },
}
