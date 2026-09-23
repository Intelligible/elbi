import { baseUtility, classStringVisitor, splitClasses } from "../class-strings.js"

const EXACT = new Set(["btn", "btn-frame", "title-row", "phc"])
const PREFIX = /^(?:btn--|phc-)/

export default {
  meta: {
    type: "problem",
    docs: { description: "No classes from the retired warehouse stylesheet" },
    messages: { legacy: '"{{cls}}" is a retired stylesheet class: use components/ui and tokens.' },
    schema: [],
  },
  create(context) {
    return classStringVisitor((node, text) => {
      for (const cls of splitClasses(text)) {
        const base = baseUtility(cls)
        if (EXACT.has(base) || PREFIX.test(base)) {
          context.report({ node, messageId: "legacy", data: { cls } })
        }
      }
    })
  },
}
