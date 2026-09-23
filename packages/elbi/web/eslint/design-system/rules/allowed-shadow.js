import { baseUtility, classStringVisitor, splitClasses } from "../class-strings.js"

const ALLOWED = new Set(["shadow-none", "shadow-panel", "shadow-elevation", "shadow-modal"])

export default {
  meta: {
    type: "problem",
    docs: { description: "Depth is borders-first; only the shadow tokens are allowed" },
    messages: { shadow: '"{{cls}}": use a border, or shadow-panel / shadow-elevation / shadow-modal.' },
    schema: [],
  },
  create(context) {
    return classStringVisitor((node, text) => {
      for (const cls of splitClasses(text)) {
        const base = baseUtility(cls)
        if (/^shadow(?:-|$)/.test(base) && !ALLOWED.has(base)) {
          context.report({ node, messageId: "shadow", data: { cls } })
        }
      }
    })
  },
}
