const MESSAGES = {
  confirm: "Use useFeedback().confirm (components/ui/feedback).",
  alert: "Use useFeedback().toast (components/ui/feedback).",
  prompt: "Use a Dialog with an Input.",
}
const NAMES = new Set(Object.keys(MESSAGES))
// window.<name> is left to no-restricted-properties; don't double-report it here.
const GLOBAL_OBJECTS = new Set(["globalThis", "self"])

// True when `name` is not bound by any enclosing scope, i.e. it really is the native global.
function isUnbound(context, node, name) {
  let scope = context.sourceCode.getScope(node)
  while (scope) {
    if (scope.set.has(name)) return false
    scope = scope.upper
  }
  return true
}

export default {
  meta: {
    type: "problem",
    docs: { description: "Use the app's feedback/dialog components instead of native browser dialogs" },
    messages: MESSAGES,
    schema: [],
  },
  create(context) {
    return {
      CallExpression(node) {
        const callee = node.callee
        if (callee.type === "Identifier" && NAMES.has(callee.name)) {
          if (isUnbound(context, node, callee.name)) {
            context.report({ node: callee, messageId: callee.name })
          }
          return
        }
        if (
          callee.type === "MemberExpression" &&
          !callee.computed &&
          callee.object.type === "Identifier" &&
          GLOBAL_OBJECTS.has(callee.object.name) &&
          callee.property.type === "Identifier" &&
          NAMES.has(callee.property.name)
        ) {
          context.report({ node: callee, messageId: callee.property.name })
        }
      },
    }
  },
}
