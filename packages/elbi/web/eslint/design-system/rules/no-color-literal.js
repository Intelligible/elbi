const COLOR = /#(?:[0-9a-fA-F]{8}|[0-9a-fA-F]{6}|[0-9a-fA-F]{3,4})\b|\b(?:rgba?|hsla?|oklch|oklab|lch|lab)\(/
const SKIP_ATTRS = new Set(["href", "to", "id", "key", "data-testid"])

function skipped(node) {
  const p = node.parent
  if (p?.type === "ImportDeclaration" || p?.type === "ExportNamedDeclaration") return true
  if (p?.type === "ExportAllDeclaration" || p?.type === "ImportExpression") return true
  return p?.type === "JSXAttribute" && SKIP_ATTRS.has(p.name.name)
}

export default {
  meta: {
    type: "problem",
    docs: { description: "No colour literals; colours come from semantic tokens" },
    messages: { literal: "Colour literal: use a semantic token from src/index.css (charts: lib/chart-theme)." },
    schema: [],
  },
  create(context) {
    const check = (node, text) => {
      if (COLOR.test(text) && !skipped(node)) context.report({ node, messageId: "literal" })
    }
    return {
      Literal(node) {
        if (typeof node.value === "string") check(node, node.value)
      },
      TemplateElement(node) {
        check(node, node.value.cooked ?? node.value.raw)
      },
    }
  },
}
