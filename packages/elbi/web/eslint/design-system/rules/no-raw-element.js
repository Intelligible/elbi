const USE = {
  button: "Button (components/ui/button) or IconButton (components/app/IconButton)",
  select: "Select (components/ui/select)",
  textarea: "Textarea (components/ui/textarea)",
  table: "Table (components/ui/table) or DataTable (components/ui/data-table)",
  input: "Input (components/ui/input) or Checkbox (components/ui/checkbox)",
}
// A file picker or a hidden field has no visual design to drift from.
const INPUT_TYPES_ALLOWED = new Set(["file", "hidden"])

export default {
  meta: {
    type: "problem",
    docs: { description: "Use the design-system component instead of a raw HTML control" },
    messages: { raw: "Raw <{{name}}>: use {{use}}." },
    schema: [],
  },
  create(context) {
    return {
      JSXOpeningElement(node) {
        if (node.name.type !== "JSXIdentifier") return
        const name = node.name.name
        if (!Object.hasOwn(USE, name)) return
        if (name === "input") {
          const type = node.attributes.find(
            (a) => a.type === "JSXAttribute" && a.name.name === "type",
          )
          if (type?.value?.type === "Literal" && INPUT_TYPES_ALLOWED.has(type.value.value)) return
        }
        context.report({ node, messageId: "raw", data: { name, use: USE[name] } })
      },
    }
  },
}
