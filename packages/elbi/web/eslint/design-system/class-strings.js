const CLASS_CALLEES = new Set(["cn", "clsx", "cva", "twMerge"])
const CLASS_ATTRS = new Set(["className", "class"])
// `const fieldClass = "..."`: class strings kept in a variable before reaching className.
const CLASS_VARIABLE = /(?:class|classes|cls)$/i

function inClassContext(node) {
  for (let p = node.parent; p; p = p.parent) {
    if (p.type === "JSXAttribute") return CLASS_ATTRS.has(p.name.name)
    if (p.type === "CallExpression" && p.callee.type === "Identifier") {
      if (CLASS_CALLEES.has(p.callee.name)) return true
    }
    if (p.type === "VariableDeclarator" && p.id.type === "Identifier") {
      return CLASS_VARIABLE.test(p.id.name)
    }
    if (p.type === "Program") return false
  }
  return false
}

function isPropertyKey(node) {
  return node.parent?.type === "Property" && node.parent.key === node
}

export function classStringVisitor(onString) {
  return {
    Literal(node) {
      if (typeof node.value !== "string" || isPropertyKey(node)) return
      if (inClassContext(node)) onString(node, node.value)
    },
    TemplateElement(node) {
      if (inClassContext(node.parent)) onString(node, node.value.cooked ?? node.value.raw)
    },
  }
}

export function splitClasses(text) {
  return text.split(/\s+/).filter(Boolean)
}

export function baseUtility(cls) {
  let depth = 0
  let cut = -1
  for (let i = 0; i < cls.length; i++) {
    const c = cls[i]
    if (c === "[" || c === "(") depth++
    else if (c === "]" || c === ")") depth--
    else if (c === ":" && depth === 0) cut = i
  }
  return cls.slice(cut + 1).replace(/^!/, "").replace(/!$/, "")
}

export function stripOpacity(base) {
  if (base.endsWith("]") || base.endsWith(")")) return base
  return base.replace(/\/[^/]+$/, "")
}
