const CLASS_CALLEES = new Set(["cn", "clsx", "cva", "twMerge"])
// Under cva(...), object keys are variant names (e.g. `default`, `icon-xs`), not class strings.
const KEY_CLASS_CALLEES = new Set(["cn", "clsx", "twMerge"])
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

// The nearest enclosing class-callee call, e.g. the `cn` in `cn({ "a": x })`.
function nearestClassCallee(node) {
  for (let p = node.parent; p; p = p.parent) {
    if (p.type === "CallExpression" && p.callee.type === "Identifier" && CLASS_CALLEES.has(p.callee.name)) {
      return p.callee.name
    }
    if (p.type === "Program") return null
  }
  return null
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
    // `cn({ "shadow-sm": active, phc: legacy })`: object keys are class strings under
    // cn/clsx/twMerge, but not cva, where keys are variant names.
    Property(node) {
      if (node.computed) return
      const key = node.key
      const text =
        key.type === "Literal" && typeof key.value === "string"
          ? key.value
          : key.type === "Identifier"
            ? key.name
            : null
      if (text === null) return
      const callee = nearestClassCallee(node)
      if (callee && KEY_CLASS_CALLEES.has(callee)) onString(key, text)
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
