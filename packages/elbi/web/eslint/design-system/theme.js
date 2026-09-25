import { readFileSync } from "node:fs"

const cache = new Map()

// Only declarations inside `@theme` blocks become utilities; `:root` variables do not.
function themeBlocks(css) {
  const blocks = []
  for (const m of css.matchAll(/@theme\b[^{]*\{/g)) {
    let depth = 1
    let i = m.index + m[0].length
    const start = i
    while (i < css.length && depth > 0) {
      if (css[i] === "{") depth++
      else if (css[i] === "}") depth--
      i++
    }
    blocks.push(css.slice(start, i - 1))
  }
  return blocks.join("\n")
}

export function readTheme(cssPath) {
  if (cache.has(cssPath)) return cache.get(cssPath)
  const body = themeBlocks(readFileSync(cssPath, "utf8"))
  const names = (prefix) =>
    new Set(
      [...body.matchAll(new RegExp(`--${prefix}-([a-z0-9-]+)\\s*:`, "g"))]
        .map((m) => m[1])
        .filter((n) => !n.includes("--")),
    )
  const theme = { colors: names("color"), textSizes: names("text") }
  cache.set(cssPath, theme)
  return theme
}
