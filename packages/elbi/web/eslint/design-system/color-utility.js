import { stripOpacity } from "./class-strings.js"

const PALETTE =
  "slate|gray|zinc|neutral|stone|red|orange|amber|yellow|lime|green|emerald|teal|cyan|sky|blue|indigo|violet|purple|fuchsia|pink|rose"
const DEFAULT_PALETTE = new RegExp(`^(?:${PALETTE})-(?:50|[1-9]00|950)$`)
const KEYWORDS = new Set(["inherit", "current", "transparent", "black", "white"])
const BANNED = {
  "text-muted-foreground": "use text-text-tertiary (same value)",
  "text-secondary": "--secondary is a background grey; use text-text-secondary",
}
const TEXT_SIZES = "xs|sm|base|lg|xl|[2-9]xl"
// Longest prefix first; each maps to the values that make it a non-colour utility.
const PREFIXES = [
  ["outline-offset", /.*/],
  ["ring-offset", /^\d+$/],
  ["divide-x", /^(?:\d+|reverse)$/],
  ["divide-y", /^(?:\d+|reverse)$/],
  ...["x", "y", "t", "r", "b", "l", "s", "e"].map((s) => [`border-${s}`, /^\d+$/]),
  ["border", /^(?:\d+|[xytrblse]|solid|dashed|dotted|double|hidden|none|collapse|separate|spacing(?:-.+)?)$/],
  ["ring", /^(?:\d+|inset)$/],
  ["outline", /^(?:\d+|none|hidden|solid|dashed|dotted|double)$/],
  ["divide", /^(?:x|y|solid|dashed|dotted|double|none)$/],
  ["text", new RegExp(`^(?:${TEXT_SIZES}|left|center|right|justify|start|end|wrap|nowrap|balance|pretty|ellipsis|clip)$`)],
  ["bg", /^(?:fixed|local|scroll|clip-.+|origin-.+|repeat.*|no-repeat|cover|contain|auto|center|top|bottom|left|right|(?:left|right)-(?:top|bottom)|none|gradient-to-.+|linear-.+|radial.*|conic.*|blend-.+)$/],
  ["fill", /^none$/],
  ["stroke", /^(?:\d+|none)$/],
  ["from", /^\d+%$/],
  ["via", /^\d+%$/],
  ["to", /^\d+%$/],
  ["decoration", /^(?:\d+|solid|double|dotted|dashed|wavy|from-font|auto|clone|slice)$/],
  ["caret", /^auto$/],
  ["accent", /^auto$/],
]

export function classifyColorUtility(cls, theme) {
  const hit = PREFIXES.find(([p]) => cls.startsWith(`${p}-`))
  if (!hit) return null
  const [prefix, nonColor] = hit
  const value = stripOpacity(cls.slice(prefix.length + 1))
  if (value.startsWith("[") || value.startsWith("(")) return null
  if (nonColor.test(value)) return null
  if (prefix === "text" && theme.textSizes.has(value)) return null
  const name = `${prefix}-${value}`
  if (Object.hasOwn(BANNED, name)) return `"${cls}": ${BANNED[name]}.`
  if (/^n-\d+$/.test(value)) return `"${cls}" uses the raw ramp: use a semantic token.`
  if (DEFAULT_PALETTE.test(value)) return `"${cls}" is Tailwind's default palette: use a semantic token (success, warning, danger, info, ...).`
  if (KEYWORDS.has(value) || theme.colors.has(value)) return null
  return `"${cls}" names no colour token in src/index.css @theme.`
}
