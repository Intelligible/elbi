// The single source of truth for chart color, derived from the design tokens in index.css.
// Charts are the product's dominant visual surface, so they must draw from the same palette and
// neutrals as everything else and adapt to the active theme, rather than carrying their own
// hardcoded colors. Tokens are authored in OKLCH (and some as color-mix), which a canvas/SVG
// chart renderer may not parse directly, so we resolve each expression to the concrete rgb()
// the browser computes for the *currently active* theme via a hidden probe.

const DATA_VARS = Array.from({ length: 15 }, (_, i) => `var(--data-${i + 1})`)

function resolve(exprs: string[]): string[] {
  const probe = document.createElement("span")
  probe.style.display = "none"
  document.body.appendChild(probe)
  const out = exprs.map((expr) => {
    probe.style.color = ""
    probe.style.color = expr
    return getComputedStyle(probe).color || expr
  })
  probe.remove()
  return out
}

export interface ChartTheme {
  text: string
  textSecondary: string
  grid: string
  domain: string
  palette: string[]
}

// Read the design tokens against the current theme. Call at render time (and re-run
// when the theme flips) so the returned colors match what the rest of the UI shows.
export function chartTheme(): ChartTheme {
  const [text, textSecondary, grid, domain, ...palette] = resolve([
    "var(--foreground)",
    "var(--text-secondary)",
    "var(--border)",
    "var(--border-strong)",
    ...DATA_VARS,
  ])
  return { text, textSecondary, grid, domain, palette }
}

// Built configs, one per resolved theme. The tokens come from computed styles, which
// forces a style recalculation to read -- once per theme is enough, and every chart on
// a page then shares the answer.
const CONFIG_CACHE = new Map<"light" | "dark", Record<string, unknown>>()

// A Vega-Lite `config` block that themes chart chrome from the tokens and sets the
// categorical/ordinal/ramp color range to the brand data-viz palette, so marks use
// the designed colors instead of Vega's defaults. Transparent background so the
// chart sits on whatever surface is behind it.
//
// `resolved` names the theme the tokens are read under. It is a real argument, not
// decoration: it keys the cache, and it is what lets a caller memoizing this config
// declare the dependency it actually has -- reading the document alone leaves that
// dependency invisible to everything, readers and tooling alike.
export function vegaThemeConfig(resolved: "light" | "dark"): Record<string, unknown> {
  const cached = CONFIG_CACHE.get(resolved)
  if (cached) return cached
  const t = chartTheme()
  const config = {
    background: "transparent",
    view: { stroke: "transparent" },
    title: { color: t.text, subtitleColor: t.textSecondary },
    axis: {
      labelColor: t.textSecondary,
      titleColor: t.text,
      gridColor: t.grid,
      domainColor: t.domain,
      tickColor: t.domain,
    },
    legend: { labelColor: t.textSecondary, titleColor: t.text },
    range: { category: t.palette, ordinal: t.palette, ramp: t.palette },
  }
  CONFIG_CACHE.set(resolved, config)
  return config
}
