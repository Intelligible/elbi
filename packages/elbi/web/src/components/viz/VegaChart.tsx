import { useEffect, useMemo, useRef, useState } from "react"
import { VegaEmbed } from "react-vega"
import type { VisualizationSpec } from "vega-embed"
import { useTheme } from "@/hooks/useTheme"
import { vegaThemeConfig } from "@/lib/chart-theme"

// Renders the model's chart by embedding its Vega-Lite spec: the full grammar of graphics, so
// any visualization is expressible. We inject the certified rows as the data (the spec never
// carries data), so the chart is always a view of exactly what the oracle checked. Values
// arrive as strings, so all-numeric columns are coerced to numbers before rendering; Vega-Lite
// derives axes, scales, and legends.

type Row = Record<string, string | number>

// Strip external data sources from every `data` object in the (LLM-authored) spec, at any
// depth: a nested layer/facet/concat sub-spec or a `lookup` transform's `from.data` can carry
// its own `url`, and overriding only the top-level data would leave those to fetch an arbitrary
// URL from the viewer's browser. Inline `values` are kept; only the certified rows we inject
// and any literal values remain.
function stripDataSources(node: unknown): unknown {
  if (Array.isArray(node)) return node.map(stripDataSources)
  if (node && typeof node === "object") {
    const out: Record<string, unknown> = {}
    for (const [key, value] of Object.entries(node as Record<string, unknown>)) {
      if (key === "data" && value && typeof value === "object" && !Array.isArray(value)) {
        const data = { ...(value as Record<string, unknown>) }
        delete data.url
        delete data.name
        out[key] = stripDataSources(data)
      } else {
        out[key] = stripDataSources(value)
      }
    }
    return out
  }
  return node
}

function coerce(rows: Row[]): Row[] {
  if (!rows.length) return rows
  const columns = Object.keys(rows[0])
  const numeric = columns.filter(
    (c) =>
      rows.some((r) => r[c] !== "" && r[c] != null) &&
      rows.every((r) => r[c] === "" || r[c] == null || Number.isFinite(Number(r[c]))),
  )
  return rows.map((r) => {
    const out: Row = { ...r }
    for (const c of numeric) {
      if (out[c] !== "" && out[c] != null) out[c] = Number(out[c])
    }
    return out
  })
}

export function VegaChart({
  spec,
  rows,
  height = 340,
  bare = false,
}: {
  spec: Record<string, unknown>
  rows: Row[]
  // A fixed pixel height (chat, default 340) or "container" to fill the parent
  // (a dashboard cell). `bare` drops the component's own border/padding so a
  // dashboard widget frame provides the chrome instead.
  height?: number | "container"
  bare?: boolean
}) {
  const { resolved } = useTheme()
  const frame = useRef<HTMLDivElement>(null)
  // Vega's own "container" height is measured once, at embed. A dashboard tile is
  // resizable, and the chart kept the height it was born with while the tile grew
  // around it, leaving the new space as padding. Measuring the frame ourselves and
  // passing pixels re-renders the chart whenever the tile changes size.
  const [measured, setMeasured] = useState(0)
  useEffect(() => {
    const element = frame.current
    if (height !== "container" || !element || typeof ResizeObserver === "undefined") return
    const observer = new ResizeObserver(([entry], _observer) => {
      setMeasured(Math.round(entry.contentRect.height))
    })
    observer.observe(element)
    return () => observer.disconnect()
  }, [height])

  const full = useMemo<VisualizationSpec>(() => {
    const authored = stripDataSources(spec) as Record<string, unknown>
    // Below about this, the axes and legend have nowhere to go and Vega warns about a
    // negative plotting area; let the tile clip instead of rendering a broken chart.
    const resolvedHeight = height === "container" ? Math.max(measured, 60) : height
    return {
      width: "container",
      height: resolvedHeight,
      autosize: { type: "fit", contains: "padding" },
      ...authored,
      config: {
        ...vegaThemeConfig(resolved),
        ...((authored.config as Record<string, unknown>) ?? {}),
      },
      data: { values: coerce(rows) },
    } as unknown as VisualizationSpec
  }, [spec, rows, height, measured, resolved])
  return (
    <div
      ref={frame}
      className={bare ? "h-full w-full" : "w-full rounded-xl border border-border p-3"}
    >
      <VegaEmbed
        spec={full}
        options={{ actions: false, mode: "vega-lite" }}
        className={bare ? "h-full w-full" : "w-full"}
      />
    </div>
  )
}
