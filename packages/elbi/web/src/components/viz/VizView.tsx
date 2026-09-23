import { lazy, Suspense } from "react"
import { VegaChart } from "./VegaChart"

// The visualization dispatcher. A verified result carries {spec, rows}: a Vega-Lite spec the
// model emitted (or an engine default) plus the certified rows. Almost everything renders
// through Vega-Lite: any mark, transform, layer. The one exception is a geographic tile map,
// which Vega-Lite can't draw: when the spec encodes latitude and longitude, it routes to
// deck.gl/MapLibre (lazy-loaded, so the heavy geospatial bundle ships only for a map).

const MapHeatmap = lazy(() => import("./MapHeatmap").then((m) => ({ default: m.MapHeatmap })))

type Field = { field?: string }

export interface VizDescriptor {
  spec: Record<string, unknown>
  rows: Record<string, string | number>[]
}

export function VizView({
  viz,
  height,
  bare = false,
}: {
  viz: VizDescriptor
  // In a dashboard cell, "container" fills the frame and `bare` drops the chart's
  // own chrome; omitted (chat), the chart keeps its default 340px bordered card.
  height?: number | "container"
  bare?: boolean
}) {
  const encoding = (viz.spec?.encoding ?? {}) as Record<string, Field>
  const lat = encoding.latitude?.field
  const long = encoding.longitude?.field
  if (lat && long) {
    const value = encoding.color?.field ?? encoding.size?.field ?? null
    return (
      <Suspense
        fallback={
          <div className="grid h-full min-h-40 place-items-center text-sm text-text-tertiary">
            loading map…
          </div>
        }
      >
        <MapHeatmap rows={viz.rows} encoding={{ lat, long, value }} />
      </Suspense>
    )
  }
  return <VegaChart spec={viz.spec} rows={viz.rows} height={height} bare={bare} />
}
