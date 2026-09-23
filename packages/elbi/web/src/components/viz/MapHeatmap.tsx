import { HexagonLayer } from "@deck.gl/aggregation-layers"
import DeckGL from "@deck.gl/react"
import { useMemo } from "react"
import { Map as MapLibre } from "react-map-gl/maplibre"
import "maplibre-gl/dist/maplibre-gl.css"

// A literal map of a verified spatial surface: the certified rows are binned into equal-area
// hexagons on the GPU and coloured by the mean of the value column: the aggregation deck.gl
// performs is deterministic, so the map is a faithful view of the data the oracle checked, not
// a smoothed interpolation that invents values.

type Row = Record<string, string | number>
type Encoding = { lat: string; long: string; value: string | null }
type Point = { position: [number, number]; value: number }

// A light-to-violet sequential ramp (brand-aligned), low mean to high mean.
const COLOR_RANGE: [number, number, number][] = [
  [237, 231, 246],
  [209, 196, 233],
  [179, 157, 219],
  [149, 117, 205],
  [126, 87, 194],
  [94, 53, 177],
]

const CARTO_POSITRON = "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json"

function toPoints(rows: Row[], encoding: Encoding): Point[] {
  const out: Point[] = []
  for (const r of rows) {
    const lng = Number(r[encoding.long])
    const lat = Number(r[encoding.lat])
    const value = encoding.value ? Number(r[encoding.value]) : 1
    if (Number.isFinite(lng) && Number.isFinite(lat) && Number.isFinite(value)) {
      out.push({ position: [lng, lat], value })
    }
  }
  return out
}

function fitView(points: Point[]) {
  let minX = Infinity,
    minY = Infinity,
    maxX = -Infinity,
    maxY = -Infinity
  for (const { position } of points) {
    minX = Math.min(minX, position[0])
    maxX = Math.max(maxX, position[0])
    minY = Math.min(minY, position[1])
    maxY = Math.max(maxY, position[1])
  }
  if (!Number.isFinite(minX)) return { longitude: 0, latitude: 0, zoom: 1 }
  const span = Math.max(maxX - minX, maxY - minY) || 0.1
  return {
    longitude: (minX + maxX) / 2,
    latitude: (minY + maxY) / 2,
    zoom: Math.min(14, Math.max(3, Math.log2(360 / span) - 1)),
    pitch: 0,
    bearing: 0,
  }
}

export function MapHeatmap({ rows, encoding }: { rows: Row[]; encoding: Encoding }) {
  const points = useMemo(() => toPoints(rows, encoding), [rows, encoding])
  const initialViewState = useMemo(() => fitView(points), [points])
  // hex radius in metres, scaled to the data's extent so a city fills with ~40 hexes
  const radius = useMemo(() => {
    const spanKm = (initialViewState.zoom ?? 10) < 9 ? 3000 : 500
    return spanKm
  }, [initialViewState])

  const layer = new HexagonLayer<Point>({
    id: "hexagon",
    data: points,
    getPosition: (d) => d.position,
    getColorWeight: (d) => d.value,
    colorAggregation: "MEAN",
    colorRange: COLOR_RANGE,
    radius,
    coverage: 0.9,
    opacity: 0.75,
    pickable: true,
  })

  if (!points.length) {
    return (
      <div className="grid h-64 place-items-center text-sm text-text-tertiary">
        no mappable coordinates in the result
      </div>
    )
  }

  return (
    <div className="relative h-[420px] w-full overflow-hidden rounded-xl border border-border">
      <DeckGL
        initialViewState={initialViewState}
        controller
        layers={[layer]}
        getTooltip={({ object }) =>
          object
            ? {
                text: `mean ${encoding.value ?? "count"}: ${
                  object.colorValue?.toLocaleString?.() ?? object.colorValue
                }\n${object.points?.length ?? 0} points`,
              }
            : null
        }
      >
        <MapLibre mapStyle={CARTO_POSITRON} />
      </DeckGL>
    </div>
  )
}
