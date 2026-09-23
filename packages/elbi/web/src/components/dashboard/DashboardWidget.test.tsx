// Pins a table widget's interactions: a row click emits its fields into the mapped
// variables (cross-filter), and Details runs the drill-through.

import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import type { Widget } from "@/lib/dashboards"

vi.mock("@/components/viz/VizView", () => ({ VizView: () => <div data-testid="viz" /> }))

import { DashboardWidget } from "./DashboardWidget"

const WIDGET: Widget = {
  id: "w1",
  type: "table",
  title: "By region",
  gridPos: { x: 0, y: 0, w: 6, h: 4 },
  interactions: {
    crossFilter: { emit: { region: "datum.region" } },
    drillThrough: { target: "d2" },
  },
}
const DATA = {
  widgetId: "w1",
  derivation: "sales",
  kind: "table",
  value: [
    { region: "west", revenue: 12 },
    { region: "east", revenue: 9 },
  ],
  dataVersion: "v1",
  error: null,
}

describe("DashboardWidget table", () => {
  it("cross-filters on the clicked row", async () => {
    const user = userEvent.setup()
    const onCrossFilter = vi.fn()
    render(
      <DashboardWidget widget={WIDGET} data={DATA} variables={{}} onCrossFilter={onCrossFilter} />,
    )
    expect(screen.getAllByRole("columnheader").map((h) => h.textContent)).toEqual([
      "region",
      "revenue",
    ])
    await user.click(screen.getByRole("cell", { name: "east" }))
    expect(onCrossFilter).toHaveBeenCalledWith({ region: "east" })
  })

  it("drills through from Details", async () => {
    const user = userEvent.setup()
    const onDrillThrough = vi.fn()
    render(
      <DashboardWidget
        widget={WIDGET}
        data={DATA}
        variables={{}}
        onDrillThrough={onDrillThrough}
      />,
    )
    await user.click(screen.getByRole("button", { name: /Details/ }))
    expect(onDrillThrough).toHaveBeenCalledOnce()
  })
})
