// Pins a table widget's interactions: a row click emits its fields into the mapped
// variables (cross-filter), and Details runs the drill-through.

import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import type { Widget, WidgetData } from "@/lib/dashboards"

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

const pos = { x: 0, y: 0, w: 12, h: 6 }

describe("a text widget", () => {
  it("renders its own content as markdown, not as literal asterisks", () => {
    const widget: Widget = {
      id: "note",
      type: "text",
      gridPos: pos,
      content: "**Read the window**, not the day.",
    }

    render(<DashboardWidget widget={widget} variables={{}} />)

    expect(screen.getByText("Read the window").tagName).toBe("STRONG")
  })

  it("renders the markdown a bound derivation returned", () => {
    const widget: Widget = {
      id: "verdict",
      type: "text",
      gridPos: pos,
      bind: { derivation: "cost_fixed_share" },
    }
    const data: WidgetData = {
      widgetId: "verdict",
      derivation: "cost_fixed_share",
      kind: "markdown",
      value: "**63%** of AWS spend does not move with customer usage.",
      dataVersion: null,
      error: null,
    }

    render(<DashboardWidget widget={widget} data={data} variables={{}} />)

    expect(screen.getByText("63%").tagName).toBe("STRONG")
    expect(screen.getByText(/of AWS spend does not move with customer usage/)).toBeInTheDocument()
  })

  it("reports a bound tile's error rather than rendering an empty card", () => {
    const widget: Widget = {
      id: "verdict",
      type: "text",
      gridPos: pos,
      bind: { derivation: "missing" },
    }
    const data: WidgetData = {
      widgetId: "verdict",
      derivation: "missing",
      kind: null,
      value: null,
      dataVersion: null,
      error: "unknown derivation 'missing'",
    }

    render(<DashboardWidget widget={widget} data={data} variables={{}} />)

    expect(screen.getByText(/unknown derivation 'missing'/)).toBeInTheDocument()
  })
})

describe("tile actions", () => {
  const widget: Widget = { id: "mrr", type: "text", gridPos: pos, content: "hi" }

  it("offers no menu on a dashboard that cannot be edited", () => {
    render(<DashboardWidget widget={widget} variables={{}} />)

    expect(screen.queryByLabelText(/Tile actions/)).toBeNull()
  })

  it("edits and deletes through the menu", async () => {
    const user = userEvent.setup()
    const onEdit = vi.fn()
    const onDelete = vi.fn()
    render(<DashboardWidget widget={widget} variables={{}} onEdit={onEdit} onDelete={onDelete} />)

    const trigger = screen.getByLabelText(/Tile actions/)
    // Radix menus close on a window blur, which jsdom fires at pointerdown when nothing
    // is focused; focusing first avoids it (jsdom only).
    trigger.focus()
    await user.click(trigger)
    await user.click(await screen.findByRole("menuitem", { name: "Edit…" }))
    expect(onEdit).toHaveBeenCalledOnce()

    trigger.focus()
    await user.click(trigger)
    await user.click(await screen.findByRole("menuitem", { name: "Delete" }))
    expect(onDelete).toHaveBeenCalledOnce()
  })

  it("keeps the menu from starting a drag", () => {
    render(<DashboardWidget widget={widget} variables={{}} onEdit={() => {}} />)

    // react-grid-layout's draggableCancel; without it, opening the menu drags the tile.
    expect(screen.getByLabelText(/Tile actions/).className).toContain("dash-no-drag")
  })
})
