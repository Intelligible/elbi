import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import type { Widget, WidgetData } from "@/lib/dashboards"
import { DashboardWidget } from "./DashboardWidget"

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

    await user.click(screen.getByLabelText(/Tile actions/))
    await user.click(screen.getByText("Edit…"))
    expect(onEdit).toHaveBeenCalledOnce()

    await user.click(screen.getByLabelText(/Tile actions/))
    await user.click(screen.getByText("Delete"))
    expect(onDelete).toHaveBeenCalledOnce()
  })

  it("keeps the menu from starting a drag", () => {
    render(<DashboardWidget widget={widget} variables={{}} onEdit={() => {}} />)

    // react-grid-layout's draggableCancel; without it, opening the menu drags the tile.
    expect(screen.getByLabelText(/Tile actions/).className).toContain("dash-no-drag")
  })
})
