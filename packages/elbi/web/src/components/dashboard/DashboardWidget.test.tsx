import { render, screen } from "@testing-library/react"
import { describe, expect, it } from "vitest"
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
