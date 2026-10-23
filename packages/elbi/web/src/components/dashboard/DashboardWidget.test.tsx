// Pins a table widget's interactions: a row click emits its fields into the mapped
// variables (cross-filter), and Details runs the drill-through.

import { fireEvent, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"

import type { Widget, WidgetData } from "@/lib/dashboards"

vi.mock("@/components/viz/VizView", () => ({ VizView: () => <div data-testid="viz" /> }))
// The real CSV encoder; only the file save is replaced, since jsdom cannot save files.
const downloadText = vi.fn()
vi.mock("@/lib/download", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/download")>()),
  downloadText: (...args: unknown[]) => downloadText(...args),
}))

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

describe("DashboardWidget metric", () => {
  const KPI: Widget = {
    id: "mrr",
    type: "metric",
    gridPos: { x: 0, y: 0, w: 6, h: 4 },
    bind: { metric: "mrr" },
  }
  const resolved = (value: unknown, format: WidgetData["format"]): WidgetData => ({
    widgetId: "mrr",
    derivation: "mrr",
    kind: "table",
    value,
    dataVersion: null,
    error: null,
    format,
  })

  it("shows the shared metric's value in the metric's own format", () => {
    render(
      <DashboardWidget
        widget={KPI}
        data={resolved([{ mrr: 0.125 }], { kind: "percent", precision: 1 })}
        variables={{}}
      />,
    )
    expect(screen.getByText("12.5%")).toBeInTheDocument()
  })

  it("says which metric came back empty rather than rendering a silent dash", () => {
    render(<DashboardWidget widget={KPI} data={resolved([], null)} variables={{}} />)
    expect(screen.getByText("no value for metric “mrr”")).toBeInTheDocument()
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

  it("downloads a data tile's rows as CSV, on a dashboard that cannot be edited too", async () => {
    const user = userEvent.setup()
    render(<DashboardWidget widget={WIDGET} data={DATA} variables={{}} />)

    const trigger = screen.getByLabelText(/Tile actions/)
    trigger.focus()
    await user.click(trigger)
    await user.click(await screen.findByRole("menuitem", { name: "Download CSV" }))
    expect(downloadText).toHaveBeenCalledWith(
      "by_region.csv",
      "region,revenue\nwest,12\neast,9",
      "text/csv",
    )
  })

  it("downloads the columns the tile shows, in its order, and keys the first row lacks", async () => {
    const user = userEvent.setup()
    const download = async (widget: Widget, value: unknown) => {
      downloadText.mockClear()
      const { unmount } = render(
        <DashboardWidget widget={widget} data={{ ...DATA, value } as WidgetData} variables={{}} />,
      )
      const trigger = screen.getByLabelText(/Tile actions/)
      trigger.focus()
      await user.click(trigger)
      await user.click(await screen.findByRole("menuitem", { name: "Download CSV" }))
      unmount()
      return downloadText.mock.calls[0][1]
    }
    const rows = [
      { region: "west", revenue: 12 },
      { region: "east", revenue: 9, note: "late" },
    ]

    expect(await download({ ...WIDGET, viz: { columns: ["revenue", "region"] } }, rows)).toBe(
      "revenue,region\n12,west\n9,east",
    )
    expect(await download(WIDGET, rows)).toBe("region,revenue,note\nwest,12,\neast,9,late")
  })

  it("keeps the menu from starting a drag", () => {
    render(<DashboardWidget widget={widget} variables={{}} onEdit={() => {}} />)

    // react-grid-layout's draggableCancel; without it, opening the menu drags the tile.
    expect(screen.getByLabelText(/Tile actions/).className).toContain("dash-no-drag")
  })
})

describe("a table tile's columns", () => {
  const colTile = (columns: unknown[]): Widget => ({
    id: "t",
    type: "table",
    title: "Evidence",
    gridPos: { x: 0, y: 0, w: 24, h: 8 },
    viz: { columns },
  })
  const colData = (rows: Record<string, string | number>[]): WidgetData => ({
    widgetId: "t",
    derivation: "evidence",
    kind: "table",
    value: rows,
    dataVersion: "v1",
    error: null,
  })
  const ROWS = [
    { unit: "chat", reason: "Top of the peer range", source: "https://modal.com/pricing" },
  ]

  it("labels a column with its configured label and keeps the field for the data", () => {
    render(
      <DashboardWidget
        widget={colTile(["unit", { field: "reason", label: "Why" }])}
        data={colData(ROWS)}
        variables={{}}
      />,
    )
    expect(screen.getAllByRole("columnheader").map((h) => h.textContent)).toEqual(["unit", "Why"])
    expect(screen.getByRole("cell", { name: "Top of the peer range" })).toBeInTheDocument()
  })

  it("gives a column its configured pixel width", () => {
    render(
      <DashboardWidget
        widget={colTile(["unit", { field: "reason", width: 420 }])}
        data={colData(ROWS)}
        variables={{}}
      />,
    )
    const header = screen.getByRole("columnheader", { name: "reason" })
    expect(header.style.width).toBe("420px")
  })

  it("sizes a fit column to its content and lets a fill column take the rest", () => {
    render(
      <DashboardWidget
        widget={colTile([
          { field: "unit", width: "fit" },
          { field: "reason", width: "fill" },
        ])}
        data={colData(ROWS)}
        variables={{}}
      />,
    )
    expect(screen.getByRole("cell", { name: "chat" })).toHaveClass("whitespace-nowrap")
    expect(screen.getByRole("columnheader", { name: "reason" }).style.width).toBe("")
  })

  it("does not wrap a column that says wrap is false", () => {
    render(
      <DashboardWidget
        widget={colTile([{ field: "reason", width: 200, wrap: false }])}
        data={colData(ROWS)}
        variables={{}}
      />,
    )
    expect(screen.getByRole("cell", { name: "Top of the peer range" })).toHaveClass(
      "whitespace-nowrap",
    )
  })

  it("downloads exactly the configured fields, whatever shape each column was given", async () => {
    const user = userEvent.setup()
    downloadText.mockClear()
    render(
      <DashboardWidget
        widget={colTile(["unit", { field: "source", width: 300 }])}
        data={colData(ROWS)}
        variables={{}}
      />,
    )
    await user.click(screen.getByRole("button", { name: /Tile actions/ }))
    await user.click(screen.getByRole("menuitem", { name: /CSV/i }))
    expect(downloadText.mock.calls[0][1]).toBe("unit,source\nchat,https://modal.com/pricing")
  })
})

describe("links in a table cell", () => {
  const tile: Widget = {
    id: "t",
    type: "table",
    gridPos: { x: 0, y: 0, w: 24, h: 8 },
    interactions: { crossFilter: { emit: { unit: "datum.unit" } } },
  }
  const data = (source: string): WidgetData => ({
    widgetId: "t",
    derivation: "evidence",
    kind: "table",
    value: [{ unit: "training", source }],
    dataVersion: "v1",
    error: null,
  })

  it("renders a web address as a link that opens in a new tab", () => {
    render(
      <DashboardWidget widget={tile} data={data("https://modal.com/pricing")} variables={{}} />,
    )
    const link = screen.getByRole("link", { name: "https://modal.com/pricing" })
    expect(link).toHaveAttribute("href", "https://modal.com/pricing")
    expect(link).toHaveAttribute("target", "_blank")
    expect(link.getAttribute("rel")).toContain("noopener")
    expect(link.getAttribute("rel")).toContain("noreferrer")
  })

  it("renders a markdown link as its label", () => {
    render(
      <DashboardWidget
        widget={tile}
        data={data("see [Modal pricing](https://modal.com/pricing)")}
        variables={{}}
      />,
    )
    expect(screen.getByRole("link", { name: "Modal pricing" })).toHaveAttribute(
      "href",
      "https://modal.com/pricing",
    )
  })

  it("does not make a link out of a script address", () => {
    render(
      <DashboardWidget widget={tile} data={data("[click](javascript:alert(1))")} variables={{}} />,
    )
    expect(screen.queryByRole("link")).toBeNull()
  })

  it("does not cross-filter when a link is clicked", async () => {
    const user = userEvent.setup()
    const onCrossFilter = vi.fn()
    render(
      <DashboardWidget
        widget={tile}
        data={data("https://modal.com/pricing")}
        variables={{}}
        onCrossFilter={onCrossFilter}
      />,
    )
    const link = screen.getByRole("link")
    link.addEventListener("click", (e) => e.preventDefault())
    await user.click(link)
    expect(onCrossFilter).not.toHaveBeenCalled()
  })
})

describe("resizing a table column", () => {
  const tile: Widget = {
    id: "t",
    type: "table",
    gridPos: { x: 0, y: 0, w: 24, h: 8 },
    viz: { columns: ["unit", { field: "reason", width: 300 }] },
  }
  const data: WidgetData = {
    widgetId: "t",
    derivation: "evidence",
    kind: "table",
    value: [{ unit: "chat", reason: "why" }],
    dataVersion: "v1",
    error: null,
  }

  it("shows no handles when nothing can save a width", () => {
    render(<DashboardWidget widget={tile} data={data} variables={{}} />)
    expect(screen.queryByRole("separator")).toBeNull()
  })

  it("saves the new width when a header edge is dragged and released", () => {
    const onColumnWidth = vi.fn()
    render(
      <DashboardWidget widget={tile} data={data} variables={{}} onColumnWidth={onColumnWidth} />,
    )
    const handle = screen.getByRole("separator", { name: "Resize column reason" })
    fireEvent.mouseDown(handle, { clientX: 100 })
    fireEvent.mouseMove(window, { clientX: 160 })
    expect(onColumnWidth).not.toHaveBeenCalled()
    fireEvent.mouseUp(window, { clientX: 160 })
    expect(onColumnWidth).toHaveBeenCalledExactlyOnceWith("reason", 360)
  })

  it("saves nothing when the edge was grabbed and not moved", () => {
    const onColumnWidth = vi.fn()
    render(
      <DashboardWidget widget={tile} data={data} variables={{}} onColumnWidth={onColumnWidth} />,
    )
    const handle = screen.getByRole("separator", { name: "Resize column reason" })
    fireEvent.mouseDown(handle, { clientX: 100 })
    fireEvent.mouseUp(window, { clientX: 100 })
    expect(onColumnWidth).not.toHaveBeenCalled()
  })

  it("returns a column to its content width on a double click", () => {
    const onColumnWidth = vi.fn()
    render(
      <DashboardWidget widget={tile} data={data} variables={{}} onColumnWidth={onColumnWidth} />,
    )
    fireEvent.doubleClick(screen.getByRole("separator", { name: "Resize column reason" }))
    expect(onColumnWidth).toHaveBeenCalledExactlyOnceWith("reason", "fit")
  })

  it("resizes from the keyboard", async () => {
    const user = userEvent.setup()
    const onColumnWidth = vi.fn()
    render(
      <DashboardWidget widget={tile} data={data} variables={{}} onColumnWidth={onColumnWidth} />,
    )
    screen.getByRole("separator", { name: "Resize column reason" }).focus()
    await user.keyboard("{ArrowRight}")
    expect(onColumnWidth).toHaveBeenLastCalledWith("reason", 310)
    await user.keyboard("{ArrowLeft}")
    expect(onColumnWidth).toHaveBeenLastCalledWith("reason", 290)
  })

  it("does not start a row click or a tile drag from the handle", () => {
    const onCrossFilter = vi.fn()
    const crossTile: Widget = {
      ...tile,
      interactions: { crossFilter: { emit: { unit: "datum.unit" } } },
    }
    render(
      <DashboardWidget
        widget={crossTile}
        data={data}
        variables={{}}
        onCrossFilter={onCrossFilter}
        onColumnWidth={vi.fn()}
      />,
    )
    const handle = screen.getByRole("separator", { name: "Resize column reason" })
    expect(handle.closest(".dash-no-drag")).not.toBeNull()
    fireEvent.click(handle)
    expect(onCrossFilter).not.toHaveBeenCalled()
  })
})
