import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"

// The provenance trail links to the derivation a tile binds.
vi.mock("./JsonEditor", () => ({
  JsonEditor: ({
    value,
    label,
    onChange,
  }: {
    value: string
    label: string
    onChange: (next: string) => void
  }) => <textarea aria-label={label} value={value} onChange={(e) => onChange(e.target.value)} />,
}))

vi.mock("@/lib/dashboards", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/dashboards")>()),
  derivationProvenance: vi.fn(async () => ({
    derivation: ["analyses_clean"],
    dataset: ["posthog_analyses"],
  })),
}))

vi.mock("@/lib/metrics", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/metrics")>()),
  listMetrics: vi.fn(async () => [
    { name: "subscriptions", type: "simple", source: "revenue_by_product" },
    { name: "mrr_usd", type: "simple", source: "revenue_by_product" },
  ]),
}))

import type { Widget } from "@/lib/dashboards"
import { derivationProvenance } from "@/lib/dashboards"
import { listMetrics } from "@/lib/metrics"
import { TileEditor, type TilePatch } from "./TileEditor"

const pos = { x: 0, y: 0, w: 12, h: 6 }
const textTile: Widget = { id: "note", type: "text", gridPos: pos, content: "before" }
const metricTile: Widget = {
  id: "mrr",
  type: "metric",
  gridPos: { x: 6, y: 2, w: 6, h: 3 },
  title: "Subscriptions (all)",
  bind: { metric: "subscriptions" },
}
const boundTile: Widget = {
  id: "mrr",
  type: "table",
  gridPos: { x: 6, y: 2, w: 6, h: 3 },
  bind: { derivation: "revenue_by_product" },
}

/**
 * The metric picker is filled from a request, and a Select opened before it lands keeps
 * the list it opened with. Wait for the metrics, then choose.
 */
async function metricsReady() {
  await waitFor(() => expect(vi.mocked(listMetrics)).toHaveBeenCalled())
  await new Promise((resolve) => setTimeout(resolve, 0))
}

async function pick(user: ReturnType<typeof userEvent.setup>, label: string, option: string) {
  await user.click(screen.getByLabelText(label))
  await user.click(await screen.findByRole("option", { name: option }))
}

function open(
  widget: Widget | null,
  props: { catalog?: string[]; onCancel?: () => void; onSave?: () => Promise<void> } = {},
) {
  const onSave = vi.fn<(id: string, patch: TilePatch) => Promise<void>>(
    props.onSave ?? (async () => {}),
  )
  const rendered = render(
    <MemoryRouter>
      <TileEditor
        widget={widget}
        catalog={props.catalog ?? ["revenue_by_product", "collected_revenue"]}
        columns={24}
        dark={false}
        onCancel={props.onCancel ?? (() => {})}
        onSave={onSave}
      />
    </MemoryRouter>,
  )
  return { onSave, ...rendered }
}

describe("TileEditor", () => {
  it("keeps the dialog open with the server's reason when a save is refused", async () => {
    const user = userEvent.setup()
    const onCancel = vi.fn()
    open(textTile, {
      onCancel,
      onSave: async () => {
        throw new Error("widget 'note': unknown type 'txt'")
      },
    })

    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(await screen.findByRole("alert")).toHaveTextContent("unknown type 'txt'")
    expect(screen.getByRole("dialog")).toBeInTheDocument()
    expect(onCancel).not.toHaveBeenCalled()
  })

  it("edits a static text tile's title and content", async () => {
    const user = userEvent.setup()
    const { onSave } = open(textTile)

    await user.type(screen.getByLabelText("Title"), "Caveat")
    const content = screen.getByLabelText("Content")
    await user.clear(content)
    await user.type(content, "after")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave).toHaveBeenCalledWith("note", {
      title: "Caveat",
      content: "after",
      gridPos: pos,
    })
  })

  it("shows the shared metric a metric tile binds", async () => {
    open(metricTile)
    await metricsReady()

    expect(screen.getByLabelText("Metric")).toHaveTextContent("subscriptions")
    expect(screen.queryByLabelText("Derivation")).toBeNull()
    expect(screen.queryByLabelText("Aggregate")).toBeNull()
  })

  it("offers the shared metrics, not columns to aggregate", async () => {
    const user = userEvent.setup()
    open(metricTile)
    await metricsReady()

    await user.click(screen.getByLabelText("Metric"))

    const offered = (await screen.findAllByRole("option")).map((o) => o.textContent)
    expect(offered).toEqual(["subscriptions", "mrr_usd"])
  })

  it("rebinds a metric tile to another metric", async () => {
    const user = userEvent.setup()
    const { onSave } = open(metricTile)
    await metricsReady()

    await pick(user, "Metric", "mrr_usd")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave).toHaveBeenCalledWith("mrr", expect.objectContaining({ metric: "mrr_usd" }))
  })

  it("asks a tile that aggregates a derivation itself to pick a shared metric", async () => {
    const user = userEvent.setup()
    const legacy: Widget = {
      ...metricTile,
      bind: { derivation: "revenue_by_product" },
      viz: { field: "subscriptions", agg: "sum" },
    }
    const { onSave } = open(legacy)
    await metricsReady()

    expect(screen.getByText(/computes its own figure from revenue_by_product/)).toBeInTheDocument()
    await pick(user, "Metric", "subscriptions")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave).toHaveBeenCalledWith("mrr", expect.objectContaining({ metric: "subscriptions" }))
  })

  it("warns about a metric that does not exist, and keeps it selectable", async () => {
    open({ ...metricTile, bind: { metric: "gone_away" } })
    await metricsReady()

    expect(await screen.findByText(/No metric named/)).toBeInTheDocument()
    expect(screen.getByLabelText("Metric")).toHaveTextContent("gone_away")
  })

  it("says when the metrics could not be loaded, rather than offering nothing", async () => {
    vi.mocked(listMetrics).mockRejectedValueOnce(new Error("503"))
    open(metricTile)

    expect(await screen.findByText(/Could not load the metrics/)).toBeInTheDocument()
    expect(screen.getByLabelText("Metric")).toHaveTextContent("subscriptions")
  })

  it("edits a table bound to a metric without writing a derivation over it", async () => {
    const user = userEvent.setup()
    const { onSave } = open({
      ...boundTile,
      bind: { metric: "subscriptions", groupBy: ["region"] },
    })

    expect(screen.queryByLabelText("Derivation")).toBeNull()
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave).toHaveBeenCalledTimes(1)
    expect(onSave.mock.calls[0]?.[1]).not.toHaveProperty("derivation")
  })

  it("resizes a tile without touching its position", async () => {
    const user = userEvent.setup()
    const { onSave } = open(boundTile)

    const width = screen.getByLabelText("Width")
    await user.clear(width)
    await user.type(width, "12")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave).toHaveBeenCalledWith(
      "mrr",
      expect.objectContaining({ gridPos: { x: 6, y: 2, w: 12, h: 3 } }),
    )
  })

  it("clamps a width wider than the page", async () => {
    const user = userEvent.setup()
    const { onSave } = open(boundTile)

    const width = screen.getByLabelText("Width")
    await user.clear(width)
    await user.type(width, "99")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave.mock.calls[0]?.[1].gridPos?.w).toBe(24)
  })

  it("rebinds a data tile rather than offering it a content box", async () => {
    const user = userEvent.setup()
    const { onSave } = open(boundTile)

    expect(screen.queryByLabelText("Content")).toBeNull()
    await pick(user, "Derivation", "collected_revenue")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave).toHaveBeenCalledWith(
      "mrr",
      expect.objectContaining({ derivation: "collected_revenue" }),
    )
  })

  it("warns about a binding nothing provides, and keeps it selectable", () => {
    // A tile bound to something the catalog no longer lists: the name has to survive
    // opening the dialog, or Save would quietly clear it.
    open(
      { ...boundTile, bind: { derivation: "no_such_derivation" } },
      {
        catalog: ["revenue_by_product"],
      },
    )

    expect(screen.getByText(/Nothing named/)).toBeInTheDocument()
    expect(screen.getByLabelText("Derivation")).toHaveTextContent("no_such_derivation")
  })

  it("shows the tile it was opened on, not the previous one", () => {
    const { rerender } = open(textTile)
    expect(screen.getByLabelText("Content")).toHaveValue("before")

    rerender(
      <MemoryRouter>
        <TileEditor
          widget={metricTile}
          catalog={[]}
          columns={24}
          dark={false}
          onCancel={() => {}}
          onSave={async () => {}}
        />
      </MemoryRouter>,
    )

    expect(screen.getByLabelText("Title")).toHaveValue("Subscriptions (all)")
    expect(screen.queryByLabelText("Content")).toBeNull()
  })

  it("comes back from the Lineage tab to the fields", async () => {
    const user = userEvent.setup()
    open(textTile)

    await user.click(screen.getByRole("button", { name: "Lineage" }))
    await user.click(screen.getByRole("button", { name: "Fields" }))

    expect(screen.getByRole("button", { name: "Fields" })).toHaveAttribute("aria-pressed", "true")
    expect(screen.getByLabelText("Content")).toHaveValue("before")
  })

  it("keeps a field edit across a visit to the Lineage tab", async () => {
    const user = userEvent.setup()
    open(textTile)
    await user.click(screen.getByRole("button", { name: "JSON" }))
    await user.click(screen.getByRole("button", { name: "Fields" }))

    await user.type(screen.getByLabelText("Title"), "Caveat")
    await user.click(screen.getByRole("button", { name: "Lineage" }))
    await user.click(screen.getByRole("button", { name: "Fields" }))

    expect(screen.getByLabelText("Title")).toHaveValue("Caveat")
  })

  it("never fills one tile's fields from the JSON of the tile opened before it", async () => {
    // The dashboard keeps one editor mounted and hands it each tile in turn.
    const user = userEvent.setup()
    const onSave = vi.fn(async () => {})
    const editor = (widget: Widget) => (
      <MemoryRouter>
        <TileEditor
          widget={widget}
          catalog={["revenue_by_product"]}
          columns={24}
          dark={false}
          onCancel={() => {}}
          onSave={onSave}
        />
      </MemoryRouter>
    )
    const { rerender } = render(editor(boundTile))
    await user.click(screen.getByRole("button", { name: "JSON" }))
    rerender(editor(textTile))

    await user.click(screen.getByRole("button", { name: "Lineage" }))
    await user.click(screen.getByRole("button", { name: "Fields" }))
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave).toHaveBeenCalledWith(
      "note",
      expect.objectContaining({ title: "", content: "before" }),
    )
  })

  it("renders nothing when no tile is open", () => {
    const { container } = open(null)
    expect(container).toBeEmptyDOMElement()
  })
})

describe("the JSON tab", () => {
  it("shows the tile exactly as the spec stores it", async () => {
    const user = userEvent.setup()
    open({ ...boundTile, viz: { mark: "bar" } })

    await user.click(screen.getByRole("button", { name: "JSON" }))

    const config = JSON.parse(
      (screen.getByLabelText("This tile's config") as HTMLTextAreaElement).value,
    )
    expect(config.id).toBe("mrr")
    expect(config.bind).toEqual({ derivation: "revenue_by_product" })
    expect(config.viz).toEqual({ mark: "bar" })
  })

  it("carries an unsaved field edit into the JSON", async () => {
    const user = userEvent.setup()
    open(metricTile)
    await metricsReady()

    await pick(user, "Metric", "mrr_usd")
    await user.click(screen.getByRole("button", { name: "JSON" }))

    const config = JSON.parse(
      (screen.getByLabelText("This tile's config") as HTMLTextAreaElement).value,
    )
    expect(config.bind).toEqual({ metric: "mrr_usd" })
  })

  it("saves the whole widget, so a removed key is really removed", async () => {
    const user = userEvent.setup()
    const { onSave } = open(boundTile)

    await user.click(screen.getByRole("button", { name: "JSON" }))
    // fireEvent, not user.type: userEvent reads `{` as a key descriptor.
    fireEvent.change(screen.getByLabelText("This tile's config"), {
      target: {
        value: JSON.stringify({ id: "mrr", type: "metric", gridPos: { x: 6, y: 2, w: 6, h: 3 } }),
      },
    })
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave).toHaveBeenCalledWith("mrr", {
      widget: { id: "mrr", type: "metric", gridPos: { x: 6, y: 2, w: 6, h: 3 } },
    })
  })

  it("refuses to save text that is not JSON, and says why", async () => {
    const user = userEvent.setup()
    const { onSave } = open(boundTile)

    await user.click(screen.getByRole("button", { name: "JSON" }))
    fireEvent.change(screen.getByLabelText("This tile's config"), {
      target: { value: "{ not json" },
    })
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave).not.toHaveBeenCalled()
    expect(screen.getByRole("alert")).toBeInTheDocument()
  })

  it("keeps the id even when the text changes it", async () => {
    const user = userEvent.setup()
    const { onSave } = open(boundTile)

    await user.click(screen.getByRole("button", { name: "JSON" }))
    fireEvent.change(screen.getByLabelText("This tile's config"), {
      target: {
        value: JSON.stringify({
          id: "renamed",
          type: "metric",
          gridPos: { x: 0, y: 0, w: 6, h: 3 },
        }),
      },
    })
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave.mock.calls[0]?.[1].widget?.id).toBe("mrr")
  })
})

describe("the Lineage tab", () => {
  it("links a bound tile to its derivation and what that reads", async () => {
    const user = userEvent.setup()
    open(boundTile)

    await user.click(screen.getByRole("button", { name: "Lineage" }))

    expect(await screen.findByRole("link", { name: "revenue_by_product" })).toHaveAttribute(
      "href",
      "/derivations/revenue_by_product",
    )
    expect(await screen.findByRole("link", { name: "analyses_clean" })).toHaveAttribute(
      "href",
      "/derivations/analyses_clean",
    )
  })

  it("says a tile that binds nothing reads nothing", async () => {
    const user = userEvent.setup()
    open(textTile)

    await user.click(screen.getByRole("button", { name: "Lineage" }))

    expect(screen.getByText(/carries its own content/)).toBeInTheDocument()
  })

  it("does not say nothing is upstream before it knows", async () => {
    const user = userEvent.setup()
    vi.mocked(derivationProvenance).mockReturnValueOnce(new Promise(() => {}))
    open(boundTile)

    await user.click(screen.getByRole("button", { name: "Lineage" }))

    expect(screen.getByText(/Loading what this derivation reads/)).toBeInTheDocument()
    expect(screen.queryByText(/Nothing upstream/)).toBeNull()
  })

  it("says the lineage could not be loaded, rather than that there is none", async () => {
    const user = userEvent.setup()
    vi.mocked(derivationProvenance).mockRejectedValueOnce(new Error("500"))
    open(boundTile)

    await user.click(screen.getByRole("button", { name: "Lineage" }))

    expect(await screen.findByText(/Could not load what this derivation reads/)).toBeInTheDocument()
    expect(screen.queryByText(/Nothing upstream/)).toBeNull()
  })

  it("stays out of the way until asked for", () => {
    // A panel over the fields fetched on every open; a tab fetches when opened.
    open(boundTile)

    expect(screen.queryByText(/Computed by/)).toBeNull()
  })

  it("says a composite metric's trail is on the Metrics page, not that the tile reads nothing", async () => {
    const user = userEvent.setup()
    vi.mocked(listMetrics).mockResolvedValueOnce([
      { name: "paid_share", type: "ratio", numerator: "paid", denominator: "signups" },
    ])
    open({ ...metricTile, bind: { metric: "paid_share" } })
    await metricsReady()

    await user.click(screen.getByRole("button", { name: "Lineage" }))

    expect(await screen.findByText(/built from other metrics/)).toBeInTheDocument()
    expect(screen.queryByText(/carries its own content/)).toBeNull()
  })
})
