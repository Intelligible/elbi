import { fireEvent, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"
import { describe, expect, it, vi } from "vitest"

// The provenance trail links to the derivation a tile binds.
vi.mock("@/lib/dashboards", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/dashboards")>()),
  derivationProvenance: vi.fn(async () => ({
    derivation: ["analyses_clean"],
    dataset: ["posthog_analyses"],
  })),
}))

import type { Widget } from "@/lib/dashboards"
import { TileEditor } from "./TileEditor"

const pos = { x: 0, y: 0, w: 12, h: 6 }
const textTile: Widget = { id: "note", type: "text", gridPos: pos, content: "before" }
const boundTile: Widget = {
  id: "mrr",
  type: "metric",
  gridPos: { x: 6, y: 2, w: 6, h: 3 },
  title: "Subscriptions (all)",
  bind: { derivation: "revenue_by_product" },
  viz: { field: "subscriptions", agg: "sum" },
}

function open(widget: Widget | null, props: { catalog?: string[]; onCancel?: () => void } = {}) {
  const onSave = vi.fn()
  const rendered = render(
    <MemoryRouter>
      <TileEditor
        widget={widget}
        catalog={props.catalog ?? ["revenue_by_product", "collected_revenue"]}
        columns={24}
        onCancel={props.onCancel ?? (() => {})}
        onSave={onSave}
      />
    </MemoryRouter>,
  )
  return { onSave, ...rendered }
}

describe("TileEditor", () => {
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

  it("shows a metric's existing column and aggregate", () => {
    open(boundTile)

    expect(screen.getByLabelText("Column")).toHaveValue("subscriptions")
    expect(screen.getByLabelText("Aggregate")).toHaveTextContent("Sum")
  })

  it("edits a metric's column", async () => {
    const user = userEvent.setup()
    const { onSave } = open(boundTile)

    const column = screen.getByLabelText("Column")
    await user.clear(column)
    await user.type(column, "mrr_usd")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave).toHaveBeenCalledWith(
      "mrr",
      expect.objectContaining({ viz: { field: "mrr_usd", agg: "sum" } }),
    )
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

    expect(onSave.mock.calls[0][1].gridPos.w).toBe(24)
  })

  it("rebinds a data tile rather than offering it a content box", async () => {
    const user = userEvent.setup()
    const { onSave } = open(boundTile)

    expect(screen.queryByLabelText("Content")).toBeNull()
    const field = screen.getByLabelText("Derivation")
    await user.clear(field)
    await user.type(field, "collected_revenue")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave).toHaveBeenCalledWith(
      "mrr",
      expect.objectContaining({ derivation: "collected_revenue" }),
    )
  })

  it("warns before saving a binding nothing provides", async () => {
    const user = userEvent.setup()
    open(boundTile, { catalog: ["revenue_by_product"] })

    const field = screen.getByLabelText("Derivation")
    await user.clear(field)
    await user.type(field, "no_such_derivation")

    expect(screen.getByText(/Nothing named/)).toBeInTheDocument()
  })

  it("shows the tile it was opened on, not the previous one", () => {
    const { rerender } = open(textTile)
    expect(screen.getByLabelText("Content")).toHaveValue("before")

    rerender(
      <MemoryRouter>
        <TileEditor
          widget={boundTile}
          catalog={[]}
          columns={24}
          onCancel={() => {}}
          onSave={() => {}}
        />
      </MemoryRouter>,
    )

    expect(screen.getByLabelText("Title")).toHaveValue("Subscriptions (all)")
    expect(screen.queryByLabelText("Content")).toBeNull()
  })

  it("renders nothing when no tile is open", () => {
    const { container } = open(null)
    expect(container).toBeEmptyDOMElement()
  })
})

describe("the JSON tab", () => {
  it("shows the tile exactly as the spec stores it", async () => {
    const user = userEvent.setup()
    open(boundTile)

    await user.click(screen.getByRole("button", { name: "JSON" }))

    const config = JSON.parse(
      (screen.getByLabelText("This tile's config") as HTMLTextAreaElement).value,
    )
    expect(config.id).toBe("mrr")
    expect(config.bind).toEqual({ derivation: "revenue_by_product" })
    expect(config.viz).toEqual({ field: "subscriptions", agg: "sum" })
  })

  it("carries an unsaved field edit into the JSON", async () => {
    const user = userEvent.setup()
    open(boundTile)

    const column = screen.getByLabelText("Column")
    await user.clear(column)
    await user.type(column, "mrr_usd")
    await user.click(screen.getByRole("button", { name: "JSON" }))

    const config = JSON.parse(
      (screen.getByLabelText("This tile's config") as HTMLTextAreaElement).value,
    )
    expect(config.viz.field).toBe("mrr_usd")
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

    expect(onSave.mock.calls[0][1].widget.id).toBe("mrr")
  })
})

describe("provenance", () => {
  it("links a bound tile to its derivation and what that reads", async () => {
    open(boundTile)

    expect(await screen.findByRole("link", { name: "revenue_by_product" })).toHaveAttribute(
      "href",
      "/derivations/revenue_by_product",
    )
    expect(await screen.findByRole("link", { name: "analyses_clean" })).toHaveAttribute(
      "href",
      "/derivations/analyses_clean",
    )
  })

  it("shows nothing for a tile that binds nothing", () => {
    open(textTile)

    expect(screen.queryByText(/Where this tile/)).toBeNull()
  })
})
