import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import type { Widget } from "@/lib/dashboards"
import { TileEditor } from "./TileEditor"

const pos = { x: 0, y: 0, w: 12, h: 6 }
const textTile: Widget = { id: "note", type: "text", gridPos: pos, content: "before" }
const boundTile: Widget = {
  id: "mrr",
  type: "metric",
  gridPos: pos,
  title: "MRR",
  bind: { derivation: "revenue_by_product" },
}

describe("TileEditor", () => {
  it("edits a static text tile's title and content", async () => {
    const user = userEvent.setup()
    const onSave = vi.fn()
    render(<TileEditor widget={textTile} catalog={[]} onCancel={() => {}} onSave={onSave} />)

    await user.type(screen.getByLabelText("Title"), "Caveat")
    const content = screen.getByLabelText("Content")
    await user.clear(content)
    await user.type(content, "after")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave).toHaveBeenCalledWith("note", { title: "Caveat", content: "after" })
  })

  it("rebinds a data tile rather than offering it a content box", async () => {
    const user = userEvent.setup()
    const onSave = vi.fn()
    render(
      <TileEditor
        widget={boundTile}
        catalog={["revenue_by_product", "collected_revenue"]}
        onCancel={() => {}}
        onSave={onSave}
      />,
    )

    expect(screen.queryByLabelText("Content")).toBeNull()
    const field = screen.getByLabelText("Derivation")
    await user.clear(field)
    await user.type(field, "collected_revenue")
    await user.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave).toHaveBeenCalledWith("mrr", {
      title: "MRR",
      derivation: "collected_revenue",
    })
  })

  it("warns before saving a binding nothing provides", async () => {
    const user = userEvent.setup()
    render(
      <TileEditor
        widget={boundTile}
        catalog={["revenue_by_product"]}
        onCancel={() => {}}
        onSave={() => {}}
      />,
    )

    const field = screen.getByLabelText("Derivation")
    await user.clear(field)
    await user.type(field, "no_such_derivation")

    expect(screen.getByText(/Nothing named/)).toBeInTheDocument()
  })

  it("shows the tile it was opened on, not the previous one", () => {
    const { rerender } = render(
      <TileEditor widget={textTile} catalog={[]} onCancel={() => {}} onSave={() => {}} />,
    )
    expect(screen.getByLabelText("Content")).toHaveValue("before")

    rerender(<TileEditor widget={boundTile} catalog={[]} onCancel={() => {}} onSave={() => {}} />)

    expect(screen.getByLabelText("Title")).toHaveValue("MRR")
    expect(screen.queryByLabelText("Content")).toBeNull()
  })

  it("renders nothing when no tile is open", () => {
    const { container } = render(
      <TileEditor widget={null} catalog={[]} onCancel={() => {}} onSave={() => {}} />,
    )
    expect(container).toBeEmptyDOMElement()
  })
})
