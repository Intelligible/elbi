// Pins the catalog list: choosing an artifact loads its lineage and impact.

import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { CatalogRecord } from "@/lib/lineage"

vi.mock("@xyflow/react", () => ({
  ReactFlow: ({ children }: { children?: React.ReactNode }) => <div>{children}</div>,
  Background: () => null,
  Controls: () => null,
}))
vi.mock("@/lib/lineage", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/lineage")>()),
  getCatalog: vi.fn(),
  getSubgraph: vi.fn(async () => ({ nodes: [], edges: [] })),
  getImpact: vi.fn(async () => ({ node: "", affected: { dashboard: ["sales"] }, count: 1 })),
}))

import { getCatalog, getImpact, getSubgraph } from "@/lib/lineage"
import { CatalogPage } from "./CatalogPage"

const record = (id: string, type: CatalogRecord["type"], name: string): CatalogRecord => ({
  id,
  type,
  name,
  verdict: null,
  certified: null,
  description: null,
  copiedFrom: null,
  upstream: 0,
})

afterEach(() => {
  vi.clearAllMocks()
})

describe("CatalogPage", () => {
  it("loads the lineage and impact of the chosen artifact", async () => {
    const user = userEvent.setup()
    vi.mocked(getCatalog).mockResolvedValue([
      record("dataset:orders", "dataset", "orders"),
      record("derivation:eff", "derivation", "eff"),
    ])
    render(<CatalogPage />)
    await user.click(await screen.findByRole("button", { name: /eff/ }))
    expect(getSubgraph).toHaveBeenCalledWith("derivation:eff")
    expect(getImpact).toHaveBeenCalledWith("derivation:eff")
    expect(await screen.findByText(/changing this affects 1 artifact/)).toBeInTheDocument()
  })
})
