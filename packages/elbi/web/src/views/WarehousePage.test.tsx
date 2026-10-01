import { render, screen, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"
import { getDatasets } from "@/lib/chat"
import type { Catalog, SourceDetail, SourceSummary } from "@/lib/warehouse"
import { deleteSource, getCatalog, getSource, listSources, updateSchema } from "@/lib/warehouse"
import { NewSourcePage, SourceDetailPage, WarehousePage } from "./WarehousePage"

vi.mock("@/lib/chat", () => ({ getDatasets: vi.fn() }))
vi.mock("@/lib/warehouse", () => ({
  deleteSource: vi.fn(),
  getCatalog: vi.fn(),
  getSource: vi.fn(),
  listSources: vi.fn(),
  registerInterest: vi.fn(),
  setSyncFrequency: vi.fn(),
  syncSource: vi.fn(),
  updateSchema: vi.fn(),
}))

const summary = (s: Partial<SourceSummary> & Pick<SourceSummary, "id" | "name">) =>
  ({
    sourceType: "postgres",
    description: "",
    prefix: "",
    syncFrequency: "manual",
    status: "idle",
    lastError: null,
    lastSyncedAt: null,
    createdAt: null,
    schemaCount: 0,
    syncedCount: 0,
    rows: 0,
    ...s,
  }) as SourceSummary

const SOURCES = [
  summary({ id: "a", name: "billing", sourceType: "stripe" }),
  summary({ id: "b", name: "warehouse-db", sourceType: "postgres" }),
]

const tile = (name: string, label: string, category: string) => ({
  name,
  label,
  category,
  fields: [],
  caption: "",
  icon: "",
  releaseStatus: "ga" as const,
  docsUrl: "",
  comingSoon: false,
})

const CATALOG: Catalog = {
  categories: ["Databases", "Payments"],
  sources: [tile("postgres", "PostgreSQL", "Databases"), tile("stripe", "Stripe", "Payments")],
}

function Where() {
  const loc = useLocation()
  return <div data-testid="where">{loc.pathname + loc.search}</div>
}

function renderAt(path: string) {
  render(
    <TooltipProvider>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/warehouse" element={<WarehousePage />} />
          <Route path="/warehouse/new-source" element={<NewSourcePage />} />
          <Route path="/warehouse/sources/:id" element={<SourceDetailPage />} />
        </Routes>
        <Where />
      </MemoryRouter>
    </TooltipProvider>,
  )
}

const where = () => screen.getByTestId("where").textContent

// Navigating mounts the destination page, which fetches too; keep those calls pending.
beforeEach(() => {
  vi.mocked(getCatalog).mockReturnValue(new Promise(() => {}))
  vi.mocked(getSource).mockReturnValue(new Promise(() => {}))
})

describe("WarehousePage", () => {
  beforeEach(() => {
    vi.mocked(getDatasets).mockResolvedValue([])
    vi.mocked(listSources).mockResolvedValue(SOURCES)
    vi.mocked(deleteSource).mockReset().mockResolvedValue({ ok: true })
  })

  it("search narrows the source list", async () => {
    renderAt("/warehouse")
    expect(await screen.findByText("billing")).toBeInTheDocument()
    await userEvent.type(screen.getByPlaceholderText("Search..."), "stripe")
    expect(screen.getByText("billing")).toBeInTheDocument()
    expect(screen.queryByText("warehouse-db")).toBeNull()
    await userEvent.clear(screen.getByPlaceholderText("Search..."))
    await userEvent.type(screen.getByPlaceholderText("Search..."), "nothing")
    expect(screen.getByText("No sources matching your search")).toBeInTheDocument()
  })

  it("a row opens the source; its delete button deletes without opening it", async () => {
    renderAt("/warehouse")
    const row = (await screen.findByText("warehouse-db")).closest("tr") as HTMLElement
    await userEvent.click(within(row).getByRole("button", { name: "Delete source" }))
    expect(deleteSource).toHaveBeenCalledWith("b")
    expect(where()).toBe("/warehouse")
    await userEvent.click(within(row).getByText("warehouse-db"))
    expect(where()).toBe("/warehouse/sources/b")
  })

  it("New source goes to the catalog", async () => {
    renderAt("/warehouse")
    await screen.findByText("billing")
    await userEvent.click(screen.getByRole("button", { name: "New source" }))
    expect(where()).toBe("/warehouse/new-source")
  })
})

describe("NewSourcePage catalog", () => {
  beforeEach(() => {
    vi.mocked(getCatalog).mockResolvedValue(CATALOG)
  })

  it("a category narrows the tiles, and a tile opens its form", async () => {
    renderAt("/warehouse/new-source")
    expect(await screen.findByText("Stripe")).toBeInTheDocument()
    await userEvent.click(screen.getByRole("tab", { name: /Databases/ }))
    expect(screen.queryByText("Stripe")).toBeNull()
    await userEvent.click(screen.getByText("PostgreSQL"))
    expect(where()).toBe("/warehouse/new-source?kind=postgres")
  })

  it("the active category tab controls the tile panel", async () => {
    renderAt("/warehouse/new-source")
    await userEvent.click(await screen.findByRole("tab", { name: /Databases/ }))
    const id = screen.getByRole("tab", { name: /Databases/ }).getAttribute("aria-controls")
    const panel = document.getElementById(id ?? "")
    expect(panel).toHaveAttribute("role", "tabpanel")
    expect(panel).toHaveTextContent("PostgreSQL")
  })
})

describe("SourceDetailPage", () => {
  it("the sync switch patches the schema", async () => {
    const detail = {
      ...summary({ id: "a", name: "billing" }),
      schemas: [
        {
          id: "sch1",
          name: "Charges",
          table: "charges",
          shouldSync: false,
          syncType: "full_refresh",
          incrementalField: null,
          incrementalFields: [],
          status: "pending",
          rowCount: null,
          lastError: null,
          lastSyncedAt: null,
        },
      ],
    } as SourceDetail
    vi.mocked(getSource).mockResolvedValue(detail)
    vi.mocked(updateSchema).mockResolvedValue({ ...detail.schemas[0], shouldSync: true })
    renderAt("/warehouse/sources/a")
    const toggle = await screen.findByRole("switch")
    expect(toggle).toHaveAttribute("aria-checked", "false")
    await userEvent.click(toggle)
    expect(updateSchema).toHaveBeenCalledWith("sch1", { should_sync: true })
    expect(await screen.findByRole("switch")).toHaveAttribute("aria-checked", "true")
  })

  const schema = (s: Partial<SourceDetail["schemas"][number]>) =>
    ({
      id: "sch1",
      name: "activation_funnel",
      table: "custom__activation_funnel",
      shouldSync: true,
      syncType: "full_refresh",
      incrementalField: null,
      incrementalFields: [],
      status: "synced",
      rowCount: 0,
      lastError: null,
      lastSyncedAt: "2026-10-01T09:00:00",
      ...s,
    }) as SourceDetail["schemas"][number]

  const open = (schemas: SourceDetail["schemas"], s: Partial<SourceSummary> = {}) => {
    vi.mocked(getSource).mockResolvedValue({
      ...summary({
        id: "a",
        name: "posthog",
        prefix: "custom",
        lastSyncedAt: "2026-10-01T09:00:00",
        ...s,
      }),
      schemas,
    })
    renderAt("/warehouse/sources/a")
  }

  const tooltipOn = async (trigger: HTMLElement) => {
    await userEvent.hover(trigger)
    return screen.findByRole("tooltip")
  }

  it("a sync that landed no rows reads as a warning, not as idle and synced", async () => {
    open([schema({ rowCount: 0 })])
    const headline = await screen.findByText("Synced, but no rows")
    expect(headline).toHaveAttribute("data-variant", "warning")
    expect(screen.queryByText("idle")).toBeNull()
    const row = screen.getByText("custom__activation_funnel").closest("tr") as HTMLElement
    expect(within(row).getByText("No rows")).toHaveAttribute("data-variant", "warning")
    expect(await tooltipOn(headline)).toHaveTextContent(
      "The last sync succeeded, but 1 table got no rows: custom__activation_funnel.",
    )
  })

  it("a failed last run is the headline and is marked on the table that failed", async () => {
    open(
      [
        schema({ status: "error", rowCount: 233, lastError: "cannot merge line items" }),
        schema({ id: "sch2", table: "custom__charges", rowCount: 5 }),
      ],
      { status: "error", lastError: "cannot merge line items" },
    )
    expect(
      await screen.findByText("Failed", { selector: "[data-slot=badge]:not(td *)" }),
    ).toHaveAttribute("data-variant", "danger")
    const failed = screen.getByText("custom__activation_funnel").closest("tr") as HTMLElement
    const badge = within(failed).getByText("Failed")
    expect(badge).toHaveAttribute("data-variant", "danger")
    expect(await tooltipOn(badge)).toHaveTextContent(
      "The last sync of this table failed: cannot merge line items",
    )
    const ok = screen.getByText("custom__charges").closest("tr") as HTMLElement
    expect(within(ok).getByText("Synced")).toHaveAttribute("data-variant", "success")
  })

  it.each([
    ["Method", "Full refresh reads every row and replaces the table on each sync."],
    ["Rows", "Rows recorded by this table's last successful sync"],
    [
      "Table",
      "Each table lands in the warehouse as custom__<stream>, for example custom__activation_funnel.",
    ],
    ["1 of 1 tables enabled", "Enabled tables are the ones a sync reads"],
    [
      "last synced 2026-10-01 09:00",
      "When a sync last finished with every enabled table succeeding.",
    ],
  ])("%s explains itself on hover, with a docs link", async (label, text) => {
    open([schema({ rowCount: 10 })])
    const tip = await tooltipOn(await screen.findByRole("button", { name: label }))
    expect(tip).toHaveTextContent(text)
  })

  it("the schedule explains that it runs only while the app does", async () => {
    open([schema({ rowCount: 10 })])
    const tip = await tooltipOn(await screen.findByRole("combobox", { name: "Sync frequency" }))
    expect(tip).toHaveTextContent("Scheduled syncs run only while the app is running")
    expect(tip).toHaveTextContent("Manual only means it syncs only when you click Sync now.")
  })

  it("a tooltip's docs link goes to the data-sources page", async () => {
    open([schema({ rowCount: 10 })])
    await userEvent.hover(await screen.findByRole("button", { name: "Method" }))
    const links = await screen.findAllByRole("link", { name: "Learn more in the docs" })
    expect(links[0]).toHaveAttribute("href", "https://docs.elbi.ai/data-sources/#incremental-sync")
  })
})
