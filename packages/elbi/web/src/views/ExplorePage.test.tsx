// Pins the explore page's controls: each test drives a control that edits the SQL, loads a
// query, switches the result view, or sends a request, so a mapping mistake shows up as a
// wrong API call rather than only as an eslint pass.

import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"
import type { QueryResult } from "@/lib/explore"

vi.mock("@/lib/explore", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/explore")>()),
  getSources: vi.fn(),
  getCatalog: vi.fn(),
  listQueries: vi.fn(),
  runQuery: vi.fn(),
  profile: vi.fn(),
  duplicateQuery: vi.fn(),
  deleteQuery: vi.fn(),
}))

vi.mock("@/lib/utils", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/utils")>()),
  copyText: vi.fn(),
}))

// CodeMirror and Vega need a real layout engine; the page only hands them values.
vi.mock("@/components/notebook/CellEditor", () => ({
  CellEditor: ({ value }: { value: string }) => <pre data-testid="sql">{value}</pre>,
}))
vi.mock("@/components/viz/VizView", () => ({ VizView: () => <div data-testid="viz" /> }))

import {
  deleteQuery,
  duplicateQuery,
  getCatalog,
  getSources,
  listQueries,
  profile,
  runQuery,
} from "@/lib/explore"
import { copyText } from "@/lib/utils"
import { ExplorePage } from "./ExplorePage"

const RESULT: QueryResult = {
  columns: ["region", "amount"],
  rows: [
    { region: "west", amount: 5 },
    { region: "east", amount: 20 },
    { region: "north", amount: 12 },
  ],
  rowCount: 3,
  truncated: false,
}

function renderPage() {
  return render(
    <TooltipProvider>
      <ExplorePage />
    </TooltipProvider>,
  )
}

const tab = (name: string) => screen.getByRole("tab", { name })
const runButton = () => screen.getByRole("button", { name: /^Run/ })
const bodyRows = () => screen.getAllByRole("row").slice(1)

describe("ExplorePage", () => {
  beforeEach(() => {
    localStorage.clear()
    vi.mocked(getSources).mockResolvedValue([{ id: null, name: "Bound datasets", kind: "project" }])
    vi.mocked(getCatalog).mockResolvedValue({
      tables: [
        {
          name: "orders",
          rows: 12,
          columns: [
            { name: "region", type: "VARCHAR" },
            { name: "amount", type: "DOUBLE" },
          ],
        },
      ],
    } as Awaited<ReturnType<typeof getCatalog>>)
    vi.mocked(listQueries).mockResolvedValue([
      {
        id: "q1",
        name: "Top orders",
        sql: "SELECT * FROM orders",
        sourceId: null,
        copiedFrom: null,
        createdAt: "2026-01-01T00:00:00Z",
        updatedAt: "2026-01-01T00:00:00Z",
      },
    ])
    vi.mocked(runQuery).mockResolvedValue(RESULT)
    vi.mocked(profile).mockResolvedValue({
      rowCount: 3,
      columns: [
        {
          name: "amount",
          count: 3,
          present: 3,
          completeness: 1,
          distinct: 3,
          inferredType: "integer",
          minimum: 5,
          maximum: 20,
          topValues: [{ value: 5, count: 1 }],
          isUnique: true,
          fractionUniqueOnce: 1,
        },
      ],
    })
    vi.mocked(duplicateQuery).mockResolvedValue(undefined as never)
    vi.mocked(deleteQuery).mockResolvedValue(undefined as never)
  })
  afterEach(() => vi.clearAllMocks())

  it("expands a table in the schema browser and inserts names into the SQL", async () => {
    const user = userEvent.setup()
    renderPage()
    const table = await screen.findByRole("button", { name: /^orders/ })
    expect(screen.queryByRole("button", { name: /^amount/ })).not.toBeInTheDocument()

    await user.click(table)
    await user.click(screen.getByRole("button", { name: /^amount/ }))
    expect(screen.getByTestId("sql")).toHaveTextContent("SELECT 1 AS example amount")

    await user.dblClick(table)
    expect(screen.getByTestId("sql")).toHaveTextContent("SELECT 1 AS example amount orders")

    await user.click(runButton())
    await waitFor(() =>
      expect(runQuery).toHaveBeenCalledWith("SELECT 1 AS example amount orders", null),
    )
  })

  it("switches result views, and profiles again each time Profile is clicked", async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(runButton())
    expect(await screen.findByText("west")).toBeInTheDocument()

    await user.click(tab("Profile"))
    await waitFor(() =>
      expect(profile).toHaveBeenCalledWith({ sql: "SELECT 1 AS example", sourceId: null }),
    )
    expect(await screen.findByText("unique")).toBeInTheDocument()
    await user.click(tab("Profile"))
    await waitFor(() => expect(profile).toHaveBeenCalledTimes(2))
    expect(tab("Profile")).toHaveAttribute("aria-selected", "true")

    await user.click(tab("Info"))
    expect(screen.getByText("Query")).toBeInTheDocument()
    expect(profile).toHaveBeenCalledTimes(2)

    await user.click(tab("Chart"))
    expect(screen.getByTestId("viz")).toBeInTheDocument()

    await user.click(tab("Results"))
    expect(screen.getByText("west")).toBeInTheDocument()
  })

  it("profiles from the keyboard", async () => {
    const user = userEvent.setup()
    renderPage()
    tab("Profile").focus()
    await user.keyboard("{Enter}")
    await waitFor(() => expect(profile).toHaveBeenCalledTimes(1))
    expect(tab("Profile")).toHaveAttribute("aria-selected", "true")
  })

  it("sorts the result grid from a column header", async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(runButton())
    await screen.findByText("west")
    const amounts = () => bodyRows().map((r) => within(r).getAllByRole("cell")[1].textContent)
    expect(amounts()).toEqual(["5", "20", "12"])

    await user.click(screen.getByRole("button", { name: "Sort by amount" }))
    const first = amounts()
    await user.click(screen.getByRole("button", { name: "Sort by amount" }))
    const second = amounts()
    expect([first, second]).toContainEqual(["5", "12", "20"])
    expect([first, second]).toContainEqual(["20", "12", "5"])
  })

  it("opens a row's details and copies one value", async () => {
    const user = userEvent.setup()
    renderPage()
    await user.click(runButton())
    await user.click(await screen.findByText("east"))

    const dialog = await screen.findByRole("dialog", { name: "Row details" })
    await waitFor(() => expect(dialog).toHaveFocus())
    expect(screen.queryByRole("tooltip")).not.toBeInTheDocument()
    await user.click(within(dialog).getByRole("button", { name: "Copy amount" }))
    expect(copyText).toHaveBeenCalledWith("20")
  })

  it("loads a query from history into the editor", async () => {
    localStorage.setItem(
      "explore.history",
      JSON.stringify([{ sql: "SELECT 42 AS answer", sourceId: null, at: Date.now() }]),
    )
    const user = userEvent.setup()
    renderPage()
    await user.click(screen.getByRole("button", { name: /History/ }))
    const dialog = await screen.findByRole("dialog", { name: "Query history" })
    await user.click(within(dialog).getByRole("button", { name: /SELECT 42 AS answer/ }))

    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(screen.getByTestId("sql")).toHaveTextContent("SELECT 42 AS answer")
    await user.click(runButton())
    await waitFor(() => expect(runQuery).toHaveBeenCalledWith("SELECT 42 AS answer", null))
  })

  it("loads, duplicates and deletes saved queries", async () => {
    const user = userEvent.setup()
    renderPage()
    const saved = await screen.findByRole("button", { name: "Saved (1)" })

    await user.click(saved)
    await user.click(screen.getByRole("button", { name: /^Duplicate Top orders/ }))
    await waitFor(() => expect(duplicateQuery).toHaveBeenCalledWith("q1"))
    await user.click(screen.getByRole("button", { name: /^Delete query/ }))
    await waitFor(() => expect(deleteQuery).toHaveBeenCalledWith("q1"))

    await user.click(screen.getByRole("button", { name: "Top orders" }))
    expect(screen.queryByRole("button", { name: "Top orders" })).not.toBeInTheDocument()
    expect(screen.getByText("Top orders")).toBeInTheDocument()
    expect(screen.getByTestId("sql")).toHaveTextContent("SELECT * FROM orders")
    await user.click(runButton())
    await waitFor(() => expect(runQuery).toHaveBeenCalledWith("SELECT * FROM orders", null))
  })
})
