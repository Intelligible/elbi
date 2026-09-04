import { render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { afterEach, describe, expect, it, vi } from "vitest"

import type { NotebookView } from "@/lib/notebooks"

vi.mock("@/lib/notebooks", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/notebooks")>()),
  getNotebook: vi.fn(),
  getNotebookVariables: vi.fn(async () => []),
}))

import { getNotebook, getNotebookVariables } from "@/lib/notebooks"
import { NotebookPage } from "./NotebookPage"

const VIEW: NotebookView = {
  id: "nb1",
  name: "Q3 report",
  folder_id: "fold1",
  folder_name: "Research",
  deps: [],
  metadata: {},
  schedule: null,
  kernel_status: "idle",
  cells: [],
  graph: { cells: {}, conflicts: {}, cycle: [] },
  environment: {
    base_env: null,
    base_environments: [],
    lock: null,
    locked: false,
  },
}

function renderPage(id = "nb1") {
  return render(
    <MemoryRouter initialEntries={[`/notebooks/${id}`]}>
      <Routes>
        <Route path="/notebooks/:name" element={<NotebookPage />} />
      </Routes>
    </MemoryRouter>,
  )
}

describe("NotebookPage back link", () => {
  it("returns to the folder the notebook lives in", async () => {
    vi.mocked(getNotebook).mockResolvedValue(VIEW)
    renderPage()
    const back = await screen.findByRole("link", { name: "Research" })
    expect(back.getAttribute("href")).toBe("/notebooks?folder=fold1")
  })

  it("returns to the notebooks root for a notebook outside any folder", async () => {
    vi.mocked(getNotebook).mockResolvedValue({
      ...VIEW,
      folder_id: null,
      folder_name: null,
    })
    renderPage()
    const back = await screen.findByRole("link", { name: "Notebooks" })
    expect(back.getAttribute("href")).toBe("/notebooks")
  })
})

describe("NotebookPage", () => {
  afterEach(() => vi.restoreAllMocks())

  it("reports a deleted notebook instead of loading forever", async () => {
    // Every request 404s: the variables fetch already swallows its own failure, so this
    // exercises the notebook load failing without enumerating each call.
    window.fetch = vi.fn(
      async () =>
        new Response(JSON.stringify({ detail: "no notebook named 'gone'" }), {
          status: 404,
          headers: { "Content-Type": "application/json" },
        }),
    ) as typeof window.fetch

    // The file-level vi.mock above exists for the back-link tests; put the real
    // fetch-backed client back so this one still proves a 404 reaches the page as a
    // rejection rather than asserting against a hand-written one.
    const real = await vi.importActual<typeof import("@/lib/notebooks")>("@/lib/notebooks")
    vi.mocked(getNotebook).mockImplementation(real.getNotebook)
    vi.mocked(getNotebookVariables).mockImplementation(real.getNotebookVariables)

    const { container } = renderPage("gone")

    // The load failing used to leave this skeleton up for good. Nothing else names the
    // bars, so data-slot (components/ui/skeleton.tsx) is what the query has to go by.
    await waitFor(() => expect(container.querySelector('[data-slot="skeleton"]')).toBeNull())
    expect(screen.getByRole("link", { name: /notebooks/i })).toBeInTheDocument()
    expect(screen.getByText(/deleted, or the link is out of date/)).toBeInTheDocument()
  })
})
