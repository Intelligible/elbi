import { render, screen, waitFor, within } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { afterEach, describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"
import { NotebooksPage } from "./NotebooksPage"

const NOTEBOOK = {
  id: "nb-1",
  name: "Q3 plan",
  folder_id: null,
  copied_from: null,
  created_at: "2026-08-01T00:00:00Z",
  updated_at: "2026-08-01T00:00:00Z",
  cell_count: 2,
}

// A minimal in-memory backend. The delete mutates the list in place so the refetch
// that follows reports what the server would, which is what lets the test tell a
// refreshed list apart from a stale one.
function stubFetch() {
  const calls: Array<{ url: string; method: string }> = []
  let notebooks: unknown[] = [NOTEBOOK]
  const reply = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    })
  window.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    const method = init?.method ?? "GET"
    calls.push({ url, method })
    if (url === "/api/notebooks/folders" && method === "GET") return reply([])
    if (url === "/api/notebooks" && method === "GET") return reply(notebooks)
    if (url === "/api/notebooks/nb-1" && method === "DELETE") {
      notebooks = []
      return reply({ ok: true })
    }
    return reply({ detail: `unexpected ${method} ${url}` }, 404)
  }) as typeof window.fetch
  return calls
}

// The probe route stands in for NotebookPage: navigation leaves no other trace the
// test can see, so the only way to assert it did not happen is to render a witness.
const mount = () =>
  render(
    <TooltipProvider>
      <MemoryRouter initialEntries={["/notebooks"]}>
        <Routes>
          <Route path="/notebooks" element={<NotebooksPage />} />
          <Route path="/notebooks/:name" element={<div>notebook detail</div>} />
        </Routes>
      </MemoryRouter>
    </TooltipProvider>,
  )

describe("NotebooksPage rows", () => {
  afterEach(() => vi.restoreAllMocks())

  it("opens the notebook when the row itself is clicked", async () => {
    stubFetch()
    mount()
    await waitFor(() => expect(screen.getByText("Q3 plan")).toBeInTheDocument())

    await userEvent.click(screen.getByText("Q3 plan"))

    expect(await screen.findByText("notebook detail")).toBeInTheDocument()
  })

  it("deletes from the row menu without opening the notebook it just deleted", async () => {
    // Radix marks the page pointer-events: none while a modal menu is open, which
    // userEvent otherwise reads as an unreachable element.
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    const calls = stubFetch()
    mount()
    await waitFor(() => expect(screen.getByText("Q3 plan")).toBeInTheDocument())

    await user.click(screen.getByLabelText("Row actions"))
    await user.click(await screen.findByRole("menuitem", { name: /delete/i }))

    await waitFor(() =>
      expect(calls.some((c) => c.url === "/api/notebooks/nb-1" && c.method === "DELETE")).toBe(
        true,
      ),
    )
    // The row's own click navigates to the notebook. A click on its menu must not
    // reach it: the notebook is gone, and its page would 404 into a dead skeleton.
    expect(screen.queryByText("notebook detail")).not.toBeInTheDocument()
    await waitFor(() => expect(screen.queryByText("Q3 plan")).not.toBeInTheDocument())
  })
})

const FOLDER = {
  id: "f1",
  name: "Research",
  parent_id: null,
  created_at: "2026-08-01T00:00:00Z",
  updated_at: "2026-08-01T00:00:00Z",
}
const INNER = { ...NOTEBOOK, id: "nb-2", name: "Inner draft", folder_id: "f1" }

// A folder holding one notebook; records every call so a test can assert what the
// server was asked to do.
function stubFolderFetch() {
  const calls: Array<{ url: string; method: string }> = []
  const reply = (body: unknown, status = 200) =>
    new Response(JSON.stringify(body), {
      status,
      headers: { "Content-Type": "application/json" },
    })
  window.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    const method = init?.method ?? "GET"
    calls.push({ url, method })
    if (url === "/api/notebooks/folders" && method === "GET") return reply([FOLDER])
    if (url === "/api/notebooks" && method === "GET") return reply([NOTEBOOK, INNER])
    if (url.startsWith("/api/notebooks/folders/f1") && method === "DELETE")
      return reply({ ok: true })
    return reply({ detail: `unexpected ${method} ${url}` }, 404)
  }) as typeof window.fetch
  return calls
}

describe("NotebooksPage folders", () => {
  afterEach(() => vi.restoreAllMocks())

  it("walks into a folder and back out through the breadcrumb", async () => {
    const user = userEvent.setup()
    stubFolderFetch()
    mount()
    await user.click(await screen.findByText("Research"))

    expect(await screen.findByText("Inner draft")).toBeInTheDocument()
    expect(screen.queryByText("Q3 plan")).not.toBeInTheDocument()
    const crumbs = screen.getByRole("navigation", { name: "Breadcrumb" })
    expect(within(crumbs).getByRole("button", { name: "Research" })).toBeInTheDocument()

    await user.click(within(crumbs).getByRole("button", { name: "Home" }))

    expect(await screen.findByText("Q3 plan")).toBeInTheDocument()
    expect(screen.queryByText("Inner draft")).not.toBeInTheDocument()
  })

  it("deletes a non-empty folder recursively only once the user confirms", async () => {
    const user = userEvent.setup({ pointerEventsCheck: 0 })
    const calls = stubFolderFetch()
    mount()
    await screen.findByText("Research")
    const deletes = () => calls.filter((c) => c.method === "DELETE")
    const openMenu = async () => {
      const trigger = screen.getAllByLabelText("Row actions")[0]
      // Radix menus close on a window blur, which jsdom fires at pointerdown when an
      // earlier test left nothing focused; focusing first avoids it (jsdom only).
      trigger.focus()
      await user.click(trigger)
      await user.click(await screen.findByRole("menuitem", { name: /delete/i }))
    }

    await openMenu()
    const dialog = await screen.findByRole("dialog")
    expect(within(dialog).getByText(/Delete "Research" and its 1 item\?/)).toBeInTheDocument()
    await user.click(within(dialog).getByRole("button", { name: "Cancel" }))
    await waitFor(() => expect(screen.queryByRole("dialog")).not.toBeInTheDocument())
    expect(deletes()).toEqual([])

    await openMenu()
    await user.click(
      within(await screen.findByRole("dialog")).getByRole("button", { name: "Confirm" }),
    )

    await waitFor(() =>
      expect(deletes()).toEqual([
        { url: "/api/notebooks/folders/f1?recursive=true", method: "DELETE" },
      ]),
    )
  })
})
