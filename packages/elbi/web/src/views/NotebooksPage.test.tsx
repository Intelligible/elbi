import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { afterEach, describe, expect, it, vi } from "vitest"

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
    <MemoryRouter initialEntries={["/notebooks"]}>
      <Routes>
        <Route path="/notebooks" element={<NotebooksPage />} />
        <Route path="/notebooks/:name" element={<div>notebook detail</div>} />
      </Routes>
    </MemoryRouter>,
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
