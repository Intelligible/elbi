import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { afterEach, describe, expect, it, vi } from "vitest"

import { FeedbackProvider } from "@/components/ui/feedback"

import { TrashSection } from "./TrashSection"

// The section is mounted inside Settings, which supplies the provider its
// "delete forever" confirmation needs; the test has to stand that up too.
const mount = () =>
  render(
    <FeedbackProvider>
      <TrashSection />
    </FeedbackProvider>,
  )

const ITEM = {
  type: "notebook",
  id: "nb-1",
  name: "Q3 plan",
  deletedAt: "2026-08-01T00:00:00Z",
}

function stubFetch(items: unknown[] = [ITEM]) {
  const calls: Array<{ url: string; method: string }> = []
  window.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    const method = init?.method ?? "GET"
    calls.push({ url, method })
    if (url === "/api/trash" && method === "GET") {
      // A restore call empties the list, matching what the server would report
      // once the item is actually gone from trash.
      const restored = calls.some((c) => c.url.includes("/restore"))
      return new Response(JSON.stringify(restored ? [] : items), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })
    }
    return new Response(JSON.stringify({ ok: true }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  }) as typeof window.fetch
  return calls
}

describe("TrashSection", () => {
  afterEach(() => vi.restoreAllMocks())

  it("shows a trashed item with its type", async () => {
    stubFetch()
    mount()
    await waitFor(() => expect(screen.getByText("Q3 plan")).toBeInTheDocument())
    expect(screen.getByText("Notebook")).toBeInTheDocument()
  })

  it("restores an item and refreshes the list", async () => {
    const calls = stubFetch()
    mount()
    await waitFor(() => expect(screen.getByText("Q3 plan")).toBeInTheDocument())

    await userEvent.click(screen.getByLabelText("Restore Q3 plan"))

    await waitFor(() =>
      expect(
        calls.some((c) => c.url === "/api/trash/notebook/nb-1/restore" && c.method === "POST"),
      ).toBe(true),
    )
    await waitFor(() => expect(screen.queryByText("Q3 plan")).not.toBeInTheDocument())
  })

  it("empty trash shows a plain message rather than an empty table", async () => {
    stubFetch([])
    mount()
    await waitFor(() => expect(screen.getByText("Trash is empty.")).toBeInTheDocument())
  })
})
