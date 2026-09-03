import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { onNotificationsChanged } from "@/lib/notifications"
import { NotificationsPage } from "./NotificationsPage"

// A minimal in-memory backend: two rows (one unread). Read and read-all mutate the array in
// place, so a refetch afterward reflects the change, the same as the real API: this is what
// lets a test prove the badge actually updated.
function makeItems() {
  return [
    {
      id: "n1",
      at: "2024-01-01T00:00:00Z",
      eventType: "metric.anomaly_detected",
      title: "Monitor 'p95' detected an anomaly",
      body: "spiked",
      targetType: "metric_monitor",
      targetId: "m1",
      verdict: "sound",
      readAt: null as string | null,
    },
    {
      id: "n2",
      at: "2024-01-01T00:00:00Z",
      eventType: "run.failed",
      title: "Orchestration run failed (manual)",
      body: "",
      targetType: "orchestration_run",
      targetId: "r1",
      verdict: null as string | null,
      readAt: "2024-01-02T00:00:00Z",
    },
  ]
}

function stubFetch(opts: { failFirst?: boolean } = {}) {
  const calls: string[] = []
  let items = makeItems()
  let failNext = opts.failFirst ?? false
  window.fetch = vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
    const url = String(input)
    calls.push(url)
    const method = init?.method ?? "GET"
    if (failNext && method === "GET") {
      failNext = false
      return new Response("", { status: 500 })
    }
    if (method === "POST" && url.endsWith("/read-all")) {
      items = items.map((n) => ({ ...n, readAt: n.readAt ?? "2024-01-03T00:00:00Z" }))
      return new Response(JSON.stringify({ ok: true, count: 1 }), { status: 200 })
    }
    const readMatch = /\/notifications\/([^/]+)\/read$/.exec(url)
    if (method === "POST" && readMatch) {
      const rowId = decodeURIComponent(readMatch[1] ?? "")
      items = items.map((n) => (n.id === rowId ? { ...n, readAt: "2024-01-03T00:00:00Z" } : n))
      return new Response(JSON.stringify({ ok: true }), { status: 200 })
    }
    const visible = url.includes("unread_only=true") ? items.filter((n) => !n.readAt) : items
    const unread = items.filter((n) => !n.readAt).length
    return new Response(JSON.stringify({ items: visible, unread }), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  }) as typeof window.fetch
  return calls
}

let realFetch: typeof window.fetch

beforeEach(() => {
  realFetch = window.fetch
})

afterEach(() => {
  window.fetch = realFetch
  vi.restoreAllMocks()
})

function renderPage() {
  return render(
    <MemoryRouter initialEntries={["/notifications"]}>
      <Routes>
        <Route path="/notifications" element={<NotificationsPage />} />
        <Route path="/monitors" element={<div>Monitors page</div>} />
      </Routes>
    </MemoryRouter>,
  )
}

describe("NotificationsPage", () => {
  it("renders rows and the unread badge from the inbox response", async () => {
    stubFetch()
    renderPage()
    expect(await screen.findByText(/detected an anomaly/)).toBeInTheDocument()
    expect(screen.getByText("1 unread")).toBeInTheDocument()
  })

  it("marks a row read, fires the change event, and navigates to its target", async () => {
    stubFetch()
    const changed = vi.fn()
    const off = onNotificationsChanged(changed)
    renderPage()
    const row = await screen.findByText(/detected an anomaly/)
    fireEvent.click(row)
    expect(await screen.findByText("Monitors page")).toBeInTheDocument()
    expect(changed).toHaveBeenCalledTimes(1)
    off()
  })

  it("marks all read and disables the button once nothing is unread", async () => {
    const calls = stubFetch()
    renderPage()
    await screen.findByText("1 unread")
    fireEvent.click(screen.getByRole("button", { name: /mark all read/i }))
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /mark all read/i })).toBeDisabled(),
    )
    expect(calls.some((u) => u.endsWith("/notifications/read-all"))).toBe(true)
  })

  it("shows a retriable error on a failed first load", async () => {
    stubFetch({ failFirst: true })
    renderPage()
    expect(await screen.findByText("Could not load notifications")).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: /retry/i }))
    expect(await screen.findByText(/detected an anomaly/)).toBeInTheDocument()
  })

  it("requests unread_only=true when the Unread filter is selected", async () => {
    const calls = stubFetch()
    renderPage()
    await screen.findByText(/detected an anomaly/)
    fireEvent.click(screen.getByRole("button", { name: "Unread" }))
    await waitFor(() => expect(calls.some((u) => u.includes("unread_only=true"))).toBe(true))
  })
})
