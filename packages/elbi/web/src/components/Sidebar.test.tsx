import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { emitNotificationsChanged } from "@/lib/notifications"
import { Sidebar } from "./Sidebar"

function stubUnread(count: number) {
  window.fetch = vi.fn(
    async () =>
      new Response(JSON.stringify({ items: [], unread: count }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      }),
  ) as typeof window.fetch
}

let realFetch: typeof window.fetch

beforeEach(() => {
  realFetch = window.fetch
})

afterEach(() => {
  window.fetch = realFetch
  vi.restoreAllMocks()
})

const BASE_PROPS = {
  conversations: [],
  hasNextPage: false,
  isFetchingNextPage: false,
  fetchNextPage: () => {},
  datasetCount: 0,
  derivationCount: 0,
  onNew: () => {},
  onDelete: () => {},
  onRename: () => {},
  onSearch: () => {},
}

function renderSidebar() {
  return render(
    <MemoryRouter initialEntries={["/warehouse"]}>
      <Sidebar {...BASE_PROPS} />
    </MemoryRouter>,
  )
}

describe("Sidebar unread pill", () => {
  it("shows the stubbed unread count next to Inbox", async () => {
    stubUnread(3)
    renderSidebar()
    expect(await screen.findByText("3")).toBeInTheDocument()
  })

  it("renders no pill when there is nothing unread", async () => {
    stubUnread(0)
    renderSidebar()
    await screen.findByText("Inbox")
    expect(screen.queryByText("0")).not.toBeInTheDocument()
  })

  it("refreshes on the change event without waiting for the 5s poll", async () => {
    stubUnread(0)
    renderSidebar()
    await screen.findByText("Inbox")
    expect(screen.queryByText("1")).not.toBeInTheDocument()
    stubUnread(1)
    emitNotificationsChanged()
    expect(await screen.findByText("1")).toBeInTheDocument()
  })
})
