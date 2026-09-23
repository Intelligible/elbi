import { act, fireEvent, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { ComponentProps } from "react"
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"
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
      <TooltipProvider>
        <Sidebar {...BASE_PROPS} />
      </TooltipProvider>
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

function Where() {
  return <span data-testid="where">{useLocation().pathname}</span>
}

function renderAt(path: string, props: Partial<ComponentProps<typeof Sidebar>> = {}) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <TooltipProvider>
        <Sidebar {...BASE_PROPS} {...props} />
        <Routes>
          <Route path="*" element={<Where />} />
        </Routes>
      </TooltipProvider>
    </MemoryRouter>,
  )
}

const CONVERSATION = {
  id: "c1",
  title: "Churn by region",
  created_at: new Date().toISOString(),
  updated_at: new Date().toISOString(),
}

describe("Sidebar controls", () => {
  beforeEach(() => {
    stubUnread(0)
    localStorage.clear()
  })

  it("marks the active nav item as the current page", () => {
    renderAt("/warehouse")
    expect(screen.getByRole("link", { name: /Data warehouse/ })).toHaveAttribute(
      "aria-current",
      "page",
    )
    expect(screen.getByRole("link", { name: "Explore" })).not.toHaveAttribute("aria-current")
  })

  it("switches to Chat from a browse page and opens the composer", async () => {
    const user = userEvent.setup()
    const onNew = vi.fn()
    renderAt("/warehouse", { onNew })
    const browse = screen.getByRole("button", { name: "Browse" })
    const chat = screen.getByRole("button", { name: "Chat" })
    expect(browse).toHaveAttribute("aria-pressed", "true")
    await user.click(chat)
    expect(onNew).toHaveBeenCalledOnce()
    expect(chat).toHaveAttribute("aria-pressed", "true")
    expect(screen.getByText("No past analyses yet")).toBeInTheDocument()
    await user.click(browse)
    expect(screen.getByRole("link", { name: /Data warehouse/ })).toBeInTheDocument()
  })

  it("stays in the open conversation when Chat is pressed on a chat route", async () => {
    const user = userEvent.setup()
    const onNew = vi.fn()
    renderAt("/conversations/c1", { onNew, conversations: [CONVERSATION] })
    await user.click(screen.getByRole("button", { name: "Chat" }))
    expect(onNew).not.toHaveBeenCalled()
    expect(screen.getByRole("link", { name: /Churn by region/ })).toHaveAttribute(
      "aria-current",
      "page",
    )
  })

  it("starts a new analysis from the history header", async () => {
    const user = userEvent.setup()
    const onNew = vi.fn()
    renderAt("/", { onNew })
    await user.click(screen.getByRole("button", { name: "New analysis" }))
    expect(onNew).toHaveBeenCalledOnce()
  })

  it("debounces the history search", () => {
    vi.useFakeTimers()
    try {
      const onSearch = vi.fn()
      renderAt("/", { onSearch })
      fireEvent.change(screen.getByRole("textbox", { name: "Search analyses" }), {
        target: { value: "churn" },
      })
      expect(onSearch).not.toHaveBeenCalled()
      act(() => {
        vi.advanceTimersByTime(250)
      })
      expect(onSearch).toHaveBeenCalledWith("churn")
      expect(screen.getByText("No matches")).toBeInTheDocument()
    } finally {
      vi.useRealTimers()
    }
  })

  it("renames a conversation from its options menu", async () => {
    const user = userEvent.setup()
    const onRename = vi.fn()
    renderAt("/", { onRename, conversations: [CONVERSATION] })
    const trigger = screen.getByRole("button", { name: "Conversation options" })
    // Radix menus close on a window blur, which jsdom fires at pointerdown when an
    // earlier test left nothing focused; focusing first avoids it (jsdom only).
    trigger.focus()
    await user.click(trigger)
    await user.click(await screen.findByRole("menuitem", { name: "Rename" }))
    const field = await screen.findByRole("textbox", { name: "Analysis title" })
    expect(field).toHaveValue("Churn by region")
    await user.clear(field)
    await user.type(field, "Churn drivers{Enter}")
    expect(onRename).toHaveBeenCalledWith("c1", "Churn drivers")
  })

  it("cycles the theme from the toggle", async () => {
    const user = userEvent.setup()
    renderAt("/warehouse")
    await user.click(screen.getByRole("button", { name: /^Theme: system theme/ }))
    expect(localStorage.getItem("elbi-theme")).toBe("light")
    await user.click(screen.getByRole("button", { name: /^Theme: light theme/ }))
    expect(localStorage.getItem("elbi-theme")).toBe("dark")
    expect(document.documentElement).toHaveClass("dark")
    await user.click(screen.getByRole("button", { name: /^Theme: dark theme/ }))
    expect(screen.getByRole("button", { name: /^Theme: system theme/ })).toBeInTheDocument()
  })
})
