// Pins the app chrome's assistant rail: its button opens and closes the assistant panel on
// a browse page, and the rail is absent on the chat routes.

import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { MemoryRouter } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

vi.mock("@/components/AssistantPanel", () => ({
  AssistantPanel: ({ onClose }: { onClose: () => void }) => (
    <div data-testid="assistant-panel">
      <button type="button" onClick={onClose}>
        close panel
      </button>
    </div>
  ),
}))
vi.mock("@/views/NotificationsPage", () => ({ NotificationsPage: () => <div>inbox page</div> }))
vi.mock("@/views/ChatView", () => ({ ChatView: () => <div>chat view</div> }))

import App from "./App"

let realFetch: typeof window.fetch

beforeEach(() => {
  realFetch = window.fetch
  window.fetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = String(input)
    const body = url.includes("/messages")
      ? []
      : url.includes("/api/conversations")
        ? { items: [], next_page_id: null }
        : url.includes("/api/notifications")
          ? { items: [], unread: 0 }
          : []
    return new Response(JSON.stringify(body), {
      status: 200,
      headers: { "Content-Type": "application/json" },
    })
  }) as typeof window.fetch
})

afterEach(() => {
  window.fetch = realFetch
})

function renderAt(path: string) {
  return render(
    <MemoryRouter initialEntries={[path]}>
      <App />
    </MemoryRouter>,
  )
}

describe("App assistant rail", () => {
  it("toggles the assistant panel from the rail button", async () => {
    const user = userEvent.setup()
    renderAt("/notifications")
    await screen.findByText("inbox page")
    expect(screen.queryByTestId("assistant-panel")).toBeNull()
    const rail = screen.getByRole("button", { name: /^Assistant/ })
    await user.click(rail)
    expect(screen.getByTestId("assistant-panel")).toBeInTheDocument()
    expect(rail).toHaveAttribute("aria-pressed", "true")
    await user.click(rail)
    expect(screen.queryByTestId("assistant-panel")).toBeNull()
  })

  it("shows no rail on the chat routes", async () => {
    renderAt("/conversations/c1")
    await screen.findByText("chat view")
    expect(screen.queryByRole("button", { name: /^Assistant/ })).toBeNull()
  })
})
