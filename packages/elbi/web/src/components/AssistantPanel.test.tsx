// Pins the assistant panel's composer: Enter and the send button both send the draft
// with the page context, Shift+Enter does not, and the button stops a running reply.

import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"

const chat = {
  messages: [] as unknown[],
  sendMessage: vi.fn(),
  status: "ready",
  stop: vi.fn(),
}

vi.mock("@ai-sdk/react", () => ({ useChat: () => chat }))

import { AssistantPanel } from "./AssistantPanel"

function renderPanel(onClose = vi.fn()) {
  render(
    <TooltipProvider>
      <AssistantPanel
        pageContext={{ label: "Notebook Q3", text: "on notebook q3" }}
        profile="default"
        onClose={onClose}
        onActed={() => {}}
      />
    </TooltipProvider>,
  )
  return { onClose }
}

beforeEach(() => {
  chat.status = "ready"
  chat.messages = []
  vi.clearAllMocks()
})

describe("AssistantPanel composer", () => {
  it("sends on Enter with the page context, and not on Shift+Enter", async () => {
    const user = userEvent.setup()
    renderPanel()
    const box = screen.getByPlaceholderText("Ask the assistant…")
    await user.type(box, "first line{Shift>}{Enter}{/Shift}")
    expect(chat.sendMessage).not.toHaveBeenCalled()
    await user.type(box, "second{Enter}")
    expect(chat.sendMessage).toHaveBeenCalledWith(
      { text: "first line\nsecond" },
      {
        body: expect.objectContaining({ profile: "default", context: "on notebook q3" }),
      },
    )
    expect(box).toHaveValue("")
  })

  it("sends from the button and ignores an empty draft", async () => {
    const user = userEvent.setup()
    renderPanel()
    const send = screen.getByRole("button", { name: "Send" })
    await user.click(send)
    expect(chat.sendMessage).not.toHaveBeenCalled()
    await user.type(screen.getByPlaceholderText("Ask the assistant…"), "hello")
    await user.click(send)
    expect(chat.sendMessage).toHaveBeenCalledWith({ text: "hello" }, expect.anything())
  })

  it("stops a streaming reply from the same button", async () => {
    const user = userEvent.setup()
    chat.status = "streaming"
    renderPanel()
    await user.click(screen.getByRole("button", { name: "Stop" }))
    expect(chat.stop).toHaveBeenCalledOnce()
  })

  it("closes from the header button", async () => {
    const user = userEvent.setup()
    const { onClose } = renderPanel()
    await user.click(screen.getByRole("button", { name: /Close/ }))
    expect(onClose).toHaveBeenCalledOnce()
  })
})
