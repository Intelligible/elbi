// Pins the chat surface's controls: each test drives a control that sends a message,
// edits and resends one, rates or regenerates an answer, or changes what the next send
// carries (context, images), so a mapping mistake shows up as a wrong call.

import { render, screen, waitFor } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import type { UIMessage } from "ai"
import { MemoryRouter } from "react-router-dom"
import { afterEach, describe, expect, it, vi } from "vitest"

import { TooltipProvider } from "@/components/ui/tooltip"

vi.mock("@/lib/chat", async (importOriginal) => ({
  ...(await importOriginal<typeof import("@/lib/chat")>()),
  setMessageFeedback: vi.fn(async () => true),
  getConversationUsage: vi.fn(async () => ({ prompt_tokens: 0, completion_tokens: 0, cost: 0 })),
  getConversationMessages: vi.fn(async () => []),
  getJobs: vi.fn(async () => []),
}))

import { setMessageFeedback } from "@/lib/chat"
import { ChatView, Turn } from "./ChatView"

afterEach(() => {
  vi.clearAllMocks()
})

function answer(): UIMessage {
  return {
    id: "a1",
    role: "assistant",
    parts: [
      { type: "data-reasoning", data: { text: "Looking at the columns first." } },
      { type: "data-tool", data: { name: "run_code", arguments: { code: "df.describe()" } } },
      { type: "data-tool-output", data: { output: "count 10" } },
      { type: "text", text: "The effect is positive." },
    ],
  } as unknown as UIMessage
}

function userMessage(text: string): UIMessage {
  return { id: "u1", role: "user", parts: [{ type: "text", text }] } as unknown as UIMessage
}

function wrap(ui: React.ReactNode) {
  return render(
    <MemoryRouter>
      <TooltipProvider>{ui}</TooltipProvider>
    </MemoryRouter>,
  )
}

function makeChat(messages: UIMessage[] = []) {
  return {
    messages,
    sendMessage: vi.fn(),
    status: "ready",
    error: undefined,
    stop: vi.fn(),
    setMessages: vi.fn(),
    regenerate: vi.fn(),
  } as unknown as Parameters<typeof ChatView>[0]["chat"]
}

function renderChat(chat = makeChat(), extra: Partial<Parameters<typeof ChatView>[0]> = {}) {
  const onContext = vi.fn()
  wrap(
    <ChatView
      chat={chat}
      conversationId=""
      datasets={[]}
      profiles={[]}
      profile="default"
      onProfile={() => {}}
      llmConfigured
      context={null}
      onContext={onContext}
      {...extra}
    />,
  )
  return { chat, onContext }
}

describe("assistant turn actions", () => {
  it("rates an answer up, toggles it off, then rates it down", async () => {
    const user = userEvent.setup()
    wrap(<Turn message={answer()} busy={false} />)
    const up = screen.getByRole("button", { name: "Good response" })
    const down = screen.getByRole("button", { name: "Bad response" })
    expect(up).toHaveAttribute("aria-pressed", "false")
    await user.click(up)
    expect(setMessageFeedback).toHaveBeenLastCalledWith("a1", "up")
    expect(up).toHaveAttribute("aria-pressed", "true")
    await user.click(up)
    expect(setMessageFeedback).toHaveBeenLastCalledWith("a1", null)
    await user.click(down)
    expect(setMessageFeedback).toHaveBeenLastCalledWith("a1", "down")
    expect(down).toHaveAttribute("aria-pressed", "true")
    expect(up).toHaveAttribute("aria-pressed", "false")
  })

  it("regenerates only when offered", async () => {
    const user = userEvent.setup()
    const onRegenerate = vi.fn()
    const { unmount } = wrap(<Turn message={answer()} busy={false} onRegenerate={onRegenerate} />)
    await user.click(screen.getByRole("button", { name: "Regenerate" }))
    expect(onRegenerate).toHaveBeenCalledOnce()
    unmount()
    wrap(<Turn message={answer()} busy={false} />)
    expect(screen.queryByRole("button", { name: "Regenerate" })).toBeNull()
  })

  it("collapses the trace and expands a tool step's input and output", async () => {
    const user = userEvent.setup()
    wrap(<Turn message={answer()} busy={false} />)
    expect(screen.getByText("Looking at the columns first.")).toBeInTheDocument()
    const step = screen.getByRole("button", { name: /Explored the data/ })
    expect(screen.queryByText("df.describe()")).toBeNull()
    await user.click(step)
    expect(screen.getByText("df.describe()")).toBeInTheDocument()
    expect(screen.getByText("count 10")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: /How it got here/ }))
    expect(screen.queryByText("Looking at the columns first.")).toBeNull()
  })
})

describe("user turn editing", () => {
  it("edits and resends the message text", async () => {
    const user = userEvent.setup()
    const onEdit = vi.fn()
    wrap(<Turn message={userMessage("old question")} busy={false} onEdit={onEdit} />)
    await user.click(screen.getByRole("button", { name: "Edit message" }))
    const box = screen.getByRole("textbox")
    expect(box).toHaveValue("old question")
    await user.clear(box)
    await user.type(box, "new question")
    await user.click(screen.getByRole("button", { name: "Send" }))
    expect(onEdit).toHaveBeenCalledWith("u1", "new question")
    expect(screen.queryByRole("textbox")).toBeNull()
  })

  it("cancels an edit without sending and disables Send on an empty draft", async () => {
    const user = userEvent.setup()
    const onEdit = vi.fn()
    wrap(<Turn message={userMessage("keep me")} busy={false} onEdit={onEdit} />)
    await user.click(screen.getByRole("button", { name: "Edit message" }))
    await user.clear(screen.getByRole("textbox"))
    expect(screen.getByRole("button", { name: "Send" })).toBeDisabled()
    await user.click(screen.getByRole("button", { name: "Cancel" }))
    expect(onEdit).not.toHaveBeenCalled()
    expect(screen.getByText("keep me")).toBeInTheDocument()
  })
})

describe("home composer", () => {
  it("sends an example prompt", async () => {
    const user = userEvent.setup()
    const { chat } = renderChat()
    await user.click(screen.getByRole("button", { name: "Do the groups differ on the metric?" }))
    expect(chat.sendMessage).toHaveBeenCalledWith(
      { text: "Do the groups differ on the metric?" },
      expect.objectContaining({ body: expect.objectContaining({ profile: "default" }) }),
    )
  })

  it("removes the attached context", async () => {
    const user = userEvent.setup()
    const { onContext } = renderChat(makeChat(), {
      context: { name: "spec.md", text: "the spec" },
    })
    expect(screen.getByText("spec.md")).toBeInTheDocument()
    await user.click(screen.getByRole("button", { name: "Remove context" }))
    expect(onContext).toHaveBeenCalledWith(null)
  })

  it("opens the file pickers from the attach buttons", async () => {
    const user = userEvent.setup()
    const click = vi.spyOn(HTMLInputElement.prototype, "click").mockImplementation(() => {})
    renderChat()
    await user.click(screen.getByRole("button", { name: /Attach a spec or data dictionary/ }))
    await user.click(screen.getByRole("button", { name: /Attach an image/ }))
    const accepts = click.mock.contexts.map((el) => (el as HTMLInputElement).accept)
    expect(accepts).toEqual([".md,.txt,.csv,.json,.yaml,.yml", "image/*"])
    click.mockRestore()
  })

  it("attaches an image and removes it again before sending", async () => {
    const user = userEvent.setup()
    const { chat } = renderChat()
    const input = document.querySelector<HTMLInputElement>('input[type="file"][accept="image/*"]')
    if (!input) throw new Error("image input missing")
    await user.upload(input, new File(["png"], "chart.png", { type: "image/png" }))
    const img = await screen.findByRole("img", { name: "attachment" })
    expect(img.getAttribute("src")).toMatch(/^data:image\/png/)
    await user.click(screen.getByRole("button", { name: "Remove image" }))
    expect(screen.queryByRole("img", { name: "attachment" })).toBeNull()
    await user.click(
      screen.getByRole("button", { name: "How well do the features predict the target?" }),
    )
    expect(chat.sendMessage).toHaveBeenCalledWith(
      expect.anything(),
      expect.objectContaining({ body: expect.objectContaining({ images: [] }) }),
    )
  })

  it("starts dictation from the mic button", async () => {
    const user = userEvent.setup()
    const start = vi.fn()
    class FakeRecognition {
      continuous = false
      interimResults = false
      lang = ""
      onresult = null
      onend = null
      onerror = null
      start = start
      stop() {}
    }
    const w = window as unknown as { SpeechRecognition?: unknown }
    w.SpeechRecognition = FakeRecognition
    try {
      renderChat()
      await user.click(screen.getByRole("button", { name: /^Dictate/ }))
      expect(start).toHaveBeenCalledOnce()
    } finally {
      delete w.SpeechRecognition
    }
  })
})

describe("conversation view", () => {
  it("renders turns and the composer below them", async () => {
    renderChat(makeChat([userMessage("hi"), answer()]), { conversationId: "c1" })
    await waitFor(() => expect(screen.getByText("The effect is positive.")).toBeInTheDocument())
    expect(screen.getByText("hi")).toBeInTheDocument()
    expect(screen.getByPlaceholderText("Ask anything about your data…")).toBeInTheDocument()
  })
})
