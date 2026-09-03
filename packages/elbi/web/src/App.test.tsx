import { act, render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { useEffect } from "react"
import { MemoryRouter, useLocation, useNavigate } from "react-router-dom"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import App from "@/App"

// The frames /api/chat actually emits, in the order it emits them (see the publisher in
// app.py: start, start-step, text-start, text-delta, text-end, finish-step, finish, [DONE]).
// Copied from the server rather than invented, so a stream the server would never send
// cannot make this test pass.
const ANSWER = "The effect is real."

const OPENING = [
  { type: "start", messageId: "m1" },
  { type: "start-step" },
  { type: "text-start", id: "t1" },
]
const CLOSING = [{ type: "text-end", id: "t1" }, { type: "finish-step" }, { type: "finish" }]

/** An /api/chat response the test feeds by hand, so an answer can be held in flight. */
function chatStream() {
  const encode = new TextEncoder()
  let sink: ReadableStreamDefaultController<Uint8Array>
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      sink = controller
    },
  })
  return {
    response: new Response(body, {
      headers: {
        "content-type": "text/event-stream",
        "x-vercel-ai-ui-message-stream": "v1",
      },
    }),
    push: (...frames: object[]) => {
      for (const f of frames) sink.enqueue(encode.encode(`data: ${JSON.stringify(f)}\n\n`))
    },
    close: () => {
      sink.enqueue(encode.encode("data: [DONE]\n\n"))
      sink.close()
    },
  }
}

/** The whole answer in one go, which is all most tests need. */
function wholeAnswer(stream: ReturnType<typeof chatStream>) {
  stream.push(...OPENING, { type: "text-delta", id: "t1", delta: ANSWER }, ...CLOSING)
  stream.close()
}

// An older conversation already in the sidebar, with one stored turn of its own.
const OLD_ID = "old-conversation"
const OLD_TITLE = "Last week's look at z"
const OLD_QUESTION = "what did we find last week"

// What each GET the app makes on mount answers with. getJson swallows anything unmatched
// into its own fallback, so only the shapes the chat screen actually reads matter here.
function payloadFor(url: string): unknown {
  if (url.includes("/api/settings/llm/profiles"))
    return { profiles: [], default: "", titleProfile: "", configured: true }
  if (/\/api\/conversations\/[^/]+\/usage/.test(url))
    return { prompt_tokens: 0, completion_tokens: 0, cost: 0 }
  if (url.includes(`/api/conversations/${OLD_ID}/messages`))
    return [{ id: "s1", role: "user", content: OLD_QUESTION, result: null }]
  if (/\/api\/conversations\/[^/]+\/messages/.test(url)) return []
  if (url.includes("/api/conversations"))
    return {
      items: [{ id: OLD_ID, title: OLD_TITLE, updated_at: "2026-08-01T00:00:00Z" }],
      next_page_id: null,
    }
  return []
}

/** The conversationId the app sent with each /api/chat request. */
let sentIds: string[]
/** The router path, so the test can see the URL the app navigated to. */
let path: string
/** One entry per /api/chat request, for a test that feeds the answer by hand. */
let streams: ReturnType<typeof chatStream>[]
/** Whether a request answers itself immediately (off for the in-flight tests). */
let autoAnswer: boolean
/** Move the router the way a nav click would, without depending on the sidebar's markup. */
let go: (to: string) => void

function RouterProbe() {
  const location = useLocation()
  const navigate = useNavigate()
  go = navigate
  useEffect(() => {
    path = location.pathname
  }, [location.pathname])
  return null
}

beforeEach(() => {
  sentIds = []
  path = ""
  streams = []
  autoAnswer = true
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: RequestInfo | URL, init?: RequestInit) => {
      const url = String(input)
      if (url.includes("/api/chat")) {
        const body = JSON.parse(String(init?.body ?? "{}")) as { conversationId?: string }
        sentIds.push(body.conversationId ?? "")
        const stream = chatStream()
        streams.push(stream)
        if (autoAnswer) wholeAnswer(stream)
        return stream.response
      }
      return new Response(JSON.stringify(payloadFor(url)), {
        headers: { "content-type": "application/json" },
      })
    }),
  )
})

afterEach(() => {
  vi.unstubAllGlobals()
})

async function ask(text: string) {
  await userEvent.type(screen.getByPlaceholderText(/Ask anything about your data/), text)
  await userEvent.click(screen.getByRole("button", { name: "Submit" }))
}

describe("asking a question from a blank chat", () => {
  it("keeps the question on screen and streams the answer into it", async () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
        <RouterProbe />
      </MemoryRouter>,
    )

    await ask("does x affect y")

    // The regression: the send moved the URL onto the new conversation, which recreated the
    // chat and took the question and its stream with it -- the user was dropped back on an
    // empty home screen and the answer never arrived.
    expect(await screen.findByText("does x affect y")).toBeInTheDocument()
    expect(await screen.findByText(ANSWER)).toBeInTheDocument()
  })

  it("sends the question to the conversation the url settled on", async () => {
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
        <RouterProbe />
      </MemoryRouter>,
    )

    await ask("does x affect y")
    await screen.findByText(ANSWER)

    // One request, addressed to the conversation the user is now looking at -- not to a
    // conversation they were bounced out of.
    expect(sentIds).toHaveLength(1)
    expect(path).toBe(`/conversations/${sentIds[0]}`)
  })
})

describe("leaving a running answer", () => {
  it("keeps it alive across a trip to another page", async () => {
    autoAnswer = false
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
        <RouterProbe />
      </MemoryRouter>,
    )

    await ask("does x affect y")
    await screen.findByText("does x affect y")
    const conversation = path

    // The answer starts arriving, then the user goes to look at their derivations.
    await act(async () => {
      streams[0].push(...OPENING, { type: "text-delta", id: "t1", delta: "Half an " })
    })
    await act(async () => {
      go("/derivations")
    })
    expect(path).toBe("/derivations")

    // The rest of the answer lands while they are away, and it is still there when they
    // come back: the Derivations page carries no conversation id, and that used to be
    // enough to throw the running chat away.
    await act(async () => {
      streams[0].push({ type: "text-delta", id: "t1", delta: "answer." }, ...CLOSING)
      streams[0].close()
    })
    await act(async () => {
      go(conversation)
    })

    expect(await screen.findByText("Half an answer.")).toBeInTheDocument()
    expect(sentIds).toEqual([conversation.replace("/conversations/", "")])
  })
})

describe("opening another conversation while an answer is running", () => {
  it("shows that conversation, not the answer streaming into the one left behind", async () => {
    autoAnswer = false
    render(
      <MemoryRouter initialEntries={["/"]}>
        <App />
        <RouterProbe />
      </MemoryRouter>,
    )

    await ask("does x affect y")
    await screen.findByText("does x affect y")
    const running = path

    await act(async () => {
      streams[0].push(...OPENING, { type: "text-delta", id: "t1", delta: "Half an " })
    })
    await screen.findByText("Half an")

    // Open the older conversation mid-answer. It has to replay its own stored turn; the
    // answer still arriving for the other one must not be written into this window.
    await act(async () => {
      go(`/conversations/${OLD_ID}`)
    })
    expect(await screen.findByText(OLD_QUESTION)).toBeInTheDocument()
    expect(screen.queryByText("Half an")).toBeNull()
    expect(screen.queryByText("does x affect y")).toBeNull()

    // And the answer was not killed by the trip: it finishes and is there on return.
    await act(async () => {
      streams[0].push({ type: "text-delta", id: "t1", delta: "answer." }, ...CLOSING)
      streams[0].close()
    })
    await act(async () => {
      go(running)
    })
    expect(await screen.findByText("Half an answer.")).toBeInTheDocument()
  })
})
