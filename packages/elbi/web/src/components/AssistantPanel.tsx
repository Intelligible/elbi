import { useChat } from "@ai-sdk/react"
import { DefaultChatTransport } from "ai"
import { ArrowUp, Loader2, Wrench, X } from "lucide-react"
import { useEffect, useRef, useState } from "react"
import { Streamdown } from "streamdown"
import { AssistantMark } from "@/components/AssistantMark"
import { IconButton } from "@/components/app/IconButton"
import { Textarea } from "@/components/ui/textarea"
import { uuid } from "@/lib/utils"

// The context the assistant is given about the page the user is on: a human label for
// the chip and the instruction text sent to the model.
export type PageContext = { label: string; text: string } | null

// Tools that change the surface the user is looking at; when the assistant calls one we
// refresh the current page so the edit appears live (a new notebook cell, say).
const MUTATING_TOOLS = new Set([
  "write_notebook",
  "run_notebook",
  "promote_notebook_cell",
  "schedule_notebook",
  "write_dashboard",
  "publish_dashboard",
  "metric_define",
  "monitor_create",
])

type Part = { type: string; text?: string; data?: { name?: string; text?: string } }

export function AssistantPanel({
  pageContext,
  profile,
  onClose,
  onActed,
}: {
  pageContext: PageContext
  profile: string
  onClose: () => void
  onActed: () => void
}) {
  // An ephemeral conversation for the assistant, kept off the URL so it never collides
  // with the main chat route. Stable for the panel's lifetime.
  const conversationId = useRef(uuid()).current
  const chat = useChat({
    transport: new DefaultChatTransport({ api: "/api/chat" }),
  })
  const { messages, sendMessage, status, stop } = chat
  const busy = status === "submitted" || status === "streaming"
  const [text, setText] = useState("")
  const scrollRef = useRef<HTMLDivElement>(null)

  const send = () => {
    if (!text.trim() || busy) return
    sendMessage({ text }, { body: { profile, context: pageContext?.text ?? "", conversationId } })
    setText("")
  }

  // Refresh the current page when the assistant runs a surface-mutating tool, so its edit
  // shows up live. Counts mutating tool calls across the thread and fires on each new one.
  const actedCount = useRef(0)
  useEffect(() => {
    let seen = 0
    for (const m of messages) {
      if (m.role !== "assistant") continue
      for (const part of m.parts as Part[]) {
        if (part.type === "data-tool" && MUTATING_TOOLS.has(part.data?.name ?? "")) {
          seen += 1
        }
      }
    }
    if (seen > actedCount.current) {
      actedCount.current = seen
      onActed()
    }
  }, [messages, onActed])

  // Follow the content rather than the state that produced it. Keyed on `messages` this
  // scrolled once per message, which misses the case that matters most: a single reply
  // streaming in grows the last message for seconds without the array ever changing, so
  // the view fell behind the text. Watching the scroll content covers both, and covers
  // anything else that changes height -- an image loading, a table expanding.
  //
  // Held at the bottom only while the reader is already there. Scrolling up is a
  // deliberate act, and yanking someone back down mid-read is the failure this guards.
  useEffect(() => {
    const box = scrollRef.current
    if (!box) return
    const NEAR_BOTTOM = 64
    let pinned = true
    const onScroll = () => {
      pinned = box.scrollHeight - box.scrollTop - box.clientHeight <= NEAR_BOTTOM
    }
    const observer = new ResizeObserver(() => {
      if (pinned) box.scrollTo({ top: box.scrollHeight })
    })
    for (const child of box.children) observer.observe(child)
    const mutations = new MutationObserver(() => {
      for (const child of box.children) observer.observe(child)
      if (pinned) box.scrollTo({ top: box.scrollHeight })
    })
    mutations.observe(box, { childList: true })
    box.addEventListener("scroll", onScroll, { passive: true })
    return () => {
      observer.disconnect()
      mutations.disconnect()
      box.removeEventListener("scroll", onScroll)
    }
  }, [])

  return (
    <aside className="flex h-full flex-col overflow-hidden rounded-lg border border-border bg-background">
      <header className="flex items-center justify-between border-b border-border px-3 py-2.5">
        <span className="flex items-center gap-1.5 text-sm font-semibold">
          <AssistantMark className="size-4 text-primary" /> Assistant
        </span>
        <IconButton label="Close assistant (⌘I)" onClick={onClose}>
          <X className="size-4" />
        </IconButton>
      </header>

      {pageContext ? (
        <div className="flex items-center gap-1.5 border-b border-border/60 bg-muted/40 px-3 py-1.5 text-xs text-text-tertiary">
          <span className="size-1.5 rounded-full bg-primary" />
          Context: {pageContext.label}
        </div>
      ) : null}

      <div ref={scrollRef} className="min-h-0 flex-1 space-y-4 overflow-y-auto px-3 py-4">
        {messages.length === 0 ? (
          <div className="pt-6 text-center text-sm text-text-tertiary">
            <AssistantMark className="mx-auto mb-2 size-6 text-text-tertiary/70" />
            Ask about what you're working on.
            {pageContext ? (
              <div className="mt-1 text-xs">
                It knows you're on <span className="font-medium">{pageContext.label}</span>.
              </div>
            ) : null}
          </div>
        ) : (
          messages.map((m) => (
            <PanelTurn key={m.id} role={m.role} parts={m.parts as Part[]} busy={busy} />
          ))
        )}
      </div>

      <div className="border-t border-border p-2.5">
        <div className="flex items-end gap-2 rounded-xl border border-border bg-card px-3 py-2 focus-within:border-primary/60">
          {/* One fixed row that scrolls, borderless inside the rounded composer. */}
          <Textarea
            value={text}
            onChange={(e) => setText(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault()
                send()
              }
            }}
            rows={1}
            placeholder="Ask the assistant…"
            className="field-sizing-fixed max-h-40 min-h-6 flex-1 resize-none rounded-none border-0 bg-transparent p-0 hover:border-0 focus-visible:ring-0"
          />
          <IconButton
            variant="default"
            label={busy ? "Stop" : "Send"}
            className="rounded-full"
            onClick={() => (busy ? stop() : send())}
          >
            {busy ? <Loader2 className="size-4 animate-spin" /> : <ArrowUp className="size-4" />}
          </IconButton>
        </div>
      </div>
    </aside>
  )
}

function PanelTurn({ role, parts, busy }: { role: string; parts: Part[]; busy: boolean }) {
  if (role === "user") {
    const text = parts
      .filter((p) => p.type === "text")
      .map((p) => p.text ?? "")
      .join("")
    return (
      <div className="flex justify-end">
        <div className="max-w-[85%] rounded-2xl rounded-br-sm bg-primary px-3 py-2 text-sm text-primary-foreground">
          {text}
        </div>
      </div>
    )
  }

  const tools: { id: number; name: string }[] = []
  const textParts: string[] = []
  for (const [partIndex, part] of parts.entries()) {
    if (part.type === "text") textParts.push(part.text ?? "")
    else if (part.type === "data-tool" && part.data?.name)
      tools.push({ id: partIndex, name: part.data.name })
  }
  const text = textParts.join("")

  return (
    <div className="space-y-2 text-sm">
      {tools.map(({ id, name }) => (
        <div
          key={id}
          className="inline-flex items-center gap-1.5 rounded-md bg-muted px-2 py-0.5 text-xs text-text-tertiary"
        >
          <Wrench className="size-3" /> {toolLabel(name)}
        </div>
      ))}
      {busy && !text && tools.length === 0 ? (
        <span className="animate-pulse text-text-tertiary">Thinking…</span>
      ) : null}
      {text ? <Streamdown className="leading-6 text-foreground/90">{text}</Streamdown> : null}
    </div>
  )
}

// A readable label for a tool call ("write_notebook" -> "Editing notebook").
function toolLabel(name: string): string {
  const labels: Record<string, string> = {
    read_notebook: "Reading notebook",
    write_notebook: "Editing notebook",
    run_notebook: "Running notebook",
    promote_notebook_cell: "Promoting a cell",
    schedule_notebook: "Scheduling notebook",
    write_dashboard: "Editing dashboard",
    publish_dashboard: "Publishing dashboard",
    metric_define: "Defining metric",
    metric_query: "Querying metric",
    monitor_create: "Creating monitor",
    derive: "Authoring a derivation",
  }
  return labels[name] ?? name.replace(/_/g, " ")
}
