import type { useChat } from "@ai-sdk/react"
import type { UIMessage } from "ai"
import {
  ChevronDown,
  ChevronRight,
  CornerDownRight,
  FileText,
  ImagePlus,
  KeyRound,
  Mic,
  Pencil,
  Plus,
  RotateCcw,
  ShieldCheck,
  ThumbsDown,
  ThumbsUp,
  X,
} from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"
import { Link } from "react-router-dom"
import { Streamdown } from "streamdown"

import { Conversation, ConversationContent } from "@/components/ai-elements/conversation"
import { Message, MessageContent } from "@/components/ai-elements/message"
import {
  PromptInput,
  PromptInputBody,
  PromptInputFooter,
  PromptInputProvider,
  PromptInputSubmit,
  PromptInputTextarea,
  PromptInputTools,
  usePromptInputController,
} from "@/components/ai-elements/prompt-input"
import { IconButton } from "@/components/app/IconButton"
import { JobsBar } from "@/components/JobsBar"
import { ProviderGlyph } from "@/components/ProviderIcon"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import { Button } from "@/components/ui/button"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Textarea } from "@/components/ui/textarea"
import { type VizDescriptor, VizView } from "@/components/viz/VizView"
import {
  certificateUrl,
  type Dataset,
  getConversationMessages,
  getConversationUsage,
  type LlmProfile,
  setMessageFeedback,
  type Usage,
} from "@/lib/chat"
import { storedToUIMessages } from "@/lib/messages"
import { useSpeechRecognition } from "@/lib/useSpeechRecognition"
import { cn, uuid } from "@/lib/utils"

type Chat = ReturnType<typeof useChat>

const EXAMPLES = [
  "What is the effect of x on y?",
  "Do the groups differ on the metric?",
  "How well do the features predict the target?",
]

const STEP_LABELS: Record<string, string> = {
  describe_dataset: "Inspected the dataset",
  structure_map: "Mapped column relationships",
  run_code: "Explored the data",
  derive: "Authored & verified a derivation",
  answer: "Wrote the finding",
  conclude_inconclusive: "Concluded",
}

interface ResultPayload {
  verified: boolean
  verdict: string
  checks: { name: string; verdict: string; detail: string }[]
  assumptions: string[]
  spec: Record<string, unknown> | null
  data_hash: string | null
}

// Coerce an untrusted result payload into a complete one, so a missing or partial field
// (a malformed server payload, or a follow-up reconstructed from storage) renders instead
// of throwing on `result.checks.map` or `result.assumptions.length`.
function normalizeResult(d: Partial<ResultPayload> | undefined | null): ResultPayload {
  return {
    verified: Boolean(d?.verified),
    verdict: d?.verdict ?? "unverified",
    checks: Array.isArray(d?.checks) ? d.checks : [],
    assumptions: Array.isArray(d?.assumptions) ? d.assumptions : [],
    spec: d?.spec ?? null,
    data_hash: d?.data_hash ?? null,
  }
}

// useChat's transport throws `new Error(await response.text())` on a non-ok
// response, so `error.message` is the raw FastAPI body -- a JSON string, not
// plain text. /api/chat's detail is either a plain string or, for a missing
// LLM config specifically, {message, code} so this view can point at Settings
// instead of a generic failure. Falls back to the raw message for anything
// that isn't JSON (a network-level failure, not an HTTP error response).
function parseChatError(message: string | undefined): { text: string; code?: string } | null {
  if (!message) return null
  try {
    const detail = (JSON.parse(message) as { detail?: unknown }).detail
    if (typeof detail === "string") return { text: detail }
    if (detail && typeof detail === "object") {
      const { message: text, code } = detail as { message?: string; code?: string }
      if (typeof text === "string") return { text, code }
    }
  } catch {
    // Not JSON -- fall through to the raw message below.
  }
  return { text: message }
}

/** `id` is the index of the streamed part that opened this item.
 *
 *  The trace is a projection of `message.parts` rebuilt on every render -- consecutive
 *  reasoning deltas coalesce into one item -- so a freshly minted id would remount the
 *  whole trace each time a token arrived. The part that opened an item never moves,
 *  which makes it the one thing about the item that is stable while its text grows. */
type TraceItem = { id: number } & (
  | { kind: "reasoning"; text: string }
  | { kind: "tool"; name: string; args: Record<string, unknown>; output?: string }
)

type PartData = {
  data: {
    text?: string
    name?: string
    arguments?: Record<string, unknown>
    output?: string
  }
}

type Context = { name: string; text: string } | null

// A 22px message action: the icon recolours on hover, with no fill.
const ACTION =
  "size-5.5 rounded text-text-tertiary hover:bg-transparent hover:text-foreground dark:hover:bg-transparent"

// A 28px composer tool with a muted fill on hover.
const TOOL =
  "size-7 rounded-lg text-text-tertiary hover:bg-muted hover:text-foreground dark:hover:bg-muted"

// A text-only disclosure toggle (the trace and receipt headers).
const DISCLOSURE =
  "flex h-auto justify-start gap-1.5 text-xs text-text-tertiary transition hover:bg-transparent hover:text-foreground dark:hover:bg-transparent"

export function ChatView({
  chat,
  conversationId,
  datasets,
  profiles,
  profile,
  onProfile,
  llmConfigured,
  context,
  onContext,
}: {
  chat: Chat
  conversationId: string
  datasets: Dataset[]
  profiles: LlmProfile[]
  profile: string
  onProfile: (name: string) => void
  // Live check of whether a chat send would actually go through right now (see
  // getLlmProfiles); drives the "no model configured yet" callout below.
  llmConfigured: boolean
  context: Context
  onContext: (c: Context) => void
}) {
  const { messages, sendMessage, status, error, stop } = chat
  const busy = status === "submitted" || status === "streaming"
  // Images attached to the next message, as data URLs (up to the server's cap of 4).
  // `{ id, src }` rather than a bare data URL: the same picture can be attached twice,
  // and removing one of a pair must take that one rather than whichever now sits at its
  // index. Client-only -- `send` maps to the plain sources the API takes.
  const [images, setImages] = useState<{ id: string; src: string }[]>([])

  function send(text: string) {
    if (!text.trim() || busy) return
    sendMessage(
      { text },
      {
        body: {
          profile,
          context: context?.text ?? "",
          conversationId,
          // The ids are the editor's; the API takes the sources.
          images: images.map((img) => img.src),
        },
      },
    )
    setImages([])
  }

  // Edit-and-resend: drop the edited message and everything after it from the view, then
  // resend the new text with its id so the server truncates the stored turns to match.
  function editAndResend(messageId: string, text: string) {
    if (!text.trim() || busy) return
    chat.setMessages((prev) => {
      const idx = prev.findIndex((m) => m.id === messageId)
      return idx === -1 ? prev : prev.slice(0, idx)
    })
    sendMessage(
      { text },
      {
        body: {
          profile,
          context: context?.text ?? "",
          conversationId,
          editMessageId: messageId,
        },
      },
    )
  }

  // A background job just certified: pull any assistant follow-up the server posted to
  // this conversation (the agent's verified finding) and append it to the thread, so a
  // finished job actually reports back instead of dropping into oblivion. Memoized so
  // JobsBar's poll effect does not tear down and restart on every streaming token.
  const setMessages = chat.setMessages
  const pullFollowup = useCallback(() => {
    if (!conversationId) return // the home screen has no conversation to pull from yet
    void getConversationMessages(conversationId).then((stored) => {
      const posted = stored.filter((m) => m.role === "assistant")
      setMessages((prev) => {
        const shown = prev.filter((m) => m.role === "assistant").length
        const extra = posted.slice(shown)
        if (extra.length === 0) return prev
        // Carry the certified verification payload, not just the text, so a background
        // follow-up shows its verdict and receipt like an inline answer.
        return [...prev, ...storedToUIMessages(extra)]
      })
    })
  }, [conversationId, setMessages])

  const parsedError = parseChatError(error?.message)
  // Shows the same callout whether caught proactively (llmConfigured, refreshed on
  // every profile fetch) or reactively (a send just failed with that specific code)
  // -- one message either way, not a generic error followed by a redundant banner.
  const needsLlmSetup = !llmConfigured || parsedError?.code === "llm_not_configured"

  const composer = (
    <div className="space-y-2">
      {needsLlmSetup ? (
        <Alert>
          <KeyRound />
          <AlertTitle>No model configured</AlertTitle>
          <AlertDescription>
            {/* One paragraph, so the description's grid lays it out as a single flowing
                item rather than a row per inline element. */}
            <p>
              Add an API key in{" "}
              <Link to="/settings/models" className="underline">
                Settings
              </Link>{" "}
              to start chatting, or set the <code>LLM_MODEL</code> and{" "}
              <code>ANTHROPIC_API_KEY</code> (or your provider's key) environment variables.
            </p>
          </AlertDescription>
        </Alert>
      ) : (
        error && (
          <p className="text-sm text-destructive">
            {parsedError?.text ?? "Something went wrong reaching the server. Please try again."}
          </p>
        )
      )}
      {context && (
        <div className="flex">
          <span className="inline-flex items-center gap-1.5 rounded-lg border border-border bg-muted px-2.5 py-1 text-xs text-text-tertiary">
            <FileText className="h-3.5 w-3.5" />
            {context.name}
            {/* The 24px hit box keeps the bare 12px icon's footprint in the chip. */}
            <IconButton
              label="Remove context"
              size="icon-xs"
              onClick={() => onContext(null)}
              className="-m-1.5 hover:bg-transparent hover:text-foreground dark:hover:bg-transparent"
            >
              <X className="size-3" />
            </IconButton>
          </span>
        </div>
      )}
      {images.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {images.map(({ id, src }) => (
            <div key={id} className="relative">
              <img
                src={src}
                alt="attachment"
                className="h-16 w-16 rounded-lg border border-border object-cover"
              />
              <IconButton
                label="Remove image"
                size="icon-xs"
                onClick={() => setImages((prev) => prev.filter((img) => img.id !== id))}
                className="absolute -top-1.5 -right-1.5 size-4 rounded-full bg-foreground/85 p-0.5 text-background hover:bg-foreground/85 hover:text-background dark:hover:bg-foreground/85"
              >
                <X className="size-3" />
              </IconButton>
            </div>
          ))}
        </div>
      )}
      <PromptInputProvider>
        <PromptInput
          onSubmit={(m) => {
            if (m.text) send(m.text)
          }}
        >
          <PromptInputBody>
            <PromptInputTextarea placeholder="Ask anything about your data…" />
          </PromptInputBody>
          <PromptInputFooter>
            <PromptInputTools>
              <AttachContext onAttach={onContext} />
              <AttachImage
                disabled={images.length >= 4}
                onAttach={(src) => setImages((prev) => [...prev, { id: uuid(), src }].slice(0, 4))}
              />
              <MicButton />
              <UsageBadge conversationId={conversationId} settled={!busy} />
              <ProfileSelect profiles={profiles} value={profile} onChange={onProfile} />
            </PromptInputTools>
            <PromptInputSubmit
              status={status}
              variant="ghost"
              aria-label={busy ? "Stop" : "Submit"}
              // While a run is streaming the button shows a stop (square) icon; clicking
              // it aborts the request (chat.stop) instead of submitting an empty message.
              // The abort disconnects the stream, and the server records the turn as
              // stopped rather than leaving it orphaned.
              onClick={(e) => {
                if (busy) {
                  e.preventDefault()
                  stop()
                }
              }}
              className="rounded-full bg-primary text-primary-foreground hover:bg-accent-hover"
            />
          </PromptInputFooter>
        </PromptInput>
      </PromptInputProvider>
    </div>
  )

  if (messages.length === 0) {
    return (
      <div className="flex flex-1 items-center justify-center px-6">
        <div className="w-full max-w-2xl pb-24">
          <h1 className="mb-7 text-center text-display font-semibold tracking-tight">
            What do you want to analyze?
          </h1>
          {composer}
          <div className="mt-5 space-y-0.5">
            {EXAMPLES.map((e) => (
              <Button
                variant="ghost"
                key={e}
                onClick={() => send(e)}
                className="flex h-auto w-full justify-start gap-2 rounded-lg px-2 py-1.5 text-left font-normal whitespace-normal text-text-tertiary transition hover:bg-transparent hover:text-foreground has-[>svg]:px-2 dark:hover:bg-transparent"
              >
                <CornerDownRight className="size-3.5 shrink-0 opacity-60" />
                {e}
              </Button>
            ))}
          </div>
          {datasets.length > 0 && (
            <p className="mt-4 text-center text-xs text-text-tertiary">
              connected: {datasets.map((d) => d.name).join(", ")}
            </p>
          )}
        </div>
      </div>
    )
  }

  return (
    <>
      <JobsBar onCertifiedComplete={pullFollowup} />
      <Conversation className="flex-1">
        <ConversationContent className="mx-auto max-w-3xl space-y-6 px-6 py-8">
          {messages.map((m, i) => (
            <Turn
              key={m.id}
              message={m}
              busy={busy}
              onEdit={editAndResend}
              onRegenerate={
                i === messages.length - 1 && !busy
                  ? () =>
                      chat.regenerate({
                        body: {
                          profile,
                          context: context?.text ?? "",
                          conversationId,
                        },
                      })
                  : undefined
              }
            />
          ))}
        </ConversationContent>
      </Conversation>
      <div>
        <div className="mx-auto max-w-3xl px-6 py-3.5">{composer}</div>
      </div>
    </>
  )
}

export function Turn({
  message,
  busy,
  onEdit,
  onRegenerate,
}: {
  message: UIMessage
  busy: boolean
  onEdit?: (messageId: string, text: string) => void
  onRegenerate?: () => void
}) {
  if (message.role === "user") {
    const text = message.parts
      .filter((p) => p.type === "text")
      .map((p) => (p as { text: string }).text)
      .join("")
    return (
      <UserTurn
        text={text}
        busy={busy}
        onEdit={onEdit ? (t) => onEdit(message.id, t) : undefined}
      />
    )
  }

  const trace: TraceItem[] = []
  const textParts: string[] = []
  let result: ResultPayload | null = null
  let viz: VizDescriptor | null = null
  for (const [partIndex, part] of message.parts.entries()) {
    if (part.type === "text") textParts.push((part as { text: string }).text)
    else if (part.type === "data-result")
      // Normalize: the payload comes from the server (LLM-driven), and a background
      // follow-up reconstructs it from storage, so missing fields must not throw on render.
      result = normalizeResult((part as { data: Partial<ResultPayload> }).data)
    else if (part.type === "data-viz") viz = (part as { data: VizDescriptor }).data
    else if (part.type === "data-reasoning")
      trace.push({ id: partIndex, kind: "reasoning", text: (part as PartData).data.text ?? "" })
    else if (part.type === "data-reasoning-delta") {
      // Streamed reasoning arrives token by token; coalesce consecutive deltas into the
      // trailing reasoning item so the model's thinking grows live. A tool item between
      // steps breaks the run, so the next step's deltas start a fresh reasoning item.
      const delta = (part as PartData).data.text ?? ""
      const last = trace[trace.length - 1]
      if (last?.kind === "reasoning") last.text += delta
      else trace.push({ id: partIndex, kind: "reasoning", text: delta })
    } else if (part.type === "data-tool")
      trace.push({
        id: partIndex,
        kind: "tool",
        name: (part as PartData).data.name ?? "",
        args: (part as PartData).data.arguments ?? {},
      })
    else if (part.type === "data-tool-output") {
      const last = trace[trace.length - 1]
      if (last?.kind === "tool") last.output = (part as PartData).data.output ?? ""
    }
  }
  const text = textParts.join("")

  return (
    <div className="space-y-3">
      {trace.length > 0 && <Trace items={trace} running={busy} />}
      {busy && !text && (
        <div className="text-sm text-text-tertiary">
          <span className="animate-pulse">Analyzing…</span>
        </div>
      )}
      {result && <Verdict result={result} />}
      {result && result.assumptions.length > 0 && <Assumptions items={result.assumptions} />}
      {text && <Streamdown className="text-sm leading-7 text-foreground/90">{text}</Streamdown>}
      {viz && <VizView viz={viz} />}
      {result && result.checks.length > 0 && <Receipt result={result} />}
      {!busy && (text || result) && (
        <MessageActions
          messageId={message.id}
          initial={
            ((message.metadata as { feedback?: "up" | "down" | null } | undefined)?.feedback ??
              null) as "up" | "down" | null
          }
          onRegenerate={onRegenerate}
        />
      )}
    </div>
  )
}

function UserTurn({
  text,
  busy,
  onEdit,
}: {
  text: string
  busy: boolean
  onEdit?: (text: string) => void
}) {
  const [editing, setEditing] = useState(false)
  const [draft, setDraft] = useState(text)
  if (editing) {
    return (
      <div className="flex flex-col items-end gap-1.5">
        {/* Sized by `rows`, from the draft's line count, rather than by its content. */}
        <Textarea
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          rows={Math.min(6, draft.split("\n").length + 1)}
          className="field-sizing-fixed min-h-0 max-w-lg rounded-2xl p-3"
        />
        <div className="flex gap-1.5">
          <Button
            variant="ghost"
            size="xs"
            onClick={() => {
              setDraft(text)
              setEditing(false)
            }}
            className="rounded-lg px-2.5 font-normal text-text-tertiary hover:bg-muted hover:text-text-tertiary dark:hover:bg-muted"
          >
            Cancel
          </Button>
          <Button
            size="xs"
            disabled={!draft.trim() || busy}
            onClick={() => {
              setEditing(false)
              onEdit?.(draft)
            }}
            className="rounded-lg px-2.5 font-normal disabled:opacity-40"
          >
            Send
          </Button>
        </div>
      </div>
    )
  }
  return (
    <div className="group flex items-center justify-end gap-1.5">
      {onEdit && (
        <IconButton
          label="Edit message"
          size="icon-xs"
          onClick={() => {
            setDraft(text)
            setEditing(true)
          }}
          className={cn(ACTION, "opacity-0 group-hover:opacity-100 focus-visible:opacity-100")}
        >
          <Pencil className="size-3.5" />
        </IconButton>
      )}
      <Message from="user">
        <MessageContent>{text}</MessageContent>
      </Message>
    </div>
  )
}

function MessageActions({
  messageId,
  initial,
  onRegenerate,
}: {
  messageId: string
  initial: "up" | "down" | null
  onRegenerate?: () => void
}) {
  const [feedback, setFeedback] = useState<"up" | "down" | null>(initial)
  // Toggle a thumb off if it is already set, else switch to it; persist by message id.
  const rate = (value: "up" | "down") => {
    const next = feedback === value ? null : value
    setFeedback(next)
    void setMessageFeedback(messageId, next)
  }
  const iconClass = (active: boolean) => cn("size-3.5", active && "text-foreground")
  return (
    <div className="flex items-center gap-1 text-text-tertiary">
      <IconButton
        label="Good response"
        size="icon-xs"
        onClick={() => rate("up")}
        aria-pressed={feedback === "up"}
        className={ACTION}
      >
        <ThumbsUp className={iconClass(feedback === "up")} />
      </IconButton>
      <IconButton
        label="Bad response"
        size="icon-xs"
        onClick={() => rate("down")}
        aria-pressed={feedback === "down"}
        className={ACTION}
      >
        <ThumbsDown className={iconClass(feedback === "down")} />
      </IconButton>
      {onRegenerate && (
        <IconButton label="Regenerate" size="icon-xs" onClick={onRegenerate} className={ACTION}>
          <RotateCcw className="size-3.5" />
        </IconButton>
      )}
    </div>
  )
}

function Trace({ items, running }: { items: TraceItem[]; running: boolean }) {
  // Open while the model works so you watch it reason and act live; collapsible after.
  const [open, setOpen] = useState(true)
  const tools = items.filter((i) => i.kind === "tool").length
  return (
    <div className="rounded-xl border border-border bg-card/50">
      <Button
        variant="ghost"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className={cn(DISCLOSURE, "w-full rounded-none px-3 py-2 has-[>svg]:px-3")}
      >
        {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
        How it got here · {tools} step{tools === 1 ? "" : "s"}
        {running && <span className="ml-1 animate-pulse text-primary">· working…</span>}
      </Button>
      {open && (
        <div className="space-y-3 border-t border-border px-3 py-3">
          {items.map((item) =>
            item.kind === "reasoning" ? (
              <p key={item.id} className="text-sm leading-6 text-foreground/80">
                {item.text}
              </p>
            ) : (
              <ToolStep key={item.id} item={item} />
            ),
          )}
        </div>
      )}
    </div>
  )
}

function ToolStep({ item }: { item: Extract<TraceItem, { kind: "tool" }> }) {
  const [open, setOpen] = useState(false)
  return (
    <div className="text-sm">
      <Button
        variant="ghost"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex h-auto justify-start gap-2 p-0 text-left font-normal whitespace-normal transition hover:bg-transparent hover:text-foreground has-[>svg]:px-0 dark:hover:bg-transparent"
      >
        <span className="h-1.5 w-1.5 shrink-0 rounded-full bg-primary/60" />
        <span className="font-medium text-foreground/80">
          {STEP_LABELS[item.name] ?? item.name}
        </span>
        <span className="font-mono text-2xs text-text-tertiary">{item.name}</span>
        {open ? (
          <ChevronDown className="size-3 text-text-tertiary" />
        ) : (
          <ChevronRight className="size-3 text-text-tertiary" />
        )}
      </Button>
      {open && (
        <div className="ml-3.5 mt-1.5 space-y-1.5">
          <Code label="input" text={formatInput(item.name, item.args)} />
          {item.output && <Code label="output" text={item.output} />}
        </div>
      )}
    </div>
  )
}

function Code({ label, text }: { label: string; text: string }) {
  const clipped = text.length > 4000 ? `${text.slice(0, 4000)}\n… (truncated)` : text
  return (
    <div>
      <div className="text-3xs font-medium uppercase tracking-wide text-text-tertiary">{label}</div>
      <pre className="mt-0.5 max-h-64 overflow-auto rounded-lg border border-border bg-muted px-2.5 py-2 font-mono text-2xs leading-relaxed whitespace-pre-wrap">
        {clipped}
      </pre>
    </div>
  )
}

function formatInput(name: string, args: Record<string, unknown>): string {
  if (name === "run_code") return String(args.code ?? "")
  if (name === "derive") {
    const claim = args.claim ? `claim: ${JSON.stringify(args.claim)}\n` : ""
    return `${args.name ?? ""}\n${claim}\n${args.source ?? ""}`.trim()
  }
  if (typeof args.dataset === "string" && Object.keys(args).length === 1)
    return `dataset: ${args.dataset}`
  return JSON.stringify(args, null, 2)
}

function Verdict({ result }: { result: ResultPayload }) {
  const { label, chip, verified } = tone(result)
  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${chip}`}
    >
      {verified ? (
        <ShieldCheck className="h-3.5 w-3.5" />
      ) : (
        <span className="h-1.5 w-1.5 rounded-full bg-current opacity-70" />
      )}
      {label}
    </span>
  )
}

function tone(result: ResultPayload): { label: string; chip: string; verified: boolean } {
  if (result.verified)
    return { label: "Verified", chip: "bg-verified-tint text-verified", verified: true }
  switch (result.verdict) {
    case "inconclusive":
      return { label: "Inconclusive", chip: "bg-warning-tint text-warning", verified: false }
    case "grounded":
      return { label: "Computed", chip: "bg-info-tint text-info", verified: false }
    case "unsound":
    case "invalid":
      return { label: "Not sound", chip: "bg-danger-tint text-danger", verified: false }
    default:
      return { label: "Unverified", chip: "bg-muted text-text-tertiary", verified: false }
  }
}

function Receipt({ result }: { result: ResultPayload }) {
  const [open, setOpen] = useState(false)
  const claim = (result.spec?.claim as Record<string, unknown>) ?? {}
  return (
    <div>
      <Button
        variant="ghost"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className={cn(DISCLOSURE, "p-0 has-[>svg]:px-0")}
      >
        {open ? <ChevronDown className="size-3.5" /> : <ChevronRight className="size-3.5" />}
        Verification: {result.checks.length} check{result.checks.length === 1 ? "" : "s"}
      </Button>
      {open && (
        <div className="mt-2.5 space-y-3 border-l-2 border-verified/30 pl-4">
          <ul className="space-y-1.5">
            {result.checks.map((c) => {
              const mark = c.verdict === "sound" ? "✓" : c.verdict === "unsound" ? "✗" : "◦"
              const color =
                c.verdict === "sound"
                  ? "text-verified"
                  : c.verdict === "unsound"
                    ? "text-danger"
                    : "text-warning"
              return (
                <li key={c.name} className="text-xs leading-5 text-text-secondary">
                  <span className={`${color} font-semibold`}>{mark}</span>{" "}
                  <span className="font-medium text-foreground/80">{c.name}</span>: {c.detail}
                </li>
              )
            })}
          </ul>
          <div className="font-mono text-3xs leading-relaxed text-text-tertiary/70">
            <span className="text-foreground/50">claim</span> {JSON.stringify(claim)}
            {result.data_hash && (
              <>
                <br />
                <span className="text-foreground/50">data</span> {result.data_hash.slice(0, 16)}…
              </>
            )}
          </div>
          {result.verified && typeof result.spec?.derivation === "string" && (
            <a
              href={certificateUrl(result.spec.derivation as string)}
              download
              className="inline-block text-xs font-medium text-text-tertiary underline-offset-2 transition hover:text-foreground hover:underline"
            >
              Download certificate
            </a>
          )}
        </div>
      )}
    </div>
  )
}

function Assumptions({ items }: { items: string[] }) {
  // Declared, sourced premises the causal reading is conditioned on: shown up front, amber to
  // signal "true only under these", with the source the user can audit.
  return (
    <div className="rounded-lg border border-[var(--caution)]/25 bg-caution-tint px-3 py-2">
      <div className="text-2xs font-medium uppercase tracking-wide text-caution">
        Under stated assumptions
      </div>
      <ul className="mt-1 space-y-0.5 text-foreground/90">
        {items.map((a) => (
          <li key={a} className="flex gap-1.5 text-xs">
            <span className="text-caution">•</span>
            {a}
          </li>
        ))}
      </ul>
    </div>
  )
}

function AttachContext({ onAttach }: { onAttach: (c: Context) => void }) {
  const input = useRef<HTMLInputElement>(null)
  return (
    <>
      <input
        ref={input}
        type="file"
        accept=".md,.txt,.csv,.json,.yaml,.yml"
        className="hidden"
        onChange={async (e) => {
          const file = e.target.files?.[0]
          if (file) onAttach({ name: file.name, text: await file.text() })
          e.target.value = ""
        }}
      />
      <IconButton
        label="Attach a spec or data dictionary for context"
        onClick={() => input.current?.click()}
        className={TOOL}
      >
        <Plus className="size-4" />
      </IconButton>
    </>
  )
}

function AttachImage({
  onAttach,
  disabled,
}: {
  onAttach: (dataUrl: string) => void
  disabled: boolean
}) {
  const input = useRef<HTMLInputElement>(null)
  return (
    <>
      <input
        ref={input}
        type="file"
        accept="image/*"
        className="hidden"
        onChange={(e) => {
          const file = e.target.files?.[0]
          if (file) {
            const reader = new FileReader()
            reader.onload = () => onAttach(String(reader.result))
            reader.readAsDataURL(file)
          }
          e.target.value = ""
        }}
      />
      <IconButton
        label="Attach an image (e.g. a chart to read)"
        disabled={disabled}
        onClick={() => input.current?.click()}
        className={cn(TOOL, "disabled:opacity-40")}
      >
        <ImagePlus className="size-4" />
      </IconButton>
    </>
  )
}

function MicButton() {
  // Live dictation: as words stream in, replace the dictated tail after the text that
  // was in the box when we started, and inject through the PromptInput controller.
  const controller = usePromptInputController()
  const base = useRef("")
  const { state, error, toggle } = useSpeechRecognition(
    (transcript) =>
      controller.textInput.setInput(base.current ? `${base.current} ${transcript}` : transcript),
    () => {
      base.current = controller.textInput.value.trim()
    },
  )
  if (state === "unsupported") return null
  return (
    <span className="flex items-center gap-1.5">
      <IconButton
        label="Dictate: real-time"
        onClick={toggle}
        aria-pressed={state === "listening"}
        className={TOOL}
      >
        <Mic className={state === "listening" ? "size-4 animate-pulse text-danger" : "size-4"} />
      </IconButton>
      {error && <span className="text-2xs text-danger">{error}</span>}
    </span>
  )
}

function UsageBadge({ conversationId, settled }: { conversationId: string; settled: boolean }) {
  const [usage, setUsage] = useState<Usage | null>(null)
  // Refetch when the conversation changes and each time a turn settles, so the running
  // token/cost total the server accumulated is reflected without a reload.
  useEffect(() => {
    if (!conversationId) {
      setUsage(null)
      return
    }
    // Not while a turn is streaming: the totals are mid-flight and the fetch would be
    // superseded the moment it lands. Waiting for the turn to settle is both the
    // cheaper call and the accurate one.
    if (!settled) return
    void getConversationUsage(conversationId).then(setUsage)
  }, [conversationId, settled])
  if (!usage) return null
  const tokens = usage.prompt_tokens + usage.completion_tokens
  if (tokens === 0) return null
  const cost = usage.cost > 0 ? ` · $${usage.cost.toFixed(usage.cost < 0.01 ? 4 : 2)}` : ""
  return (
    <span
      className="px-1 text-2xs text-text-tertiary"
      title={`${usage.prompt_tokens.toLocaleString()} in · ${usage.completion_tokens.toLocaleString()} out`}
    >
      {tokens.toLocaleString()} tok{cost}
    </span>
  )
}

function ProfileSelect({
  profiles,
  value,
  onChange,
}: {
  profiles: LlmProfile[]
  value: string
  onChange: (name: string) => void
}) {
  // With no profiles registered the app runs on its default; nothing to switch between.
  if (profiles.length === 0) return null
  const selected = profiles.find((p) => p.name === value)
  return (
    <Select value={value} onValueChange={onChange}>
      <SelectTrigger
        size="sm"
        className="gap-1.5 border-0 bg-transparent px-2 text-xs text-text-tertiary shadow-none hover:text-foreground focus-visible:ring-0"
      >
        <SelectValue>
          <span className="flex items-center gap-1.5">
            {selected && (
              <ProviderGlyph
                model={selected.model}
                provider={selected.provider}
                providerLabel={selected.providerLabel}
              />
            )}
            {value || "Default model"}
          </span>
        </SelectValue>
      </SelectTrigger>
      <SelectContent align="start" side="top">
        {profiles.map((p) => (
          <SelectItem key={p.name} value={p.name}>
            <ProviderGlyph model={p.model} provider={p.provider} providerLabel={p.providerLabel} />
            {p.name}
          </SelectItem>
        ))}
      </SelectContent>
    </Select>
  )
}
