import {
  Activity,
  Bell,
  Boxes,
  Download,
  ExternalLink,
  FileCheck2,
  FlaskConical,
  Gauge,
  Layers,
  LayoutDashboard,
  LayoutGrid,
  MessagesSquare,
  Monitor,
  Moon,
  MoreHorizontal,
  Network,
  NotebookPen,
  Pencil,
  Plus,
  Search,
  Settings2,
  Sun,
  TerminalSquare,
  Trash2,
  Warehouse,
  Workflow,
} from "lucide-react"
import { type ComponentType, type ReactNode, useEffect, useRef, useState } from "react"
import { NavLink, useLocation } from "react-router-dom"
import { Logo } from "@/components/Logo"
import { Button } from "@/components/ui/button"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"
import { useInfiniteScroll } from "@/hooks/useInfiniteScroll"
import { type Theme, useTheme } from "@/hooks/useTheme"
import { type ConversationSummary, exportConversation } from "@/lib/chat"
import { getNotifications, onNotificationsChanged } from "@/lib/notifications"

export function Sidebar({
  conversations,
  hasNextPage,
  isFetchingNextPage,
  fetchNextPage,
  datasetCount,
  derivationCount,
  onNew,
  onDelete,
  onRename,
  onSearch,
}: {
  conversations: ConversationSummary[]
  hasNextPage: boolean
  isFetchingNextPage: boolean
  fetchNextPage: () => void
  datasetCount: number
  derivationCount: number
  onNew: () => void
  onDelete: (id: string) => void
  onRename: (id: string, title: string) => void
  onSearch: (q: string) => void
}) {
  // Load the next page when the history list scrolls near its bottom.
  const scrollRef = useInfiniteScroll({
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
  })
  const unreadNotifications = useUnreadNotifications()
  // The title filter, debounced so typing does not fire a request per keystroke. The
  // first run is skipped so it does not duplicate the hook's initial load.
  const [term, setTerm] = useState("")
  const firstSearch = useRef(true)
  useEffect(() => {
    if (firstSearch.current) {
      firstSearch.current = false
      return
    }
    const t = setTimeout(() => onSearch(term), 200)
    return () => clearTimeout(t)
  }, [term, onSearch])

  // The sidebar has two modes. Browse shows the platform navigation, Chat shows the
  // conversation history. The mode follows the route, a chat route (a conversation or the home
  // composer) is Chat, anything else is Browse, so the sidebar always reflects where you are;
  // the toggle lets you switch between the two.
  const location = useLocation()
  const isChatRoute = location.pathname === "/" || location.pathname.startsWith("/conversations/")
  const [mode, setMode] = useState<"browse" | "chat">(isChatRoute ? "chat" : "browse")
  useEffect(() => setMode(isChatRoute ? "chat" : "browse"), [isChatRoute])

  const segment = (active: boolean) =>
    `flex flex-1 items-center justify-center gap-1.5 rounded-md px-2 py-1.5 text-sm ` +
    `font-medium transition ${
      active
        ? "bg-card text-foreground shadow-sm"
        : "text-sidebar-foreground/60 hover:text-sidebar-foreground"
    }`

  return (
    <aside className="hidden w-[216px] shrink-0 flex-col overflow-hidden bg-transparent text-sidebar-foreground md:flex">
      <div className="flex items-center px-4 py-4">
        <Logo />
      </div>

      {/* Browse | Chat mode toggle (replaces the old New-analysis button). Browse shows
          the platform navigation; Chat shows the conversation history. */}
      <div className="px-3">
        <div className="flex gap-1 rounded-lg bg-sidebar-accent/60 p-1">
          <button
            type="button"
            onClick={() => setMode("browse")}
            className={segment(mode === "browse")}
            aria-pressed={mode === "browse"}
          >
            <LayoutGrid className="h-4 w-4" /> Browse
          </button>
          <button
            type="button"
            onClick={() => {
              setMode("chat")
              // Entering Chat from a browse page opens the composer; if already in a
              // conversation, stay in it (a new chat is the ＋ below).
              if (!isChatRoute) onNew()
            }}
            className={segment(mode === "chat")}
            aria-pressed={mode === "chat"}
          >
            <MessagesSquare className={`h-4 w-4 ${mode === "chat" ? "text-primary" : ""}`} /> Chat
          </button>
        </div>
      </div>

      {mode === "browse" ? (
        <nav className="mt-4 flex-1 space-y-4 overflow-y-auto px-3 pb-4">
          <Section label="Data">
            <Item to="/warehouse" icon={Warehouse} label="Data warehouse" count={datasetCount} />
            <Item to="/explore" icon={TerminalSquare} label="Explore" />
          </Section>
          <Section label="Analysis">
            <Item to="/derivations" icon={FileCheck2} label="Derivations" count={derivationCount} />
            <Item to="/notebooks" icon={NotebookPen} label="Notebooks" />
          </Section>
          <Section label="Machine learning">
            <Item to="/models" icon={Boxes} label="Models" />
            <Item to="/features" icon={Layers} label="Feature store" />
            {/* The real MLflow UI, served alongside the app; a separate SPA, so a plain
                anchor into a new tab rather than a router link. */}
            <a
              href="/mlflow/"
              target="_blank"
              rel="noreferrer"
              className="flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-sidebar-foreground/70 transition hover:bg-sidebar-accent"
            >
              <FlaskConical className="h-4 w-4 shrink-0" />
              <span className="flex-1">MLflow</span>
              <ExternalLink className="h-3.5 w-3.5 shrink-0 text-sidebar-foreground/40" />
            </a>
          </Section>
          <Section label="Analytics">
            <Item to="/metrics" icon={Gauge} label="Metrics" />
            <Item to="/dashboards" icon={LayoutDashboard} label="Dashboards" />
          </Section>
          <Section label="Operations">
            <Item to="/orchestration" icon={Workflow} label="Orchestration" />
            <Item to="/monitors" icon={Activity} label="Monitors" />
            <Item to="/catalog" icon={Network} label="Catalog & lineage" />
          </Section>
        </nav>
      ) : (
        // Past conversations, newest first: each is a link to its own route, active-
        // highlighted when open, so a past analysis can be reopened and resumed.
        <div className="mt-3 flex min-h-0 flex-1 flex-col px-3">
          <div className="flex items-center justify-between px-2.5 pb-1">
            <p className="text-[11px] font-medium uppercase tracking-wide text-sidebar-foreground/45">
              Chats
            </p>
            <button
              type="button"
              onClick={onNew}
              title="New analysis"
              aria-label="New analysis"
              className="flex h-6 w-6 items-center justify-center rounded-md text-sidebar-foreground/60 transition hover:bg-sidebar-accent hover:text-sidebar-foreground"
            >
              <Plus className="h-4 w-4" />
            </button>
          </div>
          <div className="relative mb-1 px-0.5">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-sidebar-foreground/40" />
            <input
              value={term}
              onChange={(e) => setTerm(e.target.value)}
              placeholder="Search analyses"
              aria-label="Search analyses"
              className="w-full rounded-lg border border-sidebar-border bg-card py-1.5 pl-7 pr-2 text-sm outline-none placeholder:text-sidebar-foreground/40 focus:ring-1 focus:ring-ring"
            />
          </div>
          <div ref={scrollRef} className="min-h-0 flex-1 space-y-0.5 overflow-y-auto">
            {conversations.length === 0 ? (
              <p className="px-2.5 py-1.5 text-sm text-sidebar-foreground/50">
                {term ? "No matches" : "No past analyses yet"}
              </p>
            ) : (
              conversations.map((c) => (
                <ConversationLink
                  key={c.id}
                  conversation={c}
                  onDelete={onDelete}
                  onRename={onRename}
                />
              ))
            )}
            {isFetchingNextPage && (
              <p className="px-2.5 py-1.5 text-[11px] text-sidebar-foreground/40">Loading more…</p>
            )}
          </div>
        </div>
      )}

      {/* Inbox, then Settings and the theme toggle, pinned at the bottom in both
          modes. Trash lives under Settings, beside the other views of your work. */}
      <div className="mt-auto">
        <div className="space-y-0.5 border-t border-sidebar-border px-3 pt-2">
          <Item to="/notifications" icon={Bell} label="Inbox" count={unreadNotifications} />
        </div>
        <div className="flex items-center gap-1 px-3 py-2">
          <div className="min-w-0 flex-1">
            <Item to="/settings" icon={Settings2} label="Settings" />
          </div>
          <ThemeToggle />
        </div>
      </div>
    </aside>
  )
}

// The Inbox pill count: a cheap 5s poll, refreshed instantly by the change event
// the inbox page fires after marking rows read.
function useUnreadNotifications(): number {
  const [unread, setUnread] = useState(0)
  useEffect(() => {
    let alive = true
    // A slow interval poll must not overwrite a fresher change-event poll.
    let seq = 0
    const poll = async () => {
      const mySeq = ++seq
      try {
        const list = await getNotifications({ limit: 1 })
        if (alive && mySeq === seq) setUnread(list.unread)
      } catch {
        // A failed poll keeps the last known count; the next tick retries.
      }
    }
    void poll()
    const id = setInterval(() => void poll(), 5000)
    const off = onNotificationsChanged(() => void poll())
    return () => {
      alive = false
      clearInterval(id)
      off()
    }
  }, [])
  return unread
}

const _THEME_ICON = { system: Monitor, light: Sun, dark: Moon } as const
const _THEME_NEXT: Record<Theme, Theme> = {
  system: "light",
  light: "dark",
  dark: "system",
}
const _THEME_LABEL: Record<Theme, string> = {
  system: "System theme",
  light: "Light theme",
  dark: "Dark theme",
}

function ThemeToggle() {
  const { theme, setTheme } = useTheme()
  const Icon = _THEME_ICON[theme]
  const next = _THEME_NEXT[theme]
  return (
    <button
      type="button"
      onClick={() => setTheme(next)}
      title={`${_THEME_LABEL[theme]}: switch to ${_THEME_LABEL[next].toLowerCase()}`}
      aria-label={`Theme: ${_THEME_LABEL[theme].toLowerCase()}`}
      className="flex h-8 w-8 shrink-0 items-center justify-center rounded-md text-sidebar-foreground/60 transition hover:bg-sidebar-accent hover:text-sidebar-foreground"
    >
      <Icon className="h-4 w-4" />
    </button>
  )
}

function ConversationLink({
  conversation,
  onDelete,
  onRename,
}: {
  conversation: ConversationSummary
  onDelete: (id: string) => void
  onRename: (id: string, title: string) => void
}) {
  const [confirming, setConfirming] = useState(false)
  const [renaming, setRenaming] = useState(false)
  return (
    <div className="group relative flex items-center">
      <NavLink
        to={`/conversations/${conversation.id}`}
        title={conversation.title}
        className={({ isActive }) =>
          `flex min-w-0 flex-1 flex-col gap-0.5 rounded-lg py-1.5 pl-2.5 pr-8 transition ${
            isActive
              ? "bg-sidebar-accent text-sidebar-accent-foreground"
              : "text-sidebar-foreground/70 hover:bg-sidebar-accent"
          }`
        }
      >
        <span className="truncate text-sm">{conversation.title || "Untitled analysis"}</span>
        <span className="text-[11px] text-muted-foreground">
          {relativeTime(conversation.updated_at)}
        </span>
      </NavLink>
      {/* A hover-revealed options menu, kept visible while open so the click that opens
          the confirm dialog does not make the trigger vanish under the cursor. */}
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <Button
            variant="ghost"
            size="icon-xs"
            aria-label="Conversation options"
            className="absolute right-1.5 text-sidebar-foreground/50 opacity-0 transition hover:text-sidebar-foreground group-hover:opacity-100 data-[state=open]:opacity-100"
          >
            <MoreHorizontal />
          </Button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="end">
          <DropdownMenuItem onSelect={() => setRenaming(true)}>
            <Pencil /> Rename
          </DropdownMenuItem>
          <DropdownMenuItem onSelect={() => exportConversation(conversation.id)}>
            <Download /> Export
          </DropdownMenuItem>
          <DropdownMenuItem
            className="text-destructive focus:text-destructive"
            onSelect={() => setConfirming(true)}
          >
            <Trash2 /> Delete
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <ConfirmDeleteDialog
        open={confirming}
        onOpenChange={setConfirming}
        title={conversation.title}
        onConfirm={() => {
          setConfirming(false)
          onDelete(conversation.id)
        }}
      />
      <RenameConversationDialog
        open={renaming}
        onOpenChange={setRenaming}
        currentTitle={conversation.title}
        onRename={(title) => {
          setRenaming(false)
          onRename(conversation.id, title)
        }}
      />
    </div>
  )
}

function RenameConversationDialog({
  open,
  onOpenChange,
  currentTitle,
  onRename,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  currentTitle: string
  onRename: (title: string) => void
}) {
  const [value, setValue] = useState(currentTitle)
  // Reset the field to the live title each time the dialog opens, so it never shows a
  // stale value from a previous edit that was cancelled.
  useEffect(() => {
    if (open) setValue(currentTitle)
  }, [open, currentTitle])
  const save = () => {
    const trimmed = value.trim()
    if (trimmed) onRename(trimmed)
  }
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>Rename analysis</DialogTitle>
        </DialogHeader>
        <input
          autoFocus
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault()
              save()
            }
          }}
          aria-label="Analysis title"
          className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm outline-none focus:ring-1 focus:ring-ring"
        />
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button onClick={save} disabled={!value.trim()}>
            Save
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

function ConfirmDeleteDialog({
  open,
  onOpenChange,
  title,
  onConfirm,
}: {
  open: boolean
  onOpenChange: (open: boolean) => void
  title: string
  onConfirm: () => void
}) {
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="sm:max-w-sm">
        <DialogHeader>
          <DialogTitle>Delete analysis?</DialogTitle>
          <DialogDescription>
            This permanently deletes{" "}
            {title ? (
              <span className="text-foreground">&ldquo;{title}&rdquo;</span>
            ) : (
              "this analysis"
            )}{" "}
            and its messages. This cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <DialogFooter>
          <Button variant="outline" onClick={() => onOpenChange(false)}>
            Cancel
          </Button>
          <Button variant="destructive" onClick={onConfirm}>
            Delete
          </Button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  )
}

// A compact, dependency-free relative time for the conversation cards (today shows the
// clock, this week the weekday, older the date), so the list reads like a chat history.
function relativeTime(iso: string): string {
  const then = new Date(iso)
  if (Number.isNaN(then.getTime())) return ""
  const seconds = (Date.now() - then.getTime()) / 1000
  if (seconds < 60) return "just now"
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`
  if (seconds < 7 * 86400) return then.toLocaleDateString(undefined, { weekday: "short" })
  return then.toLocaleDateString(undefined, { month: "short", day: "numeric" })
}

// A labelled group of nav items, so the ~dozen browse destinations read as a few small
// sections rather than one long list (matching the grouped Settings nav).
function Section({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div className="space-y-0.5">
      <p className="px-2.5 pb-1 text-[11px] font-medium uppercase tracking-wide text-sidebar-foreground/45">
        {label}
      </p>
      {children}
    </div>
  )
}

function Item({
  to,
  icon: Icon,
  label,
  count,
  end,
}: {
  to: string
  icon: ComponentType<{ className?: string }>
  label: string
  count?: number
  end?: boolean
}) {
  return (
    <NavLink
      to={to}
      end={end}
      className={({ isActive }) =>
        `flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm transition ${
          isActive
            ? "bg-sidebar-accent font-medium text-sidebar-accent-foreground"
            : "text-sidebar-foreground/70 hover:bg-sidebar-accent"
        }`
      }
    >
      <Icon className="h-4 w-4 shrink-0" />
      <span className="flex-1">{label}</span>
      {count !== undefined && count > 0 && (
        <span className="rounded-full bg-foreground/10 px-1.5 text-[11px] text-muted-foreground">
          {count}
        </span>
      )}
    </NavLink>
  )
}
