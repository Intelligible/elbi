import { Chat, useChat } from "@ai-sdk/react"
import { DefaultChatTransport, type UIMessage } from "ai"
import { useCallback, useEffect, useRef, useState } from "react"
import { Navigate, Route, Routes, useLocation, useNavigate, useParams } from "react-router-dom"
import { AssistantContextProvider } from "@/components/AssistantContext"
import { AssistantMark } from "@/components/AssistantMark"
import { AssistantPanel, type PageContext } from "@/components/AssistantPanel"
import { InsecureContextBanner } from "@/components/InsecureContextBanner"
import { Sidebar } from "@/components/Sidebar"
import {
  ResizableHandle,
  ResizablePanel,
  ResizablePanelGroup,
  useResizableLayout,
} from "@/components/ui/resizable"
import { TooltipProvider } from "@/components/ui/tooltip"
import { usePaginatedConversations } from "@/hooks/usePaginatedConversations"
import {
  type Dataset,
  type DerivationSummary,
  getConversationMessages,
  getDatasets,
  getDerivations,
  getLlmProfiles,
  type LlmProfile,
} from "@/lib/chat"
import { storedToUIMessages } from "@/lib/messages"
import { uuid } from "@/lib/utils"
import { CatalogPage } from "@/views/CatalogPage"
import { ChatView } from "@/views/ChatView"
import { DashboardPage } from "@/views/DashboardPage"
import { DashboardsPage } from "@/views/DashboardsPage"
import { DerivationDetailPage } from "@/views/DerivationDetailPage"
import { DerivationsPage } from "@/views/DerivationsPage"
import { ExplorePage } from "@/views/ExplorePage"
import { FeatureStorePage } from "@/views/FeatureStorePage"
import { FeatureViewDetailPage } from "@/views/FeatureViewDetailPage"
import { MetricsPage } from "@/views/MetricsPage"
import { ModelDetailPage } from "@/views/ModelDetailPage"
import { ModelsPage } from "@/views/ModelsPage"
import { MonitorsPage } from "@/views/MonitorsPage"
import { NotebookPage } from "@/views/NotebookPage"
import { NotebooksPage } from "@/views/NotebooksPage"
import { NotificationsPage } from "@/views/NotificationsPage"
import { OrchestrationPage } from "@/views/OrchestrationPage"
import { SettingsPage } from "@/views/SettingsPage"
import { TrainModelPage } from "@/views/TrainModelPage"
import { NewSourcePage, SourceDetailPage, WarehousePage } from "@/views/WarehousePage"

type Context = { name: string; text: string } | null

// What the assistant panel is told about the page the user is on. The notebook/dashboard
// ids come straight from the route, and the model reaches the detail through its own
// tools (read_notebook etc.), so the context stays a short, current pointer.
function pageContextFor(pathname: string): PageContext {
  const nb = pathname.match(/^\/notebooks\/([^/]+)/)
  if (nb)
    return {
      label: "this notebook",
      text:
        `The user is viewing the notebook with id "${nb[1]}". When they ask you to ` +
        `add, change, visualize, or run something, act on THIS notebook: use ` +
        `read_notebook (id "${nb[1]}") to see its cells, write_notebook to add or edit ` +
        `cells, and run_notebook to execute it. Prefer editing this notebook over ` +
        `creating a new one.`,
    }
  const dash = pathname.match(/^\/dashboards\/([^/]+)/)
  if (dash)
    return {
      label: "this dashboard",
      text:
        `The user is viewing the dashboard with id "${dash[1]}". Use read_dashboard, ` +
        `write_dashboard, and publish_dashboard to help with it.`,
    }
  const surfaces: [RegExp, string, string][] = [
    [/^\/metrics/, "Metrics", "the Metrics page: use metric_list/metric_define/metric_query"],
    [/^\/monitors/, "Monitors", "the Monitors page: use monitor_list/monitor_create"],
    [/^\/models/, "Models", "the Models page: help train, evaluate, and promote models"],
    [
      /^\/derivations/,
      "Derivations",
      "the Derivations page: help author/understand certified derivations",
    ],
    [/^\/features/, "Feature store", "the Feature store page"],
    [
      /^\/(warehouse|explore)/,
      "the warehouse",
      "the data warehouse: help explore and query the bound data",
    ],
    [/^\/orchestration/, "Orchestration", "the Orchestration page (assets, runs, schedules)"],
    [/^\/dashboards/, "Dashboards", "the Dashboards page"],
    [/^\/notebooks/, "Notebooks", "the Notebooks page"],
  ]
  for (const [re, label, desc] of surfaces) {
    if (re.test(pathname)) return { label, text: `The user is on ${desc}.` }
  }
  return null
}

// A chat is only ever a location: "/" mints the id its blank chat will carry and redirects
// onto it, so no send ever has to move the URL out from under a chat that is already running.
function NewConversation() {
  return <Navigate to={`/conversations/${uuid()}`} replace />
}

// Keep old per-source deep links (/sources/:id) working after the move to /warehouse.
function RedirectSourceDetail() {
  const { id = "" } = useParams()
  return <Navigate to={`/warehouse/sources/${id}`} replace />
}

export default function App() {
  const [datasets, setDatasets] = useState<Dataset[]>([])
  const [derivations, setDerivations] = useState<DerivationSummary[]>([])
  const [profiles, setProfiles] = useState<LlmProfile[]>([])
  const [profile, setProfile] = useState("")
  // Defaults true so there's no flash of "not configured" before the first fetch
  // resolves; getLlmProfiles' own fallback keeps that true on a fetch failure too.
  const [llmConfigured, setLlmConfigured] = useState(true)
  const [context, setContext] = useState<Context>(null)
  const [showAssistant, setShowAssistant] = useState(false)
  // What the active page registered about itself (its live state). Falls back to a
  // route-derived pointer for pages that don't register their own context.
  const [sceneContext, setSceneContext] = useState<PageContext>(null)
  const navigate = useNavigate()
  const location = useLocation()
  // The assistant panel is for the non-chat pages; the chat routes already are the chat.
  const onChatRoute = location.pathname === "/" || location.pathname.startsWith("/conversations/")
  const pageContext = sceneContext ?? pageContextFor(location.pathname)
  // Remembers where the reader left the assistant split. Both ids are named so closing
  // and reopening the panel restores the width rather than resetting it.
  const assistantLayout = useResizableLayout({
    id: "assistant-split",
    panelIds: ["main", "assistant"],
  })

  // A page-mutating tool ran in the assistant (e.g. it edited the open notebook); tell the
  // current page to refetch so the change shows up live. Pages opt in with a listener.
  const onAssistantActed = useCallback(() => {
    window.dispatchEvent(new CustomEvent("elbi:refetch"))
  }, [])

  // ⌘I / Ctrl-I toggles the assistant, the way ⌘K opens a command palette elsewhere.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "i") {
        e.preventDefault()
        setShowAssistant((open) => !open)
      }
    }
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [])
  // The conversation open in the URL, or "" on the home screen ("/"), which carries no
  // id until a chat actually starts.
  const currentId = location.pathname.match(/^\/conversations\/([^/]+)/)?.[1] ?? ""

  // Cursor-paginated conversation history (infinite scroll in the sidebar). Reloaded from
  // the first page after a finished turn, which may create or bump one to the top.
  const {
    conversations,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    refresh,
    search,
    remove,
    rename,
    addPending,
  } = usePaginatedConversations()

  // Ids whose messages have been hydrated from the server this session. Each conversation
  // keeps its own chat instance (see `chats` below), so once one is loaded it must never be
  // re-fetched over. Held here (not in the chat view) so it survives navigating to the
  // Derivations/Data pages and back.
  const loaded = useRef<Set<string>>(new Set())

  // The conversation the chat pane owns. Read from the URL, and held in a ref so it stays
  // put on the Derivations/Data pages. Only ever assigned from the URL, so a render at a stale
  // location cannot move it.
  const chatId = useRef("")
  if (!chatId.current) chatId.current = uuid()
  if (currentId) chatId.current = currentId

  // "New analysis" and deleting the open conversation both return to "/", which mints a
  // fresh id and redirects onto it.
  const handleNew = () => navigate("/")
  const handleDelete = (id: string) => {
    void remove(id)
    if (id === currentId) navigate("/")
  }

  // One Chat per conversation, kept for the session, so messages/status/the in-flight stream
  // belong to one conversation and navigating to a different (old) chat mid-answer can
  // neither see nor abort this one's request.
  const chats = useRef(new Map<string, Chat<UIMessage>>())

  // Passing `chat` makes useChat skip its own callback refresh, so the finish handler is baked
  // in at construction and reads the current one through this ref rather than going stale. A
  // sound answer authors a derivation, so refresh both the derivation list and the
  // conversation list when a run finishes.
  const onFinish = useRef(() => {})
  onFinish.current = () => {
    getDerivations().then(setDerivations)
    getLlmProfiles().then((p) => {
      setProfiles(p.profiles)
      setLlmConfigured(p.configured)
    })
    refresh()
  }

  const chatFor = useCallback((id: string) => {
    const existing = chats.current.get(id)
    if (existing) return existing
    const created = new Chat<UIMessage>({
      id,
      transport: new DefaultChatTransport({ api: "/api/chat" }),
      onFinish: () => onFinish.current(),
    })
    chats.current.set(id, created)
    return created
  }, [])
  const chat = useChat({ chat: chatFor(chatId.current) })

  // Load a conversation's stored messages the first time it is opened, so a past id replays
  // its questions and answers (verification receipts and all).
  useEffect(() => {
    if (!currentId || loaded.current.has(currentId)) return
    loaded.current.add(currentId)
    const target = chatFor(currentId)
    void getConversationMessages(currentId).then((stored) => {
      target.messages = storedToUIMessages(stored)
    })
  }, [currentId, chatFor])

  // The sidebar's list only reloads once a turn finishes (onFinish -> refresh(), below), so
  // a conversation just started would otherwise be invisible in it -- and unclickable to
  // return to -- for the whole first turn. "submitted" is the instant a message is sent,
  // before any streaming starts; addPending no-ops once the real row has arrived.
  useEffect(() => {
    if (chat.status === "submitted") addPending(currentId)
  }, [chat.status, currentId, addPending])

  useEffect(() => {
    getDatasets().then(setDatasets)
    getDerivations().then(setDerivations)
    getLlmProfiles().then((p) => {
      setProfiles(p.profiles)
      setProfile(p.default)
      setLlmConfigured(p.configured)
    })
  }, [])

  // Profiles can be added/edited/removed on the settings page; refetch the list when the
  // user navigates back off it, so the composer's model switcher (and the "no model
  // configured yet" banner) are never stale after adding a key there.
  const path = location.pathname
  useEffect(() => {
    if (!path.startsWith("/settings")) {
      getLlmProfiles().then((p) => {
        setProfiles(p.profiles)
        setLlmConfigured(p.configured)
      })
    }
  }, [path])

  const chatPane = (
    <ChatView
      chat={chat}
      conversationId={currentId}
      datasets={datasets}
      profiles={profiles}
      profile={profile}
      onProfile={setProfile}
      llmConfigured={llmConfigured}
      context={context}
      onContext={setContext}
    />
  )

  return (
    <TooltipProvider delayDuration={200}>
      {/* App shell (MLflow-style): the content sits on a white `card` panel that floats
          above a grey `panel` canvas: the colour contrast plus a soft shadow make the
          main area read as clearly raised, the way MLflow lifts its content off the page. */}
      <div className="flex h-screen flex-col overflow-hidden bg-panel">
        <InsecureContextBanner />
        <div className="flex min-h-0 flex-1 overflow-hidden">
          <Sidebar
            conversations={conversations}
            hasNextPage={hasNextPage}
            isFetchingNextPage={isFetchingNextPage}
            fetchNextPage={fetchNextPage}
            datasetCount={datasets.length}
            derivationCount={derivations.length}
            onNew={handleNew}
            onDelete={handleDelete}
            onRename={rename}
            onSearch={search}
          />
          <AssistantContextProvider set={setSceneContext}>
            <ResizablePanelGroup {...assistantLayout} className="w-auto flex-1">
              <ResizablePanel id="main" minSize="40" className="flex min-w-0">
                <main className="m-1.5 flex min-w-0 flex-1 flex-col overflow-hidden rounded-lg border border-border bg-card shadow-panel">
                  <Routes>
                    {/* "/" mints an id and redirects onto /conversations/:id, so every chat
                is a shareable location from the first keystroke and browser back/forward
                works. */}
                    <Route path="/" element={<NewConversation />} />
                    <Route path="/conversations/:conversationId" element={chatPane} />
                    <Route path="/derivations" element={<DerivationsPage />} />
                    <Route path="/derivations/:name" element={<DerivationDetailPage />} />
                    <Route path="/notebooks" element={<NotebooksPage />} />
                    <Route path="/notebooks/:name" element={<NotebookPage />} />
                    <Route path="/models" element={<ModelsPage />} />
                    <Route path="/models/train" element={<TrainModelPage />} />
                    <Route path="/models/:name" element={<ModelDetailPage />} />
                    <Route path="/dashboards" element={<DashboardsPage />} />
                    <Route path="/dashboards/:id" element={<DashboardPage />} />
                    <Route path="/features" element={<FeatureStorePage />} />
                    <Route path="/features/:name" element={<FeatureViewDetailPage />} />
                    <Route path="/metrics" element={<MetricsPage />} />
                    <Route path="/monitors" element={<MonitorsPage />} />
                    <Route path="/notifications" element={<NotificationsPage />} />
                    {/* Trash moved into Settings; keep old links alive. */}
                    <Route path="/trash" element={<Navigate to="/settings/trash" replace />} />
                    <Route path="/catalog" element={<CatalogPage />} />
                    <Route path="/orchestration" element={<OrchestrationPage />} />
                    <Route path="/pipelines" element={<Navigate to="/orchestration" replace />} />
                    <Route path="/warehouse" element={<WarehousePage />} />
                    <Route path="/warehouse/new-source" element={<NewSourcePage />} />
                    <Route path="/warehouse/sources/:id" element={<SourceDetailPage />} />
                    <Route path="/explore" element={<ExplorePage />} />
                    {/* Datasets and Sources unified into the Data warehouse; keep old links alive. */}
                    <Route path="/datasets" element={<Navigate to="/warehouse" replace />} />
                    <Route path="/sources" element={<Navigate to="/warehouse" replace />} />
                    <Route
                      path="/sources/new-source"
                      element={<Navigate to="/warehouse/new-source" replace />}
                    />
                    <Route path="/sources/:id" element={<RedirectSourceDetail />} />
                    <Route path="/settings" element={<SettingsPage />} />
                    <Route path="/settings/:section" element={<SettingsPage />} />
                  </Routes>
                </main>
              </ResizablePanel>
              {!onChatRoute && showAssistant ? (
                <>
                  <ResizableHandle withHandle className="my-1.5" />
                  <ResizablePanel
                    id="assistant"
                    defaultSize="26"
                    minSize="18"
                    maxSize="45"
                    className="m-1.5 ml-0 flex"
                  >
                    <AssistantPanel
                      pageContext={pageContext}
                      profile={profile}
                      onClose={() => setShowAssistant(false)}
                      onActed={onAssistantActed}
                    />
                  </ResizablePanel>
                </>
              ) : null}
            </ResizablePanelGroup>
            {/* A persistent right-edge rail, always visible on non-chat pages so the
            assistant is never lost in a corner. */}
            {!onChatRoute ? (
              <div className="m-1.5 ml-0 flex w-12 shrink-0 flex-col items-center gap-1 rounded-lg border border-border bg-card shadow-panel py-3">
                <button
                  type="button"
                  onClick={() => setShowAssistant((open) => !open)}
                  title="Assistant (⌘I)"
                  className={`flex size-9 flex-col items-center justify-center rounded-md transition ${
                    showAssistant
                      ? "bg-primary text-primary-foreground"
                      : "text-muted-foreground hover:bg-muted hover:text-foreground"
                  }`}
                >
                  <AssistantMark className="size-5" />
                </button>
                <span className="text-[9px] font-medium uppercase tracking-wide text-muted-foreground">
                  AI
                </span>
              </div>
            ) : null}
          </AssistantContextProvider>
        </div>
      </div>
    </TooltipProvider>
  )
}
