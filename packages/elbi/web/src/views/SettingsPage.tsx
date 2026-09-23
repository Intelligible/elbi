import {
  Bell,
  Boxes,
  Check,
  Copy,
  Cpu,
  Database,
  Gauge,
  Info,
  Lock,
  Plug,
  Radar,
  ScrollText,
  Settings2,
  Trash,
  Trash2,
  Wallet,
  Webhook,
} from "lucide-react"
import { useCallback, useEffect, useState } from "react"
import { useNavigate, useParams } from "react-router-dom"

import { FormControl, FormField } from "@/components/app/FormField"
import { IconButton } from "@/components/app/IconButton"
import { LlmProfilesManager } from "@/components/LlmSettings"
import { Scene, SceneHeader } from "@/components/Scene"
import { ComputeUsageSection } from "@/components/settings/ComputeUsageSection"
import { SectionHeader } from "@/components/settings/SectionHeader"
import { TrashSection } from "@/components/settings/TrashSection"
import { Button } from "@/components/ui/button"
import { Checkbox } from "@/components/ui/checkbox"
import { FeedbackProvider, useFeedback } from "@/components/ui/feedback"
import { Input } from "@/components/ui/input"
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select"
import { Skeleton } from "@/components/ui/skeleton"
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table"
import { Textarea } from "@/components/ui/textarea"
import { VerdictBadge } from "@/components/VerdictBadge"
import {
  type AppVersion,
  type AuditEvent,
  type Budget,
  createDataSource,
  type DataSource,
  type DataSourceInput,
  deleteDataSource,
  deleteSecret,
  getAppVersion,
  getAudit,
  getBudget,
  getDataSources,
  getMlflowSettings,
  getSecrets,
  getWebhookSettings,
  type MlflowSettings,
  type SecretMeta,
  saveSecret,
  setBudget,
  setMlflowTrackingUri,
  setWebhookSettings,
  testDataSource,
  type WebhookSettings,
} from "@/lib/chat"
import { type BaseEnvironment, getBaseEnvironments, setBaseEnvironments } from "@/lib/notebooks"
import {
  EVENT_TYPE_LABELS,
  getNotificationPreferences,
  type NotificationPref,
  setNotificationPreferences,
} from "@/lib/notifications"
import { EMPTY, uuid } from "@/lib/utils"

type SectionId =
  | "trash"
  | "models"
  | "connections"
  | "budget"
  | "compute"
  | "tracking"
  | "webhooks"
  | "notifications"
  | "notebooks"
  | "secrets"
  | "audit"
  | "about"

// Settings are grouped under headers in the nav (Project, Governance …), so a setting is
// found by the scope it applies to. `group` names the header each section sits under;
// the render walks GROUP_ORDER so the order is stable.
const GROUP_ORDER = ["Project", "Governance", "Security", "About"] as const

const SECTIONS: {
  id: SectionId
  label: string
  group: (typeof GROUP_ORDER)[number]
  icon: typeof Plug
}[] = [
  { id: "models", label: "Models", group: "Project", icon: Cpu },
  { id: "connections", label: "Connections", group: "Project", icon: Plug },
  { id: "tracking", label: "Run tracking", group: "Project", icon: Radar },
  { id: "notebooks", label: "Notebook environments", group: "Project", icon: Boxes },
  { id: "trash", label: "Trash", group: "Project", icon: Trash },
  { id: "budget", label: "Budget", group: "Governance", icon: Wallet },
  { id: "compute", label: "Compute usage", group: "Governance", icon: Gauge },
  { id: "webhooks", label: "Webhooks", group: "Governance", icon: Webhook },
  { id: "notifications", label: "Notifications", group: "Governance", icon: Bell },
  { id: "audit", label: "Audit log", group: "Governance", icon: ScrollText },
  { id: "secrets", label: "Secrets", group: "Security", icon: Lock },
  { id: "about", label: "About", group: "About", icon: Info },
]

export function SettingsPage() {
  return (
    <FeedbackProvider>
      <SettingsBody />
    </FeedbackProvider>
  )
}

function SettingsBody() {
  const navigate = useNavigate()
  const { section: param } = useParams<{ section: string }>()
  const sections = SECTIONS
  // The active section lives in the URL (/settings/<section>), so a refresh, a bookmark,
  // or the browser back button preserve it. An unknown/missing segment falls back to the
  // first section.
  const section: SectionId = SECTIONS.some((s) => s.id === param) ? (param as SectionId) : "models"
  return (
    <Scene>
      <SceneHeader
        icon={<Settings2 className="size-5" />}
        title="Settings"
        description="Models, connections, governance, and platform configuration."
      />
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="flex items-start gap-8 px-5 py-5">
          <nav className="sticky top-0 w-52 shrink-0 self-start rounded-lg border border-border bg-card p-2">
            {GROUP_ORDER.map((group) => {
              const items = sections.filter((s) => s.group === group)
              if (items.length === 0) return null
              return (
                <div key={group} className="mb-1 last:mb-0">
                  <div className="px-2 pb-1 pt-2 text-2xs font-semibold uppercase tracking-[0.05em] text-text-tertiary">
                    {group}
                  </div>
                  {items.map((s) => (
                    <Button
                      key={s.id}
                      variant="ghost"
                      aria-current={section === s.id ? "page" : undefined}
                      onClick={() => navigate(`/settings/${s.id}`)}
                      className={`h-auto w-full justify-start px-2 py-1.5 text-left text-sm font-normal transition-colors ${
                        section === s.id
                          ? "bg-accent font-medium text-foreground hover:bg-accent hover:text-foreground dark:hover:bg-accent dark:hover:text-foreground"
                          : "text-text-secondary hover:bg-muted/60 hover:text-foreground dark:hover:bg-muted/60 dark:hover:text-foreground"
                      }`}
                    >
                      {s.label}
                    </Button>
                  ))}
                </div>
              )
            })}
          </nav>
          <div className="min-w-0 flex-1">
            {section === "trash" && <TrashSection />}
            {section === "models" && <ModelsSection />}
            {section === "connections" && <ConnectionsSection />}
            {section === "budget" && <BudgetSection />}
            {section === "compute" && <ComputeUsageSection />}
            {section === "tracking" && <TrackingSection />}
            {section === "webhooks" && <WebhooksSection />}
            {section === "notifications" && <NotificationsSection />}
            {section === "notebooks" && <NotebookEnvironmentsSection />}
            {section === "secrets" && <SecretsSection />}
            {section === "audit" && <AuditSection />}
            {section === "about" && <AboutSection />}
          </div>
        </div>
      </div>
    </Scene>
  )
}

function Empty({ children }: { children: React.ReactNode }) {
  return (
    <div className="rounded-xl border border-dashed border-border px-4 py-8 text-center text-sm text-text-tertiary">
      {children}
    </div>
  )
}

// A skeleton shown during the first fetch, so a list never flashes its empty state (e.g.
// "No API keys yet") before the rows arrive.
function Loading() {
  return (
    <div className="space-y-2">
      {[0, 1, 2].map((i) => (
        <Skeleton key={i} className="h-11 w-full rounded-lg" />
      ))}
    </div>
  )
}

function ModelsSection() {
  return (
    <>
      <SectionHeader
        title="Models"
        hint="Register a model profile per provider, then pick one per conversation from the composer. Use a provider/model string (e.g. anthropic/claude-sonnet-5)."
      />
      <LlmProfilesManager />
    </>
  )
}

const KINDS = ["postgres", "mysql", "sqlite", "mssql"]

export function ConnectionsSection() {
  const [sources, setSources] = useState<DataSource[]>([])
  const [loading, setLoading] = useState(true)
  const [editing, setEditing] = useState<DataSourceInput | null>(null)
  const fb = useFeedback()

  const refresh = useCallback(() => {
    void getDataSources().then((s) => {
      setSources(s)
      setLoading(false)
    })
  }, [])
  useEffect(refresh, [refresh])

  const remove = async (s: DataSource) => {
    if (
      !(await fb.confirm({
        title: `Delete "${s.name}"?`,
        body: "This removes the connection. Derivations that use it will stop resolving.",
        danger: true,
      }))
    )
      return
    if (await deleteDataSource(s.id)) fb.toast("ok", `Deleted "${s.name}"`)
    else fb.toast("error", `Could not delete "${s.name}"`)
    refresh()
  }

  return (
    <>
      <SectionHeader
        title="Connections"
        hint="Connect a database a derivation can read from. Passwords are encrypted at rest."
      />
      {loading ? (
        <Loading />
      ) : (
        <div className="space-y-2">
          {sources.length === 0 && !editing && (
            <Empty>No connections yet. Add one to query a database beyond the local store.</Empty>
          )}
          {sources.map((s) => (
            <div
              key={s.id}
              className="flex items-center gap-3 rounded-lg border border-border px-3 py-2.5"
            >
              <Database className="h-4 w-4 shrink-0 text-text-tertiary" />
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium">{s.name}</div>
                <div className="truncate text-xs text-text-tertiary">
                  {s.kind}
                  {s.host ? ` · ${s.host}` : ""}
                  {s.database ? `/${s.database}` : ""}
                  {s.secretSet ? " · secret set" : ""}
                </div>
              </div>
              <TestButton input={{ id: s.id }} />
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label="Delete"
                className="text-destructive"
                onClick={() => void remove(s)}
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </div>
          ))}
        </div>
      )}
      {editing ? (
        <ConnectionForm
          value={editing}
          onChange={setEditing}
          onCancel={() => setEditing(null)}
          onSaved={() => {
            setEditing(null)
            refresh()
          }}
        />
      ) : (
        <Button
          variant="outline"
          size="sm"
          className="mt-4"
          onClick={() => setEditing({ kind: "postgres" })}
        >
          Add connection
        </Button>
      )}
    </>
  )
}

function TestButton({ input }: { input: DataSourceInput | { id: string } }) {
  const [state, setState] = useState<"idle" | "testing" | "ok" | "error">("idle")
  const [error, setError] = useState("")
  const test = async () => {
    setState("testing")
    const r = await testDataSource(input)
    setState(r.ok ? "ok" : "error")
    setError(r.error ?? "")
  }
  return (
    <Button
      type="button"
      variant="outline"
      size="xs"
      onClick={test}
      title={state === "error" ? error : "Test connection"}
      className={
        state === "ok"
          ? "border-success/40 text-success hover:bg-card hover:text-success"
          : state === "error"
            ? "border-danger/40 text-danger hover:bg-card hover:text-danger"
            : "text-text-tertiary hover:text-foreground"
      }
    >
      {state === "testing"
        ? "Testing…"
        : state === "ok"
          ? "Connected"
          : state === "error"
            ? "Failed"
            : "Test"}
    </Button>
  )
}

function ConnectionForm({
  value,
  onChange,
  onCancel,
  onSaved,
}: {
  value: DataSourceInput
  onChange: (v: DataSourceInput) => void
  onCancel: () => void
  onSaved: () => void
}) {
  const [saving, setSaving] = useState(false)
  const fb = useFeedback()
  const set = (patch: Partial<DataSourceInput>) => onChange({ ...value, ...patch })
  const isSqlite = value.kind === "sqlite"
  return (
    <div className="mt-4 space-y-3 rounded-xl border border-border p-4">
      <div className="grid grid-cols-2 gap-3">
        <FormField label="Name">
          <Input value={value.name ?? ""} onChange={(e) => set({ name: e.target.value })} />
        </FormField>
        <FormField label="Kind">
          <Select value={value.kind} onValueChange={(v) => set({ kind: v })}>
            <FormControl>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
            </FormControl>
            <SelectContent>
              {KINDS.map((k) => (
                <SelectItem key={k} value={k}>
                  {k}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </FormField>
      </div>
      {isSqlite ? (
        <FormField label="Database file path">
          <Input
            value={value.database ?? ""}
            onChange={(e) => set({ database: e.target.value })}
            placeholder="/data/app.db"
          />
        </FormField>
      ) : (
        <>
          <div className="grid grid-cols-3 gap-3">
            <FormField label="Host" className="col-span-2">
              <Input value={value.host ?? ""} onChange={(e) => set({ host: e.target.value })} />
            </FormField>
            <FormField label="Port">
              <Input
                type="number"
                value={value.port ?? ""}
                onChange={(e) => set({ port: e.target.value ? Number(e.target.value) : null })}
              />
            </FormField>
          </div>
          <div className="grid grid-cols-2 gap-3">
            <FormField label="Database">
              <Input
                value={value.database ?? ""}
                onChange={(e) => set({ database: e.target.value })}
              />
            </FormField>
            <FormField label="Username">
              <Input
                value={value.username ?? ""}
                onChange={(e) => set({ username: e.target.value })}
              />
            </FormField>
          </div>
          <FormField label="Password">
            <Input
              type="password"
              value={value.secret ?? ""}
              onChange={(e) => set({ secret: e.target.value })}
              placeholder="stored encrypted (APP_SECRET_KEY)"
            />
          </FormField>
        </>
      )}
      <div className="flex items-center justify-between pt-1">
        <TestButton input={value} />
        <div className="flex gap-2">
          <Button variant="outline" size="sm" onClick={onCancel}>
            Cancel
          </Button>
          <Button
            size="sm"
            disabled={!value.name || saving}
            onClick={async () => {
              setSaving(true)
              const ok = await createDataSource(value)
              setSaving(false)
              if (ok) {
                fb.toast("ok", `Saved "${value.name}"`)
                onSaved()
              } else {
                fb.toast("error", "Could not save the connection (name may be taken)")
              }
            }}
          >
            {saving ? "Saving…" : "Save"}
          </Button>
        </div>
      </div>
    </div>
  )
}

const WINDOWS = ["1d", "7d", "30d", "1mo"]

export function BudgetSection() {
  const [budget, setB] = useState<Budget | null>(null)
  const [draft, setDraft] = useState({ max: "", window: "30d" })
  const fb = useFeedback()

  const refresh = useCallback(() => {
    void getBudget().then((b) => {
      setB(b)
      setDraft({ max: b.maxBudget ? String(b.maxBudget) : "", window: b.window })
    })
  }, [])
  useEffect(refresh, [refresh])

  const pct =
    budget && budget.maxBudget > 0 ? Math.min(100, (budget.spend / budget.maxBudget) * 100) : 0
  const tone = pct >= 100 ? "bg-danger" : pct >= 80 ? "bg-warning" : "bg-primary"

  return (
    <>
      <SectionHeader
        title="Budget"
        hint="Cap spend over a rolling window. A turn is refused once the cap is reached; the window then resets."
      />
      {budget && budget.maxBudget > 0 && (
        <div className="mb-5 rounded-xl border border-border p-4">
          <div className="mb-1.5 flex justify-between text-sm">
            <span className="text-text-tertiary">This window ({budget.window})</span>
            <span className="font-medium">
              ${budget.spend.toFixed(2)} of ${budget.maxBudget.toFixed(2)} ({Math.round(pct)}%)
            </span>
          </div>
          <div className="h-2 overflow-hidden rounded-full bg-muted">
            <div className={`h-full rounded-full ${tone}`} style={{ width: `${pct}%` }} />
          </div>
        </div>
      )}
      <div className="flex items-end gap-3">
        {/* Fixed widths restore the old native <input>/<select>'s intrinsic (UA-default)
            widths: both Input and SelectTrigger render narrower here, since neither has a
            definite-width ancestor for the flex layout to size them against. */}
        <FormField label="Cap (USD)" className="w-[253px]">
          <Input
            type="number"
            step="0.01"
            value={draft.max}
            onChange={(e) => setDraft({ ...draft, max: e.target.value })}
            placeholder="0 = no cap"
          />
        </FormField>
        <FormField label="Window" className="w-[118px]">
          <Select value={draft.window} onValueChange={(v) => setDraft({ ...draft, window: v })}>
            <FormControl>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
            </FormControl>
            <SelectContent>
              {WINDOWS.map((w) => (
                <SelectItem key={w} value={w}>
                  {w}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </FormField>
        <Button
          size="sm"
          onClick={async () => {
            if (await setBudget(Number(draft.max) || 0, draft.window)) {
              fb.toast("ok", "Budget saved")
              refresh()
            } else {
              fb.toast("error", "Could not save the budget")
            }
          }}
        >
          Save
        </Button>
      </div>
    </>
  )
}

function AboutSection() {
  const [info, setInfo] = useState<AppVersion | null>(null)
  const [copied, setCopied] = useState(false)

  useEffect(() => {
    void getAppVersion().then(setInfo)
  }, [])

  const copy = () => {
    if (!info?.install.command) return
    void navigator.clipboard.writeText(info.install.command).then(() => {
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    })
  }

  return (
    <>
      <SectionHeader
        title="About"
        hint="The version running here, and how to upgrade it. Nothing on this page checks whether a newer release exists: this app makes no outbound request you did not ask for. Run `elbi update` when you want that answer."
      />
      <dl className="grid max-w-lg grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
        <dt className="text-text-tertiary">Version</dt>
        <dd className="font-mono tabular-nums">{info?.version || "…"}</dd>
        <dt className="text-text-tertiary">Package</dt>
        <dd className="font-mono">{info?.package || "…"}</dd>
        <dt className="text-text-tertiary">Installed via</dt>
        <dd>{INSTALL_LABELS[info?.install.kind ?? ""] ?? info?.install.kind ?? "…"}</dd>
      </dl>

      {info?.install.command ? (
        <div className="mt-6 space-y-2">
          <div className="text-sm font-medium">To upgrade</div>
          <div className="flex items-center gap-2">
            <code className="flex-1 rounded-md border border-border bg-muted/40 px-3 py-2 font-mono text-sm">
              {info.install.command}
            </code>
            <Button variant="secondary" size="sm" onClick={copy}>
              {copied ? <Check className="size-4" /> : <Copy className="size-4" />}
              {copied ? "Copied" : "Copy"}
            </Button>
          </div>
          {info.install.note ? (
            <p className="text-sm text-text-tertiary">{info.install.note}</p>
          ) : null}
        </div>
      ) : null}
    </>
  )
}

// How each detected install method reads to a person. The raw values are stable
// identifiers meant for scripts, not labels.
const INSTALL_LABELS: Record<string, string> = {
  "uv-tool": "uv tool",
  pipx: "pipx",
  venv: "a virtual environment",
  pip: "pip",
  docker: "the container image",
}

function TrackingSection() {
  const [settings, setSettings] = useState<MlflowSettings | null>(null)
  const [uri, setUri] = useState("")
  const fb = useFeedback()

  const refresh = useCallback(() => {
    void getMlflowSettings().then((s) => {
      setSettings(s)
      setUri(s.trackingUri)
    })
  }, [])
  useEffect(refresh, [refresh])

  // The env var wins at export time, so when it is in effect the setting is read-only: editing
  // it here would have no effect and would only mislead.
  const envOverride = settings?.source === "env"

  return (
    <>
      <SectionHeader
        title="Run tracking"
        hint="Mirror every certified run to an MLflow tracking server. Runs are recorded in this app's own history and comparison regardless; this exports the same certified estimates to MLflow."
      />
      <div className="flex items-end gap-3">
        <FormField label="MLflow tracking URI" className="flex-1">
          <Input
            value={uri}
            onChange={(e) => setUri(e.target.value)}
            placeholder="https://mlflow.example.com (empty disables export)"
            disabled={envOverride}
          />
        </FormField>
        <Button
          size="sm"
          disabled={envOverride}
          onClick={async () => {
            if (await setMlflowTrackingUri(uri.trim())) {
              fb.toast("ok", "Tracking URI saved")
              refresh()
            } else {
              fb.toast("error", "Could not save the tracking URI")
            }
          }}
        >
          Save
        </Button>
      </div>
      <p className="mt-2 text-xs text-text-tertiary">
        The <code className="font-mono">MLFLOW_TRACKING_URI</code> environment variable overrides
        this setting.
        {envOverride && " It is set, so this field is read-only."}
      </p>
    </>
  )
}

function WebhooksSection() {
  const [settings, setSettings] = useState<WebhookSettings | null>(null)
  const [url, setUrl] = useState("")
  const [secret, setSecret] = useState("")
  const fb = useFeedback()

  const refresh = useCallback(() => {
    void getWebhookSettings().then((s) => {
      setSettings(s)
      setUrl(s.url)
      setSecret("")
    })
  }, [])
  useEffect(refresh, [refresh])

  return (
    <>
      <SectionHeader
        title="Model webhooks"
        hint="POST model-registry events (finished training runs, drift results, promotions) to a URL. Requests are signed with the secret when one is set."
      />
      <div className="space-y-3">
        <FormField label="Webhook URL">
          <Input
            value={url}
            onChange={(e) => setUrl(e.target.value)}
            placeholder="https://hooks.example.com/models (empty disables webhooks)"
          />
        </FormField>
        <FormField label="Signing secret">
          <Input
            type="password"
            value={secret}
            onChange={(e) => setSecret(e.target.value)}
            placeholder={
              settings?.secretSet
                ? "secret set, leave blank to keep it"
                : "optional, used to sign deliveries"
            }
          />
        </FormField>
        <Button
          size="sm"
          onClick={async () => {
            // An untouched secret field keeps the stored secret; only send it when set.
            const ok = await setWebhookSettings(url.trim(), secret || undefined)
            if (ok) {
              fb.toast("ok", "Webhook settings saved")
              refresh()
            } else {
              fb.toast("error", "Could not save the webhook settings")
            }
          }}
        >
          Save
        </Button>
      </div>
    </>
  )
}

export function NotificationsSection() {
  const [prefs, setPrefs] = useState<NotificationPref[] | null>(null)
  const [emailAvailable, setEmailAvailable] = useState(false)
  const [failed, setFailed] = useState(false)
  const fb = useFeedback()

  const load = useCallback(() => {
    setFailed(false)
    getNotificationPreferences()
      .then((r) => {
        setPrefs(r.prefs)
        setEmailAvailable(r.emailAvailable)
      })
      // Surface a retriable error instead of loading forever.
      .catch(() => setFailed(true))
  }, [])
  useEffect(load, [load])

  const update = (eventType: string, patch: Partial<NotificationPref>) =>
    setPrefs((current) =>
      (current ?? []).map((p) => (p.eventType === eventType ? { ...p, ...patch } : p)),
    )

  if (failed && prefs === null)
    return (
      <>
        <SectionHeader
          title="Notifications"
          hint="Which events land in your inbox, per type, and which also send you an email."
        />
        <Empty>
          Could not load the notification preferences.{" "}
          <Button size="sm" variant="outline" className="ml-2" onClick={load}>
            Retry
          </Button>
        </Empty>
      </>
    )
  if (prefs === null) return <Loading />
  return (
    <>
      <SectionHeader
        title="Notifications"
        hint="Which events land in your inbox, per type, and which also send you an email. These are your own settings; they do not affect other users or the webhook."
      />
      <div className="space-y-3">
        <div className="divide-y divide-border rounded-xl border border-border">
          <div className="flex items-center gap-4 px-4 py-2 text-xs font-medium text-text-tertiary">
            <span className="flex-1">Event</span>
            <span className="w-14 text-center">In-app</span>
            <span
              className="w-14 text-center"
              title={emailAvailable ? undefined : "SMTP is not configured on this deployment"}
            >
              Email
            </span>
          </div>
          {prefs.map((p) => {
            const meta = EVENT_TYPE_LABELS[p.eventType]
            return (
              <div key={p.eventType} className="flex items-center gap-4 px-4 py-2.5">
                <div className="min-w-0 flex-1">
                  <p className="text-sm font-medium">{meta?.label ?? p.eventType}</p>
                  <p className="truncate text-xs text-text-tertiary">
                    {meta?.description ?? p.eventType}
                  </p>
                </div>
                <span className="flex w-14 justify-center">
                  <Checkbox
                    checked={p.inApp}
                    aria-label={`In-app notifications for ${meta?.label ?? p.eventType}`}
                    onCheckedChange={(c) => update(p.eventType, { inApp: c === true })}
                  />
                </span>
                <span
                  className="flex w-14 justify-center"
                  title={
                    !emailAvailable
                      ? "SMTP is not configured on this deployment"
                      : !p.inApp
                        ? "Email requires the in-app notification to be on"
                        : undefined
                  }
                >
                  <Checkbox
                    checked={p.email && p.inApp}
                    disabled={!emailAvailable || !p.inApp}
                    aria-label={`Email notifications for ${meta?.label ?? p.eventType}`}
                    onCheckedChange={(c) => update(p.eventType, { email: c === true })}
                  />
                </span>
              </div>
            )
          })}
        </div>
        <Button
          size="sm"
          onClick={async () => {
            try {
              const r = await setNotificationPreferences(prefs)
              setPrefs(r.prefs)
              fb.toast("ok", "Notification preferences saved")
            } catch {
              fb.toast("error", "Could not save the notification preferences")
            }
          }}
        >
          Save
        </Button>
      </div>
    </>
  )
}

// The editable form of a base environment: deps are held as raw textarea text and only
// parsed into a package list on save, so typing a newline (a blank line for the next
// package) is never stripped out from under the cursor.
type EnvDraft = { id: string; name: string; depsText: string }

export function NotebookEnvironmentsSection() {
  const [drafts, setDrafts] = useState<EnvDraft[] | null>(null)
  const fb = useFeedback()

  useEffect(() => {
    void getBaseEnvironments().then((envs) =>
      setDrafts(envs.map((e) => ({ id: uuid(), name: e.name, depsText: e.deps.join("\n") }))),
    )
  }, [])

  const update = (id: string, patch: Partial<EnvDraft>) =>
    setDrafts((current) => (current ?? []).map((d) => (d.id === id ? { ...d, ...patch } : d)))

  const parseDeps = (text: string): string[] =>
    text
      .split("\n")
      .map((line) => line.trim())
      .filter(Boolean)

  if (drafts === null) return <Loading />
  return (
    <>
      <SectionHeader
        title="Notebook base environments"
        hint="Curated package baselines a notebook can build on: the trusted floor its own dependencies layer over. Each is resolved and cached, so notebooks that share a base start fast."
      />
      <div className="space-y-4">
        {drafts.map((draft) => (
          <div key={draft.id} className="space-y-2 rounded-lg border border-border p-3">
            <div className="flex items-center gap-2">
              <Input
                value={draft.name}
                placeholder="Environment name (e.g. Data science)"
                onChange={(e) => update(draft.id, { name: e.target.value })}
              />
              <IconButton
                label="Remove"
                size="icon-xs"
                className="size-7 text-text-tertiary hover:bg-muted hover:text-danger dark:hover:bg-muted"
                onClick={() => setDrafts(drafts.filter((d) => d.id !== draft.id))}
              >
                <Trash2 className="size-4" />
              </IconButton>
            </div>
            {/* ui/textarea has field-sizing-content (auto-grows to fit its value); restore
                the old fixed-height behaviour that `rows` used to give it. */}
            <Textarea
              className="field-sizing-fixed font-mono text-compact"
              rows={4}
              spellCheck={false}
              value={draft.depsText}
              placeholder="pandas&#10;numpy&#10;scikit-learn"
              onChange={(e) => update(draft.id, { depsText: e.target.value })}
            />
          </div>
        ))}
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setDrafts([...drafts, { id: uuid(), name: "", depsText: "" }])}
          >
            Add environment
          </Button>
          <Button
            size="sm"
            onClick={async () => {
              const cleaned: BaseEnvironment[] = drafts
                .filter((d) => d.name.trim())
                .map((d) => ({ name: d.name.trim(), deps: parseDeps(d.depsText) }))
              const result = await setBaseEnvironments(cleaned)
              if (result.ok) {
                fb.toast("ok", "Base environments saved")
              } else {
                fb.toast("error", "Could not save the base environments")
              }
            }}
          >
            Save
          </Button>
        </div>
      </div>
    </>
  )
}

function SecretsSection() {
  const [secrets, setSecrets] = useState<SecretMeta[]>([])
  const [loading, setLoading] = useState(true)
  const [draft, setDraft] = useState({ name: "", value: "", description: "" })
  const fb = useFeedback()

  const refresh = useCallback(() => {
    void getSecrets().then((s) => {
      setSecrets(s)
      setLoading(false)
    })
  }, [])
  useEffect(refresh, [refresh])

  const save = async () => {
    if (await saveSecret(draft.name.trim(), draft.value, draft.description)) {
      fb.toast("ok", `Saved "${draft.name.trim()}"`)
      setDraft({ name: "", value: "", description: "" })
      refresh()
    } else {
      fb.toast("error", "Could not save the secret (is APP_SECRET_KEY set?)")
    }
  }

  const remove = async (name: string) => {
    if (
      !(await fb.confirm({
        title: `Delete secret "${name}"?`,
        body: "Anything referencing this secret will lose access. This cannot be undone.",
        danger: true,
      }))
    )
      return
    if (await deleteSecret(name)) fb.toast("ok", `Deleted "${name}"`)
    else fb.toast("error", `Could not delete "${name}"`)
    refresh()
  }

  return (
    <>
      <SectionHeader
        title="Secrets"
        hint="Store credentials (tokens, passwords) encrypted at rest. Values are never shown back."
      />
      <div className="mb-4 space-y-2 rounded-xl border border-border p-4">
        <div className="grid grid-cols-2 gap-3">
          <Input
            value={draft.name}
            onChange={(e) => setDraft({ ...draft, name: e.target.value })}
            placeholder="Name"
          />
          <Input
            type="password"
            value={draft.value}
            onChange={(e) => setDraft({ ...draft, value: e.target.value })}
            placeholder="Value"
          />
        </div>
        <div className="flex gap-3">
          <Input
            value={draft.description}
            onChange={(e) => setDraft({ ...draft, description: e.target.value })}
            placeholder="Description (optional)"
          />
          <Button
            size="sm"
            disabled={!draft.name.trim() || !draft.value}
            onClick={() => void save()}
          >
            Save
          </Button>
        </div>
      </div>
      {loading ? (
        <Loading />
      ) : secrets.length === 0 ? (
        <Empty>No secrets stored.</Empty>
      ) : (
        <div className="space-y-2">
          {secrets.map((s) => (
            <div
              key={s.name}
              className="flex items-center gap-3 rounded-lg border border-border px-3 py-2.5"
            >
              <Lock className="h-4 w-4 shrink-0 text-text-tertiary" />
              <div className="min-w-0 flex-1">
                <div className="truncate text-sm font-medium">{s.name}</div>
                {s.description && (
                  <div className="truncate text-xs text-text-tertiary">{s.description}</div>
                )}
              </div>
              <span className="text-xs text-text-tertiary">••••••</span>
              <Button
                variant="ghost"
                size="icon-xs"
                aria-label="Delete"
                className="text-destructive"
                onClick={() => void remove(s.name)}
              >
                <Trash2 className="h-3.5 w-3.5" />
              </Button>
            </div>
          ))}
        </div>
      )}
    </>
  )
}
function AuditSection() {
  const [events, setEvents] = useState<AuditEvent[]>([])
  useEffect(() => {
    void getAudit(200).then(setEvents)
  }, [])
  return (
    <>
      <SectionHeader
        title="Audit log"
        hint="What ran, and whether it was sound. Read-only, newest first."
      />
      {events.length === 0 ? (
        <Empty>No activity recorded yet.</Empty>
      ) : (
        <div className="overflow-x-auto rounded-xl border border-border">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>When</TableHead>
                <TableHead>Action</TableHead>
                <TableHead>Target</TableHead>
                <TableHead>Outcome</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {events.map((e) => (
                <TableRow key={e.id}>
                  <TableCell className="whitespace-nowrap text-text-tertiary">
                    {e.at.replace("T", " ").slice(0, 16)}
                  </TableCell>
                  <TableCell className="font-medium">{e.action}</TableCell>
                  <TableCell className="max-w-40 truncate text-text-tertiary">
                    {e.targetId || EMPTY}
                  </TableCell>
                  <TableCell>{e.verdict ? <VerdictBadge verdict={e.verdict} /> : EMPTY}</TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      )}
    </>
  )
}
