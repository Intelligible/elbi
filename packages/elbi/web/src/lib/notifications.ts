// API client and types for the notification inbox: durable records of
// platform events (a monitor fired, a run failed, a training finished) with
// read/unread state and per-event-type preferences.

export interface NotificationItem {
  id: string
  at: string
  eventType: string
  title: string
  body: string
  targetType: string
  targetId: string
  verdict: string | null
  readAt: string | null
}

export interface NotificationList {
  items: NotificationItem[]
  unread: number
}

export interface NotificationPref {
  eventType: string
  inApp: boolean
  email: boolean
}

export interface PreferencesResponse {
  emailAvailable: boolean
  prefs: NotificationPref[]
}

// Human labels for the event vocabulary, keyed by the backend's event types
// (notifications.EVENT_TYPES). The preferences UI renders one row per entry.
export const EVENT_TYPE_LABELS: Record<string, { label: string; description: string }> = {
  "model_version.created": {
    label: "Training finished",
    description: "A training you started registered a new model version",
  },
  "training.failed": {
    label: "Training failed",
    description: "A training you started raised an error",
  },
  "metric.anomaly_detected": {
    label: "Monitor anomaly",
    description: "A monitor you configured detected an anomaly",
  },
  "metric.recovered": {
    label: "Monitor recovered",
    description: "A monitor you configured returned to normal",
  },
  "run.failed": {
    label: "Run failed",
    description: "An orchestration run you own failed",
  },
  "run.slow": {
    label: "Run slow",
    description: "An orchestration run took far longer than its median",
  },
  "drift.detected": {
    label: "Model drift",
    description: "Serving traffic drifted from a model's training data",
  },
  "feature.drift_detected": {
    label: "Feature drift",
    description: "A feature view's values drifted from its baseline",
  },
  "feature.expectations_failed": {
    label: "Expectations failed",
    description: "A feature view's values violated its data contract",
  },
}

async function json<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(path, init)
  if (!r.ok) {
    let detail = `${r.status}`
    try {
      const body = (await r.json()) as { detail?: string }
      if (body.detail) detail = body.detail
    } catch {
      // A non-JSON error body leaves the status code as the message.
    }
    throw new Error(detail)
  }
  return (await r.json()) as T
}

function send<T>(method: string, path: string, body?: unknown): Promise<T> {
  return json<T>(path, {
    method,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body ?? {}),
  })
}

const id = encodeURIComponent

export const getNotifications = (opts?: { limit?: number; unreadOnly?: boolean }) => {
  const params = new URLSearchParams()
  if (opts?.limit !== undefined) params.set("limit", String(opts.limit))
  if (opts?.unreadOnly) params.set("unread_only", "true")
  const qs = params.toString()
  return json<NotificationList>(`/api/notifications${qs ? `?${qs}` : ""}`)
}

export const markNotificationRead = (notificationId: string) =>
  send<{ ok: boolean }>("POST", `/api/notifications/${id(notificationId)}/read`)

export const markAllNotificationsRead = () =>
  send<{ ok: boolean; count: number }>("POST", "/api/notifications/read-all")

export const getNotificationPreferences = () =>
  json<PreferencesResponse>("/api/notifications/preferences")

export const setNotificationPreferences = (prefs: NotificationPref[]) =>
  send<PreferencesResponse>("PUT", "/api/notifications/preferences", { prefs })

// Where a notification's row click should land, from its target reference.
// Null means the notification has no page to open (stay in the inbox).
export function notificationRoute(n: NotificationItem): string | null {
  if (n.targetType === "metric_monitor") return "/monitors"
  if (n.targetType === "orchestration_run") return "/orchestration"
  if (n.targetType === "model_version") {
    const name = n.targetId.split("/")[0]
    return name ? `/models/${encodeURIComponent(name)}` : "/models"
  }
  if (n.targetType === "model") {
    return n.targetId ? `/models/${encodeURIComponent(n.targetId)}` : "/models"
  }
  if (n.targetType === "feature_view") {
    return n.targetId ? `/features/${encodeURIComponent(n.targetId)}` : "/features"
  }
  return null
}

// A window-level change signal so the sidebar's unread pill refreshes the moment
// the inbox page marks something read, instead of waiting out its poll interval.
const CHANGED_EVENT = "elbi:notifications-changed"

export function emitNotificationsChanged(): void {
  window.dispatchEvent(new CustomEvent(CHANGED_EVENT))
}

export function onNotificationsChanged(cb: () => void): () => void {
  window.addEventListener(CHANGED_EVENT, cb)
  return () => window.removeEventListener(CHANGED_EVENT, cb)
}
