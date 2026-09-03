// The notification inbox: every platform event that concerned this user (a monitor
// fired, a run failed, a training finished), newest first, with read/unread state.
// Reading happens here; delivery preferences live in Settings > Notifications.
import { Bell, CheckCheck, Settings2 } from "lucide-react"
import { useCallback, useEffect, useRef, useState } from "react"
import { Link, useNavigate } from "react-router-dom"

import { Scene, SceneBody, SceneHeader, SceneSkeleton } from "@/components/Scene"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { VerdictBadge } from "@/components/VerdictBadge"
import {
  EVENT_TYPE_LABELS,
  emitNotificationsChanged,
  getNotifications,
  markAllNotificationsRead,
  markNotificationRead,
  type NotificationItem,
  notificationRoute,
} from "@/lib/notifications"
import { relativeTime } from "@/lib/utils"

export function NotificationsPage() {
  const [items, setItems] = useState<NotificationItem[] | null>(null)
  const [unread, setUnread] = useState(0)
  const [failed, setFailed] = useState(false)
  const [filter, setFilter] = useState<"all" | "unread">("all")
  const navigate = useNavigate()

  // Drop responses that resolve after a newer request started (filter toggles).
  const requestSeq = useRef(0)
  const refresh = useCallback(async () => {
    const seq = ++requestSeq.current
    try {
      const list = await getNotifications({ limit: 200, unreadOnly: filter === "unread" })
      if (seq !== requestSeq.current) return
      setItems(list.items)
      setUnread(list.unread)
      setFailed(false)
    } catch {
      // Keep what is shown; the render distinguishes "failed" from "no rows".
      if (seq === requestSeq.current) setFailed(true)
    }
  }, [filter])

  useEffect(() => {
    void refresh()
  }, [refresh])

  const open = async (n: NotificationItem) => {
    if (!n.readAt) {
      try {
        await markNotificationRead(n.id)
        emitNotificationsChanged()
      } catch {
        // A failed mark-read must not block navigation; the row stays unread.
      }
    }
    const route = notificationRoute(n)
    if (route) navigate(route)
    else void refresh()
  }

  const readAll = async () => {
    try {
      await markAllNotificationsRead()
      emitNotificationsChanged()
    } catch {
      // Leave the list as is; the next poll shows the true state.
    }
    void refresh()
  }

  // A failed first load is an error state, not an empty inbox.
  if (failed && items === null)
    return (
      <Scene>
        <SceneHeader
          title="Notifications"
          description="What happened to the things you own: monitors, runs, trainings, and data checks."
          icon={<Bell className="size-5" />}
        />
        <SceneBody width="narrow">
          <div className="rounded-lg border border-border bg-card px-4 py-10 text-center">
            <Bell className="mx-auto size-6 text-text-tertiary" />
            <p className="mt-2 text-sm font-medium text-foreground">Could not load notifications</p>
            <p className="mt-1 text-sm text-text-secondary">
              The server did not answer. Your notifications are safe; try again.
            </p>
            <Button variant="outline" size="sm" className="mt-4" onClick={() => void refresh()}>
              Retry
            </Button>
          </div>
        </SceneBody>
      </Scene>
    )
  if (items === null) return <SceneSkeleton />

  const filterButton = (value: "all" | "unread", label: string) => (
    <button
      type="button"
      onClick={() => setFilter(value)}
      aria-pressed={filter === value}
      className={`rounded-md px-2.5 py-1 text-sm transition ${
        filter === value
          ? "bg-card font-medium text-foreground shadow-sm"
          : "text-text-tertiary hover:text-foreground"
      }`}
    >
      {label}
    </button>
  )

  return (
    <Scene>
      <SceneHeader
        title="Notifications"
        description="What happened to the things you own: monitors, runs, trainings, and data checks."
        icon={<Bell className="size-5" />}
        badges={unread > 0 ? <Badge variant="info">{unread} unread</Badge> : undefined}
        actions={
          <>
            <Button variant="outline" size="sm" asChild>
              <Link to="/settings/notifications">
                <Settings2 /> Preferences
              </Link>
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => void readAll()}
              disabled={unread === 0}
            >
              <CheckCheck /> Mark all read
            </Button>
          </>
        }
      />
      <SceneBody width="narrow">
        <div className="mb-3 inline-flex gap-1 rounded-lg bg-panel p-1">
          {filterButton("all", "All")}
          {filterButton("unread", "Unread")}
        </div>
        {/* A refetch failure must not silently show the previous list. */}
        {failed && (
          <div className="mb-3 flex items-center justify-between gap-3 rounded-lg bg-warning-tint px-3 py-2 text-sm text-warning">
            <span>Could not refresh; showing the last loaded list.</span>
            <Button variant="outline" size="sm" onClick={() => void refresh()}>
              Retry
            </Button>
          </div>
        )}
        {items.length === 0 ? (
          <div className="rounded-lg border border-border bg-card px-4 py-10 text-center">
            <Bell className="mx-auto size-6 text-text-tertiary" />
            <p className="mt-2 text-sm font-medium text-foreground">
              {filter === "unread" ? "Nothing unread" : "No notifications yet"}
            </p>
            <p className="mt-1 text-sm text-text-secondary">
              When a monitor fires, a run fails, or a training finishes, it lands here.
            </p>
          </div>
        ) : (
          <ul className="divide-y divide-border rounded-lg border border-border bg-card">
            {items.map((n) => (
              <li key={n.id}>
                <button
                  type="button"
                  onClick={() => void open(n)}
                  className="flex w-full items-start gap-3 px-4 py-3 text-left transition hover:bg-panel"
                >
                  <span
                    aria-hidden
                    className={`mt-1.5 size-2 shrink-0 rounded-full ${
                      n.readAt ? "bg-transparent" : "bg-info"
                    }`}
                  />
                  <span className="min-w-0 flex-1">
                    <span className="flex flex-wrap items-center gap-2">
                      <span
                        className={`text-sm ${
                          n.readAt ? "text-text-secondary" : "font-medium text-foreground"
                        }`}
                      >
                        {n.title}
                      </span>
                      {n.verdict && <VerdictBadge verdict={n.verdict} />}
                    </span>
                    {n.body && (
                      <span className="mt-0.5 block truncate text-sm text-text-tertiary">
                        {n.body}
                      </span>
                    )}
                    <span className="mt-0.5 block text-xs text-text-tertiary">
                      {EVENT_TYPE_LABELS[n.eventType]?.label ?? n.eventType} · {relativeTime(n.at)}
                    </span>
                  </span>
                </button>
              </li>
            ))}
          </ul>
        )}
      </SceneBody>
    </Scene>
  )
}
