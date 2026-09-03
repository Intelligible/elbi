import { createContext, useCallback, useContext, useRef, useState } from "react"
import { AlertTriangle, Check, X } from "lucide-react"

import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog"
import { Button } from "@/components/ui/button"

// A minimal toast + confirm layer so mutations surface success/failure (no silent
// failures) and destructive actions confirm first. One provider wraps the settings area.

type Toast = { id: number; kind: "ok" | "error"; text: string }
type Confirm = {
  title: string
  body: string
  danger?: boolean
  resolve: (ok: boolean) => void
}

interface Feedback {
  toast: (kind: "ok" | "error", text: string) => void
  confirm: (opts: { title: string; body: string; danger?: boolean }) => Promise<boolean>
}

const FeedbackContext = createContext<Feedback | null>(null)

export function useFeedback(): Feedback {
  const ctx = useContext(FeedbackContext)
  if (!ctx) throw new Error("useFeedback must be used within FeedbackProvider")
  return ctx
}

export function FeedbackProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([])
  const [pending, setPending] = useState<Confirm | null>(null)
  const nextId = useRef(0)

  const toast = useCallback((kind: "ok" | "error", text: string) => {
    const id = ++nextId.current
    setToasts((t) => [...t, { id, kind, text }])
    // Errors linger longer than confirmations, and never silently, but auto-clear.
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), kind === "error" ? 6000 : 3000)
  }, [])

  const confirm = useCallback(
    (opts: { title: string; body: string; danger?: boolean }) =>
      new Promise<boolean>((resolve) => setPending({ ...opts, resolve })),
    [],
  )

  const close = (ok: boolean) => {
    pending?.resolve(ok)
    setPending(null)
  }

  return (
    <FeedbackContext.Provider value={{ toast, confirm }}>
      {children}
      <div className="pointer-events-none fixed bottom-4 right-4 z-50 flex flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`pointer-events-auto flex items-center gap-2 rounded-lg border px-3 py-2 text-sm shadow-md ${
              t.kind === "error"
                ? "border-destructive/30 bg-destructive/10 text-destructive"
                : "border-border bg-background text-foreground"
            }`}
          >
            {t.kind === "error" ? (
              <AlertTriangle className="h-4 w-4 shrink-0" />
            ) : (
              <Check className="h-4 w-4 shrink-0 text-success" />
            )}
            <span>{t.text}</span>
            <button
              type="button"
              onClick={() => setToasts((x) => x.filter((y) => y.id !== t.id))}
              className="ml-1 text-muted-foreground hover:text-foreground"
              aria-label="Dismiss"
            >
              <X className="h-3.5 w-3.5" />
            </button>
          </div>
        ))}
      </div>
      <Dialog open={pending !== null} onOpenChange={(o) => !o && close(false)}>
        <DialogContent className="sm:max-w-md">
          <DialogHeader>
            <DialogTitle>{pending?.title}</DialogTitle>
            <DialogDescription>{pending?.body}</DialogDescription>
          </DialogHeader>
          <DialogFooter>
            {/* Cancel is the default/safe action to avoid an accidental Enter-confirm. */}
            <Button variant="outline" onClick={() => close(false)} autoFocus>
              Cancel
            </Button>
            <Button
              variant={pending?.danger ? "destructive" : "default"}
              onClick={() => close(true)}
            >
              Confirm
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </FeedbackContext.Provider>
  )
}
