/**
 * An undo/redo stack over whole snapshots of a value.
 *
 * Snapshots rather than diffs: a dashboard spec is small, and every edit here already
 * rebuilds it, so there is nothing to gain from recording the delta and a correctness
 * trap in getting the inverse of a delete wrong.
 */
export interface History<T> {
  past: T[]
  future: T[]
  limit: number
}

export function createHistory<T>(limit = 50): History<T> {
  return { past: [], future: [], limit }
}

/**
 * Record the value being replaced.
 *
 * Doing something new abandons the redo branch — the usual editor behaviour, and the
 * alternative is a redo that reapplies an edit to a state it was never made against.
 */
export function record<T>(history: History<T>, previous: T): void {
  history.past.push(previous)
  if (history.past.length > history.limit) history.past.shift()
  history.future.length = 0
}

/** The value to restore, or `undefined` when there is nothing to undo. */
export function undo<T>(history: History<T>, current: T): T | undefined {
  const previous = history.past.pop()
  if (previous === undefined) return undefined
  history.future.push(current)
  return previous
}

/** The value to restore, or `undefined` when there is nothing to redo. */
export function redo<T>(history: History<T>, current: T): T | undefined {
  const next = history.future.pop()
  if (next === undefined) return undefined
  history.past.push(current)
  return next
}

/**
 * Whether a keyboard event is an undo/redo gesture, and which.
 *
 * ⌘Z / Ctrl-Z undoes; ⇧⌘Z and Ctrl-Y redo. A keystroke inside a text field, or while a
 * dialog is open, is not ours: there ⌘Z means the browser's own text undo, and undoing
 * the board behind an open dialog would be invisible to whoever pressed it.
 */
export function undoGesture(event: KeyboardEvent): "undo" | "redo" | null {
  if (!(event.metaKey || event.ctrlKey) || event.altKey) return null
  const key = event.key.toLowerCase()
  const target = event.target as HTMLElement | null
  if (target?.closest?.('input, textarea, select, [contenteditable="true"]')) return null
  if (typeof document !== "undefined" && document.querySelector('[role="dialog"]')) return null
  if (key === "z") return event.shiftKey ? "redo" : "undo"
  if (key === "y" && !event.metaKey) return "redo"
  return null
}
