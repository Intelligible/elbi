import { describe, expect, it } from "vitest"

import { createHistory, record, redo, undo, undoGesture } from "./history"

const key = (init: Partial<KeyboardEvent> & { key: string }) =>
  ({ altKey: false, shiftKey: false, metaKey: false, ctrlKey: false, ...init }) as KeyboardEvent

describe("history", () => {
  it("restores the value that was replaced", () => {
    const h = createHistory<string>()
    record(h, "a")

    expect(undo(h, "b")).toBe("a")
  })

  it("walks back through several edits in order", () => {
    const h = createHistory<string>()
    record(h, "a")
    record(h, "b")

    expect(undo(h, "c")).toBe("b")
    expect(undo(h, "b")).toBe("a")
    expect(undo(h, "a")).toBeUndefined()
  })

  it("redoes what was undone", () => {
    const h = createHistory<string>()
    record(h, "a")
    const back = undo(h, "b") as string

    expect(redo(h, back)).toBe("b")
  })

  it("abandons the redo branch once something new is done", () => {
    const h = createHistory<string>()
    record(h, "a")
    undo(h, "b")

    record(h, "a")

    expect(redo(h, "c")).toBeUndefined()
  })

  it("forgets the oldest edit past the limit", () => {
    const h = createHistory<string>(2)
    record(h, "a")
    record(h, "b")
    record(h, "c")

    expect(h.past).toEqual(["b", "c"])
  })
})

describe("undoGesture", () => {
  it("reads the undo and redo chords", () => {
    expect(undoGesture(key({ key: "z", metaKey: true }))).toBe("undo")
    expect(undoGesture(key({ key: "z", ctrlKey: true }))).toBe("undo")
    expect(undoGesture(key({ key: "z", metaKey: true, shiftKey: true }))).toBe("redo")
    expect(undoGesture(key({ key: "y", ctrlKey: true }))).toBe("redo")
  })

  it("ignores a bare z and other modifiers", () => {
    expect(undoGesture(key({ key: "z" }))).toBeNull()
    expect(undoGesture(key({ key: "z", metaKey: true, altKey: true }))).toBeNull()
    expect(undoGesture(key({ key: "k", metaKey: true }))).toBeNull()
  })

  it("leaves a text field's own undo alone", () => {
    const input = document.createElement("input")
    document.body.append(input)

    expect(undoGesture(key({ key: "z", metaKey: true, target: input }))).toBeNull()

    input.remove()
  })

  it("does nothing while a dialog is open", () => {
    const dialog = document.createElement("div")
    dialog.setAttribute("role", "dialog")
    document.body.append(dialog)

    expect(undoGesture(key({ key: "z", metaKey: true }))).toBeNull()

    dialog.remove()
  })
})
