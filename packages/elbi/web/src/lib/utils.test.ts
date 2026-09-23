import { afterEach, describe, expect, it, vi } from "vitest"

import { cn, copyText, relativeTime, uuid } from "./utils"

describe("cn", () => {
  it("merges classes and lets a later Tailwind utility win a conflict", () => {
    expect(cn("px-2", "px-4")).toBe("px-4")
    expect(cn("text-sm", false, "font-medium")).toBe("text-sm font-medium")
  })

  it("knows the custom type-scale tokens are font sizes, not colours", () => {
    for (const size of ["3xs", "2xs", "compact", "title", "display"]) {
      expect(cn(`text-${size}`, "text-text-tertiary")).toBe(`text-${size} text-text-tertiary`)
    }
  })

  it("still resolves a font-size conflict between two scale tokens, later wins", () => {
    expect(cn("text-xs", "text-compact")).toBe("text-compact")
    expect(cn("text-compact", "text-sm")).toBe("text-sm")
  })
})

describe("relativeTime", () => {
  afterEach(() => vi.useRealTimers())

  it("returns 'never' for missing or unparseable input", () => {
    expect(relativeTime(null)).toBe("never")
    expect(relativeTime("not-a-date")).toBe("never")
  })

  it("formats recent timestamps in coarse buckets", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-01-01T00:00:00Z"))
    expect(relativeTime("2025-12-31T23:59:50Z")).toBe("just now")
    expect(relativeTime("2025-12-31T23:30:00Z")).toBe("30m ago")
    expect(relativeTime("2025-12-31T21:00:00Z")).toBe("3h ago")
  })
})

describe("uuid", () => {
  afterEach(() => vi.unstubAllGlobals())

  it("returns a v4 uuid from the platform when it has one", () => {
    expect(uuid()).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/)
  })

  it("still works where randomUUID does not exist", () => {
    // The shape of a page served over plain HTTP: randomUUID is secure-context-only and
    // simply absent, while getRandomValues is not restricted and remains available.
    // Before this fell back, the missing function threw during render and blanked the
    // whole app rather than degrading anything.
    vi.stubGlobal("crypto", {
      getRandomValues: globalThis.crypto.getRandomValues.bind(globalThis.crypto),
    })
    expect(typeof crypto.randomUUID).not.toBe("function")

    const ids = new Set(Array.from({ length: 200 }, () => uuid()))

    expect(ids.size).toBe(200)
    for (const id of ids) {
      expect(id).toMatch(/^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/)
    }
  })
})

describe("copyText", () => {
  afterEach(() => vi.unstubAllGlobals())

  it("uses the clipboard API when it is available", async () => {
    const writeText = vi.fn().mockResolvedValue(undefined)
    vi.stubGlobal("navigator", { clipboard: { writeText } })
    await expect(copyText("hello")).resolves.toBe(true)
    expect(writeText).toHaveBeenCalledWith("hello")
  })

  it("falls back when the clipboard API is absent, as it is over plain HTTP", async () => {
    vi.stubGlobal("navigator", {})
    const exec = vi.fn().mockReturnValue(true)
    vi.stubGlobal("document", {
      ...document,
      createElement: document.createElement.bind(document),
      body: document.body,
      execCommand: exec,
    })
    await expect(copyText("hello")).resolves.toBe(true)
    expect(exec).toHaveBeenCalledWith("copy")
  })

  it("reports failure rather than throwing when nothing can copy", async () => {
    vi.stubGlobal("navigator", {})
    vi.stubGlobal("document", {
      ...document,
      createElement: document.createElement.bind(document),
      body: document.body,
      execCommand: () => {
        throw new Error("not supported")
      },
    })
    await expect(copyText("hello")).resolves.toBe(false)
  })
})
