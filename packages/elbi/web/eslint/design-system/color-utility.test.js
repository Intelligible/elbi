// @vitest-environment node
import path from "node:path"
import { describe, expect, it } from "vitest"

import { classifyColorUtility } from "./color-utility.js"
import { readTheme } from "./theme.js"

const theme = readTheme(path.join(import.meta.dirname, "fixtures/theme.css"))
const ok = (cls) => expect(classifyColorUtility(cls, theme)).toBeNull()
const bad = (cls, re) => expect(classifyColorUtility(cls, theme)).toMatch(re)

describe("readTheme", () => {
  it("reads colour and text-size names from @theme blocks only", () => {
    expect(theme.colors.has("text-tertiary")).toBe(true)
    expect(theme.textSizes.has("2xs")).toBe(true)
    expect(theme.textSizes.has("secondary")).toBe(false)
    expect(theme.textSizes.has("2xs--line-height")).toBe(false)
  })
})

describe("classifyColorUtility", () => {
  it("accepts theme colours, keywords and opacity", () => {
    ok("text-foreground")
    ok("text-text-tertiary")
    ok("border-border-strong")
    ok("border-t-border")
    ok("bg-primary/60")
    ok("text-white")
    ok("bg-transparent")
    ok("fill-current")
    ok("bg-data-3")
  })
  it("ignores non-colour utilities that share a prefix", () => {
    for (const c of ["text-xs", "text-2xs", "text-center", "text-ellipsis", "text-wrap", "border", "border-2", "border-t", "border-dashed", "border-collapse", "ring-2", "ring-offset-2", "outline-none", "outline-offset-2", "divide-y", "stroke-2", "fill-none", "bg-cover", "bg-no-repeat", "bg-linear-to-r", "from-10%", "decoration-2", "bg-top-left", "bg-top-right", "bg-bottom-left", "bg-bottom-right", "bg-left-top", "bg-right-bottom"]) ok(c)
  })
  it("skips arbitrary values (another rule owns them)", () => {
    ok("text-[11px]")
    ok("bg-(--x)")
  })
  it("rejects undefined tokens", () => {
    bad("text-tertiary", /no colour token/)
    bad("bg-surface-9", /no colour token/)
  })
  it("rejects Tailwind's default palette, the raw ramp and banned names", () => {
    bad("text-red-500", /default palette/)
    bad("bg-emerald-50", /default palette/)
    bad("text-n-600", /raw ramp/)
    bad("text-muted-foreground", /text-text-tertiary/)
    bad("text-secondary", /text-text-secondary/)
  })
})
