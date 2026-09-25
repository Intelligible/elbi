// @vitest-environment node
import { describe, expect, it } from "vitest"

import { baseUtility, splitClasses, stripOpacity } from "./class-strings.js"

describe("class-strings", () => {
  it("splits on any whitespace", () => {
    expect(splitClasses("  a b\n  c ")).toEqual(["a", "b", "c"])
  })
  it("strips variants, including arbitrary ones with colons inside brackets", () => {
    expect(baseUtility("hover:bg-accent")).toBe("bg-accent")
    expect(baseUtility("md:dark:text-foreground")).toBe("text-foreground")
    expect(baseUtility("[&_svg:not([class*='size-'])]:size-4")).toBe("size-4")
    expect(baseUtility("!text-xs")).toBe("text-xs")
    expect(baseUtility("text-xs!")).toBe("text-xs")
  })
  it("strips an opacity suffix but not an arbitrary value", () => {
    expect(stripOpacity("bg-primary/60")).toBe("bg-primary")
    expect(stripOpacity("bg-[url(/a/b)]")).toBe("bg-[url(/a/b)]")
  })
})
