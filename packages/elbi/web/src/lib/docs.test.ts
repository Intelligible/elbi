import { describe, expect, it } from "vitest"

import { DOCS_URL, docsUrl } from "./docs"

describe("docsUrl", () => {
  it("returns the docs root with no path", () => {
    expect(docsUrl()).toBe(`${DOCS_URL}/`)
  })

  it("joins a page path with exactly one slash", () => {
    expect(docsUrl("monitoring/")).toBe("https://docs.elbi.ai/monitoring/")
    expect(docsUrl("/monitoring/")).toBe("https://docs.elbi.ai/monitoring/")
  })
})
