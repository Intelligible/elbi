import { describe, expect, it } from "vitest"

import { DOCS_URL, docsUrl } from "./docs"

describe("docsUrl", () => {
  it("builds a page's directory URL on the docs host", () => {
    expect(docsUrl("data-sources")).toBe("https://docs.elbi.ai/data-sources/")
    expect(docsUrl("/data-sources/")).toBe("https://docs.elbi.ai/data-sources/")
  })

  it("appends a heading anchor", () => {
    expect(docsUrl("data-sources", "sync-status")).toBe(
      "https://docs.elbi.ai/data-sources/#sync-status",
    )
  })

  it("an empty page is the docs home", () => {
    expect(docsUrl("")).toBe(DOCS_URL)
  })
})
