// The public docs site. Every in-app "Learn more" link goes through `docsUrl`, so the
// host lives in one place.
export const DOCS_URL = "https://docs.elbi.ai/"

// A page of the docs, as its mkdocs directory URL, with an optional heading anchor:
// `docsUrl("data-sources", "sync-status")`.
export function docsUrl(page = "", anchor?: string): string {
  const path = page.replace(/^\/+|\/+$/g, "")
  return `${DOCS_URL}${path ? `${path}/` : ""}${anchor ? `#${anchor}` : ""}`
}
