// The published user documentation. mkdocs builds with directory URLs, so a page at
// docs/monitoring.md is served at /monitoring/.
export const DOCS_URL = "https://docs.elbi.ai"

export function docsUrl(path = ""): string {
  const trimmed = path.replace(/^\/+/, "")
  return `${DOCS_URL}/${trimmed}`
}
