The dashboard JSON editors check what you type against the spec's own JSON Schema, not
just that it parses. A widget type outside the allowed set, a missing `gridPos`, a
width past the end of the grid, a key the spec does not define — each is underlined
where it is, with a message naming what is wrong, while you type rather than when you
press Save.

The schema is served from `GET /api/dashboards/schema`: the same document the server
validates with, so the editor cannot drift into accepting what a save refuses.
