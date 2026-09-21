Three faults kept a dashboard from showing what its derivations say. A `text` widget
could not bind a derivation, so the documented escape hatch — a bespoke visual belongs
in a derivation returning markdown — did not work, and narrative tiles had to hardcode
figures that go stale. A text widget did not render markdown at all, emitting its
source as preformatted text. And a tile's rows crossed the camelCase API boundary as
field names, so a derivation's `mrr_usd` reached the browser as `mrrUsd` and a widget's
`viz.field`, written as the derivation writes it, matched nothing: the tile rendered a
dash and a chart encoding silently found no data.

A text widget now takes `content` or a `bind`, not both, and renders either as
markdown. (Markdown there still reads `$…$` as inline TeX, so a dollar amount in a
bound derivation renders as an integrand; that is fixed separately.) A derivation artifact's `value` is opaque to the casing boundary, like `rows`
already was.

Separately, a page that omitted `columns` rendered on a 12-column grid while the spec,
the schema and `Page.to_manifest` all default to 24 — so every page taking the default
laid out at half width, and each widget spanned twice the intended fraction.
