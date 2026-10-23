Table tiles on a dashboard can now size their columns. Each entry in a table's
`viz.columns` may be an object with a `label`, a `width` in pixels (or `fit` or `fill`)
and `wrap`, and the same widths can be set by dragging a column header's edge, which
saves them into the dashboard spec so a pull and a sync carry them. A cell holding a web
address or a `[text](url)` link now renders as a link.
