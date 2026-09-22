A tile's editor has a **JSON** tab holding the widget exactly as the dashboard spec
stores it, so the things the fields do not reach — a chart's grammar-of-graphics
`viz`, a binding's `params`, `interactions` — are editable without leaving the tile.
Switching tabs carries an unsaved edit across in both directions. The JSON saves the
whole widget, so a key deleted in the text is really deleted; the tile's `id` is kept
whatever the text says, since it addresses the tile in the layout.

A bound tile also shows where its numbers come from: a link to the derivation, and to
the derivations and datasets that derivation reads.
