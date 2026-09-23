A tile's editor has a **JSON** tab holding the widget exactly as the dashboard spec
stores it, so the things the fields do not reach — a chart's grammar-of-graphics
`viz`, a binding's `params`, `interactions` — are editable without leaving the tile.
Switching tabs carries an unsaved edit across in both directions. The JSON saves the
whole widget, so a key deleted in the text is really deleted; the tile's `id` is kept
whatever the text says, since it addresses the tile in the layout.

A **Lineage** tab shows where a bound tile's numbers come from: the derivation that
computed them, and the derivations and warehouse tables that derivation reads. It is a
tab rather than a panel over the fields, so it costs a click and buys room to say what
each link is — and nothing is fetched until someone asks.
