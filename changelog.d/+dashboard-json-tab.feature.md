The tile editor gained a **JSON** tab holding the widget exactly as the dashboard spec
stores it, so a chart's `viz`, a binding's `params` and a tile's `interactions` became
editable without leaving the tile. It and **Edit spec** were syntax highlighted, and
checked what was typed against the dashboard spec's schema, so an unknown widget type,
a missing `gridPos` or a tile wider than the grid was underlined where it was, before
Save. A **Lineage** tab showed where a bound tile's numbers came from: the derivation
that computed them, and the derivations and warehouse tables that one reads.
