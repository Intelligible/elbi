A dashboard's tiles can be edited from the dashboard itself. Each tile carries a menu
with **Edit…** and **Delete**. The editor changes a tile's title, a text tile's
markdown, which derivation a bound tile reads, a metric's column, aggregate and
number format, and the tile's size in grid columns and rows. It warns before saving a
derivation nothing provides, and before formatting a column that already holds a
percentage as one.

Deleting asks first and says what it does not remove. Dragging and resizing already
persisted, but the resize handle was an invisible corner: it is now a visible grip in
the tile's bottom-right, shown on hover or keyboard focus. Laying a dashboard out no
longer means editing the spec JSON — a chart's grammar-of-graphics `viz`, params and
interactions still live there.
