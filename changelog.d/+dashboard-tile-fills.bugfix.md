A chart or table grows when its tile does. A chart took the height it was embedded at
and kept it, so dragging a tile taller left the new space empty below the plot; its
frame is now measured and the chart re-rendered to fit. A table was sized to its rows
rather than to the tile, which left the same dead band under a short table — it now
fills the tile and scrolls, so a taller tile shows more rows.
