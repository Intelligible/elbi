A dashboard's grid is twice as fine vertically. A tile's height, and the step a
vertical resize moves in, was one 44px row plus a 16px gap — 60px a notch, enough that
dragging an edge felt chunky and the tiles below moved in visible jumps. The row is
now 14px, so the step is 30px.

**The unit of `gridPos.h` changed with it.** The same pixel height is twice as many
rows: a tile written as `h: 3` is now `h: 6`. A dashboard authored against the old unit
renders at about half its intended height until its `h` and `y` values are doubled.
