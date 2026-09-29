A dashboard's grid became twice as fine vertically, so a vertical resize moved in 30px
steps instead of 60px. **The unit of `gridPos.h` changed with it:** the same height is
twice as many rows, so a tile written as `h: 3` is now `h: 6`. A dashboard authored
against the old unit renders at about half its height until its `h` and `y` values are
doubled.
