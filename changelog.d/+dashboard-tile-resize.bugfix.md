Dashboard tiles resized properly. A tile could be resized from any edge or corner
rather than only an invisible bottom-right corner, and its handles were visible on
hover. A chart or table grew with its tile instead of keeping the height it first
rendered at, which had left the new space empty; a table filled the tile and scrolled.
