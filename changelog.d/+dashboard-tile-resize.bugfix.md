A dashboard tile resizes from any edge or corner, not only the bottom-right. Widening
a tile leftwards meant dragging its right edge and then moving it back; now the left
edge does it directly, and the top and bottom edges change height. The handles are
visible: react-grid-layout draws them as faint corner triangles at `opacity: 0`, which
against either theme read as nothing, so a tile looked like it could only be moved. An
edge handle now runs the whole length of its edge, so the entire side is grabbable.
