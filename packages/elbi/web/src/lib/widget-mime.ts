// The MIME type an ipywidgets display carries; its payload names the model to render.
// Kept in its own tiny module so a cell-output renderer can test for it without pulling
// in the heavy @jupyter-widgets manager (which is dynamically imported only on the
// notebook route).
export const WIDGET_MIME = "application/vnd.jupyter.widget-view+json"
