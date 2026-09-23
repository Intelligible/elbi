import { createContext, useContext, useEffect, useRef } from "react"

import type { NotebookWidgetManager } from "@/lib/widgets"

// The live widget manager for the open notebook, provided by NotebookPage so any cell
// output can render a widget view without threading the manager through props.
export const WidgetManagerContext = createContext<NotebookWidgetManager | null>(null)

// Renders the ipywidgets model named by a widget-view output into a host div, driven by
// the notebook's manager. Interactivity flows over the manager's comm WebSocket.
export function WidgetView({ modelId }: { modelId: string }) {
  const manager = useContext(WidgetManagerContext)
  const host = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const el = host.current
    if (!manager || !el) return
    void manager.renderWidget(modelId, el)
    return () => el.replaceChildren()
  }, [manager, modelId])

  if (!manager) {
    return <div className="text-xs text-text-tertiary">[widget: kernel not connected]</div>
  }
  return <div ref={host} className="jupyter-widgets-view" />
}
