import { createContext, useContext, useEffect } from "react"

import type { PageContext } from "@/components/AssistantPanel"

// A page publishes what the assistant should know about it, as a small React
// context that each scene fills in for itself. The assistant panel reads
// whatever the active page last registered, so the model always knows the live thing the
// user is looking at (a notebook and its cells, a dashboard, ...) rather than just a URL.
type Registry = { set: (context: PageContext) => void }

const AssistantContextRegistry = createContext<Registry | null>(null)

export function AssistantContextProvider({
  set,
  children,
}: {
  set: (context: PageContext) => void
  children: React.ReactNode
}) {
  return (
    <AssistantContextRegistry.Provider value={{ set }}>
      {children}
    </AssistantContextRegistry.Provider>
  )
}

// A scene calls this to register its live context with the assistant; the registration is
// cleared when the scene unmounts (navigating away), mirroring `locationChanged`. Pass a
// memoized value so this only re-registers when the context actually changes.
export function useAssistantContext(context: PageContext): void {
  const registry = useContext(AssistantContextRegistry)
  useEffect(() => {
    registry?.set(context)
    return () => registry?.set(null)
  }, [registry, context])
}
