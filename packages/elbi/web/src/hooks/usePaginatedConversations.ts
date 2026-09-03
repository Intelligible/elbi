import { useCallback, useEffect, useRef, useState } from "react"

import {
  type ConversationSummary,
  deleteConversation,
  getConversations,
  renameConversation,
} from "@/lib/chat"

// A plain-fetch paginated conversation list: it accumulates pages, exposes
// fetchNextPage/hasNextPage for infinite scroll, refresh() to reload from the first page
// (used after a turn creates or bumps a conversation), and search() to filter by title.
// The app has no react-query, so this holds the accumulated pages in useState, matching
// the plain-fetch style.
export function usePaginatedConversations(limit = 20) {
  const [conversations, setConversations] = useState<ConversationSummary[]>([])
  const [hasNextPage, setHasNextPage] = useState(false)
  const [isFetchingNextPage, setIsFetchingNextPage] = useState(false)
  const nextPageId = useRef<string | null>(null)
  const loading = useRef(false)
  // The active title filter, held in a ref so load()/fetchNextPage() read the current
  // value without being torn down and recreated as the query changes.
  const query = useRef("")

  const load = useCallback(
    async (pageId?: string) => {
      if (loading.current) return
      loading.current = true
      setIsFetchingNextPage(true)
      try {
        const page = await getConversations(limit, pageId, query.current || undefined)
        // No pageId means "reload from the top", so replace; otherwise append the page.
        setConversations((prev) => (pageId ? [...prev, ...page.items] : page.items))
        nextPageId.current = page.next_page_id
        setHasNextPage(page.next_page_id !== null)
      } finally {
        loading.current = false
        setIsFetchingNextPage(false)
      }
    },
    [limit],
  )

  const refresh = useCallback(() => {
    void load()
  }, [load])

  const fetchNextPage = useCallback(() => {
    if (nextPageId.current) void load(nextPageId.current)
  }, [load])

  // Set the title filter and reload from the top. An empty query clears the filter.
  const search = useCallback(
    (q: string) => {
      query.current = q
      void load()
    },
    [load],
  )

  // Delete a conversation, dropping its row from the list at once (optimistic) so the
  // sidebar feels responsive; if the request fails, reload from the top to restore the
  // true list rather than leaving the row wrongly hidden.
  const remove = useCallback(
    async (id: string) => {
      setConversations((prev) => prev.filter((c) => c.id !== id))
      const ok = await deleteConversation(id)
      if (!ok) void load()
    },
    [load],
  )

  // Rename a conversation, showing the new title at once; reload to restore truth on
  // failure. A loading guard means a rename mid-load can be dropped, so reload either
  // way keeps the list honest.
  const rename = useCallback(
    async (id: string, title: string) => {
      setConversations((prev) => prev.map((c) => (c.id === id ? { ...c, title } : c)))
      const ok = await renameConversation(id, title)
      if (!ok) void load()
    },
    [load],
  )

  // A brand-new conversation's row does not exist on the server until its first message
  // lands, and the list otherwise only reloads once that turn finishes (refresh(), called
  // from onFinish).
  const addPending = useCallback((id: string) => {
    setConversations((prev) => {
      if (prev.some((c) => c.id === id)) return prev
      const now = new Date().toISOString()
      return [{ id, title: "", created_at: now, updated_at: now }, ...prev]
    })
  }, [])

  useEffect(() => {
    void load()
  }, [load])

  return {
    conversations,
    hasNextPage,
    isFetchingNextPage,
    fetchNextPage,
    refresh,
    search,
    remove,
    rename,
    addPending,
  }
}
