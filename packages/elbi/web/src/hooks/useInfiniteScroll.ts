import { useCallback, useEffect, useRef } from "react"

// Attach the returned ref to a scroll container; when the user scrolls within
// `threshold` px of the bottom, load the next page. Dependency-free, so the sidebar's
// conversation history pages in as it is scrolled.
export function useInfiniteScroll({
  hasNextPage,
  isFetchingNextPage,
  fetchNextPage,
  threshold = 120,
}: {
  hasNextPage: boolean
  isFetchingNextPage: boolean
  fetchNextPage: () => void
  threshold?: number
}) {
  const containerRef = useRef<HTMLDivElement>(null)

  const handleScroll = useCallback(() => {
    const el = containerRef.current
    if (!el || isFetchingNextPage || !hasNextPage) return
    const { scrollTop, scrollHeight, clientHeight } = el
    if (scrollTop + clientHeight >= scrollHeight - threshold) fetchNextPage()
  }, [hasNextPage, isFetchingNextPage, fetchNextPage, threshold])

  useEffect(() => {
    const el = containerRef.current
    if (!el) return undefined
    el.addEventListener("scroll", handleScroll)
    return () => el.removeEventListener("scroll", handleScroll)
  }, [handleScroll])

  return containerRef
}
