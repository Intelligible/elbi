import { useRef } from "react"

/**
 * Stable React keys for rows that carry no identity of their own.
 *
 * A query result is a list of plain objects: no id column is promised, and two rows can
 * be field-for-field identical. Position is the usual stand-in, but position is exactly
 * what changes when a list is filtered or sorted, and React then reuses the previous
 * occupant's DOM for whatever lands at that index.
 *
 * What the rows do have is object identity. The fetch that produced them allocated each
 * one, and a later fetch allocates new ones, which is precisely the lifetime a key
 * should track. So keys are handed out per object and remembered in a `WeakMap`: the
 * same row keeps its key for as long as it is rendered, a re-fetch produces new rows and
 * new keys, and rows dropped from the list are collected without bookkeeping.
 *
 * Cheaper than wrapping: no row is copied, no downstream type gains a field, and the
 * lookup is a hash on the reference.
 */
export function useRowKeys(): (row: object) => string {
  const keys = useRef(new WeakMap<object, string>())
  const next = useRef(0)
  return (row: object) => {
    const seen = keys.current.get(row)
    if (seen !== undefined) return seen
    const minted = `r${next.current++}`
    keys.current.set(row, minted)
    return minted
  }
}
