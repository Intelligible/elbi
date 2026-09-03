import { type ClassValue, clsx } from "clsx"
import { twMerge } from "tailwind-merge"

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs))
}

// What a table cell shows when it has no value. An em dash reads as "no value" where a
// hyphen would read as a minus sign in a numeric column.
export const EMPTY = "—"

// A compact relative time ("just now", "5m ago", "3d ago") for freshness columns.
export function relativeTime(iso: string | null): string {
  if (!iso) return "never"
  const then = new Date(iso).getTime()
  if (Number.isNaN(then)) return "never"
  const seconds = Math.round((Date.now() - then) / 1000)
  if (seconds < 45) return "just now"
  const minutes = seconds / 60
  if (minutes < 60) return `${Math.round(minutes)}m ago`
  const hours = minutes / 60
  if (hours < 24) return `${Math.round(hours)}h ago`
  const days = hours / 24
  if (days < 7) return `${Math.round(days)}d ago`
  return `${Math.round(days / 7)}w ago`
}

// A v4 UUID, including where `crypto.randomUUID` does not exist. `randomUUID` is restricted to
// secure contexts, and a self-hosted deployment reached over plain HTTP on an internal address
// is not one, which is the normal shape behind a load balancer that terminates nothing. There
// the call is not merely unavailable, it throws during render and takes the page down with it.
// `getRandomValues` carries no such restriction, so the fallback is the same randomness reached
// by a different name rather than a downgrade to `Math.random`.
export function uuid(): string {
  if (typeof crypto.randomUUID === "function") return crypto.randomUUID()
  const bytes = crypto.getRandomValues(new Uint8Array(16))
  bytes[6] = (bytes[6] & 0x0f) | 0x40 // version 4
  bytes[8] = (bytes[8] & 0x3f) | 0x80 // variant 1
  const hex = Array.from(bytes, (b) => b.toString(16).padStart(2, "0")).join("")
  return [
    hex.slice(0, 8),
    hex.slice(8, 12),
    hex.slice(12, 16),
    hex.slice(16, 20),
    hex.slice(20),
  ].join("-")
}

// Copy to the clipboard, including where `navigator.clipboard` does not exist.
//
// Secure-context-only for the same reason as `uuid` above, so on a plaintext internal
// address every copy button silently does nothing. `execCommand("copy")` is deprecated
// and is the only thing that works there; it is tried second, never first.
export async function copyText(text: string): Promise<boolean> {
  if (navigator.clipboard?.writeText) {
    try {
      await navigator.clipboard.writeText(text)
      return true
    } catch {
      // Fall through: permission can be refused even in a secure context.
    }
  }
  const staging = document.createElement("textarea")
  staging.value = text
  // Off-screen rather than hidden: a display:none element cannot be selected.
  staging.style.position = "fixed"
  staging.style.top = "-9999px"
  document.body.appendChild(staging)
  staging.select()
  try {
    return document.execCommand("copy")
  } catch {
    return false
  } finally {
    staging.remove()
  }
}
