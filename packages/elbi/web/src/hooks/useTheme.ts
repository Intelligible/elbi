import { useCallback, useEffect, useState } from "react"

// Light / dark / system theme, applied by toggling `.dark` on the document root (which the CSS
// keys its color tokens off). The choice is stored so it survives reloads; the no-flash init in
// index.html applies it before React mounts, and this hook keeps it in sync afterwards:
// including following the OS when the choice is "system".

export type Theme = "light" | "dark" | "system"

const STORAGE_KEY = "elbi-theme"

function prefersDark(): boolean {
  return window.matchMedia("(prefers-color-scheme: dark)").matches
}

function apply(theme: Theme): void {
  const dark = theme === "dark" || (theme === "system" && prefersDark())
  document.documentElement.classList.toggle("dark", dark)
}

function stored(): Theme {
  const value = localStorage.getItem(STORAGE_KEY)
  return value === "light" || value === "dark" ? value : "system"
}

export function useTheme(): {
  theme: Theme
  resolved: "light" | "dark"
  setTheme: (theme: Theme) => void
} {
  const [theme, setThemeState] = useState<Theme>(stored)

  useEffect(() => apply(theme), [theme])

  // When following the system, react to the OS switching between light and dark.
  useEffect(() => {
    if (theme !== "system") return
    const media = window.matchMedia("(prefers-color-scheme: dark)")
    const onChange = () => apply("system")
    media.addEventListener("change", onChange)
    return () => media.removeEventListener("change", onChange)
  }, [theme])

  const setTheme = useCallback((next: Theme) => {
    localStorage.setItem(STORAGE_KEY, next)
    setThemeState(next)
  }, [])

  const resolved = theme === "system" ? (prefersDark() ? "dark" : "light") : theme
  return { theme, resolved, setTheme }
}
