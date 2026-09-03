# Design system

The contract behind the UI. The authoritative values live in `src/index.css` as CSS
custom properties, exposed to Tailwind utilities via `@theme inline`. Components must
consume the **semantic** tokens (below), never the raw ramp or hardcoded colors: that
is what keeps light/dark, states, and every surface reading as one family.

## Color

**Neutral ramp**: the spine of the UI. A true-grey OKLCH ramp `--n-25 … --n-950`
(chroma 0). Never used directly in components; it backs the semantic tokens.

**Surfaces**: layered page → panel → card, climbing toward the reader:
`--background` (canvas) · `--panel` (shell field) · `--card` / `--popover` (white, lifted)
· `--surface-secondary` · `--surface-tertiary`.

**Text: three tiers**, not a foreground/muted binary:
`--foreground` (primary) · `--text-secondary` · `--text-tertiary`.
Tailwind: `text-foreground` / `text-secondary` / `text-tertiary`.

**Brand accent**: the mark's cocoa; states are lightness shifts that warm toward its
gold layer, never new colors: `--primary` → `--accent-hover` → `--accent-active`.

**Borders, two weights**: `--border` (quiet hairline, the default) and
`--border-strong` (structural). Depth is borders-first (see below), so reach for a
border before a shadow.

**Semantics**: each ships with a low-opacity `-tint` companion, so a status is always
{saturated ink + soft fill} and is theme-adaptive via `color-mix`:
`--success` / `--warning` / `--danger` / `--info` + `--*-tint`.

**Verdict**: the product's signature vocabulary, mapped onto the semantics so a word
means one thing everywhere: `--verified` (= success) · `--caution` (= warning). Only a
certified-sound run wears the verified tint; the UI never dresses an unproven run as
verified. Render via `<VerdictBadge>`, never an ad-hoc badge.

**Data-viz palette**: 15 accent-anchored, evenly-spaced hues `--data-1 … --data-15`,
tuned separately for light and dark. **Charts must draw from this palette**, not the
renderer's defaults: go through `lib/chart-theme.ts` (`vegaThemeConfig()`), which
resolves the tokens for the active theme and sets the chart's color range + chrome.

## Radius

Base `--radius: 0.375rem`. Scale via `rounded-sm|md|lg|xl` → `--radius-sm` 0.25 /
`-md` 0.375 / `-lg` 0.5 / `-xl` 0.625rem.

## Typography

`--font-sans` = Geist, `--font-mono` = Geist Mono (Tailwind `font-sans` / `font-mono`).

## Depth & elevation: borders-first, shadows rare

Separation comes from surface + border first; shadows are reserved for genuinely
floating things. Three resting/elevation tokens:
`--shadow-panel` (shallow resting lift for content panels) ·
`--shadow-elevation` (popovers, menus, dropdowns) · `--shadow-modal` (dialogs, with
`--modal-backdrop`). Buttons get a tactile `--chrome-depth` (0.1875rem bottom depth,
collapsed on press: the Lemon-UI key-press feel).

## Spacing

Tailwind's default 4px base scale. Prefer utilities over ad-hoc pixel values.

## Dark mode

`.dark` on the root overrides the same semantic tokens (and the data palette). Because
components read only semantic tokens, dark mode is automatic; never branch on theme in
a component (charts read the resolved tokens at runtime instead).

## Rules

- Consume semantic tokens (`bg-card`, `text-secondary`, `border-border`, …). Never a
  raw `--n-*`, hex, or `oklch(...)` literal in a component (charts excepted, via
  `chart-theme.ts`).
- A status = ink + its `-tint` fill. A verdict = `<VerdictBadge>`.
- Borders before shadows. Shadows only for floating surfaces.
