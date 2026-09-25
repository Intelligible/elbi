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
Tailwind: `text-foreground` / `text-text-secondary` / `text-text-tertiary`.
Banned: `text-muted-foreground` (same value; app code uses `text-text-tertiary`) and `text-secondary` (`--secondary` is a background).

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

## Type scale

| Token | Value |
|---|---|
| `text-3xs` | 0.625rem (10px) |
| `text-2xs` | 0.6875rem (11px) |
| `text-compact` | 0.8125rem (13px) |
| `text-title` | 1.35rem |
| `text-display` | 1.7rem |

Plus Tailwind's own `text-xs` … `text-3xl`. Never an arbitrary `text-[…]`.

## Components

Primitives come from `components/ui/*` (shadcn over Radix). For the composed pieces,
reach for `components/app/FormField`, `IconButton`, `EmptyState` before hand-rolling one.
Raw `<button>` `<select>` `<input>` `<textarea>` `<table>` are not used in app code.

- **Select** is controlled with a string: `value={v}`, `""` shows the placeholder. A
  meaningful empty choice is an item with a named sentinel, mapped to `""` only at the
  `value` / `onValueChange` boundary. Label it via `FormField` + `FormControl` around the
  `SelectTrigger`.
- **Icons** in `xs` / `icon-xs` buttons carry `size-*`; without it the button sizes them to 12px.
- **Hit boxes** of adjacent icon buttons never overlap; keep negative-margin footprints to
  one axis.
- **State semantics**: `aria-pressed` on segmented-control options, `aria-current` on the
  active navigation item; every `TabsTrigger` has its `TabsContent`.
- **Parity**: moving markup onto a component accepts that component's defaults, and restores
  any layout the page loses.

## Enforcement

`npm run lint` runs Biome, then ESLint (`eslint/design-system`). `src/test/design-system.test.ts`
runs the same rules again in `npm test`, so a violation fails the suite, not only the editor.
Escapes are `// eslint-disable-next-line ds/<rule> -- <reason>` only.

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

- Consume semantic tokens (`bg-card`, `text-text-secondary`, `border-border`, …). Never a
  raw `--n-*`, hex, or `oklch(...)` literal in a component (charts excepted, via
  `chart-theme.ts`).
- A status = ink + its `-tint` fill. A verdict = `<VerdictBadge>`.
- Borders before shadows. Shadows only for floating surfaces.
