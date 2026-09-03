// Shared stylesheet + wrapper for the warehouse "Sources" surface. The layout, density and
// control shapes are derived from an external stylesheet, which NOTICE attributes. The
// *colours* are this app's own: a button here is the same cocoa as a button anywhere else, so
// the page reads as part of the product rather than as a borrowed screen. Scoped under `.phc`
// so nothing leaks the other way. Anything tied to the brand resolves through the app's tokens
// rather than being restated, which is also why the dark block below redefines only the
// neutrals: the accent follows `--primary`, and that already changes with the theme.

import type { ReactNode } from "react"

export const PHC_CSS = `
.phc {
  /* The canvas field tracks the app shell background, so this surface matches the rest
     of the app rather than carrying the upstream stylesheet's own tint. */
  --ph-bg: var(--panel);
  --ph-surface: #fff;
  --ph-text: #111;
  --ph-text-secondary: rgb(17 17 17 / 65%);
  --ph-muted: rgb(17 17 17 / 60%);
  --ph-border: hsl(78deg 13% 85%);
  --ph-border-strong: hsl(69deg 8% 65%);
  --ph-link: var(--primary);
  --ph-radius: 0.375rem;
  --ph-font: var(--font-sans);
  /* The primary button: the app's accent, kept in Lemon's raised shape; the frame is
     the 3px edge beneath it, so it wants the pressed state rather than the resting one. */
  --ph-primary-bg: var(--primary); --ph-primary-border: var(--primary); --ph-primary-frame: var(--accent-active); --ph-primary-border-hover: var(--accent-hover); --ph-primary-text: var(--primary-foreground);
  --ph-secondary-bg: #f3f4ef; --ph-secondary-border: #ccc; --ph-secondary-frame: #e1dddd; --ph-secondary-border-hover: #aaa;
  --ph-row-hover: rgb(0 0 0 / 3%);

  position: absolute; inset: 0; overflow-y: auto;
  background: var(--ph-bg); color: var(--ph-text);
  font-family: var(--ph-font); font-size: 14px; line-height: 1.4;
  -webkit-font-smoothing: antialiased;
}
.dark .phc {
  /* Only the neutrals: the accent follows --primary, which the theme already swaps. */
  --ph-bg: var(--panel);
  --ph-surface: hsl(235deg 8% 15%);
  --ph-text: hsl(0deg 0% 90%);
  --ph-text-secondary: hsl(0deg 0% 65%);
  --ph-muted: rgb(255 255 255 / 50%);
  --ph-border: hsl(230deg 8% 20%);
  --ph-border-strong: hsl(230deg 8% 40%);
  --ph-secondary-bg: #1d1f27; --ph-secondary-border: #4a4c52; --ph-secondary-frame: #323232; --ph-secondary-border-hover: #5e6064;
  --ph-row-hover: rgb(255 255 255 / 4%);
}
.phc * { box-sizing: border-box; }
.phc .wrap { padding: 1rem 1.25rem 3rem; }

/* Breadcrumb + scene title */
.phc .crumb { display: inline-flex; align-items: center; gap: 0.25rem; color: var(--ph-muted); font-size: 0.8125rem; font-weight: 600; background: none; border: 0; cursor: pointer; padding: 0.25rem 0; margin-bottom: 0.25rem; }
.phc .crumb:hover { color: var(--ph-text); }
.phc .title-row { display: flex; align-items: center; justify-content: space-between; gap: 1rem; margin-bottom: 1.25rem; }
.phc h1 { font-size: 1.5rem; font-weight: 700; letter-spacing: -0.01em; margin: 0; }

/* Buttons (LemonButton) */
.phc .btn { display: inline-flex; align-items: center; justify-content: center; gap: 0.375rem; height: 2.3125rem; padding: 0 0.75rem; border-radius: var(--ph-radius); font-family: var(--ph-font); font-size: 0.875rem; font-weight: 600; line-height: 1; cursor: pointer; user-select: none; white-space: nowrap; transition: transform 120ms ease, box-shadow 120ms ease, border-color 120ms ease; }
.phc .btn:disabled { opacity: 0.5; cursor: not-allowed; }
.phc .btn--sm { height: 2.0625rem; }
.phc .btn--secondary { background: var(--ph-secondary-bg); color: var(--ph-text); border: 1px solid var(--ph-secondary-border); box-shadow: 0 3px 0 -1px var(--ph-secondary-frame); }
.phc .btn--secondary:not(:disabled):hover { border-color: var(--ph-secondary-border-hover); transform: translateY(-0.5px); }
.phc .btn--secondary:not(:disabled):active { transform: translateY(3px); box-shadow: 0 0 0 0 var(--ph-secondary-frame); }
.phc .btn--primary { background: var(--ph-primary-bg); color: var(--ph-primary-text); border: 1px solid var(--ph-primary-border); box-shadow: 0 3px 0 -1px var(--ph-primary-frame); }
.phc .btn--primary:not(:disabled):hover { border-color: var(--ph-primary-border-hover); transform: translateY(-0.5px); }
.phc .btn--primary:not(:disabled):active { transform: translateY(3px); box-shadow: 0 0 0 0 var(--ph-primary-frame); }

/* Wizard header */
.phc .src-head { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 1rem; }
.phc .src-head h4 { font-size: 1.125rem; font-weight: 600; margin: 0; }
.phc .src-head p { font-size: 0.875rem; color: var(--ph-muted); margin: 0.125rem 0 0; }
.phc .caption { font-size: 0.875rem; color: var(--ph-text-secondary); margin: 0.5rem 0 0.75rem; line-height: 1.5; }
.phc a { color: var(--ph-link); text-decoration: none; }
.phc a:hover { text-decoration: underline; }
.phc .docs-row { display: flex; align-items: center; gap: 0.5rem; font-size: 0.8125rem; margin: 0 0 1rem; }

/* SceneSection */
.phc .check {
  display: flex; align-items: center; gap: 0.5rem; cursor: pointer;
  font-weight: 500; margin: 0;
}
.phc .check input[type="checkbox"] {
  width: 1rem; height: 1rem; margin: 0; cursor: pointer;
  accent-color: var(--ph-primary-bg);
}
.phc .section { margin-bottom: 1.75rem; }
.phc .section > h2 { font-size: 1rem; font-weight: 600; margin: 0 0 0.25rem; }
.phc .section > p { font-size: 0.875rem; color: var(--ph-text-secondary); margin: 0 0 0.75rem; max-width: 65ch; line-height: 1.5; }
.phc hr.divider { border: 0; border-top: 1px solid var(--ph-border); margin: 1.5rem 0; }

/* Fields (LemonField) */
.phc .field { display: flex; flex-direction: column; gap: 0.375rem; margin-bottom: 1rem; }
.phc .field > label { font-size: 0.8125rem; font-weight: 600; color: var(--ph-text); }
.phc .field .opt { color: var(--ph-muted); font-weight: 400; }
.phc .help { font-size: 0.75rem; color: var(--ph-muted); margin-top: 0.125rem; line-height: 1.4; }
.phc .upload-row { display: flex; align-items: center; gap: 0.625rem; margin-top: 0.125rem; }
.phc .upload-btn { cursor: pointer; }
.phc .upload-row .help { margin-top: 0; }
.phc .err-inline { font-size: 0.75rem; color: #b42318; line-height: 1.4; }
.dark .phc .err-inline { color: #f9a8a1; }

/* Inputs / selects (LemonInput) */
.phc .input, .phc .select {
  width: 100%; height: calc(2.125rem + 3px); padding: 0.25rem 0.5rem;
  background: var(--ph-surface); color: var(--ph-text);
  border: 1px solid var(--ph-border); border-radius: var(--ph-radius);
  font-size: 0.875rem; font-family: var(--ph-font); outline: none;
}
.phc textarea.input { height: auto; min-height: 4rem; resize: vertical; }
.phc .input::placeholder { color: var(--ph-muted); }
.phc .input:hover, .phc .select:hover, .phc .input:focus, .phc .select:focus { border-color: var(--ph-border-strong); }
.phc .select { appearance: none; cursor: pointer; background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='gray' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E"); background-repeat: no-repeat; background-position: right 0.5rem center; background-size: 1rem; padding-right: 2rem; }
.phc .search { max-width: 20rem; margin-bottom: 0.5rem; }

/* Table (LemonTable) */
.phc .tbl { width: 100%; border-collapse: collapse; font-size: 13px; background: var(--ph-surface); border: 1px solid var(--ph-border); border-radius: var(--ph-radius); overflow: hidden; }
.phc .tbl thead th { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.03125rem; font-weight: 600; color: var(--ph-text-secondary); text-align: left; padding: 0.5rem; white-space: nowrap; }
.phc .tbl tbody td { padding: 0.5rem; border-top: 1px solid var(--ph-border); color: var(--ph-text); vertical-align: middle; }
.phc .tbl tbody tr { cursor: pointer; }
.phc .tbl tbody tr:hover { background: var(--ph-row-hover); }
.phc .tbl .num { text-align: right; font-variant-numeric: tabular-nums; }
.phc .link-title { color: var(--ph-text); font-weight: 600; }
.phc .tbl tbody tr:hover .link-title { color: var(--ph-link); }
.phc .sub { font-size: 0.75rem; color: var(--ph-muted); }
.phc .empty { display: flex; flex-direction: column; align-items: center; gap: 0.5rem; padding: 2rem 1rem; color: var(--ph-text-secondary); font-size: 0.875rem; }

/* Tags (LemonTag) */
.phc .tag { display: inline-flex; align-items: center; gap: 0.25rem; padding: 0.0625rem 0.375rem; border-radius: var(--ph-radius); font-size: 0.75rem; font-weight: 500; line-height: 1.4; }
.phc .tag--muted { background: rgb(0 0 0 / 6%); color: var(--ph-text-secondary); }
.phc .tag--success { background: rgb(16 185 129 / 15%); color: #047857; }
.phc .tag--danger { background: rgb(180 35 24 / 15%); color: #b42318; }
.phc .tag--info { background: rgb(47 128 237 / 15%); color: #1d4ed8; }
.dark .phc .tag--muted { background: rgb(255 255 255 / 8%); }
.dark .phc .tag--success { color: #34d399; }
.dark .phc .tag--danger { color: #f9a8a1; }
.dark .phc .tag--info { color: #93c5fd; }

/* Error banner + footer */
.phc .err { background: #fde8e8; color: #b42318; border: 1px solid #f4c7c3; border-radius: var(--ph-radius); padding: 0.5rem 0.75rem; margin-bottom: 1rem; font-size: 0.8125rem; }
.dark .phc .err { background: rgb(180 35 24 / 15%); color: #f9a8a1; border-color: rgb(180 35 24 / 40%); }
.phc .section {
  font-size: 0.8125rem; font-weight: 600; letter-spacing: 0.01em;
  color: var(--ph-muted); text-transform: none;
  margin: 1.5rem 0 0.25rem; padding-top: 1rem;
  border-top: 1px solid var(--ph-border);
}
.phc .footer { display: flex; justify-content: flex-end; gap: 0.5rem; margin: 1rem 0; }
`

/** Full-panel surface (absolute inset-0), with the scoped stylesheet applied. */
export function Phc({ children }: { children: ReactNode }) {
  return (
    <div className="phc">
      <style>{PHC_CSS}</style>
      <div className="wrap">{children}</div>
    </div>
  )
}
