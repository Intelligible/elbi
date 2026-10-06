"""Render a derivation's output as one self-contained, read-only HTML page.

The derivation page shows its finding and output as markdown; this is that same
markdown as a file someone without access to the app can open. The page carries no
script and its Content-Security-Policy forbids every request, so raw HTML inside the
markdown renders but cannot run or fetch anything.
"""

from __future__ import annotations

import html

from markdown_it import MarkdownIt

_CSP = (
    "default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
    "base-uri 'none'; form-action 'none'"
)

# GFM tables and strikethrough, which is what the app's own renderer (Streamdown) reads.
_md = MarkdownIt("commonmark").enable("table").enable("strikethrough")


def render(
    *,
    name: str,
    title: str,
    output: str,
    finding: str = "",
    verdict: str | None = None,
    exported_at: str,
) -> str:
    """The page for a derivation's rendered output, as HTML text."""
    verified = verdict == "sound"
    sections = []
    if finding:
        sections.append(_section("Finding", finding))
    sections.append(_section("Verified output" if verified else "Output", output))
    return _TEMPLATE.format(
        csp=_CSP,
        title=html.escape(title),
        name=html.escape(name),
        status="Verified" if verified else "Not verified",
        exported_at=html.escape(exported_at),
        body="".join(sections),
    )


def _section(heading: str, markdown: str) -> str:
    return (
        f'<section><h2 class="label">{heading}</h2>'
        f'<div class="card md">{_md.render(markdown)}</div></section>'
    )


# Elbi's own tokens (web/src/index.css), so the page reads as the app. System fonts
# rather than Geist: the CSP allows no request, fonts included.
_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="{csp}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{
    --bg: oklch(0.972 0 0); --card: #ffffff; --fg: oklch(0.185 0 0);
    --text-3: oklch(0.535 0 0); --border: oklch(0.902 0 0); --code-bg: oklch(0.955 0 0);
    --primary: oklch(0.42 0.062 50);
    --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    --mono: ui-monospace, "SF Mono", Menlo, monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --bg: oklch(0.185 0 0); --card: oklch(0.225 0 0); --fg: oklch(0.97 0 0);
      --text-3: oklch(0.66 0 0); --border: oklch(0.29 0 0); --code-bg: oklch(0.2 0 0);
      --primary: oklch(0.72 0.088 65);
    }}
  }}
  body {{ margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.6 var(--sans); }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: 28px 16px 48px; }}
  header {{ margin-bottom: 22px; }}
  .eyebrow, .label {{ font: 500 11px/1 var(--mono); letter-spacing: .08em;
    text-transform: uppercase; color: var(--primary); margin: 0; }}
  h1.title {{ margin: 8px 0 4px; font-size: 22px; font-weight: 600; }}
  .meta {{ color: var(--text-3); font-size: 12px; margin: 0; }}
  section {{ margin-bottom: 18px; }}
  .label {{ margin-bottom: 8px; }}
  .card {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 16px; overflow-x: auto; }}
  .md > :first-child {{ margin-top: 0; }}
  .md > :last-child {{ margin-bottom: 0; }}
  .md h1 {{ font-size: 20px; }}
  .md h2 {{ font-size: 17px; }}
  .md h3 {{ font-size: 15px; }}
  .md table {{ border-collapse: collapse; font-size: 13px; margin: 12px 0; }}
  .md th, .md td {{ border: 1px solid var(--border); padding: 5px 10px;
    text-align: left; }}
  .md th {{ background: var(--code-bg); font-weight: 600; }}
  .md code {{ font-family: var(--mono); font-size: 12.5px; background: var(--code-bg);
    padding: 1px 4px; border-radius: 4px; }}
  .md pre {{ background: var(--code-bg); padding: 10px 12px; border-radius: 6px;
    overflow-x: auto; }}
  .md pre code {{ background: none; padding: 0; }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <p class="eyebrow">Elbi derivation · {name}</p>
  <h1 class="title">{title}</h1>
  <p class="meta">{status}. Exported {exported_at}. A read-only copy of the output.</p>
</header>
{body}
</div>
</body>
</html>
"""
