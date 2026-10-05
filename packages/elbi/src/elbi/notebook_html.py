"""Render a notebook's ``.ipynb`` payload as one self-contained, read-only HTML page.

The notebook counterpart of a dashboard snapshot: something to hand to a person with no
access to the app. The input is :meth:`NotebookService.export_ipynb`'s payload, so the
page carries outputs exactly when that export does and the deployment's
``NOTEBOOK_EXPORT_OUTPUTS`` decision holds for HTML too.

Rendered server-side, so the page needs no script of its own, and its
Content-Security-Policy forbids every script. That is what makes embedding kernel HTML
safe without a sanitizer: a DataFrame's ``_repr_html_`` renders, a ``<script>`` or an
``onerror=`` in an output does nothing. The app gets the same guarantee from DOMPurify.
"""

from __future__ import annotations

import html
import re
from collections.abc import Mapping
from typing import Any

from markdown_it import MarkdownIt

_CSP = (
    "default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
    "base-uri 'none'; form-action 'none'"
)

#: Kernels colour tracebacks and some streams with ANSI escapes; a page shows them raw.
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

_md = MarkdownIt("commonmark").enable("table").enable("strikethrough")


def _text(value: Any) -> str:
    """A multiline nbformat field, which is stored as a string or a list of lines."""
    return "".join(value) if isinstance(value, list) else str(value or "")


def _plain(value: Any) -> str:
    """Terminal text for a <pre>: ANSI escapes dropped, then HTML-escaped."""
    return html.escape(_ANSI.sub("", _text(value)))


def _render_data(data: Mapping[str, Any]) -> str:
    """The richest MIME type the page can show, in the same order the app prefers."""
    if "text/html" in data:
        return f'<div class="out-html">{_text(data["text/html"])}</div>'
    if "image/svg+xml" in data:
        return f'<div class="out-html">{_text(data["image/svg+xml"])}</div>'
    for mime in ("image/png", "image/jpeg"):
        if mime in data:
            src = f"data:{mime};base64,{_text(data[mime]).strip()}"
            return f'<img class="out-img" alt="" src="{html.escape(src)}">'
    if "text/markdown" in data:
        return f'<div class="md">{_md.render(_text(data["text/markdown"]))}</div>'
    if "text/plain" in data:
        return f"<pre>{html.escape(_text(data['text/plain']))}</pre>"
    return ""


def _render_output(output: Mapping[str, Any]) -> str:
    kind = output.get("output_type")
    if kind == "stream":
        stream = "stderr" if output.get("name") == "stderr" else "stdout"
        return f'<pre class="{stream}">{_plain(output.get("text"))}</pre>'
    if kind == "error":
        trace = "\n".join(_text(line) for line in output.get("traceback") or [])
        body = trace or f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
        return f'<pre class="stderr">{_plain(body)}</pre>'
    return _render_data(output.get("data") or {})


def _render_cell(cell: Mapping[str, Any]) -> str:
    source = _text(cell.get("source"))
    if cell.get("cell_type") == "markdown":
        return f'<section class="cell md">{_md.render(source)}</section>'
    if cell.get("cell_type") != "code":
        return f'<section class="cell"><pre>{html.escape(source)}</pre></section>'
    count = cell.get("execution_count")
    prompt = f"[{count}]" if isinstance(count, int) else "[ ]"
    outputs = "".join(_render_output(o) for o in cell.get("outputs") or [])
    return (
        '<section class="cell code">'
        f'<div class="prompt">{prompt}</div>'
        f'<pre class="src"><code>{html.escape(source)}</code></pre>'
        + (f'<div class="outputs">{outputs}</div>' if outputs else "")
        + "</section>"
    )


def render(payload: Mapping[str, Any], *, title: str, exported_at: str) -> str:
    """The page for a notebook's ``.ipynb`` payload, as HTML text."""
    cells = payload.get("cells") or []
    has_outputs = any(c.get("outputs") for c in cells)
    note = (
        ""
        if has_outputs
        else '<p class="note">Code only: this deployment does not export cell '
        "outputs.</p>"
    )
    body = "".join(_render_cell(c) for c in cells)
    return _TEMPLATE.format(
        csp=_CSP,
        title=html.escape(title),
        exported_at=html.escape(exported_at),
        note=note,
        body=body,
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
    --primary: oklch(0.42 0.062 50); --danger: oklch(0.56 0.2 26);
    --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    --mono: ui-monospace, "SF Mono", Menlo, monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --bg: oklch(0.185 0 0); --card: oklch(0.225 0 0); --fg: oklch(0.97 0 0);
      --text-3: oklch(0.66 0 0); --border: oklch(0.29 0 0); --code-bg: oklch(0.2 0 0);
      --primary: oklch(0.72 0.088 65); --danger: oklch(0.7 0.17 26);
    }}
  }}
  body {{ margin: 0; background: var(--bg); color: var(--fg);
    font: 14px/1.55 var(--sans); }}
  .wrap {{ max-width: 960px; margin: 0 auto; padding: 28px 16px 48px; }}
  header {{ margin-bottom: 20px; }}
  .eyebrow {{ font: 500 11px/1 var(--mono); letter-spacing: .08em;
    text-transform: uppercase; color: var(--primary); }}
  h1.title {{ margin: 8px 0 4px; font-size: 22px; font-weight: 600; }}
  .meta, .note {{ color: var(--text-3); font-size: 12px; margin: 0; }}
  .cell {{ background: var(--card); border: 1px solid var(--border); border-radius: 8px;
    padding: 12px 16px; margin-bottom: 12px; overflow-x: auto; }}
  .cell.code {{ position: relative; }}
  .prompt {{ font: 11px var(--mono); color: var(--text-3); margin-bottom: 6px; }}
  pre {{ margin: 0; font: 12.5px/1.5 var(--mono); white-space: pre-wrap;
    word-break: break-word; }}
  pre.src {{ background: var(--code-bg); border-radius: 6px; padding: 10px 12px; }}
  .outputs {{ margin-top: 10px; padding-top: 10px; border-top: 1px dashed var(--border);
    display: grid; gap: 8px; }}
  .stderr {{ color: var(--danger); }}
  .out-img {{ max-width: 100%; height: auto; }}
  .out-html {{ overflow-x: auto; contain: paint; }}
  .out-html table, .md table {{ border-collapse: collapse; font-size: 12.5px; }}
  .out-html th, .out-html td, .md th, .md td {{ border: 1px solid var(--border);
    padding: 4px 8px; text-align: right; }}
  .md :first-child {{ margin-top: 0; }}
  .md :last-child {{ margin-bottom: 0; }}
  .md code {{ font-family: var(--mono); font-size: 12.5px; background: var(--code-bg);
    padding: 1px 4px; border-radius: 4px; }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <div class="eyebrow">Elbi notebook</div>
  <h1 class="title">{title}</h1>
  <p class="meta">Exported {exported_at}. A read-only copy; nothing here runs.</p>
  {note}
</header>
{body}
</div>
</body>
</html>
"""
