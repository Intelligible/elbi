"""Render a derivation's page as one self-contained, read-only HTML snapshot.

The snapshot carries what the derivation page shows, in the same order: question,
verdict, data hash, assumptions, finding, output, verification checks, result history,
claim and source. It is for someone without access to the app. Raw HTML inside the
markdown sections is sanitized the way the app's renderer does it, and behind that the
page carries no script and its Content-Security-Policy forbids every request.
"""

from __future__ import annotations

import html
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

import nh3
from markdown_it import MarkdownIt

_CSP = (
    "default-src 'none'; img-src data:; style-src 'unsafe-inline'; "
    "base-uri 'none'; form-action 'none'"
)

# GFM tables and strikethrough, which is what the app's own renderer (Streamdown) reads.
_md = MarkdownIt("commonmark").enable("table").enable("strikethrough")

# Table cells carry synced data verbatim, so raw HTML in an output is untrusted. nh3's
# allowlist keeps safe markup (<b>, links) and drops anything that could run, navigate
# or restyle the page (<script>, <meta>, <style> and its contents), as the app does.
# The one addition: markdown-it aligns a GFM table column with an inline text-align.
_ATTRIBUTES = {
    **nh3.ALLOWED_ATTRIBUTES,
    "th": nh3.ALLOWED_ATTRIBUTES["th"] | {"style"},
    "td": nh3.ALLOWED_ATTRIBUTES["td"] | {"style"},
}


# The app's verdict badge labels (web/src/components/VerdictBadge.tsx), so the page
# names a verdict as the derivation page does. An unknown verdict shows as-is, as there.
_VERDICT_LABELS = {
    "sound": "Verified",
    "verified": "Verified",
    "computed": "Computed",
    "inconclusive": "Inconclusive",
    "unsound": "Not sound",
    "invalid": "Invalid",
    "unverified": "Unverified",
}


def _markdown(text: str) -> str:
    return nh3.clean(
        _md.render(text), attributes=_ATTRIBUTES, filter_style_properties={"text-align"}
    )


def render(
    *,
    name: str,
    title: str,
    output: str = "",
    finding: str = "",
    verdict: str | None = None,
    exported_at: datetime,
    question: str = "",
    data_hash: str | None = None,
    assumptions: Sequence[str] = (),
    checks: Sequence[Mapping[str, Any]] = (),
    history: Sequence[Mapping[str, Any]] = (),
    claim: Mapping[str, Any] | None = None,
    source: str = "",
) -> str:
    """The snapshot page for a derivation, as HTML text."""
    # In UTC and said so: the page runs no script, so it cannot localize for a reader.
    at = exported_at.astimezone(timezone.utc)
    verified = verdict == "sound"
    # No verdict, no label: the app shows no badge then either.
    label = _label(verdict)
    sections = []
    if data_hash:
        sections.append(
            _plain_section("Data hash", f'<code class="hash">{_e(data_hash)}</code>')
        )
    if assumptions:
        items = "".join(f"<li>{_e(a)}</li>" for a in assumptions)
        sections.append(
            _plain_section(
                "Sound under stated assumptions", f"<ul>{items}</ul>", "caution"
            )
        )
    if finding:
        sections.append(_section("Finding", finding))
    if output:
        sections.append(_section("Verified output" if verified else "Output", output))
    if checks:
        sections.append(
            _plain_section(
                f"Verification · {len(checks)} checks run by the oracle",
                '<ul class="rows">' + "".join(_check(c) for c in checks) + "</ul>",
            )
        )
    if history:
        n = len(history)
        sections.append(
            _plain_section(
                f"Result history · {n} version{'' if n == 1 else 's'}",
                '<ul class="rows">' + "".join(_version(v) for v in history) + "</ul>",
            )
        )
    if claim:
        note = "" if verdict else '<p class="meta">Not checked by the oracle.</p>'
        body = f"<pre>{_e(json.dumps(claim, indent=2))}</pre>{note}"
        sections.append(_plain_section("Claim", body, "code"))
    if source:
        sections.append(_plain_section("Source", f"<pre>{_e(source)}</pre>", "code"))
    return _TEMPLATE.format(
        csp=_CSP,
        title=_e(title),
        name=_e(name),
        question=f'<p class="question">{_e(question)}</p>' if question else "",
        status=f"{_e(label)}. " if label else "",
        # at.day, not %-d: the unpadded-day directive does not exist on Windows.
        exported_at=f"{at.day} {at:%b %Y, %H:%M} UTC",
        exported_iso=at.isoformat(timespec="seconds"),
        body="".join(sections),
    )


def _e(text: object) -> str:
    return html.escape(str(text))


def _label(verdict: str | None) -> str:
    return _VERDICT_LABELS.get(verdict.lower(), verdict) if verdict else ""


def _section(heading: str, markdown: str) -> str:
    return (
        f'<section><h2 class="label">{heading}</h2>'
        f'<div class="card md">{_markdown(markdown)}</div></section>'
    )


def _plain_section(heading: str, inner: str, kind: str = "") -> str:
    """A section whose body is already-escaped HTML, not markdown."""
    return (
        f'<section><h2 class="label">{_e(heading)}</h2>'
        f'<div class="card {kind}">{inner}</div></section>'
    )


def _check(check: Mapping[str, Any]) -> str:
    verdict = str(check.get("verdict") or "")
    mark = {"sound": "✓", "unsound": "✗"}.get(verdict, "◦")
    label, detail = _e(check.get("name", "")), _e(check.get("detail", ""))
    return (
        f'<li><span class="mark {_e(verdict)}">{mark}</span>'
        f"<span><b>{label}</b>: {detail}</span></li>"
    )


def _version(run: Mapping[str, Any]) -> str:
    """One certified version, as the history section shows it."""
    estimate = run.get("estimate")
    value = "" if estimate is None else f"{estimate:,.4g}"
    shown = (
        " ".join(x for x in (value, str(run.get("estimate_label") or "")) if x) or "—"
    )
    changed = list(run.get("changed") or [])
    tags = (
        "".join(f'<span class="tag">{_e(d)} changed</span>' for d in changed)
        if changed
        else '<span class="meta">first certified version</span>'
    )
    created = str(run.get("created_at") or "")[:10]
    return (
        f"<li><code>{_e(run.get('short_version', ''))}</code>"
        f'<span class="badge">{_e(_label(run.get("verdict")))}</span>'
        f'<span class="grow">{_e(shown)}</span><span class="meta">{_e(created)}</span>'
        f'<div class="tags">{tags}</div></li>'
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
    --primary: oklch(0.42 0.062 50); --verified: oklch(0.55 0.13 150);
    --danger: oklch(0.56 0.2 26); --caution: oklch(0.7 0.14 75);
    --sans: ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif;
    --mono: ui-monospace, "SF Mono", Menlo, monospace;
  }}
  @media (prefers-color-scheme: dark) {{
    :root {{
      color-scheme: dark;
      --bg: oklch(0.185 0 0); --card: oklch(0.225 0 0); --fg: oklch(0.97 0 0);
      --text-3: oklch(0.66 0 0); --border: oklch(0.29 0 0); --code-bg: oklch(0.2 0 0);
      --primary: oklch(0.72 0.088 65); --verified: oklch(0.72 0.13 150);
      --danger: oklch(0.7 0.17 26); --caution: oklch(0.78 0.13 75);
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
  .question {{ margin: 2px 0 6px; font-size: 14px; }}
  .card.code {{ background: var(--code-bg); }}
  .card pre {{ margin: 0; font: 12px/1.5 var(--mono); white-space: pre-wrap;
    word-break: break-word; }}
  .card ul {{ margin: 0; padding-left: 18px; }}
  .card.caution {{ border-color: var(--caution); }}
  .hash {{ font: 12px var(--mono); word-break: break-all; color: var(--text-3); }}
  ul.rows {{ list-style: none; padding: 0; margin: -16px; }}
  ul.rows li {{ display: flex; flex-wrap: wrap; align-items: center; gap: 10px;
    padding: 10px 14px; border-top: 1px solid var(--border); }}
  ul.rows li:first-child {{ border-top: 0; }}
  .mark {{ color: var(--text-3); }}
  .mark.sound {{ color: var(--verified); }}
  .mark.unsound {{ color: var(--danger); }}
  .badge {{ font-size: 11px; padding: 1px 8px; border-radius: 999px;
    background: var(--code-bg); }}
  .grow {{ flex: 1; min-width: 0; }}
  .tags {{ flex-basis: 100%; display: flex; gap: 4px; }}
  .tag {{ font-size: 11px; padding: 1px 8px; border-radius: 999px;
    background: var(--code-bg); color: var(--text-3); }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <p class="eyebrow">Elbi derivation · {name}</p>
  <h1 class="title">{title}</h1>
  {question}
  <p class="meta">{status}Exported
    <time datetime="{exported_iso}">{exported_at}</time>.
    A read-only snapshot of the derivation page.</p>
</header>
{body}
</div>
</body>
</html>
"""
