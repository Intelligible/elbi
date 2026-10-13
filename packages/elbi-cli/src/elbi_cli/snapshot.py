"""Render a dashboard export as one self-contained, read-only HTML page.

The input is the app's own export document (``/api/exports/dashboards/{id}``): the spec
plus every page's resolved values. Only the published spec (or the draft, for a
dashboard never published) and those values reach the page. The version history and
the other spec stay behind, so a snapshot shares what the dashboard shows and nothing
it once showed.

The page carries no dependency and makes no request. Charts are drawn from the widget's
vega-lite encoding by a small renderer covering bar and line marks; any other chart
falls back to its data as a table, so a widget is never silently blank.
"""

from __future__ import annotations

import html
import json
import math
import re
from importlib import resources
from typing import Any

#: The fields of a resolved widget that describe what was shown and where it came from.
#: ``format`` is the bound metric's own display format, which a metric tile renders in.
_VALUE_FIELDS = ("value", "error", "kind", "derivation", "data_version", "format")

#: The fields of a widget spec the page needs to draw it. A metric tile reads its value
#: by ``bind.metric``; a filter tile names its ``variable``.
_WIDGET_FIELDS = (
    "id",
    "type",
    "title",
    "content",
    "gridPos",
    "viz",
    "bind",
    "variable",
)

#: What JSON inside a <script> must not hold raw: "<", since "</script" ends the
#: element, and the rest of the set Rails' json_escape uses (">", "&", U+2028, U+2029).
_SCRIPT_ESCAPES = {ord(c): f"\\u{ord(c):04x}" for c in "<>&\u2028\u2029"}


def _finite(value: Any) -> Any:
    """``value`` with every NaN and infinity as ``None``, which the page shows as empty.

    JSON has neither, so the bare ``NaN`` that ``json.dumps`` writes by default would
    make the page's ``JSON.parse`` reject the whole payload and leave the page blank.
    """
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def build(document: dict[str, Any]) -> dict[str, Any]:
    """The snapshot payload: pages, their widgets, and each widget's resolved value."""
    published = document.get("status") == "published"
    spec = (
        (document.get("published_spec") if published else None)
        or document.get("spec")
        or {}
    )
    values = document.get("values") or {}
    pages = []
    for page in spec.get("pages", []):
        resolved = {str(w.get("widget_id")): w for w in values.get(page["name"], [])}
        widgets = []
        for widget in page.get("widgets", []):
            out = {k: widget[k] for k in _WIDGET_FIELDS if k in widget}
            value = resolved.get(str(widget.get("id")))
            if value is not None:
                out["data"] = {k: value.get(k) for k in _VALUE_FIELDS}
            widgets.append(out)
        pages.append(
            {
                "name": page["name"],
                "title": page.get("title") or page["name"],
                "columns": page.get("columns"),
                "widgets": widgets,
            }
        )
    title = document.get("title")
    version = document.get("version")
    updated = document.get("updated_at")
    if published:
        # A draft save moves the row's title, version and edit time, but the page shows
        # the published spec, so it names that version (``versions`` is newest first).
        title = spec.get("title") or spec.get("name") or title
        shown = next(
            (
                v
                for v in document.get("versions") or []
                if v.get("label") == "published"
            ),
            None,
        )
        if shown is not None:
            version = shown.get("version", version)
            updated = shown.get("created_at") or updated
    return {
        "name": document.get("name"),
        "title": title or spec.get("title") or document.get("name"),
        "description": spec.get("description"),
        "version": version,
        "status": document.get("status"),
        "updatedAt": updated,
        "snapshotAt": document.get("created_at"),
        "hasVariables": bool(spec.get("variables")),
        "variables": {v["name"]: v.get("default") for v in spec.get("variables") or []},
        "pages": pages,
    }


def render(document: dict[str, Any]) -> str:
    """The snapshot page for an export document, as HTML text."""
    payload = build(document)
    template = (
        resources.files("elbi_cli")
        .joinpath("snapshot_template.html")
        .read_text("utf-8")
    )
    # Each escape is a JSON string escape, so JSON.parse reads back the same text.
    data = json.dumps(_finite(payload), ensure_ascii=False, allow_nan=False).translate(
        _SCRIPT_ESCAPES
    )
    # One pass, so a title that happens to read "__DATA__" is not itself substituted.
    fields = {"__TITLE__": html.escape(str(payload["title"])), "__DATA__": data}
    return re.sub("__TITLE__|__DATA__", lambda m: fields[m.group()], template)
