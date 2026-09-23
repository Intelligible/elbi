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
from importlib import resources
from typing import Any

#: The fields of a resolved widget that describe what was shown and where it came from.
_VALUE_FIELDS = ("value", "error", "kind", "derivation", "data_version")

#: The fields of a widget spec the page needs to draw it.
_WIDGET_FIELDS = ("id", "type", "title", "content", "gridPos", "viz")


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
    return {
        "name": document.get("name"),
        "title": document.get("title") or spec.get("title") or document.get("name"),
        "description": spec.get("description"),
        "version": document.get("version"),
        "status": document.get("status"),
        "updatedAt": document.get("updated_at"),
        "snapshotAt": document.get("created_at"),
        "hasVariables": bool(spec.get("variables")),
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
    # JSON inside a <script> ends at the first "</script"; escaping every "<" closes
    # that off without changing what the parser reads back.
    data = json.dumps(payload, ensure_ascii=False).replace("<", "\\u003c")
    return template.replace("__TITLE__", html.escape(str(payload["title"]))).replace(
        "__DATA__", data
    )
