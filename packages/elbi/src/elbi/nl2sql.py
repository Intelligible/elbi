"""Generate SQL from a natural-language question, grounded in the schema.

A single tool-free LLM completion: given the tables and columns available and the SQL
dialect, return one query and nothing else. This is a convenience for the exploration
workbench, not a governed path. The SQL it writes is a starting point the person reads,
edits, and runs; only a promoted query is ever certified. It drives whatever
``LLMClient`` the app was built with, so it works for any provider backing that seam.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from elbi_agent import LLMClient, Transcript

_SYSTEM = (
    "You translate a data question into a single SQL query. Return ONLY the SQL: no "
    "prose, no markdown fences, no explanation, no trailing semicolon. Use only the "
    "tables and columns listed; do not invent names. Prefer an explicit ORDER BY on a "
    "stable column so the result is reproducible."
)

#: The one-completion draft: SQL plus the statistical claim the question asserts, so
#: the verification oracle can rule on the drafted derivation, not just run it.
_DRAFT_SYSTEM = (
    "You draft one SQL query for a data question and state the statistical claim the "
    "question asserts, if any. Reply with ONLY a JSON object, no prose and no fences: "
    '{"sql": "...", "claim": {"x": "column", "y": "column"} | null}. Declare a claim '
    "only when the question asks whether one listed column drives, affects, predicts, "
    "or varies with another; the roles are x (the driver), y (the outcome), and an "
    'optional "controls" list, all exact column names from the schema. When you '
    "declare a claim, the query MUST return the claim's columns as row-level values "
    "(one row per record; never aggregate them away), so the claim can be checked "
    "against the output. Use only the tables and columns listed; do not invent names. "
    "Prefer an explicit ORDER BY on a stable column; no trailing semicolon."
)


def generate_sql(
    question: str,
    tables: Sequence[Mapping[str, Any]],
    client: LLMClient,
    *,
    dialect: str = "DuckDB",
) -> str:
    """Return one ``dialect`` SQL query answering ``question`` over ``tables``.

    ``tables`` is the catalog shape (``{"name", "columns": [{"name"}]}``). Raises
    whatever the client raises; the caller decides how to surface a failure.
    """
    transcript = Transcript(system=_SYSTEM)
    transcript.add_user_text(
        f"Dialect: {dialect}.\n\n"
        f"Available tables and columns:\n{_render_schema(tables)}\n\n"
        f"Question: {question.strip()}\n\n"
        f"Return one {dialect} SQL query."
    )
    step = client.step(transcript, ())
    return _clean_sql(step.text)


def draft_query(
    question: str,
    tables: Sequence[Mapping[str, Any]],
    client: LLMClient,
    *,
    dialect: str = "DuckDB",
) -> tuple[str, dict[str, Any] | None]:
    """Draft SQL for ``question`` plus the statistical claim it asserts, if any.

    One completion returns both: row-level SQL and the claim roles the oracle needs
    to rule on the draft, or ``(sql, None)`` for a question that asserts nothing.
    Surprises degrade instead of failing: a non-JSON reply is treated as bare SQL,
    and an uncheckable claim is dropped (computation-only verification).
    """
    transcript = Transcript(system=_DRAFT_SYSTEM)
    transcript.add_user_text(
        f"Dialect: {dialect}.\n\n"
        f"Available tables and columns:\n{_render_schema(tables)}\n\n"
        f"Question: {question.strip()}\n\n"
        f"Return the JSON object with one {dialect} SQL query and the claim (or null)."
    )
    step = client.step(transcript, ())
    return _parse_draft(step.text, tables)


def _parse_draft(
    text: str | None, tables: Sequence[Mapping[str, Any]]
) -> tuple[str, dict[str, Any] | None]:
    """Parse the draft reply into ``(sql, claim)``, degrading safely on surprises."""
    body = _strip_fence(text)
    if not body:
        return "", None
    try:
        obj = json.loads(body)
    except ValueError:
        return _clean_sql(body), None  # the model answered with bare SQL
    if not isinstance(obj, dict):
        return _clean_sql(body), None
    sql = _clean_sql(str(obj.get("sql") or ""))
    claim = obj.get("claim")
    if not isinstance(claim, dict):
        return sql, None
    known = {
        c["name"] for table in tables for c in table.get("columns", []) if "name" in c
    }
    x, y = claim.get("x"), claim.get("y")
    if not (isinstance(x, str) and isinstance(y, str) and x != y and {x, y} <= known):
        return sql, None  # unknown columns or x==y: nothing the oracle could judge
    controls = [
        c
        for c in claim.get("controls") or ()
        if isinstance(c, str) and c in known and c not in (x, y)
    ]
    roles: dict[str, Any] = {"x": x, "y": y}
    if controls:
        roles["controls"] = controls
    return sql, roles


def _render_schema(tables: Sequence[Mapping[str, Any]]) -> str:
    """Render the catalog as ``- table(col, col, ...)`` lines for the prompt."""
    lines = [
        f"- {table['name']}({', '.join(c['name'] for c in table.get('columns', []))})"
        for table in tables
    ]
    return "\n".join(lines) or "(no tables)"


def _strip_fence(text: str | None) -> str:
    """Drop an outer markdown code fence (any language tag) and outer whitespace."""
    if not text or not text.strip():
        return ""
    body = text.strip()
    if body.startswith("```"):
        body = body[3:]
        first, _, rest = body.partition("\n")
        if first.strip().lower() in ("", "sql", "json"):
            body = rest
        body = body.split("```", 1)[0]
    return body.strip()


def _clean_sql(text: str | None) -> str:
    """Strip a model reply to bare SQL: drop any code fence and outer whitespace."""
    return _strip_fence(text).rstrip(";").strip()
