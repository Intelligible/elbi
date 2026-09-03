"""Tests for serve contracts: manifest emission and rendering."""

from __future__ import annotations

import json

from elbi_core import serve
from elbi_core.artifact import Artifact


def test_table_manifest_only_emits_relevant_fields() -> None:
    contract = serve.table(title="T", columns=["a", "b"], max_rows=10, max_cells=200)
    manifest = contract.to_manifest()
    assert manifest == {
        "format": "table",
        "title": "T",
        "columns": ["a", "b"],
        "maxRows": 10,
        "maxCells": 200,
    }


def test_json_manifest_emits_indent() -> None:
    assert serve.json(indent=4).to_manifest() == {"format": "json", "indent": 4}


def test_text_manifest_is_minimal() -> None:
    assert serve.text().to_manifest() == {"format": "text"}


def test_render_table_with_columns_and_title() -> None:
    contract = serve.table(title="Scores", columns=["id", "risk"])
    rows = [{"id": "a", "risk": 0.9}, {"id": "b", "risk": 0.1, "extra": "ignored"}]
    rendered = contract.render(Artifact.table(rows))
    assert rendered.startswith("# Scores")
    assert "| id | risk |" in rendered
    assert "extra" not in rendered


def test_render_table_truncates_to_max_rows() -> None:
    contract = serve.table(max_rows=1)
    rows = [{"n": 1}, {"n": 2}, {"n": 3}]
    rendered = contract.render(Artifact.table(rows))
    assert "Showing 1 of 3 rows" in rendered


def test_render_table_escapes_pipes() -> None:
    rendered = serve.table().render(Artifact.table([{"x": "a|b"}]))
    assert "a\\|b" in rendered


def test_render_json_is_parseable() -> None:
    rendered = serve.json().render(Artifact.json({"k": [1, 2]}))
    assert json.loads(rendered) == {"k": [1, 2]}


def test_render_text_and_markdown() -> None:
    assert serve.text().render(Artifact.text("hi")) == "hi"
    rendered = serve.markdown(title="Doc").render(Artifact.markdown("body"))
    assert rendered == "# Doc\n\nbody"


def test_render_empty_table_columns() -> None:
    rendered = serve.table().render(Artifact.table([]))
    assert "(no columns)" in rendered


# --- Preview + structured content (Gap 4: result-shaping wire format) ---


def test_preview_returns_full_table_within_budget() -> None:
    contract = serve.table(max_cells=100)
    rows = [{"n": i} for i in range(5)]  # 5 rows x 1 col = 5 cells, under budget
    preview = contract.preview(Artifact.table(rows))
    assert "Preview: first" not in preview
    assert preview == contract.render(Artifact.table(rows))


def test_preview_trims_to_cell_budget_and_notes_total() -> None:
    contract = serve.table(max_cells=4)  # 2 columns → 2 preview rows
    rows = [{"a": i, "b": i} for i in range(50)]
    preview = contract.preview(Artifact.table(rows))
    assert "Preview: first 2 of 50 rows" in preview
    assert "structured content" in preview


def test_preview_non_table_matches_render() -> None:
    contract = serve.json()
    artifact = Artifact.json({"k": 1})
    assert contract.preview(artifact) == contract.render(artifact)


def test_structured_returns_rows_and_count_for_table() -> None:
    contract = serve.table(max_rows=2)
    rows = [{"id": "a"}, {"id": "b"}, {"id": "c"}]
    structured = contract.structured(Artifact.table(rows))
    assert structured == {"rows": [{"id": "a"}, {"id": "b"}], "row_count": 3}


def test_structured_projects_to_declared_columns() -> None:
    contract = serve.table(columns=["id"])
    rows = [{"id": "a", "secret": "x"}]
    structured = contract.structured(Artifact.table(rows))
    assert structured == {"rows": [{"id": "a"}], "row_count": 1}


def test_structured_is_none_for_non_table() -> None:
    assert serve.text().structured(Artifact.text("hi")) is None
