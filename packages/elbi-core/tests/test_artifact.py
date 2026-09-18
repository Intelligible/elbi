"""Tests for Artifact construction and coercion."""

from __future__ import annotations

from elbi_core.artifact import DISPLAY_ROWS, Artifact, coerce_artifact


def test_table_constructor_copies_rows() -> None:
    rows = [{"a": 1}]
    artifact = Artifact.table(rows)
    rows.append({"a": 2})
    assert artifact.kind == "table"
    assert artifact.value == [{"a": 1}]


def test_text_and_markdown_and_json_constructors() -> None:
    assert Artifact.text(123).value == "123"
    assert Artifact.markdown("# hi").kind == "markdown"
    assert Artifact.json({"k": "v"}).value == {"k": "v"}


def test_opaque_constructor_holds_object() -> None:
    model = object()
    artifact = Artifact.opaque(model)
    assert artifact.kind == "opaque"
    assert artifact.value is model


def test_coerce_passthrough_for_artifact() -> None:
    artifact = Artifact.text("x")
    assert coerce_artifact(artifact) is artifact


def test_coerce_str_to_text() -> None:
    assert coerce_artifact("hello").kind == "text"


def test_coerce_list_of_dicts_to_table() -> None:
    assert coerce_artifact([{"a": 1}, {"a": 2}]).kind == "table"


def test_coerce_other_to_json() -> None:
    assert coerce_artifact({"a": 1}).kind == "json"
    assert coerce_artifact([1, 2, 3]).kind == "json"


def test_a_table_renders_as_an_html_table() -> None:
    html = Artifact.table([{"env": "preview", "failed": 670}])._repr_html_()
    assert "<th>env</th>" in html
    assert "<td>preview</td>" in html
    assert "<td>670</td>" in html


def test_a_table_escapes_cell_content() -> None:
    html = Artifact.table([{"name": "<script>"}])._repr_html_()
    assert "<script>" not in html
    assert "&lt;script&gt;" in html


def test_a_wide_table_uses_the_union_of_every_row_s_keys() -> None:
    html = Artifact.table([{"a": 1}, {"b": 2}])._repr_html_()
    assert "<th>a</th>" in html
    assert "<th>b</th>" in html


def test_a_long_table_renders_a_prefix_and_says_so() -> None:
    artifact = Artifact.table([{"n": n} for n in range(DISPLAY_ROWS + 5)])
    html = artifact._repr_html_()
    assert f"<td>{DISPLAY_ROWS - 1}</td>" in html
    assert f"<td>{DISPLAY_ROWS}</td>" not in html
    assert f"showing {DISPLAY_ROWS:,} of {DISPLAY_ROWS + 5:,} rows" in html


def test_an_empty_table_says_so_rather_than_rendering_nothing() -> None:
    assert "no rows" in Artifact.table([])._repr_html_()


def test_markdown_renders_as_markdown_not_html() -> None:
    artifact = Artifact.markdown("# Reliability")
    assert artifact._repr_markdown_() == "# Reliability"
    assert artifact._repr_html_() is None


def test_kinds_without_a_rich_rendering_offer_none() -> None:
    for artifact in (
        Artifact.text("x"),
        Artifact.json({"a": 1}),
        Artifact.opaque(object()),
    ):
        assert artifact._repr_html_() is None
        assert artifact._repr_markdown_() is None
