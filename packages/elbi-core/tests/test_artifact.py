"""Tests for Artifact construction and coercion."""

from __future__ import annotations

from elbi_core.artifact import Artifact, coerce_artifact


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
