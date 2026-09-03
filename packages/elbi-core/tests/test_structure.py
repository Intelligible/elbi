"""Tests for structure discovery, at the behavior boundary (analyze a dataset).

The properties that matter for grounding an analysis: a determiner is found, an
exactly-derived column is flagged, an identifier is recognized, related columns
surface by mutual information, and a high-cardinality column does NOT spuriously
dominate the dependency graph (the bias the binning fixes).
"""

from __future__ import annotations

from elbi_core import analyze_structure
from elbi_core.structure import StructureMap


def _rows(**columns: list[object]) -> tuple[list[dict[str, str]], list[str]]:
    names = list(columns)
    n = len(next(iter(columns.values())))
    rows = [{k: str(columns[k][i]) for k in names} for i in range(n)]
    return rows, names


def test_functional_dependency_detected() -> None:
    # zip determines city: city is constant within each zip group.
    rows, cols = _rows(
        zip=["1", "1", "2", "2", "3"],
        city=["A", "A", "B", "B", "C"],
        sales=["10", "20", "30", "40", "50"],
    )
    sm = analyze_structure(rows, cols)
    assert ("zip", "city") in sm.functional_dependencies
    # zip does NOT determine sales (zip 1 -> sales 10 and 20), so a real constancy
    # check must exclude it; this fails if every pair is reported as a dependency.
    assert ("zip", "sales") not in sm.functional_dependencies
    assert ("city", "sales") not in sm.functional_dependencies


def test_derived_column_detected() -> None:
    rows, cols = _rows(
        total=["3", "7", "12"],
        part_a=["1", "3", "5"],
        part_b=["2", "4", "7"],
    )
    sm = analyze_structure(rows, cols)
    assert "total = part_a + part_b" in sm.derived_columns


def test_categorical_identifier_is_a_candidate_key() -> None:
    rows, cols = _rows(
        order_id=[f"o{i}" for i in range(50)],
        region=["west", "east"] * 25,
    )
    sm = analyze_structure(rows, cols)
    assert "order_id" in sm.candidate_keys
    assert "region" not in sm.candidate_keys


def test_continuous_column_is_not_a_key_unless_named() -> None:
    # A continuous measure is near-unique by nature; it must not be called a key.
    rows, cols = _rows(
        score=[str(i + 0.5) for i in range(80)],  # all distinct, continuous
        record_id=[str(i) for i in range(80)],  # named like an id
    )
    sm = analyze_structure(rows, cols)
    assert "score" not in sm.candidate_keys
    assert "record_id" in sm.candidate_keys  # 'id' in the name


def test_dependency_graph_surfaces_related_columns() -> None:
    # y tracks x (same bin), z is unrelated; x~y must rank, x~z must not appear.
    xs = list(range(60))
    rows, cols = _rows(
        x=[str(v) for v in xs],
        y=[str(v * 2) for v in xs],  # perfectly co-monotonic with x
        z=[str(v % 2) for v in xs],  # alternating, independent of x's magnitude
    )
    sm = analyze_structure(rows, cols)
    pairs = {(a, b) for a, b, _ in sm.dependencies}
    assert ("x", "y") in pairs
    assert ("x", "z") not in pairs


def test_high_cardinality_column_does_not_spuriously_determine() -> None:
    # The bias the binning fixes: a unique continuous column trivially "predicts"
    # an independent flag under naive entropy. Binned MI must NOT rank x ~ flag.
    n = 200
    rows, cols = _rows(
        uid=[str(i + 0.5) for i in range(n)],  # unique continuous
        flag=[str(i % 2) for i in range(n)],  # independent of uid magnitude
    )
    sm = analyze_structure(rows, cols)
    strengths = {(a, b): s for a, b, s in sm.dependencies}
    assert strengths.get(("flag", "uid"), 0.0) < 0.3
    # And uid, being ~unique, is not reported as determining flag.
    assert ("uid", "flag") not in sm.functional_dependencies


def test_column_kinds_classified() -> None:
    rows, cols = _rows(
        code=["0", "1", "0", "1", "1"],  # small integer set -> discrete code
        amount=["10.5", "20.1", "33.7", "9.2", "88.0"],  # continuous
        label=["a", "b", "a", "c", "b"],  # categorical
    )
    sm = analyze_structure(rows, cols)
    assert sm.kinds == {
        "code": "discrete",
        "amount": "continuous",
        "label": "categorical",
    }


def test_render_includes_sections() -> None:
    rows, cols = _rows(total=["3", "7", "12"], a=["1", "3", "5"], b=["2", "4", "7"])
    text = analyze_structure(rows, cols).render()
    assert "# Data structure" in text
    assert "total = a + b" in text


def test_empty_dataset() -> None:
    sm = analyze_structure([], ["a", "b"])
    assert isinstance(sm, StructureMap)
    assert sm.n_rows == 0
    assert sm.dependencies == ()
