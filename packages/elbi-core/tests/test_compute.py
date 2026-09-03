"""The engine-agnostic compute seam: reference semantics and cross-engine parity.

The load-bearing property is *parity*: the DuckDB and Polars backends must return
exactly what the ``list[dict]`` reference backend returns, for every profile field and
every located violation. These tests assert that over diverse data (nulls, empties,
duplicates, mixed types) and as a Hypothesis property, so a backend that diverges from
the reference is caught. Engine backends are optional, so their cases are generated only
when the extra is installed (and run under CI's ``--all-extras``).
"""

from __future__ import annotations

import sys
import tempfile
from collections.abc import Callable
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elbi_core.compute import RowsFrame, backend_for, key_set
from elbi_core.compute.arrow import is_arrow_source

Make = Callable[[list[dict[str, Any]]], Any]


def _engines() -> list[tuple[str, Make]]:
    """The engine backends available in this environment, each as (id, factory)."""
    out: list[tuple[str, Make]] = []
    if find_spec("duckdb") and find_spec("pyarrow"):
        import pyarrow as pa

        from elbi_core.compute.duckdb_backend import DuckDBFrame

        out.append(
            ("duckdb", lambda rows: DuckDBFrame.from_arrow(pa.Table.from_pylist(rows)))
        )
    if find_spec("polars"):
        import polars as pl

        from elbi_core.compute.polars_backend import PolarsFrame

        out.append(("polars", lambda rows: PolarsFrame(pl.from_dicts(rows))))
    return out


ENGINES = _engines()
ENGINE_PARAMS = [pytest.param(make, id=name) for name, make in ENGINES]

_DATASETS = [
    [{"id": 1, "g": "A", "s": "90.0"}, {"id": 2, "g": "B", "s": "70.5"}],
    [{"id": 1}, {"id": 1}, {"id": 2}],  # duplicates
    [{"x": "1"}, {"x": ""}, {"x": None}, {"x": "3"}],  # missing (empty and null)
    [{"v": "a@b.com"}, {"v": "nope"}, {"v": "c@d.io"}],  # patterns
    [{"n": "-5"}, {"n": "10"}, {"n": "notnum"}],  # numeric + non-numeric
    [{"b": "true"}, {"b": "false"}, {"b": "yes"}],  # booleans
]


def _assert_parity(rows: list[dict[str, Any]], make: Make) -> None:
    ref = RowsFrame(rows)
    eng = make(rows)
    assert eng.row_count() == ref.row_count()
    assert eng.columns() == ref.columns()
    for column in ref.columns():
        assert eng.present_count(column) == ref.present_count(column)
        assert eng.missing_indices(column) == ref.missing_indices(column)
        assert sorted(eng.type_violations(column, "integer")) == sorted(
            ref.type_violations(column, "integer")
        )
        assert sorted(eng.range_violations(column, 0, None)) == sorted(
            ref.range_violations(column, 0, None)
        )
        assert sorted(eng.pattern_violations(column, "@")) == sorted(
            ref.pattern_violations(column, "@")
        )
        assert sorted(eng.enum_violations(column, ["A", "1"])) == sorted(
            ref.enum_violations(column, ["A", "1"])
        )
        assert sorted(eng.duplicate_indices([column])) == sorted(
            ref.duplicate_indices([column])
        )
        rp, ep = ref.profile([column])[column], eng.profile([column])[column]
        for attribute in ("present", "distinct", "inferred_type", "minimum", "maximum"):
            assert getattr(ep, attribute) == getattr(rp, attribute), (
                column,
                attribute,
            )
        assert {v for v, _ in ep.top_values} == {v for v, _ in rp.top_values}


@pytest.mark.parametrize("make", ENGINE_PARAMS)
@pytest.mark.parametrize("rows", _DATASETS)
def test_engine_matches_the_reference(rows: list[dict[str, Any]], make: Make) -> None:
    _assert_parity(rows, make)


@pytest.mark.parametrize("make", ENGINE_PARAMS)
def test_reference_integrity_matches_the_reference(make: Make) -> None:
    rows = [{"fk": "1"}, {"fk": "2"}, {"fk": "9"}]
    allowed = key_set([{"id": "1"}, {"id": "2"}], ["id"])
    ref, eng = RowsFrame(rows), make(rows)
    assert sorted(eng.reference_violations(["fk"], allowed)) == sorted(
        ref.reference_violations(["fk"], allowed)
    )


@pytest.mark.parametrize("make", ENGINE_PARAMS)
def test_sample_is_deterministic_and_matches_the_reference(make: Make) -> None:
    rows = [{"id": i} for i in range(2000)]
    ref, eng = RowsFrame(rows), make(rows)
    ref_ids = sorted(int(r["id"]) for r in ref.sample_rows(200))
    eng_a = sorted(int(r["id"]) for r in eng.sample_rows(200))
    eng_b = sorted(int(r["id"]) for r in eng.sample_rows(200))
    assert eng_a == eng_b == ref_ids
    assert 100 < len(ref_ids) < 300  # roughly the cap, position-uniform


@given(
    rows=st.lists(
        st.fixed_dictionaries(
            {
                "k": st.integers(min_value=0, max_value=5),
                "v": st.sampled_from(["a", "b", "", "10", "2.5"]),
            }
        ),
        min_size=1,
        max_size=40,
    )
)
@settings(max_examples=40, deadline=None)
def test_profile_parity_property(rows: list[dict[str, Any]]) -> None:
    for _name, make in ENGINES:
        ref, eng = RowsFrame(rows), make(rows)
        for column in ("k", "v"):
            rp, ep = ref.profile([column])[column], eng.profile([column])[column]
            assert (ep.present, ep.distinct, ep.inferred_type) == (
                rp.present,
                rp.distinct,
                rp.inferred_type,
            )
        assert sorted(eng.duplicate_indices(["k"])) == sorted(
            ref.duplicate_indices(["k"])
        )


def test_out_of_core_file_scan_matches_loading_the_rows() -> None:
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    import pyarrow.parquet as pq

    from elbi_core.compute.duckdb_backend import DuckDBFrame

    rows = [{"id": i, "g": ["A", "B", "C"][i % 3]} for i in range(3000)]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "data.parquet"
        pq.write_table(pa.Table.from_pylist(rows), path)
        scanned = DuckDBFrame.from_path(path)  # scans the file, does not load it
        loaded = RowsFrame(rows)
        assert scanned.row_count() == loaded.row_count()
        assert (
            scanned.profile(["g"])["g"].distinct == loaded.profile(["g"])["g"].distinct
        )


def test_backend_for_dispatches_by_type() -> None:
    assert isinstance(backend_for([{"a": 1}]), RowsFrame)
    with pytest.raises(TypeError, match="no compute backend"):
        backend_for(object())


def test_arrow_source_detection() -> None:
    pa = pytest.importorskip("pyarrow")
    assert is_arrow_source(pa.Table.from_pylist([{"a": 1}]))
    assert not is_arrow_source([{"a": 1}])


@pytest.mark.skipif(not ENGINES, reason="no compute engine installed")
def test_backend_for_selects_an_engine_for_arrow_and_files() -> None:
    pa = pytest.importorskip("pyarrow")
    table = pa.Table.from_pylist([{"a": 1}, {"a": 2}])
    backend = backend_for(table)
    assert backend.row_count() == 2 and not isinstance(backend, RowsFrame)


# --- more coverage of the engine branches and the dispatch -----------------------

_MIXED = [
    {"code": "AB", "grade": "A", "fk": "1", "n": "5"},
    {"code": "ABCDE", "grade": "Z", "fk": "9", "n": "-3"},
    {"code": "C", "grade": "B", "fk": "2", "n": "notnum"},
]


@pytest.mark.parametrize("make", ENGINE_PARAMS)
def test_length_enum_and_range_variants_match(make: Make) -> None:
    ref, eng = RowsFrame(_MIXED), make(_MIXED)
    assert sorted(eng.length_violations("code", 2, 3)) == sorted(
        ref.length_violations("code", 2, 3)
    )
    assert sorted(eng.enum_violations("grade", ["A", "B"])) == sorted(
        ref.enum_violations("grade", ["A", "B"])
    )
    assert sorted(eng.range_violations("n", -1, 4)) == sorted(
        ref.range_violations("n", -1, 4)
    )
    assert eng.key_missing_indices(["code"]) == ref.key_missing_indices(["code"])


@pytest.mark.parametrize("make", ENGINE_PARAMS)
def test_reference_with_no_allowed_keys_flags_every_present_row(make: Make) -> None:
    ref, eng = RowsFrame(_MIXED), make(_MIXED)
    assert sorted(eng.reference_violations(["fk"], set())) == sorted(
        ref.reference_violations(["fk"], set())
    )


@pytest.mark.parametrize("make", ENGINE_PARAMS)
def test_materialization_and_small_sample(make: Make) -> None:
    eng = make(_MIXED)
    assert len(eng.to_rows(limit=2)) == 2
    assert len(eng.to_rows()) == 3
    assert len(eng.sample_rows(100)) == 3  # cap exceeds row count: every row


def test_profile_columns_accepts_a_backend_directly() -> None:
    from elbi_core.quality.profile import profile_columns

    backend = RowsFrame([{"x": "1"}, {"x": "2"}])
    profiles = profile_columns(backend)
    assert profiles[0].distinct == 2


def test_arrow_helpers() -> None:
    pa = pytest.importorskip("pyarrow")
    from elbi_core.compute.arrow import require_pyarrow, rows_to_arrow

    assert require_pyarrow() is pa
    table = rows_to_arrow([{"a": 1}, {"a": 2}])
    assert table.num_rows == 2


def test_backend_for_dispatches_to_polars_and_registered_backends() -> None:
    pl = pytest.importorskip("polars")
    from elbi_core.compute import register_backend
    from elbi_core.compute.polars_backend import PolarsFrame

    assert isinstance(backend_for(pl.from_dicts([{"a": 1}])), PolarsFrame)
    assert isinstance(backend_for(pl.LazyFrame({"a": [1]})), PolarsFrame)

    class _Boxed:
        def __init__(self) -> None:
            self.rows = [{"a": 1}]

    register_backend(
        "boxed", lambda d: isinstance(d, _Boxed), lambda d: RowsFrame(d.rows)
    )
    assert backend_for(_Boxed()).row_count() == 1


def test_backend_for_scans_a_file_path_and_falls_back_without_an_engine() -> None:
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    import sys

    import pyarrow.parquet as pq

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "d.parquet"
        pq.write_table(pa.Table.from_pylist([{"a": 1}, {"a": 2}]), path)
        assert backend_for(path).row_count() == 2  # a Path scans via the engine

    # With no DuckDB visible, an Arrow table falls back to the reference backend.
    table = pa.Table.from_pylist([{"a": 1}])
    saved = sys.modules.pop("duckdb", None)
    try:
        assert isinstance(backend_for(table), RowsFrame)
    finally:
        if saved is not None:
            sys.modules["duckdb"] = saved


def test_duckdb_from_relation_and_file_readers() -> None:
    duckdb = pytest.importorskip("duckdb")
    pytest.importorskip("pyarrow")
    import csv

    from elbi_core.compute.duckdb_backend import DuckDBFrame

    relation = duckdb.sql("SELECT 1 AS a UNION ALL SELECT 2")
    assert DuckDBFrame(relation).row_count() == 2

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "d.csv"
        with path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["a", "b"])
            writer.writerow(["1", "x"])
        assert DuckDBFrame.from_path(path).row_count() == 1
        with pytest.raises(ValueError, match="cannot scan"):
            DuckDBFrame.from_path(Path("nope.txt"))


def test_load_backend_binds_a_file_as_a_scanning_handle() -> None:
    pa = pytest.importorskip("pyarrow")
    pytest.importorskip("duckdb")
    import csv

    import pyarrow.parquet as pq

    from elbi_core.config import DataBindings

    with tempfile.TemporaryDirectory() as directory:
        base = Path(directory)
        rows = [{"id": i, "g": ["A", "B"][i % 2]} for i in range(500)]
        pq.write_table(pa.Table.from_pylist(rows), base / "p.parquet")
        with (base / "r.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["id"])
            writer.writerow(["1"])

        bindings = DataBindings(bindings={"p": "p.parquet", "r": "r.csv"})
        parquet_handle = bindings.load_backend("p", base)
        assert parquet_handle.row_count() == 500  # scanned, not loaded
        assert not isinstance(parquet_handle, RowsFrame)
        assert bindings.load_backend("r", base).row_count() == 1


def test_missing_engine_dependencies_raise_a_clear_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elbi_core.compute import arrow
    from elbi_core.compute.duckdb_backend import _require_duckdb
    from elbi_core.compute.polars_backend import _pl

    monkeypatch.setitem(sys.modules, "pyarrow", None)
    with pytest.raises(ImportError, match="pyarrow"):
        arrow.require_pyarrow()
    monkeypatch.setitem(sys.modules, "duckdb", None)
    with pytest.raises(ImportError, match="duckdb"):
        _require_duckdb()
    monkeypatch.setitem(sys.modules, "polars", None)
    with pytest.raises(ImportError, match="polars"):
        _pl()


def test_backend_for_a_file_without_duckdb_raises_a_type_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "duckdb", None)
    with pytest.raises(TypeError, match="needs the 'duckdb' engine"):
        backend_for(Path("events.parquet"))
