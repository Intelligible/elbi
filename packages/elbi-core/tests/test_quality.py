"""Verifiable data cleaning: the contract spec, the checker, profiling, and suggestion.

Tests drive the public surface (``verify_contract``, ``suggest_contract``,
``profile_columns``, and the authoring gate) so they survive refactors, and add
property tests for the invariants a coverage number cannot capture: the verdict is a
conjunction, tolerance is monotone, and a suggested threshold never exceeds the rate it
was drawn from.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from elbi_core import (
    Constraints,
    Context,
    DataContract,
    FieldSpec,
    ForeignKey,
    Reference,
    Registry,
    Runner,
    TableSpec,
    derivation,
    profile_columns,
    serve,
    suggest_contract,
    verify_contract,
)
from elbi_core.authoring import verify
from elbi_core.errors import SpecValidationError
from elbi_core.quality._stats import (
    is_missing,
    parses_as,
    to_float,
    wilson_lower_bound,
)
from elbi_core.quality.backend import RowsBackend, backend_for
from elbi_core.quality.contract import load_data_contract_schema

REPO_ROOT = Path(__file__).resolve().parents[3]


# --- the contract spec -----------------------------------------------------------


def _full_contract() -> DataContract:
    return DataContract(
        fields=(
            FieldSpec(
                "id", "integer", Constraints(required=True, unique=True, minimum=1)
            ),
            FieldSpec(
                "email",
                "string",
                Constraints(required=True, pattern="@", max_length=254),
                mostly=0.9,
            ),
            FieldSpec("status", "string", Constraints(enum=("active", "churned"))),
        ),
        table=TableSpec(
            primary_key=("id",),
            row_count_min=1,
            foreign_keys=(ForeignKey(("plan_id",), Reference("plans", ("id",))),),
        ),
    )


def test_contract_round_trips_through_its_manifest() -> None:
    contract = _full_contract()
    assert DataContract.from_manifest(contract.to_manifest()) == contract


def test_per_type_constraint_legality_is_rejected() -> None:
    # A string-only constraint (pattern) on an integer field is not spec-conformant.
    bad = {
        "specVersion": "1.0",
        "kind": "DataContract",
        "fields": [{"name": "n", "type": "integer", "constraints": {"pattern": "^x$"}}],
    }
    with pytest.raises(SpecValidationError):
        DataContract.from_manifest(bad)


def test_numeric_range_on_a_string_is_rejected() -> None:
    bad = {
        "specVersion": "1.0",
        "kind": "DataContract",
        "fields": [{"name": "s", "type": "string", "constraints": {"minimum": 0}}],
    }
    with pytest.raises(SpecValidationError):
        DataContract.from_manifest(bad)


def test_duplicate_field_name_is_rejected() -> None:
    dup = {
        "specVersion": "1.0",
        "kind": "DataContract",
        "fields": [{"name": "x", "type": "string"}, {"name": "x", "type": "integer"}],
    }
    with pytest.raises(SpecValidationError, match="duplicate field name"):
        DataContract.from_manifest(dup)


def test_bundled_data_contract_schema_matches_the_canonical_spec() -> None:
    canonical = json.loads(
        (REPO_ROOT / "spec" / "data_contract.schema.json").read_text(encoding="utf-8")
    )
    assert load_data_contract_schema() == canonical


# --- the checker (verify_contract) -----------------------------------------------

_CLEAN = [
    {"id": 1, "email": "a@b.com", "status": "active", "plan_id": 10},
    {"id": 2, "email": "c@d.com", "status": "churned", "plan_id": 20},
]
_PLANS = [{"id": 10}, {"id": 20}]


def test_clean_data_is_sound() -> None:
    report = verify_contract(_CLEAN, _full_contract(), references={"plans": _PLANS})
    assert report.verdict == "sound"


@pytest.mark.parametrize(
    ("row", "failing_clause"),
    [
        ({"id": 2, "email": "bad", "status": "active", "plan_id": 10}, "email.pattern"),
        (
            {"id": 2, "email": "e@f.com", "status": "GHOST", "plan_id": 10},
            "status.enum",
        ),
        ({"id": 1, "email": "e@f.com", "status": "active", "plan_id": 10}, "id.unique"),
        ({"id": 2, "email": "e@f.com", "status": "active", "plan_id": 99}, None),
    ],
)
def test_a_violating_row_makes_the_contract_unsound(
    row: dict[str, object], failing_clause: str | None
) -> None:
    report = verify_contract(
        [_CLEAN[0], row], _full_contract(), references={"plans": _PLANS}
    )
    assert report.verdict == "unsound"
    if failing_clause is not None:
        verdicts = {c.name: c.verdict for c in report.clauses}
        assert verdicts[failing_clause] == "unsound"


def test_missing_required_value_is_unsound_and_located() -> None:
    rows = [_CLEAN[0], {"id": 2, "status": "active", "plan_id": 20}]  # no email
    report = verify_contract(rows, _full_contract(), references={"plans": _PLANS})
    assert report.verdict == "unsound"
    email_missing = [v for v in report.violations if v.clause == "email.required"]
    assert email_missing and email_missing[0].row_number == 2


def test_a_missing_declared_column_is_inconclusive_not_a_pass() -> None:
    contract = DataContract(
        fields=(FieldSpec("ghost", "string", Constraints(required=True)),)
    )
    report = verify_contract(_CLEAN, contract)
    assert report.verdict == "inconclusive"


def test_a_foreign_key_without_its_reference_is_inconclusive() -> None:
    # The referenced rows are not supplied, so referential integrity cannot be checked.
    report = verify_contract(_CLEAN, _full_contract())
    fk = {c.name: c.verdict for c in report.clauses}["table.foreign_key[plan_id]"]
    assert fk == "inconclusive"
    assert report.verdict == "inconclusive"


def test_row_count_bound_is_enforced() -> None:
    contract = DataContract(
        fields=(FieldSpec("id", "integer"),), table=TableSpec(row_count_min=5)
    )
    assert verify_contract(_CLEAN, contract).verdict == "unsound"


def test_columns_match_flags_schema_drift() -> None:
    contract = DataContract(
        fields=(FieldSpec("id", "integer"),), table=TableSpec(columns_match=True)
    )
    report = verify_contract(_CLEAN, contract)  # data has extra columns
    assert report.verdict == "unsound"
    assert {c.name for c in report.clauses} >= {"table.columns_match"}


def test_tolerance_admits_a_bounded_share_of_violations() -> None:
    # Four of five present values match; mostly=0.8 is exactly met (sound), 0.81 is not.
    rows = [{"e": v} for v in ["a@x", "b@x", "c@x", "d@x", "bad"]]
    ok = DataContract(
        fields=(FieldSpec("e", "string", Constraints(pattern="@"), mostly=0.8),)
    )
    strict = DataContract(
        fields=(FieldSpec("e", "string", Constraints(pattern="@"), mostly=0.81),)
    )
    assert verify_contract(rows, ok).verdict == "sound"
    assert verify_contract(rows, strict).verdict == "unsound"


# --- profiling -------------------------------------------------------------------


def test_profile_infers_the_narrowest_type() -> None:
    rows = [
        {"i": "1", "f": "1.5", "s": "x", "b": "true", "d": "2024-01-02"},
        {"i": "2", "f": "2.0", "s": "y", "b": "false", "d": "2024-01-03"},
    ]
    kinds = {p.name: p.inferred_type for p in profile_columns(rows)}
    assert kinds == {
        "i": "integer",
        "f": "number",
        "s": "string",
        "b": "boolean",
        "d": "date",
    }


def test_profile_reports_completeness_distinct_and_range() -> None:
    rows = [{"x": "1"}, {"x": "3"}, {"x": ""}, {"x": "3"}]
    (profile,) = profile_columns(rows)
    assert profile.present == 3
    assert profile.completeness == pytest.approx(0.75)
    assert profile.distinct == 2
    assert (profile.minimum, profile.maximum) == (1.0, 3.0)


# --- suggestion ------------------------------------------------------------------


def _sample() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i in range(1, 121):
        rows.append(
            {
                "id": str(i),
                "grade": ["A", "B", "C"][i % 3],
                "score": str(i % 50),
                "note": "" if i % 4 == 0 else "seen",
            }
        )
    return rows


def test_a_suggested_contract_certifies_the_data_it_was_drawn_from() -> None:
    rows = _sample()
    contract = suggest_contract(rows)
    assert verify_contract(rows, contract).verdict in ("sound", "inconclusive")


def test_suggestion_detects_the_primary_key_and_a_categorical_enum() -> None:
    contract = suggest_contract(_sample())
    assert contract.table.primary_key == ("id",)
    grade = next(f for f in contract.fields if f.name == "grade")
    assert grade.constraints.enum is not None
    assert set(grade.constraints.enum) == {"A", "B", "C"}


def test_a_partly_complete_column_gets_a_tolerance_below_its_observed_rate() -> None:
    # `note` is present in 3/4 of rows; the suggested completeness tolerance is a Wilson
    # lower bound, so it sits strictly below the observed 0.75 rather than pinning it.
    contract = suggest_contract(_sample())
    note = next(f for f in contract.fields if f.name == "note")
    assert note.constraints.required
    assert 0.0 < note.mostly < 0.75


# --- the authoring gate ----------------------------------------------------------


def test_a_contract_gates_certification() -> None:
    good = DataContract(
        fields=(
            FieldSpec("id", "integer", Constraints(required=True, unique=True)),
            FieldSpec("email", "string", Constraints(required=True, pattern="@")),
        )
    )
    reg = Registry()

    @derivation(name="clean_ok", serve=serve.table(), contract=good, registry=reg)
    def clean_ok(ctx: Context) -> list[dict[str, object]]:
        """Two valid rows."""
        return [{"id": 1, "email": "a@b.com"}, {"id": 2, "email": "c@d.com"}]

    @derivation(name="clean_bad", serve=serve.table(), contract=good, registry=reg)
    def clean_bad(ctx: Context) -> list[dict[str, object]]:
        """A duplicate id and a malformed email."""
        return [{"id": 1, "email": "a@b.com"}, {"id": 1, "email": "nope"}]

    runner = Runner(reg)
    ok = verify(runner, reg.get("clean_ok"))
    assert ok.ok and ok.contract_verdict == "sound"
    assert ok.contract_attestation is not None

    bad = verify(runner, reg.get("clean_bad"))
    assert not bad.ok and bad.contract_verdict == "unsound"


# --- properties ------------------------------------------------------------------


@given(
    k=st.integers(min_value=0, max_value=10_000),
    n=st.integers(min_value=1, max_value=10_000),
)
def test_wilson_lower_bound_never_exceeds_the_sample_rate(k: int, n: int) -> None:
    k = min(k, n)
    lower = wilson_lower_bound(k, n)
    assert 0.0 <= lower <= 1.0
    assert lower <= k / n + 1e-9


@given(n=st.integers(min_value=1, max_value=5000))
def test_wilson_lower_bound_tightens_with_evidence(n: int) -> None:
    # The same observed rate over more rows yields a bound at least as high (more
    # evidence, less discount). Compare n against 4n at a fixed 90% success rate.
    small = wilson_lower_bound(int(0.9 * n), n)
    large = wilson_lower_bound(int(0.9 * 4 * n), 4 * n)
    assert large >= small - 1e-9


@given(mostly=st.floats(min_value=0.0, max_value=1.0))
def test_tolerance_is_monotone(mostly: float) -> None:
    # Half the present values violate the pattern. A contract is sound only when the
    # tolerance admits at least that half, so a stricter `mostly` can never re-pass a
    # contract that a looser one failed.
    rows = [{"e": "a@x"}, {"e": "bad"}]
    contract = DataContract(
        fields=(FieldSpec("e", "string", Constraints(pattern="@"), mostly=mostly),)
    )
    verdict = verify_contract(rows, contract).verdict
    assert verdict == ("sound" if mostly <= 0.5 else "unsound")


@given(bad_value=st.text(min_size=1).filter(lambda s: "@" not in s and s.strip() != ""))
def test_a_guaranteed_violation_is_never_sound(bad_value: str) -> None:
    # A conjunction: a field the data provably violates cannot yield a sound contract,
    # whatever the other clauses say.
    rows = [{"e": "a@b.com"}, {"e": bad_value}]
    contract = DataContract(
        fields=(FieldSpec("e", "string", Constraints(pattern="@")),)
    )
    assert verify_contract(rows, contract).verdict == "unsound"


@given(
    value=st.integers() | st.floats(allow_nan=False, allow_infinity=False) | st.text()
)
def test_parses_as_string_accepts_any_scalar(value: object) -> None:
    assert parses_as(value, "string")


# --- type parsing edges ----------------------------------------------------------


def test_parses_as_covers_each_type_and_its_rejections() -> None:
    assert parses_as("5", "integer") and not parses_as("5.3", "integer")
    assert parses_as("5.3", "number") and not parses_as("x", "number")
    assert parses_as("yes", "boolean") and not parses_as("maybe", "boolean")
    assert parses_as(_dt.date(2024, 1, 2), "date")
    assert not parses_as(_dt.datetime(2024, 1, 2), "date")  # a datetime is not a date
    assert parses_as(_dt.datetime(2024, 1, 2, 3), "datetime")
    assert parses_as("2024-01-02T03:04", "datetime")
    assert not parses_as("nope", "datetime")
    assert not parses_as(["a"], "string")  # a container is no scalar type
    assert parses_as("anything", "unknown-type")  # an unknown type constrains nothing


def test_is_missing_and_to_float_edges() -> None:
    assert is_missing(None) and is_missing("") and not is_missing("0")
    assert to_float(True) is None  # a boolean is not a number
    nan = to_float("nan")
    assert nan != nan  # parses to NaN, which is not equal to itself
    assert to_float("x") is None and to_float([1]) is None


def test_backend_selection_and_errors() -> None:
    assert isinstance(backend_for([{"a": 1}]), RowsBackend)
    assert isinstance(backend_for(({"a": 1},)), RowsBackend)  # a tuple works too
    with pytest.raises(TypeError, match="no compute backend"):
        backend_for("not a table")


# --- remaining check kinds -------------------------------------------------------


def test_string_length_bounds() -> None:
    rows = [{"code": "AB"}, {"code": "ABCDE"}]
    contract = DataContract(
        fields=(FieldSpec("code", "string", Constraints(min_length=2, max_length=3)),)
    )
    report = verify_contract(rows, contract)
    assert report.verdict == "unsound"  # "ABCDE" is too long
    assert {c.name for c in report.clauses} >= {"code.length"}


def test_primary_key_requires_presence_and_uniqueness() -> None:
    rows = [{"id": 1}, {"id": None}]  # a missing key value
    contract = DataContract(
        fields=(FieldSpec("id", "integer"),), table=TableSpec(primary_key=("id",))
    )
    assert verify_contract(rows, contract).verdict == "unsound"


def test_unique_key_over_a_composite_of_columns() -> None:
    rows = [{"a": 1, "b": "x"}, {"a": 1, "b": "x"}]  # the (a, b) pair repeats
    contract = DataContract(
        fields=(FieldSpec("a", "integer"), FieldSpec("b", "string")),
        table=TableSpec(unique_keys=(("a", "b"),)),
    )
    report = verify_contract(rows, contract)
    assert report.verdict == "unsound"
    assert {c.name for c in report.clauses} >= {"table.unique[a+b]"}


def test_a_key_over_an_absent_column_is_inconclusive() -> None:
    contract = DataContract(
        fields=(FieldSpec("a", "integer"),), table=TableSpec(primary_key=("ghost",))
    )
    report = verify_contract([{"a": 1}], contract)
    pk = {c.name: c.verdict for c in report.clauses}["table.primary_key"]
    assert pk == "inconclusive"


def test_foreign_key_passes_when_every_value_is_present() -> None:
    contract = DataContract(
        fields=(FieldSpec("fk", "integer"),),
        table=TableSpec(foreign_keys=(ForeignKey(("fk",), Reference("ref", ("id",))),)),
    )
    report = verify_contract([{"fk": 1}], contract, references={"ref": [{"id": 1}]})
    assert report.verdict == "sound"


# --- suggestion edges ------------------------------------------------------------


def test_a_sparse_column_is_not_required() -> None:
    # Present in only ~10% of rows: below the floor, so it is optional, not required.
    rows = [{"x": "v" if i == 0 else ""} for i in range(20)]
    contract = suggest_contract(rows)
    (field,) = contract.fields
    assert not field.constraints.required and field.mostly == 1.0


def test_a_negative_numeric_column_gets_no_zero_floor() -> None:
    rows = [{"n": str(i - 5)} for i in range(10)]  # spans negatives
    (field,) = suggest_contract(rows).fields
    assert field.constraints.minimum is None


def test_a_high_cardinality_string_gets_no_enum() -> None:
    rows = [{"s": f"val{i}"} for i in range(40)]  # every value distinct
    (field,) = suggest_contract(rows).fields
    assert field.constraints.enum is None


# --- report, and remaining defensive branches ------------------------------------


def test_report_renders_and_attests_with_located_violations() -> None:
    rows = [{"id": 1}, {"id": 1}]  # a duplicate
    contract = DataContract(
        fields=(FieldSpec("id", "integer", Constraints(unique=True)),)
    )
    report = verify_contract(rows, contract)
    rendered = report.render()
    assert "UNSOUND" in rendered and "id.unique" in rendered
    attestation = report.attestation()
    assert attestation["schema"] == "elbi.quality/v1"
    assert attestation["verdict"] == "unsound"
    located = attestation["violations"][0]
    assert located["column"] == "id" and located["rowNumber"] in (1, 2)


def test_a_table_level_violation_attests_and_renders_without_a_row() -> None:
    contract = DataContract(
        fields=(FieldSpec("id", "integer"),), table=TableSpec(row_count_min=5)
    )
    report = verify_contract(_CLEAN, contract)  # 2 rows, below the minimum
    assert "table" in report.render()  # located at the table, with no row number
    attested = [
        v
        for v in report.attestation()["violations"]
        if v["clause"] == "table.row_count"
    ]
    assert attested and "rowNumber" not in attested[0] and "column" not in attested[0]


def test_columns_out_of_declared_order_is_flagged() -> None:
    rows = [{"a": 1, "b": "x"}]
    # The same columns, declared in the opposite order.
    contract = DataContract(
        fields=(FieldSpec("b", "string"), FieldSpec("a", "integer")),
        table=TableSpec(columns_match=True),
    )
    report = verify_contract(rows, contract)
    detail = {c.name: c.detail for c in report.clauses}["table.columns_match"]
    assert report.verdict == "unsound" and "order" in detail


def test_a_foreign_key_over_an_absent_local_column_is_inconclusive() -> None:
    contract = DataContract(
        fields=(FieldSpec("a", "integer"),),
        table=TableSpec(
            foreign_keys=(ForeignKey(("ghost",), Reference("ref", ("id",))),)
        ),
    )
    report = verify_contract([{"a": 1}], contract, references={"ref": [{"id": 1}]})
    fk = {c.name: c.verdict for c in report.clauses}["table.foreign_key[ghost]"]
    assert fk == "inconclusive"


def test_required_check_over_an_empty_table_is_inconclusive() -> None:
    from elbi_core.quality.checks import check_required

    result = check_required(RowsBackend([]), "x.required", "x", 1.0, 10)
    assert result.verdict == "inconclusive"


def test_wilson_bound_of_no_evidence_is_zero() -> None:
    assert wilson_lower_bound(0, 0) == 0.0


def test_profile_of_an_all_missing_column_infers_string() -> None:
    (profile,) = profile_columns([{"x": ""}, {"x": None}])
    assert profile.present == 0 and profile.inferred_type == "string"


def test_profile_serializes_to_a_dict() -> None:
    (profile,) = profile_columns([{"x": "1"}, {"x": "2"}])
    payload = profile.to_dict()
    assert payload["name"] == "x" and payload["inferredType"] == "integer"
    assert payload["topValues"][0]["count"] == 1


def test_every_serializable_constraint_round_trips() -> None:
    # Exercise the manifest branches for maximum, minLength, description, unique keys,
    # a row-count max, columnsMatch, and a contract description.
    contract = DataContract(
        fields=(
            FieldSpec(
                "n",
                "integer",
                Constraints(minimum=0, maximum=10),
                description="a bounded count",
            ),
            FieldSpec("s", "string", Constraints(min_length=1, max_length=8)),
        ),
        table=TableSpec(
            unique_keys=(("n", "s"),), row_count_max=100, columns_match=True
        ),
        description="a fully populated contract",
    )
    contract.validate()  # a programmatic contract validates against the spec
    assert DataContract.from_manifest(contract.to_manifest()) == contract


def test_validate_data_contract_tolerates_a_malformed_fields_value() -> None:
    from elbi_core.quality.contract import validate_data_contract

    # `fields` is not a list, and a field has no name: both are spec violations, and
    # the duplicate-name scan must not itself raise on the malformed input.
    with pytest.raises(SpecValidationError):
        validate_data_contract(
            {"specVersion": "1.0", "kind": "DataContract", "fields": "x"}
        )
    with pytest.raises(SpecValidationError):
        validate_data_contract(
            {
                "specVersion": "1.0",
                "kind": "DataContract",
                "fields": [{"type": "string"}],
            }
        )


def test_type_parsing_covers_every_branch() -> None:
    assert parses_as(5, "integer") and not parses_as(True, "integer")
    assert parses_as(True, "boolean")  # a native bool
    assert not parses_as(5, "boolean")  # a bare int is not a boolean spelling
    assert not parses_as(5, "date")  # a non-string, non-date value
    assert not parses_as(5, "datetime")  # a non-string, non-datetime value
    assert not parses_as("bad", "date")  # an unparseable date string
    assert parses_as(_dt.date(2024, 1, 2), "date")  # a native date
    assert parses_as(_dt.datetime(2024, 1, 2, 3), "datetime")  # a native datetime
    assert not parses_as(_dt.datetime(2024, 1, 1), "date")


def test_a_numeric_range_violation_is_located() -> None:
    rows = [{"n": 5}, {"n": -1}, {"n": "notnum"}]
    contract = DataContract(fields=(FieldSpec("n", "number", Constraints(minimum=0)),))
    report = verify_contract(rows, contract)
    assert report.verdict == "unsound"
    offending = {v.row_number for v in report.violations if v.clause == "n.range"}
    assert offending == {2, 3}  # the negative value and the non-numeric one


def test_columns_match_passes_and_names_missing_columns() -> None:
    ok = DataContract(
        fields=(FieldSpec("a", "integer"),), table=TableSpec(columns_match=True)
    )
    assert verify_contract([{"a": 1}], ok).verdict == "sound"

    drift = DataContract(
        fields=(FieldSpec("a", "integer"), FieldSpec("b", "string")),
        table=TableSpec(columns_match=True),
    )
    report = verify_contract([{"a": 1}], drift)  # column b is absent
    detail = {c.name: c.detail for c in report.clauses}["table.columns_match"]
    assert report.verdict == "unsound" and "missing" in detail


def test_a_value_constraint_over_no_present_values_passes_vacuously() -> None:
    # The column is entirely missing but not required, so its pattern check has nothing
    # to reject and passes rather than blocking on absent data.
    rows = [{"e": ""}, {"e": None}]
    contract = DataContract(
        fields=(FieldSpec("e", "string", Constraints(pattern="@")),)
    )
    verdicts = {c.name: c.verdict for c in verify_contract(rows, contract).clauses}
    assert verdicts["e.pattern"] == "sound"


# --- the authoring gate: references, non-row output, and the policy --------------


def test_a_foreign_key_resolves_through_the_runner_during_certification() -> None:
    reg = Registry()

    @derivation(name="plans", serve=serve.table(), registry=reg)
    def plans(ctx: Context) -> list[dict[str, object]]:
        """The reference table."""
        return [{"id": 1}, {"id": 2}]

    fk_contract = DataContract(
        fields=(FieldSpec("plan_id", "integer"),),
        table=TableSpec(
            foreign_keys=(ForeignKey(("plan_id",), Reference("plans", ("id",))),)
        ),
    )

    @derivation(name="rows_ok", serve=serve.table(), contract=fk_contract, registry=reg)
    def rows_ok(ctx: Context) -> list[dict[str, object]]:
        """Every plan_id exists in plans."""
        return [{"plan_id": 1}, {"plan_id": 2}]

    @derivation(
        name="rows_bad", serve=serve.table(), contract=fk_contract, registry=reg
    )
    def rows_bad(ctx: Context) -> list[dict[str, object]]:
        """A dangling plan_id."""
        return [{"plan_id": 99}]

    runner = Runner(reg)
    assert verify(runner, reg.get("rows_ok")).contract_verdict == "sound"
    assert verify(runner, reg.get("rows_bad")).contract_verdict == "unsound"


def test_a_contract_over_non_row_output_is_inconclusive() -> None:
    reg = Registry()
    contract = DataContract(fields=(FieldSpec("x", "integer"),))

    @derivation(name="not_rows", serve=serve.json(), contract=contract, registry=reg)
    def not_rows(ctx: Context) -> list[int]:
        """A list, but not of row dicts, so no contract can be checked over it."""
        return [1, 2, 3]

    result = verify(Runner(reg), reg.get("not_rows"))
    assert result.contract_verdict == "inconclusive" and not result.ok


def test_require_sound_contract_policy_gates_on_the_contract() -> None:
    from elbi_core import RequireSoundContract
    from elbi_core.authoring import VerificationResult

    reg = Registry()

    @derivation(name="d", serve=serve.table(), registry=reg)
    def d(ctx: Context) -> list[dict[str, object]]:
        """A stand-in derivation."""
        return [{"x": 1}]

    policy = RequireSoundContract()
    deriv = reg.get("d")
    assert policy.should_certify(
        deriv, VerificationResult(ok=True, contract_verdict="sound")
    )
    assert not policy.should_certify(
        deriv, VerificationResult(ok=True, contract_verdict="unsound")
    )
    assert not policy.should_certify(
        deriv, VerificationResult(ok=False, contract_verdict="sound")
    )
