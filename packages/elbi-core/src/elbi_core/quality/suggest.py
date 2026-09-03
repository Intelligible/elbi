"""Propose a contract from a table by profiling it and applying deterministic rules.

The suggestion is a starting point a human or agent edits, never a certified verdict.
Every threshold it proposes is a confidence bound, not the observed sample rate: a
column 92% complete over a few hundred rows yields a looser floor than the same rate
over millions, so a contract proposed from a sample does not overfit it and false-alarm
on the next batch. The rules are conservative -- they propose only constraints the data
clearly supports (a complete column is required, a fully distinct one is unique, a
non-negative numeric one has a zero floor, a low-cardinality string one has an enum) and
leave the rest to the author.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from ._stats import wilson_lower_bound
from .contract import Constraints, DataContract, FieldSpec, TableSpec
from .profile import ColumnProfile, profile_columns

#: A column at least this complete (but not fully) is proposed as required, with a
#: completeness tolerance set to the Wilson lower bound of its observed rate. Below it,
#: the column is treated as legitimately optional and not required at all.
_REQUIRED_FLOOR = 0.2
#: A string column with at most this many distinct values, most of them recurring, is
#: proposed as an enum over the values seen.
_ENUM_MAX_CARDINALITY = 15
_ENUM_MIN_ROWS = 20
#: At most this share of distinct values may appear exactly once for a column to read as
#: categorical (an identifier, where every value is unique, sits near 1.0).
_ENUM_MAX_UNIQUE_ONCE = 0.1


def suggest_contract(data: Any) -> DataContract:
    """Profile ``data`` and propose a :class:`DataContract` for it.

    ``data`` is anything a compute backend handles (``list[dict]`` rows, or an engine
    handle over a larger-than-memory file). The result is validated against the spec, so
    it is always well-formed. It asserts nothing on its own; run it through the checker
    (or the authoring loop) to get a verdict.
    """
    profiles = profile_columns(data)
    total = profiles[0].count if profiles else 0
    primary_key = _primary_key(profiles)
    fields = tuple(
        _field_for(profile, is_primary=profile.name == primary_key)
        for profile in profiles
    )
    table = TableSpec(
        primary_key=(primary_key,) if primary_key is not None else (),
        row_count_min=1 if total > 0 else None,
    )
    contract = DataContract(fields=fields, table=table)
    contract.validate()
    return contract


def _primary_key(profiles: Sequence[ColumnProfile]) -> str | None:
    """The first fully present, fully distinct column, if any (a key candidate)."""
    for profile in profiles:
        if profile.completeness == 1.0 and profile.is_unique:
            return profile.name
    return None


def _field_for(profile: ColumnProfile, *, is_primary: bool) -> FieldSpec:
    """Propose one field's type, constraints, and completeness tolerance."""
    required, mostly = _completeness(profile)
    constraints = Constraints(
        required=required,
        # The primary key carries uniqueness at the table level; other fully distinct
        # complete columns get a field-level unique constraint.
        unique=profile.is_unique and profile.completeness == 1.0 and not is_primary,
        minimum=_nonnegative_floor(profile),
        enum=_enum(profile),
    )
    return FieldSpec(
        name=profile.name,
        type=profile.inferred_type,
        constraints=constraints,
        mostly=mostly,
    )


def _completeness(profile: ColumnProfile) -> tuple[bool, float]:
    """Whether to require the column, and the completeness/validity tolerance.

    A fully present column is required strictly. A partly present one (above the floor)
    is required with a tolerance set to the Wilson lower bound of its completeness, so
    the proposal accounts for sample size rather than pinning the observed rate. A
    sparsely present column is not required.
    """
    completeness = profile.completeness
    if completeness >= 1.0:
        return True, 1.0
    if completeness >= _REQUIRED_FLOOR:
        return True, _floor2(wilson_lower_bound(profile.present, profile.count))
    return False, 1.0


def _nonnegative_floor(profile: ColumnProfile) -> float | None:
    """A zero minimum for a numeric column whose values are all non-negative."""
    if profile.inferred_type in ("integer", "number") and profile.minimum is not None:
        return 0.0 if profile.minimum >= 0 else None
    return None


def _enum(profile: ColumnProfile) -> tuple[str, ...] | None:
    """A closed value set for a low-cardinality, recurring string column."""
    if profile.inferred_type != "string":
        return None
    if profile.present < _ENUM_MIN_ROWS or profile.distinct > _ENUM_MAX_CARDINALITY:
        return None
    if profile.fraction_unique_once > _ENUM_MAX_UNIQUE_ONCE:
        return None
    return tuple(str(value) for value, _ in profile.top_values)


def _floor2(value: float) -> float:
    """Round down to two decimals, so a proposed tolerance is a clean lower bound."""
    return math.floor(value * 100) / 100
