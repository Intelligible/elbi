"""Property tests for the copy-name generators (:mod:`elbi.duplicate`).

These are the mechanism behind two acceptance criteria: a duplicate always gets a
distinct name rather than erroring (never raises, never returns something already
taken), and a dashboard/metric duplicate's name stays valid against the Open Derivation
Spec identifier pattern the manifest schemas enforce.
"""

from __future__ import annotations

import re

from hypothesis import given
from hypothesis import strategies as st

from elbi.duplicate import copy_identifier, copy_label

#: The same pattern `dashboard.schema.json` and `metric.schema.json` enforce on `name`.
_IDENTIFIER_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")

#: Wider than the ``limit`` the tests pass below, so the strategy reaches the
#: truncation path where a suffix competes with the base for room.
_names = st.text(min_size=0, max_size=60)
_taken_sets = st.sets(st.text(min_size=1, max_size=20), max_size=25)


@given(base=_names, taken=_taken_sets)
def test_copy_identifier_is_schema_valid_and_fresh(base: str, taken: set[str]) -> None:
    result = copy_identifier(base, taken, limit=32)
    assert result not in taken
    assert _IDENTIFIER_PATTERN.match(result)
    assert len(result) <= 32


@st.composite
def _labels_and_limits(draw: st.DrawFn) -> tuple[str, int]:
    """A base name paired with a limit drawn around its own length.

    Correlating the two is what reaches the boundary: a suffix stops fitting beside
    the base only within a few characters of the limit, and two independent draws
    land in that window far too rarely to exercise it. Floored well above the suffix
    length, since no generator can produce a distinct name in a handful of
    characters. The real limits are 128 and 256.
    """
    base = draw(st.text(min_size=0, max_size=60))
    width = len(base.strip() or "Untitled")
    offset = draw(st.integers(min_value=-8, max_value=8))
    return base, max(16, width + offset)


@given(case=_labels_and_limits(), taken=_taken_sets)
def test_copy_label_is_fresh(case: tuple[str, int], taken: set[str]) -> None:
    base, limit = case
    # The source's own name is always among the taken ones in practice, since the
    # caller builds the set from artifacts that include the one being copied.
    held = taken | {base}
    result = copy_label(base, held, limit=limit)
    assert result not in held
    assert len(result) <= limit


def test_copy_label_is_fresh_when_the_base_exceeds_the_limit() -> None:
    """A name at the limit still yields a copy distinct from it.

    The suffix has to displace part of the base rather than being truncated off
    the end, or every attempt collapses to the same string and the generator
    hands back the name it was asked to avoid.
    """
    base = "A" * 256
    assert copy_label(base, {base}) != base


def test_copy_identifier_first_choice_is_readable() -> None:
    assert copy_identifier("revenue", set()) == "revenue_copy"
    assert copy_identifier("Revenue", {"revenue_copy"}) == "revenue_copy_2"


def test_copy_label_first_choice_is_readable() -> None:
    assert copy_label("Revenue", set()) == "Revenue (copy)"
    assert copy_label("Revenue", {"Revenue (copy)"}) == "Revenue (copy 2)"


def test_copy_identifier_survives_a_name_with_no_letters() -> None:
    # An all-symbol base has nothing left after slugifying; "copy" is the floor rather
    # than an empty (and schema-invalid) identifier.
    assert copy_identifier("!!!", set()) == "copy_copy"


def test_copy_identifier_exhausts_the_numbered_sequence_without_raising() -> None:
    taken = {"x_copy", *(f"x_copy_{n}" for n in range(2, 101))}
    result = copy_identifier("x", taken)
    assert result not in taken
    assert _IDENTIFIER_PATTERN.match(result)


def test_copy_label_exhausts_the_numbered_sequence_without_raising() -> None:
    taken = {"x (copy)", *(f"x (copy {n})" for n in range(2, 101))}
    result = copy_label("x", taken)
    assert result not in taken
