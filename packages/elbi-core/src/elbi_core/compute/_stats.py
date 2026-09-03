"""Pure-Python value helpers shared by every compute backend.

Kept dependency-free (standard library only) so the ``list[dict]`` reference backend
runs without numpy or a dataframe engine. Type parsing here *validates* a value against
a declared type without mutating it: a verifying oracle must be able to report a
violation, never silently coerce it away.
"""

from __future__ import annotations

import datetime as _dt
import math
from typing import Any

#: Values treated as missing (absent), judged only by a field's ``required``
#: constraint and never by its value constraints. An explicit ``None`` and the empty
#: string cover the common cases: a JSON null and a blank CSV cell.
MISSING: tuple[Any, ...] = (None, "")


def is_missing(value: Any) -> bool:
    """Whether a cell counts as missing (absent), not merely falsy."""
    return value is None or value == ""


def to_float(value: Any) -> float | None:
    """Parse a value as a float, or ``None`` if it does not parse.

    Booleans do not count as numbers: ``True`` is not the value ``1`` here, so a
    boolean column is never mistaken for numeric.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


_BOOL_TRUE = {"true", "1", "yes", "t", "y"}
_BOOL_FALSE = {"false", "0", "no", "f", "n"}


def parses_as(value: Any, type_name: str) -> bool:
    """Whether a present value conforms to a declared field type, without mutating it.

    ``string`` accepts any scalar; ``integer`` accepts whole numbers (including a
    float like ``5.0`` and the string ``"5"``, but not ``"5.3"``); ``number`` accepts
    any real; ``boolean`` accepts the common truthy/falsey spellings; ``date`` and
    ``datetime`` accept ISO-8601 values (or native ``date``/``datetime`` objects).
    A container value (list/dict) conforms to no scalar type.
    """
    if isinstance(value, list | dict):
        return False
    if type_name == "string":
        return isinstance(value, str | int | float | bool)
    if type_name == "integer":
        return _is_integer(value)
    if type_name == "number":
        return to_float(value) is not None
    if type_name == "boolean":
        return _is_boolean(value)
    if type_name == "date":
        return _parses_date(value)
    if type_name == "datetime":
        return _parses_datetime(value)
    return True


def _is_integer(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    if isinstance(value, int):
        return True
    number = to_float(value)
    return number is not None and number.is_integer()


def _is_boolean(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, str):
        return value.strip().lower() in (_BOOL_TRUE | _BOOL_FALSE)
    return False


def _parses_date(value: Any) -> bool:
    if isinstance(value, _dt.datetime):
        return False  # a datetime is not a bare date
    if isinstance(value, _dt.date):
        return True
    if isinstance(value, str):
        try:
            _dt.date.fromisoformat(value.strip())
            return True
        except ValueError:
            return False
    return False


def _parses_datetime(value: Any) -> bool:
    if isinstance(value, _dt.datetime):
        return True
    if isinstance(value, str):
        try:
            _dt.datetime.fromisoformat(value.strip())
            return True
        except ValueError:
            return False
    return False


def wilson_lower_bound(successes: int, total: int, *, z: float = 1.96) -> float:
    """The lower bound of the Wilson score interval for a binomial proportion.

    A defensible floor for an observed success rate ``successes / total``: it accounts
    for sample size, so a rate seen over few rows yields a looser bound than the same
    rate over many. Suggested tolerances use this rather than the raw sample rate, so a
    contract proposed from a sample does not overfit it and false-alarm on the next
    batch. Returns 0.0 for an empty sample (no evidence, no floor).
    """
    if total <= 0:
        return 0.0
    phat = successes / total
    z2 = z * z
    denom = 1.0 + z2 / total
    center = phat + z2 / (2.0 * total)
    margin = z * math.sqrt((phat * (1.0 - phat) + z2 / (4.0 * total)) / total)
    lower = (center - margin) / denom
    return max(0.0, min(1.0, lower))
