"""A derivation's result history: how its certified estimate changed, and what moved it.

Comparing two certified versions of the same derivation answers the question tracking
exists for: the estimate moved from one version to the next, and *why*: the data
refreshed, the code changed, or the adjustment set changed. Because a version is a hash
of those components and each is stored on the run, the attribution is exact.

This needs no tracking server: a derivation's history is the runs already recorded for
its name, newest first, and the diff is a comparison of adjacent versions.
"""

from __future__ import annotations

from collections.abc import Sequence

from .run import CertifiedRun

#: The input dimensions a version can differ on, in the order they are reported.
_DIMENSIONS = ("data", "code", "controls", "params", "claim")


def changed_dimensions(newer: CertifiedRun, older: CertifiedRun) -> tuple[str, ...]:
    """Which inputs differ between two certified versions of a derivation.

    Returns any of ``data`` (an input dataset version changed), ``code`` (the derivation
    source changed), ``controls`` (the adjustment set changed), ``params``, or
    ``claim``, in a stable order. A dimension whose component was not captured on either
    run (e.g. a backfilled run predating component capture) is not reported, so the diff
    degrades to what it can attribute rather than over-claiming a change.
    """
    changes: list[str] = []
    if newer.input_versions != older.input_versions:
        changes.append("data")
    if (
        newer.code_version is not None
        and older.code_version is not None
        and newer.code_version != older.code_version
    ):
        changes.append("code")
    if set(newer.adjusted_for) != set(older.adjusted_for):
        changes.append("controls")
    if newer.params != older.params:
        changes.append("params")
    if newer.claim != older.claim:
        changes.append("claim")
    return tuple(c for c in _DIMENSIONS if c in changes)


def with_changes(
    runs: Sequence[CertifiedRun],
) -> list[tuple[CertifiedRun, tuple[str, ...]]]:
    """Pair each run with what changed from the next-older one.

    ``runs`` is expected newest first (the store's order). Each run is paired with the
    dimensions that differ from its immediate predecessor in time; the oldest run pairs
    with an empty tuple, since there is nothing before it to have moved.
    """
    out: list[tuple[CertifiedRun, tuple[str, ...]]] = []
    for index, run in enumerate(runs):
        older = runs[index + 1] if index + 1 < len(runs) else None
        changed = changed_dimensions(run, older) if older is not None else ()
        out.append((run, changed))
    return out
