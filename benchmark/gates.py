"""Dispatch from a fixture's ``gate`` string to an oracle verification function.

Every gate here takes ``rows`` plus the fixture's ``claim`` as keyword arguments and
returns a report exposing ``.verdict`` and ``.render()`` (``CompositeReport`` from
``verify_all``, ``VerificationReport`` from a direct gate, or ``MultiverseReport``).
The harness reads those two members uniformly, so no gate needs special casing.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol

from elbi_core import verify_all
from elbi_core.verification import (
    verify_experiment,
    verify_extrapolation,
    verify_leakage,
    verify_multiverse,
    verify_powerlaw,
    verify_proportional_hazards,
    verify_rtm,
)

# verify_spatial is reached via verify_all's catalog and is not re-exported from the
# package facade, so it is imported from its own module for direct-gate traps.
from elbi_core.verification.spatial import verify_spatial


class Report(Protocol):
    """The shared surface every gate report exposes, read by the harness."""

    verdict: str

    def render(self) -> str:
        """Render the report as markdown."""
        ...


#: Fixture ``gate`` string -> the verification function it dispatches to.
GATES: dict[str, Callable[..., Report]] = {
    "verify_all": verify_all,
    "verify_experiment": verify_experiment,
    "verify_leakage": verify_leakage,
    "verify_multiverse": verify_multiverse,
    "verify_powerlaw": verify_powerlaw,
    "verify_spatial": verify_spatial,
    "verify_extrapolation": verify_extrapolation,
    "verify_proportional_hazards": verify_proportional_hazards,
    "verify_rtm": verify_rtm,
}


def run_gate(gate: str, rows: list[dict[str, str]], claim: dict[str, Any]) -> Report:
    """Run the named gate over ``rows`` with ``claim`` as keyword arguments."""
    return GATES[gate](rows, **claim)
