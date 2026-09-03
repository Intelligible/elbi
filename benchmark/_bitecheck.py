"""Bite-gate test command: exit non-zero only when a mutation flips a trap's outcome.

A mutant is "killed" only when it flips the trap's verdict or drops its pivotal reason,
so the trap genuinely depends on the mutated behaviour. A mutant that instead breaks the
module's import or crashes the gate is reported as survived, so neither masquerades as a
real dependency.

  exit 1  the trap no longer passes -> a real kill
  exit 0  the trap still passes, or the gate raised/failed to import -> survived

This assumes the trap passes on unmutated code, which the required ``test_traps`` job
enforces; a trap red at baseline is caught there, not here.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        return 0
    trap_id = argv[0]
    try:
        # Import inside the guard: importing benchmark loads the (mutated) gate module,
        # so an import-breaking mutant is caught here (survives), not counted as a kill.
        from .harness import score
        from .loader import load_traps

        matches = [t for t in load_traps() if t.id == trap_id]
        if not matches:
            return 0
        result = score(matches[0], {})
    except Exception:
        # A mutation that breaks the gate's import or crashes it is not a real kill.
        return 0
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
