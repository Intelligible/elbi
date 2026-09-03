"""Blocking bite gate: every trap must kill at least one mutant of its gate module.

Per trap, Cosmic Ray mutates the one verification module the trap targets
(``Trap.bite_module``), running only that trap's case. A trap that kills a mutant
depends on its gate; a trap that kills none passes regardless of its gate and is
rejected (non-zero exit). Survivors are fine: they are gate behaviour a trap need not
exercise, so this is a non-vacuity check, not a zero-survivor gate that fires on
unrelated code.

Cosmic Ray mutates source in place, so this runner is serial (two sessions cannot share
a module in one checkout); parallelism comes from the CI matrix, one module per runner.

  python -m benchmark.bite                      # every trap
  python -m benchmark.bite --module effect.py   # one module
  python -m benchmark.bite --trap simpsons_ab_reversal
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from cosmic_ray.mutating import mutate_and_test
from cosmic_ray.work_db import WorkDB, use_db
from cosmic_ray.work_item import TestOutcome

from .loader import BENCHMARK_DIR, load_traps
from .trap import Trap

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cosmic_ray.work_item import WorkItem

REPO_ROOT = BENCHMARK_DIR.parent
VERIFICATION_DIR = REPO_ROOT / "packages/elbi-core/src/elbi_core/verification"

# _bitecheck exits non-zero only when the mutation flips the trap's outcome (a real
# kill). Bare `python` (not `uv run`) skips re-resolving the env per mutant.
_TEST_TEMPLATE = "python -m benchmark._bitecheck {trap_id}"
_MUTANT_TIMEOUT = 30.0


@dataclass(frozen=True)
class BiteResult:
    """One trap's outcome against its scoped gate module."""

    trap_id: str
    module: str | None
    killed: int
    survived: int
    total: int
    skipped: str | None  # reason the trap has no bite target, else None

    @property
    def passed(self) -> bool:
        """One kill proves the trap depends on its gate; a skipped trap never fails."""
        return self.skipped is not None or self.killed >= 1


def _config_toml(module_path: Path, trap_id: str) -> str:
    test_command = _TEST_TEMPLATE.format(trap_id=trap_id)
    return (
        "[cosmic-ray]\n"
        f'module-path = "{module_path}"\n'
        f"timeout = {_MUTANT_TIMEOUT}\n"
        f'test-command = "{test_command}"\n'
        "excluded-modules = []\n"
        "\n"
        "[cosmic-ray.distributor]\n"
        'name = "local"\n'
    )


def _raise_interrupt() -> None:
    raise KeyboardInterrupt


def _cosmic_ray(*args: str) -> None:
    exe = shutil.which("cosmic-ray")
    if exe is None:
        raise RuntimeError("cosmic-ray not on PATH; run under `uv run`")
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [exe, *args],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"cosmic-ray {args[0]} failed ({proc.returncode}):\n{proc.stderr}"
        )


def _mutant_key(item: WorkItem) -> bytes:
    """A stable hash of the mutation, so mutant order does not drift between runs.

    The job id is a fresh UUID per ``init``; the operator, occurrence, and position are
    not, so hashing them gives the same interleaving each run.
    """
    parts = [
        f"{m.module_path}:{m.operator_name}:{m.occurrence}:{m.start_pos}"
        for m in item.mutations
    ]
    return hashlib.sha1("|".join(parts).encode(), usedforsecurity=False).digest()


def _work_items(session: Path) -> list[WorkItem]:
    """The module's mutants, interleaved by a stable hash so a kill surfaces early.

    In file order a trap's kills cluster in one region and it wades through many
    survivors first; the shuffle brings a kill within a handful.
    """
    with use_db(str(session), WorkDB.Mode.open) as db:
        items = list(db.work_items)
    items.sort(key=_mutant_key)
    return items


@contextmanager
def _chdir(target: Path) -> Iterator[None]:
    """Run the mutant test command from the repo root (test-command is relative)."""
    prev = Path.cwd()
    os.chdir(target)
    try:
        yield
    finally:
        os.chdir(prev)


def run_bite(trap: Trap, *, keep: bool = False) -> BiteResult:
    """Mutate the trap's gate module, stopping at the first mutant the trap kills.

    A vacuous trap kills nothing and runs every mutant. ``mutate_and_test`` restores the
    source per mutant.
    """
    if trap.bite_module is None:
        return BiteResult(trap.id, None, 0, 0, 0, skipped="no bite module declared")
    module_path = VERIFICATION_DIR / trap.bite_module
    if not module_path.exists():
        raise FileNotFoundError(f"{trap.id}: bite module not found: {module_path}")
    tmp = Path(tempfile.mkdtemp(prefix=f"bite-{trap.id}-"))
    config = tmp / "config.toml"
    session = tmp / "session.sqlite"
    config.write_text(_config_toml(module_path, trap.id), encoding="utf-8")
    test_command = _TEST_TEMPLATE.format(trap_id=trap.id)
    killed = survived = total = 0
    # An interrupt can break mid-mutation, so snapshot and restore unconditionally.
    original = module_path.read_bytes()
    try:
        _cosmic_ray("init", str(config), str(session))
        items = _work_items(session)
        total = len(items)
        with _chdir(REPO_ROOT):
            for item in items:
                result = mutate_and_test(item.mutations, test_command, _MUTANT_TIMEOUT)
                if result.test_outcome == TestOutcome.KILLED:
                    killed += 1
                    break
                if result.test_outcome == TestOutcome.SURVIVED:
                    survived += 1
    finally:
        module_path.write_bytes(original)
        if not keep:
            shutil.rmtree(tmp, ignore_errors=True)
    return BiteResult(trap.id, trap.bite_module, killed, survived, total, None)


def _select(traps: list[Trap], module: str | None, trap_id: str | None) -> list[Trap]:
    chosen = traps
    if module is not None:
        chosen = [t for t in chosen if t.bite_module == module]
    if trap_id is not None:
        chosen = [t for t in chosen if t.id == trap_id]
    return chosen


def _render(results: list[BiteResult]) -> str:
    # `ran` is how many mutants executed before we stopped (a live trap stops at the
    # first kill); `mutants` is the full set Cosmic Ray generated for the module.
    width = max((len(r.trap_id) for r in results), default=4)
    lines = [f"{'trap':{width}}  status  killed  ran  mutants  module"]
    for r in results:
        if r.skipped is not None:
            status, detail = "SKIP", r.skipped
        else:
            status, detail = ("ok" if r.passed else "REJECT"), (r.module or "")
        ran = r.killed + r.survived
        lines.append(
            f"{r.trap_id:{width}}  {status:6}  "
            f"{r.killed:6}  {ran:3}  {r.total:7}  {detail}"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Run the bite gate; exit non-zero if any trap kills zero mutants."""
    parser = argparse.ArgumentParser(
        description="Blocking gate: each trap must kill a mutant of its gate module."
    )
    parser.add_argument("--module", help="only traps whose bite module is this file")
    parser.add_argument("--trap", help="only this trap id")
    parser.add_argument(
        "--keep", action="store_true", help="keep temp Cosmic Ray sessions"
    )
    args = parser.parse_args(argv)

    # Promote SIGTERM (a CI timeout) to KeyboardInterrupt so run_bite's finally
    # restores the source.
    signal.signal(signal.SIGTERM, lambda *_: _raise_interrupt())

    traps = _select(load_traps(), args.module, args.trap)
    if not traps:
        print("no traps matched the selection", file=sys.stderr)
        return 1

    results: list[BiteResult] = []
    for trap in traps:
        print(f"biting {trap.id} ({trap.bite_module}) ...", file=sys.stderr, flush=True)
        results.append(run_bite(trap, keep=args.keep))

    print(_render(results))
    rejected = [r for r in results if not r.passed]
    if rejected:
        ids = ", ".join(r.trap_id for r in rejected)
        print(
            f"\n{len(rejected)} trap(s) killed no mutants of their gate "
            f"(they pass regardless of the gate): {ids}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
