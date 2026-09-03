"""Resolving a notebook's dependencies to a pinned lock, the reproducibility capstone.

A notebook declares a loose environment spec (``["scikit-learn", "matplotlib"]``), which
is convenient but not reproducible: re-resolving it next month picks up new versions.
Locking runs ``uv pip compile`` over the spec to a fully-pinned ``name==version`` set
(the resolved transitive graph), so a scheduled or parameterized rerun provisions the
exact same environment. The kernel then installs from the lock in preference to the
loose spec: the same declarative-spec → lockfile → cached-environment pattern the modern
platforms use, with uv making the resolve fast enough to be routine.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

from ..errors import DerivationError
from ..executor import _DEP_RE

#: The interpreter the lock resolves against. It matches the default container image
#: (``python:3.12-slim``) so a lock made on the host installs cleanly in the sandbox.
DEFAULT_PYTHON_VERSION = "3.12"


def resolve_lock(
    deps: Sequence[str],
    *,
    python_version: str = DEFAULT_PYTHON_VERSION,
    timeout: float = 180.0,
    uv_bin: str = "uv",
) -> list[str]:
    """Pin ``deps`` to an exact ``name==version`` set via ``uv pip compile``.

    Returns the resolved specs (empty for empty input). Raises
    :class:`~elbi.errors.DerivationError` when a dependency is malformed, uv is absent,
    or resolution fails (an unsatisfiable constraint, a typo'd package), the caller
    surfaces that so the notebook keeps its previous lock rather than a broken one.
    """
    specs = [str(dep).strip() for dep in deps if str(dep).strip()]
    if not specs:
        return []
    for spec in specs:
        if not _DEP_RE.match(spec):
            raise DerivationError(f"refusing to lock invalid dependency {spec!r}")
    try:
        completed = subprocess.run(  # noqa: S603
            [
                uv_bin,
                "pip",
                "compile",
                "-",
                "--quiet",
                "--no-header",
                "--python-version",
                python_version,
            ],
            input="\n".join(specs) + "\n",
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError as exc:
        raise DerivationError(
            f"uv executable {uv_bin!r} not found; cannot lock the environment"
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise DerivationError("locking the environment timed out") from exc
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip().splitlines()
        message = detail[-1] if detail else "resolution failed"
        raise DerivationError(f"could not resolve the environment: {message}")
    return _parse_pins(completed.stdout)


def _parse_pins(output: str) -> list[str]:
    """Extract the pinned ``name==version`` lines from ``uv pip compile`` output.

    The compiler interleaves comments (``# via ...``) and options (``--index-url``); a
    pinned requirement is a line that starts at column zero with a package spec.
    """
    pins: list[str] = []
    for line in output.splitlines():
        stripped = line.strip()
        if not stripped or line[:1].isspace() or stripped.startswith(("#", "-")):
            continue
        # Drop any inline ``# via`` comment uv appends after the spec on the same line.
        pins.append(stripped.split("#", 1)[0].strip())
    return pins
