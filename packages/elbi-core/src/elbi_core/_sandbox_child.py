"""Child entry point for :class:`~elbi.executor.SubprocessExecutor`.

Not a public API. Reads a pickled job from ``<io_dir>/job.pkl`` (written by the
trusted parent) and writes the result to ``<io_dir>/result.json``. Two job modes
share this one hardened sandbox:

* a **derivation** job (``{source, fn_name, inputs, params}``) runs a derivation's
  compute function and returns its artifact, and
* an **exploration** job (``{mode: "exec", code, data}``) runs arbitrary code with
  the named datasets in scope and returns its stdout and ``result`` value, for the
  iterative explore/debug loop that precedes authoring a derivation.

The result is JSON, never a pickle: the parent runs untrusted code here, so it
must not unpickle anything this process produces.
"""

from __future__ import annotations

import contextlib
import io
import json
import pickle
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

#: Cap stdout and the repr'd result so a runaway print does not flood the parent.
_MAX_STDOUT = 8000
_MAX_RESULT = 4000


def main(io_dir: str) -> int:
    """Run one sandboxed job; return 0 on success, 1 on failure."""
    base = Path(io_dir)
    # The job is written by the trusted parent into a private temp dir.
    job = pickle.loads((base / "job.pkl").read_bytes())  # noqa: S301
    result_path = base / "result.json"
    restore = None if job.get("allow_network", False) else _deny_network()
    try:
        if job.get("mode") == "exec":
            return _run_code(job, result_path)
        return _run_derivation(job, result_path)
    finally:
        # Restore the original socket so in-process callers (tests) are unaffected.
        if restore is not None:
            restore()


def _run_derivation(job: dict[str, Any], result_path: Path) -> int:
    try:
        from .artifact import coerce_artifact
        from .context import Context

        namespace: dict[str, Any] = {}
        exec(compile(job["source"], "<derivation>", "exec"), namespace)  # noqa: S102
        fn = namespace[job["fn_name"]]
        artifact = coerce_artifact(fn(Context(job["inputs"], job["params"])))
        _write(
            result_path, {"ok": True, "kind": artifact.kind, "value": artifact.value}
        )
    except BaseException as exc:  # any failure is reported back to the parent
        _write(
            result_path,
            {
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        )
        return 1
    return 0


def _run_code(job: dict[str, Any], result_path: Path) -> int:
    """Run exploratory code; capture stdout, the ``result`` value, and any error.

    A failure in the code is reported back as ``error`` (a traceback) rather than a
    sandbox failure, so the agent can read it and iterate, as in a REPL.
    """
    buf = io.StringIO()
    namespace: dict[str, Any] = {"data": job.get("data", {})}
    error: str | None = None
    with contextlib.redirect_stdout(buf):
        try:
            exec(compile(job["code"], "<exploration>", "exec"), namespace)  # noqa: S102
        except BaseException:  # report it, do not fail the sandbox
            error = traceback.format_exc()
    result = namespace.get("result")
    _write(
        result_path,
        {
            "ok": True,
            "mode": "exec",
            "stdout": buf.getvalue()[:_MAX_STDOUT],
            "result": None if result is None else repr(result)[:_MAX_RESULT],
            "error": error,
        },
    )
    return 0


def _deny_network() -> Callable[[], None]:
    """Block network egress in the sandbox child and return a restore callable.

    Best-effort defense in depth, not a security boundary: replacing
    ``socket.socket`` with one that raises denies the usual egress stacks
    (requests, urllib, http.client), which all construct it.
    """
    import socket

    saved = socket.socket

    class _BlockedSocket(socket.socket):
        def __init__(self, *_args: object, **_kwargs: object) -> NoReturn:
            raise OSError("network access is disabled in the elbi sandbox")

    # Write the module's own dict; mypy rejects assigning to the ``socket`` type.
    socket.__dict__["socket"] = _BlockedSocket

    def restore() -> None:
        socket.__dict__["socket"] = saved

    return restore


def _write(path: Path, payload: dict[str, Any]) -> None:
    # default=str degrades non-JSON values (e.g. datetimes) instead of raising;
    # the result is fingerprint-grade only, matching the rest of the SDK.
    path.write_text(json.dumps(payload, default=str), encoding="utf-8")


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main(sys.argv[1]))
