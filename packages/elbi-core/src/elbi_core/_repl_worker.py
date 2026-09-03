"""Child entry point for a stateful exploration session (a persistent REPL).

Not a public API. Unlike :mod:`elbi._sandbox_child`, which runs one job and
exits, this process persists: it reads newline-delimited JSON requests on stdin,
executes each in a namespace that survives across requests (so a variable or import from
one call is available in the next), and writes one newline-delimited JSON response on
stdout. The user code's own stdout is captured, so the process's real stdout carries
only protocol messages. It is stdlib-only, so it runs by file path in a uv-provisioned
environment without needing ``elbi`` installed there.

State persistence is deliberate and scoped to exploration: it gives an agent the
iterative, notebook-like loop that data work needs. It is never the path to a verified
answer; that is a derivation, which runs one-shot and hermetic in
:mod:`elbi._sandbox_child`.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import traceback
from pathlib import Path
from typing import Any, NoReturn

#: Cap stdout and the repr'd result so a runaway print cannot flood the parent.
_MAX_STDOUT = 8000
_MAX_RESULT = 4000


def _run_once(namespace: dict[str, Any], code: str) -> dict[str, Any]:
    """Execute one request in the persistent namespace; capture stdout and result.

    ``result`` is popped, not read, so it does not silently persist into the next
    request; everything else in the namespace stays for continuity.
    """
    buf = io.StringIO()
    error: str | None = None
    with contextlib.redirect_stdout(buf):
        try:
            exec(compile(code, "<session>", "exec"), namespace)  # noqa: S102
        except BaseException:  # report it; the session survives so the agent iterates
            error = traceback.format_exc()
    result = namespace.pop("result", None)
    return {
        "ok": True,
        "stdout": buf.getvalue()[:_MAX_STDOUT],
        "result": None if result is None else repr(result)[:_MAX_RESULT],
        "error": error,
    }


def _deny_network() -> None:
    """Block network egress in the session, matching the one-shot sandbox default.

    Best-effort defense in depth (replacing ``socket.socket`` denies the usual egress
    stacks), not a security boundary; kernel-level isolation is the container backend's.
    """
    import socket

    class _BlockedSocket(socket.socket):
        def __init__(self, *_a: object, **_k: object) -> NoReturn:
            raise OSError("network access is disabled in the elbi sandbox")

    socket.__dict__["socket"] = _BlockedSocket


def main(argv: list[str]) -> int:
    """Serve requests until stdin closes or a shutdown request arrives.

    ``argv`` may carry a single path to a JSON file of datasets to seed as ``data`` in
    the namespace (the exploration inputs), and ``--allow-network`` to skip the guard.
    """
    args = list(argv)
    if "--allow-network" in args:
        args.remove("--allow-network")
    else:
        _deny_network()

    namespace: dict[str, Any] = {"data": {}}
    if args:
        data = json.loads(Path(args[0]).read_text(encoding="utf-8"))
        namespace["data"] = data

    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        request = json.loads(stripped)
        if request.get("shutdown"):
            break
        response = _run_once(namespace, str(request.get("code", "")))
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main(sys.argv[1:]))
