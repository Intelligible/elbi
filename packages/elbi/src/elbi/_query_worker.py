"""Child process that answers warehouse queries, so the app never holds a scan.

Not a public API. Reads one JSON request per line from stdin and writes one JSON reply
per line to stdout::

    {"sql": "select 1", "max_rows": 1000} {"columns": ["1"], "rows": [{"1": 1}],
    "truncated": false}

It rebuilds its own :class:`~elbi.warehouse.service.WarehouseService` from
``DB_URI``, the same way the app does, so nothing has to be shipped across the boundary
except the query and its result. The service it builds has no runner of its own, which
is what stops a worker from delegating back into another worker.

Why a separate process at all: DuckDB is an in-process engine, so a query embedded in
the web server competes for the same pages as request handling and an out-of-memory
query takes the whole service down with it. A worker moves that failure to a process the
app can replace. DuckDB 1.5.2 added a native client-server protocol for the same reason;
this uses the line protocol already proven by the notebook kernel rather than adding a
protobuf dependency for one call site.
"""

from __future__ import annotations

import json
import sys
from typing import Any


def _reply(payload: dict[str, Any]) -> None:
    """Write one reply line and flush, so the parent never waits on a buffer."""
    sys.stdout.write(json.dumps(payload, default=str) + "\n")
    sys.stdout.flush()


def main() -> int:
    """Serve queries until stdin closes."""
    # Imported here rather than at module scope so a failure to build the service is
    # reported as a reply the parent can surface, not a traceback on a dead pipe.
    from .db import open_store
    from .env import env
    from .warehouse.service import WarehouseError, WarehouseService

    service: WarehouseService | None = None
    build_error = ""
    try:
        service = WarehouseService(open_store(env("DB_URI") or "sqlite:app.db"))
    except Exception as exc:
        build_error = f"{type(exc).__name__}: {exc}"

    for line in sys.stdin:
        stripped = line.strip()
        if not stripped:
            continue
        try:
            request = json.loads(stripped)
        except ValueError:
            _reply({"error": "the query worker received a malformed request"})
            continue
        if service is None:
            _reply(
                {
                    "error": "the query worker could not open the warehouse: "
                    + build_error
                }
            )
            continue
        try:
            columns, rows, truncated = service.query(
                str(request.get("sql", "")), max_rows=int(request.get("max_rows", 1000))
            )
        except WarehouseError as exc:
            # A query error is the user's, and belongs in their result rather than
            # killing the worker that every other request shares.
            _reply({"error": str(exc)})
            continue
        _reply({"columns": columns, "rows": rows, "truncated": truncated})
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main())
