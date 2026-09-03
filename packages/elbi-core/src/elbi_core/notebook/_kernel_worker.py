"""Child entry point for a notebook kernel: a persistent REPL with rich output.

Not a public API. Like :mod:`elbi._repl_worker` it is a long-lived child that holds a
namespace across requests, but it speaks a richer protocol built for a notebook UI: it
reproduces the pieces IPython gives a Jupyter kernel, the last expression of a cell
shown as a result, the ``display()`` / ``_repr_*_`` MIME protocol so a DataFrame or a
Vega chart renders, inline matplotlib figures, and clean tracebacks, without importing
IPython, so it stays stdlib-only and runs by file path in a uv-provisioned environment.

Output streams as it is produced: each ``print`` and each ``display`` emits a message on
the protocol channel immediately (the child's real stdout, captured before user code can
touch it), and the cell ends with a single ``done`` message. That live channel is what
lets the browser show a long cell's output as it happens rather than only at the end. A
SIGINT interrupts the running cell (raising ``KeyboardInterrupt``) without killing the
process, so its namespace survives an interrupt; only a restart replaces the
interpreter.
"""

from __future__ import annotations

import ast
import base64
import builtins
import contextlib
import getpass
import io
import json
import os
import sys
import threading
import traceback
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

# The worker is launched by file path (its package is not installed in the kernel's
# ephemeral uv environment), so its own directory is ``sys.path[0]`` and sibling helper
# modules import as top-level names. Under type checking the package-qualified import is
# followed instead, so mypy resolves them without a second module root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

if TYPE_CHECKING:
    from elbi_core.notebook import _comm, _data, _introspect, _magics
else:
    import _comm
    import _data
    import _introspect
    import _magics

#: Dispatches inbound comm messages into ipywidgets' comm manager. Set at startup when
#: the ``comm`` package is importable (ipywidgets present), else ``None``.
_COMM_DISPATCH: _comm.Dispatcher | None = None

#: Coalesce stdout writes up to this size before flushing a stream message, so a tight
#: print loop does not emit a protocol message per character; flushed on newline anyway.
_STREAM_CHUNK = 4096
#: Cap a single ``text/plain`` repr so a huge object cannot flood the transport.
_MAX_TEXT = 200_000

#: The protocol channel: the child's real stdout, captured before user code can replace
#: ``sys.stdout``. Every message to the parent goes here as one line of JSON.
_CHANNEL = sys.stdout

#: Serializes writes to the channel: the fd-capture reader threads call ``_send``
#: concurrently with the main thread, so without this their JSON lines interleave and
#: the parent cannot parse them.
_CHANNEL_LOCK = threading.Lock()


def _send(message: dict[str, Any]) -> None:
    """Write one newline-delimited JSON message to the parent and flush it."""
    line = json.dumps(message) + "\n"
    with _CHANNEL_LOCK:
        _CHANNEL.write(line)
        _CHANNEL.flush()


#: File-descriptor capture relies on ``os.dup2`` and pipe reader threads, which behave
#: on POSIX; Windows keeps only the Python-level ``sys.stdout``/``sys.stderr`` capture.
_FD_CAPTURE = sys.platform != "win32"


def _install_protocol_channel() -> None:
    """Move the parent protocol channel off fd 1 so a cell can capture fd 1 safely.

    The parent reads the worker's stdout (fd 1) as the JSON protocol. A cell that writes
    to fd 1 directly (a C extension's ``printf``, a logging handler bound to the real
    stream) must not corrupt that, so the protocol is duplicated onto a private fd and
    every ``_send`` goes there; fd 1 is then free to be redirected into a capture pipe.
    """
    global _CHANNEL
    duplicate = os.dup(_CHANNEL.fileno())
    _CHANNEL = os.fdopen(duplicate, "w", encoding="utf-8", errors="replace")


def _pump_fd(read_fd: int, name: str) -> None:
    """Forward everything written to a captured fd to the parent as stream output."""
    try:
        while True:
            chunk = os.read(read_fd, 65536)
            if not chunk:
                break
            _send(
                {"type": "stream", "name": name, "text": chunk.decode(errors="replace")}
            )
    finally:
        os.close(read_fd)


@contextlib.contextmanager
def _capture_fds() -> Iterator[None]:
    """Stream writes to fd 1 and fd 2 during a cell to the parent, live.

    Redirecting the real file descriptors (not just ``sys.stdout``) catches output the
    Python-level capture cannot: a C extension's ``printf`` and a logging handler bound
    to the original stream both write to the fd by number, which is where FLAML,
    LightGBM, and XGBoost emit their progress. Each fd is swapped for a pipe drained by
    a thread; on exit the fd is restored, closing the pipe's only writer so the reader
    drains the last bytes and stops before the cell reports ``done``.
    """
    if not _FD_CAPTURE:
        yield
        return
    watchers: list[tuple[threading.Thread, int]] = []
    saved: dict[int, int] = {}
    try:
        for fd, name in ((1, "stdout"), (2, "stderr")):
            read_fd, write_fd = os.pipe()
            saved[fd] = os.dup(fd)
            os.dup2(write_fd, fd)
            os.close(write_fd)
            thread = threading.Thread(
                target=_pump_fd, args=(read_fd, name), daemon=True
            )
            thread.start()
            watchers.append((thread, fd))
        yield
    finally:
        # Restoring each fd drops the pipe's last writer, so the reader hits EOF and
        # drains what is buffered; join before returning so it lands before ``done``.
        for _thread, fd in watchers:
            os.dup2(saved[fd], fd)
            os.close(saved[fd])
        for thread, _fd in watchers:
            thread.join(timeout=2.0)


def _build_mimebundle(obj: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    """Assemble a MIME bundle for a value, the way IPython's display formatters do.

    ``_repr_mimebundle_`` is preferred when present (it lets a library offer every
    representation at once: Vega, Plotly, Altair), then the individual ``_repr_*_``
    hooks, and finally ``repr()`` as the always-present ``text/plain`` fallback.
    """
    data: dict[str, Any] = {}
    meta: dict[str, Any] = {}
    if hasattr(obj, "_repr_mimebundle_"):
        with contextlib.suppress(Exception):
            bundle = obj._repr_mimebundle_()
            bdata, bmeta = bundle if isinstance(bundle, tuple) else (bundle, {})
            if isinstance(bdata, dict) and bdata:
                bdata.setdefault("text/plain", _safe_repr(obj))
                return bdata, dict(bmeta or {})

    text_reprs = {
        "text/html": "_repr_html_",
        "text/markdown": "_repr_markdown_",
        "image/svg+xml": "_repr_svg_",
        "text/latex": "_repr_latex_",
        "application/javascript": "_repr_javascript_",
    }
    for mime, method in text_reprs.items():
        payload = _call_repr(obj, method)
        if payload is not None:
            value, extra = payload
            data[mime] = value
            if extra:
                meta[mime] = extra
    for mime, method in (("image/png", "_repr_png_"), ("image/jpeg", "_repr_jpeg_")):
        payload = _call_repr(obj, method)
        if payload is not None:
            value, _ = payload
            data[mime] = value if isinstance(value, str) else _b64(value)
    json_payload = _call_repr(obj, "_repr_json_")
    if json_payload is not None:
        data["application/json"] = json_payload[0]

    data["text/plain"] = _safe_repr(obj)
    return data, meta


def _call_repr(obj: Any, method: str) -> tuple[Any, dict[str, Any] | None] | None:
    """Call a ``_repr_*_`` hook if present; return ``(payload, metadata)`` or None."""
    hook = getattr(obj, method, None)
    if not callable(hook):
        return None
    try:
        out = hook()
    except Exception:
        return None
    if out is None:
        return None
    if isinstance(out, tuple):
        return out[0], (out[1] if len(out) > 1 else None)
    return out, None


def _b64(raw: Any) -> str:
    """Base64-encode raw image bytes for an ``image/*`` MIME payload."""
    if isinstance(raw, (bytes, bytearray)):
        return base64.b64encode(bytes(raw)).decode("ascii")
    return str(raw)


def _safe_repr(obj: Any) -> str:
    """A length-capped repr that never raises, even on a broken ``__repr__``."""
    try:
        text = repr(obj)
    except Exception as exc:  # a broken __repr__ must not break the cell's output
        text = f"<unreprable {type(obj).__name__}: {exc}>"
    return text[:_MAX_TEXT]


class _StreamWriter(io.TextIOBase):
    """A stdout/stderr replacement that emits ``stream`` messages as text is written.

    Buffered lightly and flushed on newline or when it grows past a chunk, so a long
    computation's prints reach the browser promptly without a message per character.
    """

    def __init__(self, name: str) -> None:
        self._name = name
        self._buffer: list[str] = []
        self._size = 0

    def write(self, text: str) -> int:
        if not text:
            return 0
        self._buffer.append(text)
        self._size += len(text)
        if "\n" in text or self._size >= _STREAM_CHUNK:
            self.flush()
        return len(text)

    def flush(self) -> None:
        if self._buffer:
            _send({"type": "stream", "name": self._name, "text": "".join(self._buffer)})
            self._buffer.clear()
            self._size = 0

    def writable(self) -> bool:
        return True


def _display(*objects: Any, **_: Any) -> None:
    """The injected ``display()`` builtin: push a ``display_data`` output per object."""
    for obj in objects:
        data, meta = _build_mimebundle(obj)
        _send({"type": "display_data", "data": data, "metadata": meta})


def _capture_figures() -> None:
    """Emit any open matplotlib figures as inline PNGs, then close them (like %inline).

    Only runs when matplotlib is actually loaded, so a kernel that never plots pays
    nothing and the worker carries no matplotlib dependency of its own.
    """
    module = sys.modules.get("matplotlib.pyplot")
    if module is None:
        return
    # A plotting-capture glitch must never mask the cell's real output.
    with contextlib.suppress(Exception):
        for num in module.get_fignums():
            figure = module.figure(num)
            buffer = io.BytesIO()
            figure.savefig(buffer, format="png", bbox_inches="tight", dpi=100)
            png = base64.b64encode(buffer.getvalue()).decode("ascii")
            _send(
                {
                    "type": "display_data",
                    "data": {"image/png": png, "text/plain": "<Figure>"},
                    "metadata": {},
                }
            )
            module.close(figure)


def _emit_result(value: Any, execution_count: int) -> None:
    """Emit a last-expression value as an ``execute_result`` (None shows nothing)."""
    if value is None:
        return
    data, meta = _build_mimebundle(value)
    _send(
        {
            "type": "execute_result",
            "execution_count": execution_count,
            "data": data,
            "metadata": meta,
        }
    )


def _emit_error(exc: BaseException) -> None:
    """Emit an ``error`` output, the worker's own frames stripped from the traceback."""
    # Drop the frame for this module's ``exec`` call so the user sees only their code.
    tb = exc.__traceback__.tb_next if exc.__traceback__ else None
    lines = traceback.format_exception(type(exc), exc, tb)
    flattened = [line for chunk in lines for line in chunk.rstrip("\n").split("\n")]
    _send(
        {
            "type": "error",
            "ename": type(exc).__name__,
            "evalue": str(exc),
            "traceback": flattened,
        }
    )


class _MagicContext:
    """The worker's :class:`elbi.notebook._magics.MagicContext` implementation.

    A magic emits through the same redirected ``sys.stdout``/``sys.stderr`` as ordinary
    cell output, so its text interleaves in order; rich output goes straight to the
    protocol channel as ``display_data``.
    """

    def __init__(self, namespace: dict[str, Any], execution_count: int) -> None:
        self.namespace = namespace
        self.execution_count = execution_count

    def emit_stream(self, name: str, text: str) -> None:
        (sys.stdout if name == "stdout" else sys.stderr).write(text)

    def emit_display(self, data: dict[str, Any], metadata: dict[str, Any]) -> None:
        _send({"type": "display_data", "data": data, "metadata": metadata})

    def run_code(self, code: str) -> Any:
        """Exec ``code`` in the namespace, returning a trailing expression's value."""
        tree = ast.parse(code)
        last_expr = _split_last_expr(tree)
        exec(compile(tree, "<cell>", "exec"), self.namespace)  # noqa: S102
        if last_expr is not None:
            return eval(compile(last_expr, "<cell>", "eval"), self.namespace)  # noqa: S307
        return None


#: The magic context for the cell currently running, so the namespace's seeded shell and
#: line-magic helpers reach the live streams and namespace. Set at the top of each run.
_CURRENT_CONTEXT: _MagicContext | None = None


def _split_last_expr(tree: ast.Module) -> ast.Expression | None:
    """Pop a trailing bare expression off ``tree`` as an ``Expression`` node."""
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        expr_stmt = tree.body[-1]
        tree.body.pop()
        last_expr = ast.Expression(expr_stmt.value)
        ast.copy_location(last_expr, last_expr.body)
        return last_expr
    return None


def _run_cell(namespace: dict[str, Any], code: str, execution_count: int) -> str:
    """Execute one cell, streaming its outputs; return ``ok``/``error``/``interrupted``.

    A ``%%`` first line runs the whole cell as a cell magic. Otherwise the cell's
    ``%line-magic`` and ``!shell`` lines are rewritten to helper calls and the body runs
    in ``exec`` mode; when its final statement is a bare expression, that expression is
    evaluated separately so its value becomes the result: the REPL behavior a notebook
    user expects (``df.head()`` on the last line shows the table). The namespace
    persists, so a later cell sees what this one defined.
    """
    global _CURRENT_CONTEXT
    ctx = _MagicContext(namespace, execution_count)
    _CURRENT_CONTEXT = ctx
    out, err = _StreamWriter("stdout"), _StreamWriter("stderr")
    status = "ok"
    try:
        with (
            _capture_fds(),
            contextlib.redirect_stdout(out),
            contextlib.redirect_stderr(err),
        ):
            cell_magic = _magics.split_cell_magic(code)
            if cell_magic is not None:
                name, args, body = cell_magic
                _magics.run_cell_magic(name, args, body, ctx)
                _capture_figures()
            else:
                tree = ast.parse(_magics.transform_lines(code))
                last_expr = _split_last_expr(tree)
                exec(compile(tree, "<cell>", "exec"), namespace)  # noqa: S102
                value = None
                if last_expr is not None:
                    value = eval(compile(last_expr, "<cell>", "eval"), namespace)  # noqa: S307
                _capture_figures()
                if value is not None:
                    namespace["_"] = value
                    _emit_result(value, execution_count)
    except SyntaxError as exc:
        status = "error"
        _emit_error(exc)
    except _magics.UsageError as exc:
        status = "error"
        _emit_error(exc)
    except KeyboardInterrupt as exc:
        status = "interrupted"
        _emit_error(exc)
    except BaseException as exc:  # report it; the kernel survives for the next cell
        status = "error"
        _emit_error(exc)
    finally:
        out.flush()
        err.flush()
        _CURRENT_CONTEXT = None
    return status


def _tracking_configured() -> bool:
    """Whether this kernel was handed a tracking URI it can reach."""
    return bool(os.environ.get("MLFLOW_TRACKING_URI", "").strip())


def _register_unavailable(*_a: object, **_k: object) -> None:
    """Sentinel bound only when there is no tracking URI.

    Without one MLflow does not fail: it writes to ``./mlruns`` inside the sandbox and
    reports success, so the run is lost when the kernel stops.
    """
    raise RuntimeError(
        "There is no experiment-tracking URI, so a model logged here would be written "
        "inside this sandbox and lost when the kernel stops. Set "
        "NOTEBOOK_MLFLOW_TRACKING_URI to somewhere durable. With it set, "
        "`mlflow.<flavor>.log_model(..., registered_model_name=...)` registers into "
        "the app's registry and the version appears on the Models page."
    )


def _seed_namespace(
    data_path: str | None, dataset_names: Sequence[str] | None = None
) -> dict[str, Any]:
    """Start the namespace, seeding bound datasets as ``data`` and ``display``."""
    namespace: dict[str, Any] = {"__name__": "__main__", "data": {}}
    # Bound datasets are fetched on first access, and `sql` runs where the engine is.
    # Both go through the same host round-trip; see notebook/_data.py for why.
    # `data_path` is the older eager form, a JSON file of every dataset, kept so a host
    # that has not been updated (or a test seeding rows) keeps working.
    if dataset_names is not None:
        namespace["data"] = _data.LazyData(
            _request_query,
            tuple(dataset_names),
            on_materialise=lambda: _note_data_mode("materialised"),
        )
        namespace["sql"] = _nb_sql
    elif data_path:
        with contextlib.suppress(Exception):
            namespace["data"] = json.loads(Path(data_path).read_text(encoding="utf-8"))
    # ``display`` is a builtin in a Jupyter kernel; inject it so notebook code can call
    # it.
    builtins.display = _display  # type: ignore[attr-defined]
    if not _tracking_configured():
        # With tracking configured, MLflow's own calls are the path and these names must
        # not shadow them.
        builtins.train_model = _register_unavailable  # type: ignore[attr-defined]
        builtins.register_model = _register_unavailable  # type: ignore[attr-defined]
    # Route input() and getpass through the frontend, the way a Jupyter kernel does.
    builtins.input = _nb_input
    getpass.getpass = _nb_getpass
    # The targets a cell's rewritten `%magic`/`!shell` lines call into. They dispatch
    # against whichever cell is running, so they read the module-level current context.
    namespace[_magics.LINE_MAGIC_FN] = _nb_line_magic
    namespace[_magics.SYSTEM_FN] = _nb_system
    namespace[_magics.GETOUTPUT_FN] = _nb_getoutput
    return namespace


def _nb_input(prompt: object = "") -> str:
    """The kernel's ``input()``: ask the frontend for a line and block until it replies.

    Emits an ``input_request`` and reads the reply the parent writes to the worker's
    stdin. The cell's execution thread owns stdin while it runs (the serve loop is
    parked inside the cell), so this read never races the protocol reader.
    """
    return _request_input(str(prompt), password=False)


def _nb_getpass(prompt: str = "", stream: object = None) -> str:
    """The kernel's ``getpass.getpass``: an ``input_request`` with echo suppressed."""
    return _request_input(str(prompt), password=True)


def _nb_sql(sql: str, limit: int = _data._UNLIMITED) -> _data.QueryResult:
    """Run read-only SQL against the warehouse and return the result.

    The scan, join and aggregation happen server-side over the Delta tables. The whole
    result crosses into this kernel, bounded only by the host's ceiling; rendering it
    shows the first thousand rows and says how many there are. Pass ``limit`` to fetch
    less on purpose, and prefer aggregating in the query to fetching and reducing here.
    """
    _note_data_mode("pushed_down")
    return _data.run_query(_request_query, sql=sql, limit=limit)


#: How the cell now running reached its data, reported on ``done`` so a notebook can say
#: which side of the boundary a cell is on. Hex makes this a pill on the cell, and the
#: reason to copy that is the same reason the split is explicit at all: a lazy handle
#: whose materialisation is invisible only moves the memory failure later.
_DATA_MODE: str = ""


def _note_data_mode(mode: str) -> None:
    """Record how this cell got its data. ``materialised`` outranks ``pushed_down``.

    A cell that does both has materialised rows into the kernel, and that is the fact
    worth surfacing: the cheaper half does not undo the expensive half.
    """
    global _DATA_MODE
    if mode == "materialised" or not _DATA_MODE:
        _DATA_MODE = mode


def _request_query(*, sql: str = "", table: str = "", limit: int) -> dict[str, Any]:
    """Ask the host to run a query server-side and block for its reply.

    Same shape as :func:`_request_input`: emit a request, then read the parent's reply
    off stdin while still servicing widget traffic, so a query inside a cell that also
    has live widgets does not deadlock or mistake a comm message for its answer.
    """
    _send({"type": "query_request", "sql": sql, "table": table, "limit": limit})
    while True:
        line = sys.stdin.readline()
        if not line:
            return {"error": "the kernel host closed before answering the query"}
        try:
            message = json.loads(line)
        except ValueError:
            continue
        op = message.get("op")
        if op in ("comm_open", "comm_msg", "comm_close") and _COMM_DISPATCH is not None:
            _COMM_DISPATCH(op, message.get("content", {}), message.get("buffers", []))
            continue
        if op != "query_reply":
            continue
        return dict(message.get("content") or {})


def _request_input(prompt: str, password: bool) -> str:
    _send({"type": "input_request", "prompt": prompt, "password": password})
    while True:
        line = sys.stdin.readline()
        if not line:
            raise EOFError("no input available")
        try:
            message = json.loads(line)
        except ValueError:
            return line.rstrip("\n")
        op = message.get("op")
        if op in ("comm_open", "comm_msg", "comm_close") and _COMM_DISPATCH is not None:
            # A widget interaction landed while a cell is blocked on input(); service it
            # and keep waiting for the actual reply rather than mistaking it for input.
            _COMM_DISPATCH(op, message.get("content", {}), message.get("buffers", []))
            continue
        if op is not None and op != "input_reply":
            continue
        value = message.get("value", "")
        if value == "\x04":  # the frontend's cancel/EOF sentinel
            raise EOFError("input cancelled")
        return str(value)


def _current_context() -> _MagicContext:
    """The running cell's magic context; magics are only reachable during a run."""
    if _CURRENT_CONTEXT is None:  # pragma: no cover - a rewritten line runs in a cell
        raise RuntimeError("magics and shell escapes run only inside a cell")
    return _CURRENT_CONTEXT


def _nb_line_magic(name: str, line: str) -> Any:
    """Namespace hook for a rewritten ``%name args`` line magic."""
    return _magics.run_line_magic(name, line, _current_context())


def _nb_system(cmd: str) -> None:
    """Namespace hook for a rewritten ``!cmd`` shell escape."""
    _magics.run_system(cmd, _current_context())


def _nb_getoutput(cmd: str) -> _magics.ShellList:
    """Namespace hook for a rewritten ``!!cmd`` / ``x = !cmd`` shell capture."""
    return _magics.get_output(cmd, _current_context())


def _deny_network() -> None:
    """Best-effort egress block for the subprocess backend (defense in depth)."""
    import socket

    class _Blocked(socket.socket):
        def __init__(self, *_a: object, **_k: object) -> None:
            raise OSError("network is disabled in the elbi notebook sandbox")

    socket.socket = _Blocked  # type: ignore[misc]


def main(argv: list[str]) -> int:
    """Serve cell requests until stdin closes or a shutdown message arrives."""
    global _COMM_DISPATCH
    # Move the protocol onto a private fd before any cell can redirect fd 1 to capture
    # its output; every _send from here on reaches the parent regardless of capture.
    _install_protocol_channel()
    args = list(argv)
    if "--allow-network" in args:
        args.remove("--allow-network")
    else:
        _deny_network()
    # argv: <data.json> [<names.json>]. The second is present only in lazy mode, where
    # it lists the bound dataset names and data.json is empty. A runner with no shared
    # filesystem -- a pod, which cannot mount anything from the host that launched it --
    # passes the same list in the environment instead.
    names: Sequence[str] | None = None
    if len(args) > 1:
        with contextlib.suppress(OSError, ValueError):
            names = list(json.loads(Path(args[1]).read_text(encoding="utf-8")))
    elif os.environ.get("ELBI_DATASET_NAMES"):
        with contextlib.suppress(ValueError):
            names = list(json.loads(os.environ["ELBI_DATASET_NAMES"]))
    namespace = _seed_namespace(args[0] if args else None, names)
    # Wire ipywidgets' comm channel to the protocol relay (a no-op without ipywidgets).
    _COMM_DISPATCH = _comm.install(_send)

    while True:
        try:
            line = sys.stdin.readline()
        except KeyboardInterrupt:
            # A SIGINT that lands while idle (between cells) interrupts no running code;
            # ignore it and keep serving.
            continue
        if not line:
            break
        stripped = line.strip()
        if not stripped:
            continue
        request = json.loads(stripped)
        if request.get("shutdown"):
            break
        op = request.get("op")
        if op == "input_reply":
            # A reply with no cell parked in input() (the serve loop, not a blocked
            # input(), read it): nothing is waiting, so drop it.
            continue
        if op in ("comm_open", "comm_msg", "comm_close"):
            if _COMM_DISPATCH is not None:
                _COMM_DISPATCH(
                    op, request.get("content", {}), request.get("buffers", [])
                )
            continue
        if op == "complete":
            reply = _introspect.complete(
                namespace,
                str(request.get("code", "")),
                int(request.get("cursor_pos", 0)),
            )
            _send({"type": "complete_reply", **reply})
            continue
        if op == "inspect":
            reply = _introspect.inspect_token(
                namespace,
                str(request.get("code", "")),
                int(request.get("cursor_pos", 0)),
                int(request.get("detail_level", 0)),
            )
            _send({"type": "inspect_reply", **reply})
            continue
        if op == "variables":
            _send({"type": "variables_reply", **_introspect.variables(namespace)})
            continue
        count = int(request.get("execution_count", 0))
        global _DATA_MODE
        _DATA_MODE = ""
        status = _run_cell(namespace, str(request.get("code", "")), count)
        done: dict[str, Any] = {
            "type": "done",
            "execution_count": count,
            "status": status,
        }
        if _DATA_MODE:
            done["data_mode"] = _DATA_MODE
        _send(done)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess
    raise SystemExit(main(sys.argv[1:]))
