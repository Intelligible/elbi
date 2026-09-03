"""IPython-style magics and shell escapes for the notebook kernel worker.

The worker is a plain interpreter, not IPython, so this module reproduces the pieces a
data scientist reaches for from a cell: line magics (``%time``, ``%timeit``, ``%who``,
``%pwd``, ``%cd``, ``%env``, ``%reset``, ``%matplotlib``, ``%lsmagic``), cell magics
(``%%time``, ``%%timeit``, ``%%bash``/``%%sh``, ``%%capture``, ``%%writefile``,
``%%html``, ``%%latex``, ``%%markdown``, ``%%javascript``, ``%%python``), and shell
escapes (``!cmd``, ``!!cmd``, and ``x = !cmd`` capture).

The design mirrors IPython's input transformer: a cell's magic and shell lines are
rewritten to calls on helper functions seeded into the namespace, so magics compose with
ordinary Python in the same cell and the worker's normal ``exec`` path runs the result.
Cell magics (a ``%%`` first line) own the whole cell and are dispatched directly. Kept
stdlib-only, so it runs in any uv-provisioned environment with no extra dependency.
"""

from __future__ import annotations

import contextlib
import io
import os
import re
import shlex
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol


class MagicContext(Protocol):
    """The worker primitives a magic needs, passed in so this module stays decoupled."""

    namespace: dict[str, Any]
    execution_count: int

    def emit_stream(self, name: str, text: str) -> None:
        """Write to the cell's ``stdout``/``stderr`` stream output."""

    def emit_display(self, data: dict[str, Any], metadata: dict[str, Any]) -> None:
        """Emit a rich ``display_data`` output (a MIME bundle)."""

    def run_code(self, code: str) -> Any:
        """Exec ``code`` in the namespace; return a final bare expression's value."""


class UsageError(Exception):
    """A malformed or unknown magic, surfaced to the cell like IPython's UsageError."""


# A shell command that produced a non-zero exit status. The list-valued form (``!!cmd``)
# still returns its output; the plain form (``!cmd``) streams and swallows the code, as
# IPython does.
class ShellList(list[str]):
    """The line list ``!!cmd`` / ``x = !cmd`` yields; prints as newline-joined text."""

    def __str__(self) -> str:
        return "\n".join(self)


_LINE_MAGIC_RE = re.compile(r"^(\s*)%(\w+)(.*)$")
_CELL_MAGIC_RE = re.compile(r"^%%(\w+)([^\n]*)\n?(.*)\Z", re.DOTALL)
_BANG_ASSIGN_RE = re.compile(r"^(\s*)([\w\[\].]+\s*=\s*)!(!?)(.*)$")
_BANG_RE = re.compile(r"^(\s*)!(!?)(.*)$")

#: Names seeded into the execution namespace that line/shell transforms call into.
LINE_MAGIC_FN = "__nb_line_magic__"
SYSTEM_FN = "__nb_system__"
GETOUTPUT_FN = "__nb_getoutput__"


def split_cell_magic(code: str) -> tuple[str, str, str] | None:
    """Return ``(name, args, body)`` when the first line is a ``%%`` cell magic."""
    if not code.lstrip().startswith("%%"):
        return None
    match = _CELL_MAGIC_RE.match(code.lstrip("\n"))
    if match is None:
        return None
    name, args, body = match.group(1), match.group(2).strip(), match.group(3)
    return name, args, body


def transform_lines(code: str) -> str:
    """Rewrite each ``%line-magic`` and ``!shell`` line to a helper-function call.

    Ordinary Python lines pass through untouched, so magics and code coexist in one cell
    exactly as they do in IPython.
    """
    out: list[str] = []
    for line in code.splitlines():
        out.append(_transform_line(line))
    return "\n".join(out)


def _transform_line(line: str) -> str:
    assign = _BANG_ASSIGN_RE.match(line)
    if assign is not None:
        indent, target, bang, cmd = assign.groups()
        return f"{indent}{target}{GETOUTPUT_FN}({_q(cmd.strip())})"
    bang = _BANG_RE.match(line)
    if bang is not None:
        indent, double, cmd = bang.groups()
        fn = GETOUTPUT_FN if double else SYSTEM_FN
        return f"{indent}{fn}({_q(cmd.strip())})"
    magic = _LINE_MAGIC_RE.match(line)
    if magic is not None:
        indent, name, rest = magic.groups()
        return f"{indent}{LINE_MAGIC_FN}({_q(name)}, {_q(rest.strip())})"
    return line


def _q(text: str) -> str:
    """Quote a string as a Python literal for the rewritten call."""
    return repr(text)


# --- line magics ---------------------------------------------------------------------
def _magic_time(args: str, ctx: MagicContext) -> Any:
    start = time.perf_counter()
    value = ctx.run_code(args)
    elapsed = time.perf_counter() - start
    ctx.emit_stream("stdout", f"Wall time: {elapsed * 1000:.3f} ms\n")
    return value


def _magic_timeit(args: str, ctx: MagicContext) -> None:
    number, repeat = 1000, 5
    code = compile(args, "<timeit>", "eval" if _is_expr(args) else "exec")
    # A quick calibration run keeps a slow statement from taking the full 1000 loops.
    calib_start = time.perf_counter()
    exec(code, ctx.namespace)  # noqa: S102
    if time.perf_counter() - calib_start > 0.2:
        number = 10
    best = min(_time_loops(code, ctx.namespace, number) for _ in range(repeat))
    per = best / number
    ctx.emit_stream(
        "stdout",
        f"{_fmt_time(per)} per loop (mean of {repeat} runs, {number} loops each)\n",
    )


def _time_loops(code: Any, namespace: dict[str, Any], number: int) -> float:
    start = time.perf_counter()
    for _ in range(number):
        exec(code, namespace)  # noqa: S102
    return time.perf_counter() - start


def _magic_who(args: str, ctx: MagicContext) -> None:
    names = sorted(
        name
        for name, value in ctx.namespace.items()
        if not name.startswith("_") and not callable_module(value)
    )
    ctx.emit_stream(
        "stdout",
        ("\t".join(names) + "\n") if names else "Interactive namespace is empty.\n",
    )


def _magic_whos(args: str, ctx: MagicContext) -> None:
    rows = [
        (name, type(value).__name__, _short_repr(value))
        for name, value in sorted(ctx.namespace.items())
        if not name.startswith("_") and not callable_module(value)
    ]
    if not rows:
        ctx.emit_stream("stdout", "Interactive namespace is empty.\n")
        return
    width = max(len(name) for name, _, _ in rows)
    kind = max(len(t) for _, t, _ in rows)
    header = f"{'Variable'.ljust(width)}   {'Type'.ljust(kind)}   Data/Info\n"
    body = "".join(
        f"{name.ljust(width)}   {t.ljust(kind)}   {info}\n" for name, t, info in rows
    )
    ctx.emit_stream("stdout", header + body)


def _magic_pwd(args: str, ctx: MagicContext) -> str:
    return str(Path.cwd())


def _magic_cd(args: str, ctx: MagicContext) -> None:
    target = Path(args.strip()).expanduser() if args.strip() else Path.home()
    os.chdir(target)
    ctx.emit_stream("stdout", str(Path.cwd()) + "\n")


def _magic_env(args: str, ctx: MagicContext) -> Any:
    args = args.strip()
    if not args:
        return dict(os.environ)
    if "=" in args:
        key, value = args.split("=", 1)
        os.environ[key.strip()] = value.strip()
        return None
    return os.environ.get(args)


def _magic_reset(args: str, ctx: MagicContext) -> None:
    if "-f" not in args and "--force" not in args:
        raise UsageError("%reset needs -f to confirm clearing the namespace")
    keep = {"__name__", "__builtins__", "data"}
    for name in [n for n in ctx.namespace if n not in keep]:
        del ctx.namespace[name]


def _magic_matplotlib(args: str, ctx: MagicContext) -> None:
    # Figures are always captured inline by the worker, so the only sensible mode is
    # inline; accept the call so `%matplotlib inline` from a pasted notebook is a no-op.
    mode = args.strip() or "inline"
    if mode not in {"inline", "agg", "notebook", "widget"}:
        raise UsageError(
            f"unsupported matplotlib backend {mode!r}; only inline is available"
        )


def _magic_lsmagic(args: str, ctx: MagicContext) -> None:
    line = "  ".join("%" + name for name in sorted(LINE_MAGICS))
    cell = "  ".join("%%" + name for name in sorted(CELL_MAGICS))
    ctx.emit_stream(
        "stdout", f"Available line magics:\n{line}\n\nAvailable cell magics:\n{cell}\n"
    )


LINE_MAGICS: dict[str, Callable[[str, MagicContext], Any]] = {
    "time": _magic_time,
    "timeit": _magic_timeit,
    "who": _magic_who,
    "whos": _magic_whos,
    "pwd": _magic_pwd,
    "cd": _magic_cd,
    "env": _magic_env,
    "reset": _magic_reset,
    "matplotlib": _magic_matplotlib,
    "lsmagic": _magic_lsmagic,
}


# --- cell magics ---------------------------------------------------------------------
def _cell_bash(args: str, body: str, ctx: MagicContext) -> None:
    _run_shell(body, ctx, shell="/bin/bash" if args.strip() == "" else args)


def _cell_time(args: str, body: str, ctx: MagicContext) -> None:
    start = time.perf_counter()
    ctx.run_code(transform_lines(body))
    ctx.emit_stream(
        "stdout", f"Wall time: {(time.perf_counter() - start) * 1000:.3f} ms\n"
    )


def _cell_timeit(args: str, body: str, ctx: MagicContext) -> None:
    _magic_timeit(body if _is_expr(body) else _as_exec_wrapper(body), ctx)


def _cell_capture(args: str, body: str, ctx: MagicContext) -> None:
    target = args.strip()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
        ctx.run_code(transform_lines(body))
    if target:
        ctx.namespace[target] = buffer.getvalue()
    else:  # no target: swallow output, matching IPython's %%capture with no name
        pass


def _cell_writefile(args: str, body: str, ctx: MagicContext) -> None:
    parts = shlex.split(args)
    append = "-a" in parts or "--append" in parts
    paths = [p for p in parts if not p.startswith("-")]
    if not paths:
        raise UsageError("%%writefile needs a filename")
    mode = "a" if append else "w"
    with Path(paths[0]).open(mode, encoding="utf-8") as handle:
        handle.write(body)
    ctx.emit_stream("stdout", f"{'Appending to' if append else 'Writing'} {paths[0]}\n")


def _cell_html(args: str, body: str, ctx: MagicContext) -> None:
    ctx.emit_display({"text/html": body, "text/plain": body}, {})


def _cell_latex(args: str, body: str, ctx: MagicContext) -> None:
    ctx.emit_display({"text/latex": body, "text/plain": body}, {})


def _cell_markdown(args: str, body: str, ctx: MagicContext) -> None:
    ctx.emit_display({"text/markdown": body, "text/plain": body}, {})


def _cell_javascript(args: str, body: str, ctx: MagicContext) -> None:
    ctx.emit_display({"application/javascript": body, "text/plain": body}, {})


def _cell_python(args: str, body: str, ctx: MagicContext) -> None:
    ctx.run_code(transform_lines(body))


CELL_MAGICS: dict[str, Callable[[str, str, MagicContext], None]] = {
    "bash": _cell_bash,
    "sh": _cell_bash,
    "time": _cell_time,
    "timeit": _cell_timeit,
    "capture": _cell_capture,
    "writefile": _cell_writefile,
    "html": _cell_html,
    "latex": _cell_latex,
    "markdown": _cell_markdown,
    "javascript": _cell_javascript,
    "js": _cell_javascript,
    "python": _cell_python,
    "python3": _cell_python,
}


def run_line_magic(name: str, args: str, ctx: MagicContext) -> Any:
    """Dispatch a ``%name args`` line magic; raise ``UsageError`` if unknown."""
    handler = LINE_MAGICS.get(name)
    if handler is None:
        raise UsageError(f"Line magic function `%{name}` not found.")
    return handler(args, ctx)


def run_cell_magic(name: str, args: str, body: str, ctx: MagicContext) -> None:
    """Dispatch a ``%%name`` cell magic; raise ``UsageError`` if unknown."""
    handler = CELL_MAGICS.get(name)
    if handler is None:
        raise UsageError(f"Cell magic function `%%{name}` not found.")
    handler(args, body, ctx)


def run_system(cmd: str, ctx: MagicContext) -> None:
    """The ``!cmd`` escape: run a shell command, streaming its output into the cell."""
    _run_shell(cmd, ctx)


def get_output(cmd: str, ctx: MagicContext) -> ShellList:
    """The ``!!cmd`` / ``x = !cmd`` escape: return the command's stdout lines."""
    completed = subprocess.run(  # noqa: S602
        cmd, shell=True, capture_output=True, text=True, check=False
    )
    if completed.stderr:
        ctx.emit_stream("stderr", completed.stderr)
    return ShellList(completed.stdout.splitlines())


def _run_shell(cmd: str, ctx: MagicContext, shell: str | None = None) -> None:
    executable = shell if shell and Path(shell).exists() else None
    process = subprocess.Popen(  # noqa: S602
        cmd,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        executable=executable,
    )
    if process.stdout is not None:
        for line in process.stdout:
            ctx.emit_stream("stdout", line)
    process.wait()


# --- small helpers -------------------------------------------------------------------
def _is_expr(code: str) -> bool:
    try:
        compile(code, "<magic>", "eval")
    except SyntaxError:
        return False
    return True


def _as_exec_wrapper(body: str) -> str:
    # %%timeit over a multi-statement body: time the whole body, not one expression.
    return body


def _fmt_time(seconds: float) -> str:
    for unit, scale in (("s", 1.0), ("ms", 1e-3), ("µs", 1e-6), ("ns", 1e-9)):
        if seconds >= scale:
            return f"{seconds / scale:.3f} {unit}"
    return f"{seconds / 1e-9:.3f} ns"


def callable_module(value: Any) -> bool:
    import types

    return isinstance(value, types.ModuleType)


def _short_repr(value: Any, limit: int = 50) -> str:
    try:
        text = repr(value)
    except Exception:
        return f"<{type(value).__name__}>"
    text = text.replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."
