"""Completion and object inspection for the notebook kernel worker.

Stdlib-only equivalents of a Jupyter kernel's ``complete_request`` and
``inspect_request`` (the editor's tab-completion and Shift-Tab help): completions drawn
from the live namespace (attributes of an object, names, builtins, and keywords) and
inspection of the name under the cursor into a signature-plus-docstring text bundle. No
IPython or Jedi, so it runs in any uv-provisioned environment.

The functions take the execution namespace and return the ``content`` payload of the
matching Jupyter reply (``matches``/``cursor_start``/``cursor_end`` for completion;
``found``/``data`` for inspection), so the worker can relay them unchanged.
"""

from __future__ import annotations

import contextlib
import inspect
import keyword
import re
import rlcompleter
import types
from typing import Any

#: Names the worker seeds into every namespace (bound data, the display/magic helpers);
#: they are infrastructure, not the user's variables, so the inspector hides them.
_INJECTED = {"data", "display", "train_model", "register_model"}

#: Types the variable inspector skips: modules and callables are imports and
#: definitions, not the data a data scientist inspects.
_SKIP_TYPES = (
    types.ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
    type,
)


def variables(namespace: dict[str, Any]) -> dict[str, Any]:
    """The user's data variables in ``namespace``: name, type, and a short summary.

    Skips private names, injected helpers, modules, and callables, so what remains is
    the DataFrames, arrays, and values a cell produced: the notebook's live state.
    """
    out: list[dict[str, str]] = []
    for name, obj in list(namespace.items()):
        if name.startswith("_") or name in _INJECTED:
            continue
        if isinstance(obj, _SKIP_TYPES):
            continue
        out.append(
            {"name": name, "type": type(obj).__name__, "summary": _summarize(obj)}
        )
    out.sort(key=lambda v: v["name"])
    return {"variables": out}


def _summarize(obj: Any) -> str:
    """One line: shape for a frame/array, length for a container, else a repr."""
    with contextlib.suppress(Exception):
        shape = getattr(obj, "shape", None)
        if isinstance(shape, tuple) and shape and all(type(n) is int for n in shape):
            return " x ".join(str(n) for n in shape)  # DataFrame / ndarray / Series
        if isinstance(obj, (list, tuple, set, frozenset, dict)):
            count = len(obj)
            return f"{count} item{'' if count == 1 else 's'}"
        if isinstance(obj, (bool, int, float)):
            return str(obj)
        text = obj if isinstance(obj, str) else repr(obj)
        return text if len(text) <= 60 else text[:57] + "..."
    return ""


#: The identifier-or-attribute token immediately left of the cursor, and the identifier
#: continuing to its right: together they bound the name completion/inspection acts on.
_TOKEN_LEFT = re.compile(r"[\w.]*$")
_TOKEN_RIGHT = re.compile(r"^\w*")


def complete(namespace: dict[str, Any], code: str, cursor_pos: int) -> dict[str, Any]:
    """Completions for the token ending at ``cursor_pos`` (``complete_reply`` content).

    A dotted token (``obj.attr``) completes against the attributes of the resolved base;
    a bare token against the namespace, builtins, and keywords. ``cursor_start`` marks
    where the replaced text begins, so the frontend swaps the token span for the match.
    """
    match = _TOKEN_LEFT.search(code[:cursor_pos])
    token = match.group() if match else ""
    if "." in token:
        base, _, attr = token.rpartition(".")
        matches = _attr_matches(namespace, base, attr)
        cursor_start = cursor_pos - len(attr)
    else:
        matches = _name_matches(namespace, token)
        cursor_start = cursor_pos - len(token)
    return {"matches": matches, "cursor_start": cursor_start, "cursor_end": cursor_pos}


def inspect_token(
    namespace: dict[str, Any], code: str, cursor_pos: int, detail_level: int = 0
) -> dict[str, Any]:
    """Inspect the name spanning ``cursor_pos`` (an ``inspect_reply`` content).

    Resolves the dotted name straddling the cursor and renders its type, signature, and
    docstring as ``text/plain``; ``detail_level`` 1 appends the source when it can be
    found. ``found`` is False when the cursor is not on a resolvable name.
    """
    left = _TOKEN_LEFT.search(code[:cursor_pos])
    right = _TOKEN_RIGHT.match(code[cursor_pos:])
    name = ((left.group() if left else "") + (right.group() if right else "")).strip(
        "."
    )
    if not name:
        return {"found": False, "data": {}, "metadata": {}}
    try:
        obj = eval(name, namespace)  # noqa: S307 - the user's own name in their namespace
    except Exception:
        return {"found": False, "data": {}, "metadata": {}}
    return {
        "found": True,
        "data": {"text/plain": _describe(obj, detail_level)},
        "metadata": {},
    }


def _attr_matches(namespace: dict[str, Any], base: str, attr: str) -> list[str]:
    try:
        obj = eval(base, namespace)  # noqa: S307 - resolving the user's own expression
    except Exception:
        return []
    names = [name for name in dir(obj) if name.startswith(attr)]
    # Public names first, dunders last, the order a completer's user expects.
    public = [n for n in names if not n.startswith("_")]
    private = [n for n in names if n.startswith("_")]
    return public + private


def _name_matches(namespace: dict[str, Any], prefix: str) -> list[str]:
    if not prefix:
        return []
    completer = rlcompleter.Completer(namespace)
    seen: set[str] = set()
    out: list[str] = []
    state = 0
    while True:
        match = completer.complete(prefix, state)
        if match is None:
            break
        state += 1
        # rlcompleter appends "(" to callables and ": " to keys; keep the bare name.
        name = match.rstrip("(").rstrip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    for kw in keyword.kwlist:
        if kw.startswith(prefix) and kw not in seen:
            out.append(kw)
    return out


def _describe(obj: Any, detail_level: int) -> str:
    parts: list[str] = []
    name = getattr(obj, "__name__", None) or type(obj).__name__
    with contextlib.suppress(TypeError, ValueError):
        parts.append(f"Signature: {name}{inspect.signature(obj)}")
    parts.append(f"Type: {type(obj).__name__}")
    doc = inspect.getdoc(obj)
    if doc:
        parts.append("\n" + doc)
    if detail_level >= 1:
        with contextlib.suppress(OSError, TypeError):
            parts.append("\nSource:\n" + inspect.getsource(obj))
    return "\n".join(parts)
