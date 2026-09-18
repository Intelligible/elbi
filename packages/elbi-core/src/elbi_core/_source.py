"""Reading a derivation's source back out of the module that defined it.

``inspect.getsource(fn)`` returns a function and nothing else, which is lossy for a
derivation: the decorator, the contract it declares, the constants it reads and the
helpers it calls all live at module level and none of them come with it. Read back that
way, a derivation's "source" cannot be run, only looked at — and anything that pastes it
somewhere executable (opening one in a notebook) gets a ``NameError`` on line one.

So the preamble is captured with it: every module-level statement the function actually
depends on, in the order the file declares them. Sibling derivations are left out, since
they are their own records and pulling one in would drag the whole module along with it.

A relative import is resolved rather than copied. ``from .shared import CEILING`` is a
statement a notebook cell cannot execute at all — there is no parent package there — so
the names it brings in are followed into the sibling module and emitted as the
definitions they refer to.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from typing import Any

#: Node types that bind a name at module level and are worth carrying: imports, the
#: constants and objects a derivation reads, and the helpers it calls.
_BINDING = (
    ast.Import,
    ast.ImportFrom,
    ast.Assign,
    ast.AnnAssign,
    ast.FunctionDef,
    ast.AsyncFunctionDef,
    ast.ClassDef,
)


def _bound_names(node: ast.stmt) -> set[str]:
    """The module-level names one statement binds."""
    if isinstance(node, ast.Import | ast.ImportFrom):
        return {(alias.asname or alias.name).split(".")[0] for alias in node.names}
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


def _is_derivation(node: ast.stmt) -> bool:
    """Whether a statement defines a derivation, which is its own stored record."""
    if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
        return False
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = getattr(target, "id", None) or getattr(target, "attr", None)
        if name == "derivation":
            return True
    return False


def _free_names(source: str) -> set[str]:
    """Every name a fragment reads, whether or not it also binds it."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {
        n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
    }


def _preamble(
    module: Any, wanted: set[str], seen_modules: set[str]
) -> tuple[list[str], set[str]]:
    """Module-level statements binding any of ``wanted``, in declaration order.

    Returns the statements and the names still unaccounted for, so a caller can follow
    a relative import into the module that satisfies it.
    """
    if module is None or getattr(module, "__name__", None) in seen_modules:
        return [], wanted
    seen_modules.add(getattr(module, "__name__", ""))
    try:
        module_source = inspect.getsource(module)
        tree = ast.parse(module_source)
    except (OSError, TypeError, SyntaxError):
        return [], wanted

    keep: list[str] = []
    # Backwards, so a helper's own dependencies are wanted by the time the statement
    # that declares them is reached: `_environment` needs `_PREVIEW`, declared above it.
    for node in reversed(tree.body):
        if _is_derivation(node) or not isinstance(node, _BINDING):
            continue
        bound = _bound_names(node)
        if not bound & wanted:
            continue
        segment = ast.get_source_segment(module_source, node) or ""

        if isinstance(node, ast.ImportFrom) and node.level:
            # A cell has no parent package, so this line cannot run there. Follow it
            # into the sibling and take the definitions instead of the import.
            try:
                sibling = importlib.import_module(
                    "." * node.level + (node.module or ""),
                    package=getattr(module, "__package__", None),
                )
            except (ImportError, TypeError, ValueError):
                continue
            inner, _ = _preamble(sibling, set(bound & wanted), seen_modules)
            keep.extend(reversed(inner))
            wanted -= bound
            continue

        keep.append(segment)
        wanted = (wanted - bound) | _free_names(segment)

    return list(reversed(keep)), wanted


def split_stored(source: str) -> tuple[list[str], str]:
    """Split a stored source back into its preamble statements and its derivation.

    :func:`compute_source` joins the two, and the store keeps the joined string. A
    caller laying several derivations out together needs them apart again, and parsing
    is how: everything up to the decorated function is context, the rest is the
    derivation. A source with no decorated function is all derivation and no context,
    which is what an agent-authored one looks like.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return [], source

    for node in tree.body:
        if not _is_derivation(node):
            continue
        cut = min([node.lineno, *(d.lineno for d in node.decorator_list)]) - 1
        lines = source.splitlines()
        preamble = [
            segment
            for statement in tree.body
            if statement.lineno < cut + 1
            and statement is not node
            and (segment := ast.get_source_segment(source, statement))
        ]
        return preamble, "\n".join(lines[cut:]).strip("\n")
    return [], source


def source_parts(fn: Any) -> tuple[list[str], str]:
    """A derivation's compute, and the module-level statements it needs, separately.

    Kept apart for callers that lay the two out themselves: seeding a derivation and its
    upstream into one notebook would otherwise repeat a shared import, constant and
    helper in every cell, since each source is self-contained on its own.
    """
    try:
        own = textwrap.dedent(inspect.getsource(fn))
    except (OSError, TypeError):
        return [], ""

    statements, _ = _preamble(inspect.getmodule(fn), _free_names(own), set())
    return statements, own


def compute_source(fn: Any) -> str:
    """A derivation's compute with the module-level context it needs to run.

    Best-effort: an empty string when the source is unavailable (a REPL- or C-defined
    compute), and the bare function when its module cannot be read, which is what this
    returned before the preamble existed.
    """
    statements, own = source_parts(fn)
    preamble = "\n".join(statements)
    return f"{preamble}\n\n{own}" if preamble else own
