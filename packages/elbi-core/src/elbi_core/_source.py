"""Reading a derivation's source back out of the module that defined it.

``inspect.getsource(fn)`` returns a function and nothing else, which is lossy for a
derivation: the decorator, the contract it declares, the constants it reads and the
helpers it calls all live at module level and none of them come with it. Read back that
way, a derivation's "source" cannot be run, only looked at, and anything that pastes it
somewhere executable (opening one in a notebook) gets a ``NameError`` on line one.

So the preamble is captured with it: every module-level statement the function actually
depends on, in the order the file declares them. Sibling derivations are left out, since
they are their own records and pulling one in would drag the whole module along with it.

A relative import is resolved rather than copied. ``from .shared import CEILING`` is a
statement a notebook cell cannot execute at all (there is no parent package there), so
the names it brings in are followed into the sibling module and emitted as the
definitions they refer to, rebound under their ``as`` name where the import renames one.
A submodule, as in ``from . import helpers``, is rebuilt from its file instead.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import textwrap
from typing import Any, TypeGuard

from .derivation import Derivation

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


def _is_derivation(
    node: ast.stmt,
) -> TypeGuard[ast.FunctionDef | ast.AsyncFunctionDef]:
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
    name = getattr(module, "__name__", "")
    if module is None or name in seen_modules:
        return [], wanted
    try:
        module_source = inspect.getsource(module)
        tree = ast.parse(module_source)
    except (OSError, TypeError, SyntaxError):
        return [], wanted

    futures: list[str] = []
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
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            # It has to stay the first statement, ahead of the `__file__` binding.
            futures.append(segment)
            wanted -= bound
            continue

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
            there: set[str] = set()
            for alias in node.names:
                local = alias.asname or alias.name
                if local not in wanted:
                    continue
                member = getattr(sibling, alias.name, None)
                if (
                    inspect.ismodule(member)
                    and member.__name__ == f"{sibling.__name__}.{alias.name}"
                ):
                    # `from . import helpers` binds a submodule, not a definition.
                    rebuilt = _rebuilt_module(local, member, seen_modules | {name})
                    if rebuilt:
                        keep.append(rebuilt)
                    continue
                if local != alias.name and not isinstance(member, Derivation):
                    # `from .ci import _window as _ci_window`: the sibling defines the
                    # original name. Not a derivation, which is a later cell of its own.
                    keep.append(f"{local} = {alias.name}")
                there.add(alias.name)
            if there:
                # Only the modules on this route count as seen: a cycle stops, and a
                # second route into the same sibling still gets what it needs.
                inner, _ = _preamble(sibling, there, seen_modules | {name})
                keep.extend(reversed(inner))
            wanted -= bound
            continue

        keep.append(segment)
        wanted = (wanted - bound) | _free_names(segment)

    path = getattr(module, "__file__", None)
    if "__file__" in wanted and path:
        # The import system binds `__file__`, and a cell has no file of its own, so
        # the statements lifted out of this module get the one it was loaded from.
        keep.append(f"__file__ = {path!r}")
        wanted = wanted - {"__file__"}
    return [*futures, *reversed(keep)], wanted


def _rebuilt_module(local: str, module: Any, seen_modules: set[str]) -> str:
    """A statement binding ``local`` to ``module``, rebuilt from its file.

    A cell cannot import a sibling module, having no parent package, but it can do what
    the import system does: create the module and run its definitions in it. Those are
    read the way any preamble is, so the module's own relative imports are resolved
    too. Empty when the source cannot be read.
    """
    try:
        tree = ast.parse(inspect.getsource(module))
    except (OSError, TypeError, SyntaxError):
        return ""
    names = set().union(*(_bound_names(node) for node in tree.body))
    statements, _ = _preamble(module, names, seen_modules)
    source = "\n".join(statements)
    return (
        "import types as _types\n"
        f"{local} = _types.ModuleType({module.__name__!r})\n"
        f"{local}.__file__ = {module.__file__!r}\n"
        f"exec(compile({source!r}, {local}.__file__, 'exec'), vars({local}))"
    )


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
