"""Static dataflow analysis over notebook cells: the basis for reactive execution.

A reactive notebook is a graph, not a script. Each cell *defines* some global names and
*references* others; cell ``A`` depends on cell ``B`` when ``A`` reads a name ``B``
binds.
Running (or editing) a cell then re-runs exactly its transitive dependents, in a correct
order, and nothing else: the model marimo popularized, and the one that makes a
notebook
reproducible regardless of the order a human clicked cells in.

The analysis is purely static: we never run a cell to learn what it touches. Definitions
come from a scope-aware walk of the module-level bindings (so names bound only inside a
function, class, or comprehension do not leak out as globals), and references come from
CPython's own :mod:`symtable`, which resolves free variables through nested scopes the
way the compiler does, so ``def f(): return external`` correctly depends on whatever
cell defines ``external``. The one thing static analysis cannot see is *mutation*:
``lst.append(x)`` in another cell reads ``lst`` but does not redefine it, so it creates
no edge. That is a fundamental limit (tracking mutation reliably is impossible in
Python), surfaced to the user as guidance rather than silently mistracked.
"""

from __future__ import annotations

import ast
import builtins
import symtable
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

_BUILTINS = frozenset(dir(builtins))

#: Comprehension nodes. CPython's :mod:`symtable` leaks a list comprehension's loop
#: variable into the enclosing scope as a referenced global (a long-standing quirk), so
#: such a name would otherwise show up as a spurious cross-cell reference; we detect and
#: drop the ones used only as comprehension targets.
_COMPREHENSIONS = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)


@dataclass(frozen=True)
class CellDeps:
    """What one cell binds, reads, and deletes at notebook-global scope.

    ``refs`` excludes names the cell also defines (a cell that reads its own variable
    does not depend on another for it): except an augmented-assignment target such as
    ``total += 1``, which reads the prior value and rebinds, so it appears in both.
    ``syntax_error`` is set (and the sets left empty) when the source will not parse, so
    a half-written cell degrades to "no known dependencies" rather than raising.
    """

    defs: frozenset[str]
    refs: frozenset[str]
    deletes: frozenset[str] = frozenset()
    syntax_error: str | None = None


class _ModuleBindings(ast.NodeVisitor):
    """Collect names bound at module scope, without descending into nested scopes.

    A function, class, lambda, or comprehension body has its own scope, so an assignment
    there is not a notebook global; we record such a construct's *name* (a real module
    binding) but do not walk its body. Walrus targets are the exception that must be
    chased
    through expressions, since ``if (n := len(x)):`` binds ``n`` in this scope.
    """

    def __init__(self) -> None:
        self.defs: set[str] = set()
        self.deletes: set[str] = set()
        self.augmented: set[str] = set()

    def _bind_target(self, target: ast.expr) -> None:
        """Record every name a (possibly nested) assignment target binds."""
        if isinstance(target, ast.Name):
            self.defs.add(target.id)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self._bind_target(element)
        # Attribute/Subscript targets (``obj.a =``, ``d[k] =``) bind nothing new: they
        # mutate an existing object, which is a *reference* to it, captured via
        # symtable.

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._bind_target(target)
        self.visit(node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        # ``x: int = 1`` binds x; a bare ``x: int`` annotation does not, matching
        # Python.
        if node.value is not None:
            self._bind_target(node.target)
            self.visit(node.value)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        # ``x += 1`` reads the old x and rebinds it: both a definition and a reference.
        if isinstance(node.target, ast.Name):
            self.defs.add(node.target.id)
            self.augmented.add(node.target.id)
        else:
            self._bind_target(node.target)
        self.visit(node.value)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        # Walrus binds in the containing scope (even from inside a comprehension).
        if isinstance(node.target, ast.Name):
            self.defs.add(node.target.id)
        self.visit(node.value)

    def visit_For(self, node: ast.For) -> None:
        self._bind_target(node.target)
        self.visit(node.iter)
        for statement in node.body + node.orelse:
            self.visit(statement)

    visit_AsyncFor = visit_For  # type: ignore[assignment]

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            if item.optional_vars is not None:
                self._bind_target(item.optional_vars)
            self.visit(item.context_expr)
        for statement in node.body:
            self.visit(statement)

    visit_AsyncWith = visit_With  # type: ignore[assignment]

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            # ``import a.b.c`` binds the top package ``a``; ``import a.b as c`` binds
            # ``c``.
            self.defs.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            # ``from m import *`` is opaque; there is no name to bind statically.
            if alias.name != "*":
                self.defs.add(alias.asname or alias.name)

    def _visit_named_scope(self, node: ast.AST) -> None:
        """Record a def/class/lambda's own name, then stop: its body is another scope.

        Decorators, defaults, and base classes evaluate in *this* scope, so their reads
        still matter, but symtable already accounts for them, and here we only collect
        bindings, so declining to descend is exactly right.
        """
        name = getattr(node, "name", None)
        if isinstance(name, str):
            self.defs.add(name)

    visit_FunctionDef = _visit_named_scope
    visit_AsyncFunctionDef = _visit_named_scope
    visit_ClassDef = _visit_named_scope
    visit_Lambda = _visit_named_scope

    def visit_Delete(self, node: ast.Delete) -> None:
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.deletes.add(target.id)
            else:
                self.visit(target)


def _referenced_globals(table: symtable.SymbolTable, into: set[str]) -> None:
    """Collect every global name read anywhere in ``table`` or its child scopes.

    A free variable inside a function is marked global-and-referenced by the compiler,
    so
    recursing the table tree is how a reference buried in a nested ``def`` still
    surfaces
    as a notebook-level dependency.
    """
    for symbol in table.get_symbols():
        if symbol.is_referenced() and symbol.is_global():
            into.add(symbol.get_name())
    for child in table.get_children():
        _referenced_globals(child, into)


def _comprehension_only(tree: ast.AST) -> set[str]:
    """Names that appear only as comprehension loop targets, never loaded elsewhere.

    These are the list-comprehension variables :mod:`symtable` spuriously reports at
    module
    scope; a name also loaded outside a comprehension is a genuine reference and is left
    alone.
    """
    targets: set[str] = set()
    loaded_outside: set[str] = set()

    def walk(node: ast.AST, in_comp: bool) -> None:
        if isinstance(node, _COMPREHENSIONS):
            for generator in node.generators:
                for inner in ast.walk(generator.target):
                    if isinstance(inner, ast.Name):
                        targets.add(inner.id)
            in_comp = True
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and not in_comp
        ):
            loaded_outside.add(node.id)
        for child in ast.iter_child_nodes(node):
            walk(child, in_comp)

    walk(tree, False)
    return targets - loaded_outside


def analyze_code(source: str) -> CellDeps:
    """Compute the notebook-global names a code cell defines, reads, and deletes."""
    try:
        tree = ast.parse(source)
        table = symtable.symtable(source, "<cell>", "exec")
    except SyntaxError as exc:
        return CellDeps(defs=frozenset(), refs=frozenset(), syntax_error=str(exc))

    bindings = _ModuleBindings()
    bindings.visit(tree)
    defs = frozenset(bindings.defs)

    referenced: set[str] = set()
    _referenced_globals(table, referenced)
    referenced -= _comprehension_only(tree)
    # A read of a name the cell also binds is not a cross-cell dependency: except an
    # augmented target, which genuinely reads a prior (external) value before rebinding.
    refs = (referenced - _BUILTINS - defs) | (bindings.augmented - _BUILTINS)
    return CellDeps(
        defs=defs,
        refs=frozenset(refs),
        deletes=frozenset(bindings.deletes),
    )


@dataclass
class DependencyGraph:
    """The dataflow graph over a notebook's code cells, in document order.

    Construct it from ``(cell_id, source)`` pairs for the code cells only (markdown and
    raw cells bind nothing). It resolves each name to the cell that produces it, records
    where two cells produce the same name (a conflict the UI should surface, since it
    makes
    execution order-dependent), and exposes the queries the reactive engine needs: a
    topological order, the transitive dependents of a change, and any dependency cycle.
    """

    #: cell id -> the names it defines / references, in document order.
    order: list[str]
    deps: dict[str, CellDeps]
    #: name -> id of the cell that defines it (last writer wins, so a redefinition in a
    #: later cell shadows an earlier one, matching top-to-bottom execution).
    producers: dict[str, str] = field(init=False)
    #: name -> every cell id that defines it, when more than one does.
    conflicts: dict[str, list[str]] = field(init=False)
    #: cell id -> the ids it directly depends on (its upstream producers).
    upstream: dict[str, set[str]] = field(init=False)

    def __init__(self, cells: Sequence[tuple[str, str]]) -> None:
        self.order = [cell_id for cell_id, _ in cells]
        self.deps = {cell_id: analyze_code(source) for cell_id, source in cells}
        definers: dict[str, list[str]] = {}
        for cell_id in self.order:
            for name in self.deps[cell_id].defs:
                definers.setdefault(name, []).append(cell_id)
        self.producers = {name: ids[-1] for name, ids in definers.items()}
        self.conflicts = {name: ids for name, ids in definers.items() if len(ids) > 1}
        self.upstream = {}
        for cell_id in self.order:
            producers = {
                self.producers[name]
                for name in self.deps[cell_id].refs
                if name in self.producers and self.producers[name] != cell_id
            }
            self.upstream[cell_id] = producers

    def downstream(self, cell_id: str) -> set[str]:
        """The cells that directly depend on ``cell_id`` (read a name it defines)."""
        return {
            other for other in self.order if cell_id in self.upstream.get(other, ())
        }

    def topo_order(self) -> list[str]:
        """Every cell in a dependency-respecting order; cycle members fall to the end.

        Kahn's algorithm over document order, so among cells with no ordering constraint
        the notebook's own layout is preserved (stable and predictable for the user).
        """
        indegree = {cell_id: len(self.upstream[cell_id]) for cell_id in self.order}
        ready = [cell_id for cell_id in self.order if indegree[cell_id] == 0]
        result: list[str] = []
        while ready:
            current = ready.pop(0)
            result.append(current)
            for other in self.order:
                if current in self.upstream.get(other, ()):
                    indegree[other] -= 1
                    if indegree[other] == 0:
                        ready.append(other)
        # Anything left has an unsatisfied in-edge: it sits on a cycle. Append in
        # document order so the run plan is still total (the reactive engine reports the
        # cycle).
        result.extend(cell_id for cell_id in self.order if cell_id not in result)
        return result

    def cycle_members(self) -> set[str]:
        """Cells that cannot be topologically ordered because they form a cycle."""
        ordered: set[str] = set()
        indegree = {cell_id: len(self.upstream[cell_id]) for cell_id in self.order}
        ready = [cell_id for cell_id in self.order if indegree[cell_id] == 0]
        while ready:
            current = ready.pop(0)
            ordered.add(current)
            for other in self.order:
                if current in self.upstream.get(other, ()) and other not in ordered:
                    indegree[other] -= 1
                    if indegree[other] == 0:
                        ready.append(other)
        return {cell_id for cell_id in self.order if cell_id not in ordered}

    def descendants(self, roots: Iterable[str]) -> set[str]:
        """Every cell transitively downstream of ``roots`` (not including the roots)."""
        seen: set[str] = set()
        stack = list(roots)
        while stack:
            current = stack.pop()
            for child in self.downstream(current):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        return seen

    def run_plan(self, changed: Iterable[str]) -> list[str]:
        """The cells to re-execute when ``changed`` runs: them plus all dependents.

        Returned in topological order, so a cell never runs before a cell it depends on.
        This is the reactive execution set: running one cell propagates to exactly the
        cells whose inputs it could have altered, and no others.
        """
        roots = set(changed)
        affected = roots | self.descendants(roots)
        return [cell_id for cell_id in self.topo_order() if cell_id in affected]
