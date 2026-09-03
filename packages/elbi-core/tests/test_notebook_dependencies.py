"""Tests for the notebook dataflow analysis (defs/refs and the cell graph).

Correctness of reactive execution rests entirely on this: a missed edge under-runs (a
stale downstream), a spurious edge over-runs. The cases here pin the scoping rules that
are easy to get wrong (augmented assignment as read-and-write, a free variable inside a
nested function, a comprehension variable that must not leak) and the graph queries the
engine drives (topological order, transitive dependents, cycles, redefinition
conflicts).
"""

from __future__ import annotations

from elbi_core.notebook import DependencyGraph, analyze_code


def test_defs_and_refs_basic() -> None:
    deps = analyze_code("import pandas as pd\ntotal = sum(r['x'] for r in data)\ntotal")
    assert deps.defs == frozenset({"pd", "total"})
    assert deps.refs == frozenset({"data"})


def test_augmented_assignment_is_read_and_write() -> None:
    # ``total += 1`` reads the prior (external) value and rebinds it: both def and ref.
    deps = analyze_code("total += 1")
    assert "total" in deps.defs
    assert "total" in deps.refs


def test_free_variable_in_nested_function_is_a_reference() -> None:
    # A name read inside a function body but defined by another cell is a dependency.
    deps = analyze_code("def f():\n    return external_data.sum()\nout = f()")
    assert "external_data" in deps.refs
    assert deps.defs == frozenset({"f", "out"})


def test_comprehension_variable_does_not_leak() -> None:
    # The list-comp loop variable is local; it must not appear as a def or a ref.
    deps = analyze_code("ys = [z for z in xs]")
    assert deps.defs == frozenset({"ys"})
    assert deps.refs == frozenset({"xs"})


def test_syntax_error_degrades_to_no_dependencies() -> None:
    deps = analyze_code("x = ")
    assert deps.syntax_error is not None
    assert deps.defs == frozenset()
    assert deps.refs == frozenset()


def test_graph_edges_and_topo_order() -> None:
    graph = DependencyGraph(
        [("a", "df = load()"), ("b", "m = df.mean()"), ("c", "print('unrelated')")]
    )
    assert graph.upstream["b"] == {"a"}
    assert graph.upstream["c"] == set()
    order = graph.topo_order()
    assert order.index("a") < order.index("b")


def test_run_plan_includes_transitive_dependents_in_order() -> None:
    graph = DependencyGraph(
        [("a", "x = 1"), ("b", "y = x + 1"), ("c", "z = y + 1"), ("d", "w = 9")]
    )
    plan = graph.run_plan(["a"])
    assert plan == ["a", "b", "c"]  # d is independent, so it is not re-run
    assert graph.run_plan(["d"]) == ["d"]


def test_redefinition_is_reported_as_a_conflict() -> None:
    graph = DependencyGraph([("a", "df = 1"), ("b", "df = 2")])
    assert "df" in graph.conflicts
    assert graph.conflicts["df"] == ["a", "b"]


def test_cycle_is_detected() -> None:
    graph = DependencyGraph([("a", "x = y"), ("b", "y = x")])
    assert graph.cycle_members() == {"a", "b"}


def test_mutation_creates_no_edge_documented_limit() -> None:
    # Static analysis cannot see mutation: cell b mutates df but does not redefine it,
    # so it depends on df (a read) but does not itself become a producer of a new name.
    graph = DependencyGraph([("a", "df = frame()"), ("b", "df['c'] = df['a'] + 1")])
    assert graph.upstream["b"] == {"a"}
    assert "c" not in graph.producers  # the column is invisible to the analysis
