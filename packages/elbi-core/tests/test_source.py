"""Reading a derivation's source back with the module context it needs."""

from __future__ import annotations

import textwrap

from elbi_core._source import compute_source


def _module(tmp_path, body: str):
    """Import a throwaway module so its functions carry a real file to read back."""
    import importlib.util
    import sys
    import uuid

    name = f"mod_under_test_{uuid.uuid4().hex}"
    path = tmp_path / f"{name}.py"
    path.write_text(textwrap.dedent(body), encoding="utf-8")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # Registered, because `inspect.getmodule` resolves a function's module through
    # `sys.modules`: without this every case silently exercises the fallback.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_the_imports_a_function_uses_come_with_it(tmp_path) -> None:
    # `inspect.getsource` alone returns the function, so a paste of it dies on the
    # first name the module imported.
    module = _module(
        tmp_path,
        """
        from collections import defaultdict
        import json

        def counts():
            return defaultdict(int)
        """,
    )

    source = compute_source(module.counts)

    assert "from collections import defaultdict" in source
    assert "def counts():" in source


def test_an_import_it_does_not_use_is_left_behind(tmp_path) -> None:
    # Carrying the whole module would make a derivation's source unreadable.
    module = _module(
        tmp_path,
        """
        from collections import defaultdict
        import json

        def counts():
            return defaultdict(int)
        """,
    )

    assert "import json" not in compute_source(module.counts)


def test_constants_and_helpers_come_with_it(tmp_path) -> None:
    module = _module(
        tmp_path,
        """
        FAILED = "analysis_failed"

        def _environment(stage):
            return stage

        def outcome(row):
            return _environment(row), FAILED
        """,
    )

    source = compute_source(module.outcome)

    assert 'FAILED = "analysis_failed"' in source
    assert "def _environment(stage):" in source


def test_a_helper_brings_what_it_needs_in_turn(tmp_path) -> None:
    # `outcome` never names `_PREVIEW`; only the helper it calls does. Walking the
    # module once, top to bottom, would drop it.
    module = _module(
        tmp_path,
        """
        import re

        _PREVIEW = re.compile(r"^pr\\d+$")

        def _environment(stage):
            return bool(_PREVIEW.match(stage))

        def outcome(stage):
            return _environment(stage)
        """,
    )

    source = compute_source(module.outcome)

    assert "_PREVIEW = re.compile" in source
    assert "import re" in source


def test_declaration_order_is_preserved(tmp_path) -> None:
    # A constant must appear above the helper that reads it, or the paste cannot run.
    module = _module(
        tmp_path,
        """
        LIMIT = 3

        def _cap(n):
            return min(n, LIMIT)

        def capped(n):
            return _cap(n)
        """,
    )

    source = compute_source(module.capped)

    assert source.index("LIMIT = 3") < source.index("def _cap(n):")
    assert source.index("def _cap(n):") < source.index("def capped(n):")


def test_a_sibling_derivation_is_not_dragged_in(tmp_path) -> None:
    # Each derivation is its own stored record; pulling a sibling's body in here would
    # duplicate it and, transitively, drag the whole module along.
    module = _module(
        tmp_path,
        """
        def derivation(**kwargs):
            return lambda fn: fn

        @derivation()
        def upstream(ctx):
            return []

        @derivation()
        def downstream(ctx):
            return upstream(ctx)
        """,
    )

    source = compute_source(module.downstream)

    assert "def downstream(ctx):" in source
    assert "def upstream(ctx):" not in source.split("def downstream")[0]


def test_an_unreadable_compute_yields_nothing(tmp_path) -> None:
    # A REPL- or C-defined compute has no source; callers want "" rather than a raise.
    assert compute_source(len) == ""


def _package(tmp_path, files: dict[str, str]):
    """Import a throwaway package, so relative imports between modules are real."""
    import importlib
    import sys
    import uuid

    name = f"pkg_under_test_{uuid.uuid4().hex}"
    root = tmp_path / name
    root.mkdir()
    (root / "__init__.py").write_text("", encoding="utf-8")
    for stem, body in files.items():
        (root / f"{stem}.py").write_text(textwrap.dedent(body), encoding="utf-8")
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module(name)
    finally:
        sys.path.remove(str(tmp_path))


def test_a_relative_import_is_resolved_not_copied(tmp_path) -> None:
    """`from .sibling import x` cannot run in a notebook cell.

    A cell has no parent package, so copying the line verbatim raises
    "attempted relative import with no known parent package" — a failure with no
    relationship to what the reader was trying to do. The names it brings in have to
    arrive as definitions instead.
    """
    import importlib

    package = _package(
        tmp_path,
        {
            "shared": "CEILING = 1.0\n\ndef helper(x):\n    return x\n",
            "main": (
                "from .shared import CEILING, helper\n\n"
                "def total(n):\n    return helper(n) + CEILING\n"
            ),
        },
    )
    main = importlib.import_module(f"{package.__name__}.main")

    source = compute_source(main.total)

    assert "from .shared import" not in source
    assert "CEILING = 1.0" in source
    assert "def helper(x):" in source
    assert source.index("CEILING = 1.0") < source.index("def total(n):")


def test_an_absolute_import_is_still_copied(tmp_path) -> None:
    # Only relative imports are unrunnable in a cell; `from collections import ...`
    # works there exactly as it does in the module.
    import importlib

    package = _package(
        tmp_path,
        {
            "main": (
                "from collections import defaultdict\n\n"
                "def counts():\n    return defaultdict(int)\n"
            )
        },
    )
    main = importlib.import_module(f"{package.__name__}.main")

    assert "from collections import defaultdict" in compute_source(main.counts)
