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
    "attempted relative import with no known parent package", a failure with no
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


def test_a_relatively_imported_module_comes_with_it(tmp_path) -> None:
    # `from . import helpers` binds a submodule, not a definition in one, and a cell has
    # no package to import it from, so the module has to come with the source.
    import importlib

    package = _package(
        tmp_path,
        {
            "helpers": "def double(x):\n    return 2 * x\n",
            "main": (
                "from . import helpers\n\ndef total(n):\n    return helpers.double(n)\n"
            ),
        },
    )
    main = importlib.import_module(f"{package.__name__}.main")
    namespace = {"__name__": "__main__"}

    exec(compile(compute_source(main.total), "<cell>", "exec"), namespace)

    assert namespace["total"](3) == 6


def test_a_relatively_imported_module_brings_its_own_relative_imports(tmp_path) -> None:
    # The rebuilt module has no package either, so what it imports from a sibling has to
    # arrive as definitions, the same as for the derivation's own module.
    package = _package(
        tmp_path,
        {
            "shared": "FACTOR = 3\n",
            "helpers": (
                "from .shared import FACTOR\n\ndef triple(x):\n    return FACTOR * x\n"
            ),
            "main": (
                "from . import helpers\n\nLIMIT = 5\n\n"
                "def total():\n    return helpers.triple(LIMIT)\n"
            ),
        },
    )

    assert _run(package, "total") == 15


def test_a_rebuilt_module_keeps_its_future_import_first(tmp_path) -> None:
    # `from __future__` has to be the first statement the rebuilt module runs, ahead of
    # the `__file__` its own lines read.
    package = _package(
        tmp_path,
        {
            "helpers": (
                "from __future__ import annotations\n\nfrom pathlib import Path\n\n"
                "NAME = Path(__file__).stem\n"
            ),
            "main": "from . import helpers\n\ndef name():\n    return helpers.NAME\n",
        },
    )

    assert _run(package, "name") == "helpers"


def test_a_constant_reading_its_own_file_still_finds_it(tmp_path) -> None:
    # The import system binds `__file__`, not a statement, so a cell running the lifted
    # `Path(__file__)` line needs it bound to the module's own file.
    module = _module(
        tmp_path,
        """
        from pathlib import Path

        HERE = Path(__file__).parent

        def here():
            return HERE
        """,
    )
    namespace = {"__name__": "__main__"}

    exec(compile(compute_source(module.here), "<cell>", "exec"), namespace)

    assert namespace["here"]() == tmp_path


def _run(package, fn_name: str, *args):
    """Exec ``main.<fn_name>``'s source the way a notebook cell does, then call it."""
    import importlib

    main = importlib.import_module(f"{package.__name__}.main")
    namespace = {"__name__": "__main__"}
    exec(compile(compute_source(getattr(main, fn_name)), "<cell>", "exec"), namespace)
    return namespace[fn_name](*args)


def test_a_module_a_sibling_imports_is_still_imported(tmp_path) -> None:
    # `json` is the sibling's import, not a submodule of the package, so its import line
    # is what comes along, not the module's own source.
    package = _package(
        tmp_path,
        {
            "shared": "import json\n",
            "main": (
                "from .shared import json\n\ndef dump(x):\n    return json.dumps(x)\n"
            ),
        },
    )

    assert _run(package, "dump", {"a": 1}) == '{"a": 1}'


def test_a_renamed_relative_import_is_bound_under_its_new_name(tmp_path) -> None:
    # The sibling defines `CEILING`; the function reads it as `CAP`.
    package = _package(
        tmp_path,
        {
            "shared": "CEILING = 10\n",
            "main": (
                "from .shared import CEILING as CAP\n\ndef cap():\n    return CAP\n"
            ),
        },
    )

    assert _run(package, "cap") == 10


def test_two_routes_into_one_sibling_each_get_what_they_need(tmp_path) -> None:
    # `main` wants `CEILING` from `shared`, and so does `helpers`, for `FLOOR`. Reaching
    # `shared` a second time is not a cycle.
    package = _package(
        tmp_path,
        {
            "shared": "CEILING = 10\nFLOOR = 1\n",
            "helpers": (
                "from .shared import FLOOR\n\ndef clamp(x):\n    return max(FLOOR, x)\n"
            ),
            "main": (
                "from .shared import CEILING\nfrom .helpers import clamp\n\n"
                "def capped(x):\n    return min(CEILING, clamp(x))\n"
            ),
        },
    )

    assert _run(package, "capped", 50) == 10
    assert _run(package, "capped", -5) == 1


def test_a_renamed_upstream_derivation_leaves_the_context_runnable(tmp_path) -> None:
    # An upstream derivation is its own cell, after the context, so the context cannot
    # bind a new name to it: that line would fail and take the rest of the context down.
    import importlib

    from elbi_core._source import split_stored

    derivation = (
        "from elbi_core import derivation\n"
        "from elbi_core.registry import Registry\n\n"
        "REGISTRY = Registry()\n\n"
    )
    package = _package(
        tmp_path,
        {
            "up": derivation
            + "@derivation(registry=REGISTRY)\ndef upstream(ctx):\n    return []\n",
            "main": derivation
            + (
                "from .up import upstream as up\n\n"
                "@derivation(inputs={'u': up}, registry=REGISTRY)\n"
                "def down(ctx):\n    return []\n"
            ),
        },
    )
    main = importlib.import_module(f"{package.__name__}.main")
    context, _ = split_stored(compute_source(main.down.compute))
    namespace = {"__name__": "__main__"}

    exec(compile("\n".join(context), "<setup>", "exec"), namespace)

    assert "REGISTRY" in namespace
