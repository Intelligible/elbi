"""Every import in a documented example resolves against the installed packages.

The runtime ships as ``elbi_core`` and the app as ``elbi``. The docs were written
before that split and kept importing the runtime API from ``elbi``, so every example
in the README and most of ``docs/`` raised ``ImportError`` on its first line against a
released wheel. Nothing caught it, because no test reads the docs.

Imports are the part worth executing. A full example needs datasets and a registry,
but the import line is what a reader runs first and what silently rots when a symbol
moves between packages.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

#: A fenced Python block in Markdown.
_BLOCK = re.compile(r"```python\n(.*?)```", re.S)
#: An import of one of this project's packages, at the top level of a block.
_IMPORT = re.compile(
    r"^(from\s+elbi[\w.]*\s+import\s+[^\n(]+|import\s+elbi[\w.]*)$", re.M
)


def _documents() -> list[Path]:
    return [ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))]


def _imports() -> list[tuple[str, str]]:
    """Every project import statement in a documented example, with its file."""
    found: list[tuple[str, str]] = []
    for path in _documents():
        if not path.is_file():
            continue
        for block in _BLOCK.findall(path.read_text()):
            for statement in _IMPORT.findall(block):
                found.append((str(path.relative_to(ROOT)), statement.strip()))
    return found


DOCUMENTED_IMPORTS = _imports()


def test_the_documents_carry_examples_to_check() -> None:
    """A regex that stops matching would make every other test here vacuous."""
    assert len(DOCUMENTED_IMPORTS) > 10


@pytest.mark.parametrize(
    ("document", "statement"),
    DOCUMENTED_IMPORTS,
    ids=[f"{doc}::{stmt[:40]}" for doc, stmt in DOCUMENTED_IMPORTS],
)
def test_a_documented_import_resolves(document: str, statement: str) -> None:
    try:
        exec(statement, {})
    except ImportError as exc:
        pytest.fail(f"{document} documents an import that fails: {statement!r} ({exc})")
