"""The four packages release together, so their eight version strings must agree.

Each package carries its version twice: ``version =`` in its ``pyproject.toml``, which
is what the wheel is stamped with and what PyPI indexes, and ``__version__`` in its
``__init__.py``, which is what ``elbi --version``, the update check's User-Agent and
``/api/version`` report. Nothing else holds the eight together, and RELEASING.md names
only one of them.

Read as text rather than imported or parsed: CI still runs Python 3.10, which has no
``tomllib``, and importing ``elbi`` loads FastAPI to compare a string.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("elbi", "elbi-agent", "elbi-cli", "elbi-core")


def _versions() -> dict[str, str]:
    found: dict[str, str] = {}
    for name in PACKAGES:
        package = ROOT / "packages" / name
        metadata = (package / "pyproject.toml").read_text()
        init = (package / "src" / name.replace("-", "_") / "__init__.py").read_text()
        for label, text, pattern in (
            ("pyproject.toml", metadata, r'^version = "([^"]+)"$'),
            ("__init__.py", init, r'^__version__ = "([^"]+)"$'),
        ):
            match = re.search(pattern, text, re.MULTILINE)
            assert match, f"{name}/{label} has no static version"
            found[f"{name}/{label}"] = match.group(1)
    return found


def test_every_version_string_agrees() -> None:
    versions = _versions()
    assert len(set(versions.values())) == 1, versions
