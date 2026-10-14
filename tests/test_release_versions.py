"""The packages and the web app release together, so their version strings must agree.

Each package carries its version twice: ``version =`` in its ``pyproject.toml``, which
is what the wheel is stamped with and what PyPI indexes, and ``__version__`` in its
``__init__.py``, which is what ``elbi --version``, the update check's User-Agent and
``/api/version`` report. The web app ships inside the ``elbi`` wheel and carries its own
in ``package.json``, mirrored twice in ``package-lock.json``. Nothing else holds the
eleven together, and RELEASING.md names only one of them.

Read as text rather than imported or parsed: CI still runs Python 3.10, which has no
``tomllib``, and importing ``elbi`` loads FastAPI to compare a string.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("elbi", "elbi-agent", "elbi-cli", "elbi-core")
WEB = ROOT / "packages" / "elbi" / "web"


def _versions() -> dict[str, str]:
    version_by_location: dict[str, str] = {}
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
            version_by_location[f"{name}/{label}"] = match.group(1)
    web_manifest = json.loads((WEB / "package.json").read_text())
    web_lock = json.loads((WEB / "package-lock.json").read_text())
    version_by_location["web/package.json"] = web_manifest["version"]
    version_by_location["web/package-lock.json"] = web_lock["version"]
    web_lock_root = web_lock["packages"][""]
    version_by_location["web/package-lock.json root package"] = web_lock_root["version"]
    return version_by_location


def test_every_version_string_agrees() -> None:
    versions = _versions()
    assert len(set(versions.values())) == 1, versions
