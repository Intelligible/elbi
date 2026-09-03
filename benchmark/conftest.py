"""Put the repo root on sys.path so tests can import the ``benchmark`` package.

The harness lives in a top-level ``benchmark/`` directory (not an installed
workspace package), mirroring ``spec/``. Under pytest's importlib mode nothing adds
the repo root automatically, so this makes ``import benchmark.<module>`` resolve.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
