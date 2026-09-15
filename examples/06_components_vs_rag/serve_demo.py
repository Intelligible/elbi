"""Stand-alone launcher for the split-screen demo.

Builds a bare FastAPI app and mounts this example's extension on it via the real
`elbi.extensions.ExtensionContext`/`install_extension` protocol -- see
`extension.py`'s module docstring for why this calls it directly instead of going
through entry-point discovery. `elbi serve` itself, unmodified, would mount the
same extension the same way if this were registered as a real installed package.

Usage:
    uv run --package elbi python examples/06_components_vs_rag/serve_demo.py
    # then open http://localhost:8000/demo/components-vs-rag
"""

from __future__ import annotations

import sys
from pathlib import Path

EXAMPLE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(EXAMPLE_ROOT))


def build_app():
    import extension
    from fastapi import FastAPI

    from elbi.extensions import ExtensionContext

    app = FastAPI(title="elbi (components-vs-rag demo)")
    context = ExtensionContext(app=app, store=None)
    extension.install_extension(context)
    return app


def main() -> None:
    import uvicorn

    uvicorn.run(build_app(), host="127.0.0.1", port=8000)


if __name__ == "__main__":
    main()
