"""elbi-app: the FastAPI surface over the verifying runtime.

:func:`create_app` builds the ASGI app: a Server-Sent-Events chat endpoint over the
runtime, an optionally mounted MCP server, and the built SPA. Importing it pulls in only
FastAPI and the runtime; the launcher (:mod:`elbi.serve`) adds the project loader and
LLM client.
"""

from __future__ import annotations

from .app import create_app

__version__ = "0.1.0"

__all__ = ["__version__", "create_app"]
