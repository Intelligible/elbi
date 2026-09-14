"""Mounts the components-vs-RAG split-screen demo onto a running elbi app.

Uses the real `elbi.extensions` protocol (`packages/elbi/src/elbi/extensions.py`)
rather than reinventing one: `install_extension(context)` mounts routes on
`context.app`, the same call the shipped app makes once at startup for any
installed extension. Nothing here touches the shipped app's own routes, store, or
services -- this demo needs none of them (every non-`app` field on
`ExtensionContext` is optional, and none is used below).

A productized version of this would register itself as a real entry point
(`[project.entry-points."elbi.extensions"]` in a package's `pyproject.toml`);
example directories in this repo aren't installable packages (their numeric
prefixes aren't even valid Python identifiers), so `serve_demo.py` calls
`install_extension` directly on a bare app instead -- exactly what entry-point
discovery would do on the shipped app, minus the indirection.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

EXAMPLE_ROOT = Path(__file__).resolve().parent
for _path in (EXAMPLE_ROOT, EXAMPLE_ROOT / "agent", EXAMPLE_ROOT / "rag"):
    sys.path.insert(0, str(_path))

import demo_agent  # noqa: E402
import pipeline  # noqa: E402
import publish_update  # noqa: E402

if TYPE_CHECKING:
    from elbi.extensions import ExtensionContext

ROUTE_PREFIX = "/demo/components-vs-rag"


class AskRequest(BaseModel):
    question: str


class DemoState:
    """Runtime paths plus a lazily-built embedder and RAG vector store.

    Built once per app process (`install_extension` constructs one and closes
    over it in the route handlers below), so the embedding model loads at most
    once and the RAG store persists -- including across a "publish update" --
    exactly like a real running index would.
    """

    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = publish_update.ensure_runtime(runtime_dir)
        self._embedder: Any = None
        self._store: pipeline.VectorStore | None = None

    @property
    def embedder(self) -> Any:
        if self._embedder is None:
            from elbi_core.retrieval import OnnxEmbedder

            self._embedder = OnnxEmbedder()
        return self._embedder

    @property
    def store(self) -> pipeline.VectorStore:
        if self._store is None:
            self._store = pipeline.build_store(
                self.runtime_dir / "rag_docs", embedder=self.embedder
            )
        return self._store

    def ask_both(self, question: str) -> dict[str, Any]:
        components = demo_agent.load_all_components(
            self.runtime_dir / "rate_schedule.csv"
        )
        elbi_answer = demo_agent.ask(question, components, embedder=self.embedder)
        rag_answer = pipeline.ask(question, self.store, embedder=self.embedder)
        return {
            "rag": {
                "answer": rag_answer.answer,
                "chunks": [
                    {"source": c.source, "heading": c.heading, "text": c.text}
                    for c in rag_answer.chunks
                ],
            },
            "elbi": {
                "declined": elbi_answer.declined,
                "answer": elbi_answer.answer,
                "components": [
                    {
                        "id": c["id"],
                        "statement": c["statement"],
                        "effective_date": c["structure"]["effective_date"],
                        "citation": c["evidence"].get("citation"),
                        "relations": c["relations"],
                    }
                    for c in elbi_answer.components
                ],
            },
        }

    def publish(self) -> dict[str, Any]:
        result = publish_update.run(self.runtime_dir)
        if result.applied and self._store is not None:
            # Extend the already-built store in place with just the new document's
            # chunks -- never rebuild from scratch, never touch what's already
            # indexed. See rag/pipeline.py's module docstring for why.
            new_path = self.runtime_dir / "rag_docs" / "amendment_1.md"
            self._store.add(pipeline.chunk_document(new_path), self.embedder)
        return {
            "applied": result.applied,
            "detail": result.detail,
            "old_component_id": result.old_component_id,
            "new_component_id": result.new_component_id,
        }


def install_extension(
    context: ExtensionContext, *, runtime_dir: Path = EXAMPLE_ROOT / "runtime"
) -> None:
    """Mount the demo's page and API onto `context.app`.

    `runtime_dir` defaults to this example's own gitignored working copy (what a
    live `serve_demo.py` uses); tests pass an isolated `tmp_path` instead, so a
    test run never touches the shared live-demo state.
    """
    state = DemoState(runtime_dir)
    router = APIRouter()

    @router.get(ROUTE_PREFIX)
    def index() -> FileResponse:
        return FileResponse(EXAMPLE_ROOT / "static" / "index.html")

    @router.post(ROUTE_PREFIX + "/api/ask")
    def ask(body: AskRequest) -> JSONResponse:
        return JSONResponse(state.ask_both(body.question))

    @router.post(ROUTE_PREFIX + "/api/publish-update")
    def publish() -> JSONResponse:
        return JSONResponse(state.publish())

    context.app.include_router(router)
