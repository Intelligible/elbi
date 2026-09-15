"""A plain retrieval-augmented-generation baseline, the RAG side of this demo.

Chunk a directory of Markdown documents by section, embed each chunk with a real
contextual embedding model, hold the vectors in a local in-memory store, and answer
a question from the top-k most similar chunks. This is the shape a RAG vendor would
actually ship (chunk -> embed -> top-k -> LLM call), not a strawman: see
`../llm.py` for why the same prompt template is used on both sides of this demo, and
the module docstring in `../agent/demo_agent.py` for the two things this pipeline
deliberately does *not* have that the Elbi side does.

Deliberately absent, on purpose (this is the point of the demo, not an oversight):
  - No recency tracking. A document is ingested by adding its chunks to the store;
    nothing marks an earlier chunk about the same fact as superseded, and nothing
    removes it. `../publish_update.py`'s RAG-side action is exactly this: add a new
    document's chunks, touch nothing already indexed. That is a common real-world
    failure mode (most minimal RAG deployments have no invalidation step at all),
    not an artificially crippled pipeline.
  - No relevance floor. Top-k always returns k chunks, however low their similarity
    to the query -- there is no threshold below which the pipeline declines to
    retrieve anything, so it always hands the LLM *something* to answer from, right
    or wrong.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from elbi_core.components import Embedder

EXAMPLE_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = EXAMPLE_ROOT / "fixtures" / "rag_docs"

_SECTION_SPLIT = re.compile(r"(?m)^## ")


@dataclass
class Chunk:
    source: str  # file name, e.g. "base_tariff.md"
    heading: str
    text: str


@dataclass
class VectorStore:
    chunks: list[Chunk] = field(default_factory=list)
    vectors: list[list[float]] = field(default_factory=list)

    def add(self, chunks: list[Chunk], embedder: Embedder) -> None:
        """Append `chunks` to the store. Never removes or re-scores what's already
        indexed -- see the module docstring."""
        if not chunks:
            return
        new_vectors = embedder.embed([f"{c.heading}\n{c.text}" for c in chunks])
        self.chunks.extend(chunks)
        self.vectors.extend(new_vectors)

    def top_k(self, query: str, embedder: Embedder, k: int = 3) -> list[Chunk]:
        if not self.chunks:
            return []
        query_vector = embedder.embed([query])[0]
        scored = sorted(
            range(len(self.chunks)),
            key=lambda i: -_cosine(query_vector, self.vectors[i]),
        )
        return [self.chunks[i] for i in scored[:k]]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def chunk_document(path: Path) -> list[Chunk]:
    """Split one Markdown file into one chunk per `##` section.

    A fixed-size character-window chunker is more common in production, but a
    heading-delimited chunker is just as real a choice (most RAG vendors offer
    "structure-aware" chunking as an option) and makes each chunk's boundary
    legible in this demo's UI -- the failure this demo shows up doesn't depend on
    which chunking strategy is used; both share the same missing piece (nothing
    marks a chunk as superseded).
    """
    text = path.read_text(encoding="utf-8")
    sections = _SECTION_SPLIT.split(text)
    chunks = []
    for section in sections[1:]:  # sections[0] is the doc's H1 title, not a section
        lines = section.strip().splitlines()
        heading, body = lines[0], "\n".join(lines[1:]).strip()
        # A section with a blank line between two fee paragraphs (the fuel-surcharge
        # section) becomes two chunks, one per paragraph -- each fee independently
        # retrievable, matching how the other eight sections already chunk.
        for paragraph in re.split(r"\n\s*\n", body):
            paragraph = paragraph.strip()
            if paragraph:
                chunks.append(Chunk(source=path.name, heading=heading, text=paragraph))
    return chunks


def build_store(docs_dir: Path = DOCS_DIR, *, embedder: Embedder) -> VectorStore:
    """Chunk and embed every document currently in `docs_dir`, in filename order.

    Called once at startup; `publish_update.py` calls `VectorStore.add` directly
    for an in-place update rather than rebuilding, so the store's history
    (including anything a rebuild-from-scratch approach would have thrown away) is
    exactly what a running index would actually accumulate.
    """
    store = VectorStore()
    for path in sorted(docs_dir.glob("*.md")):
        store.add(chunk_document(path), embedder)
    return store


@dataclass
class RagAnswer:
    question: str
    answer: str
    chunks: list[Chunk] = field(default_factory=list)


def ask(
    question: str,
    store: VectorStore,
    *,
    embedder: Embedder,
    use_llm: bool = True,
    k: int = 3,
) -> RagAnswer:
    """Answer `question` from the top-k chunks, with no gate and no recency check."""
    import sys

    sys.path.insert(0, str(EXAMPLE_ROOT))
    from llm import phrase_answer

    hits = store.top_k(question, embedder, k=k)
    context = "\n\n".join(f"[{c.source} / {c.heading}]\n{c.text}" for c in hits)
    answer = phrase_answer(question, context) if use_llm else None
    if answer is None:
        # Extractive fallback, same spirit as the Elbi side: whatever scored
        # highest, verbatim -- this is what makes beat three's failure mode
        # (confidently wrong, not garbled) visible without a live model: the
        # top chunk is handed back even though it doesn't cover the question.
        answer = hits[0].text if hits else "No documents indexed."

    return RagAnswer(question=question, answer=answer, chunks=hits)
