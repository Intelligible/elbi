"""Shared test fixtures for the app package."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator, Sequence

import pytest
from prometheus_client import REGISTRY

from elbi.search.document_map import EMBED_DIM


@pytest.fixture(autouse=True)
def _reset_prometheus_registry() -> Iterator[None]:
    """Clear the global Prometheus registry between tests.

    ``create_app`` registers metrics on ``prometheus_client``'s default registry; the
    suite builds the app many times, so without this reset the second build raises
    "Duplicated timeseries in CollectorRegistry". Clearing before each test keeps them
    independent.
    """
    for collector in list(REGISTRY._collector_to_names):
        REGISTRY.unregister(collector)
    yield


@pytest.fixture(autouse=True)
def _no_spa_build(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep booting a server out of npm.

    ``web/dist`` is not checked in, so the first test to call ``serve.build`` runs
    ``npm ci`` and ``npm run build``. Under ``--dist=loadfile`` the files that boot a
    server land on different workers, which then race on the same build directory and
    one of them hits the timeout.
    """
    from elbi import serve

    monkeypatch.setattr(serve, "_ensure_spa_built", lambda *_: None)


class _OfflineEmbedder:
    """A deterministic stand-in for ``OnnxEmbedder`` at the real width.

    Vectors are derived from a hash of the text, so the same text embeds the same way
    on every run and every machine. They carry no semantics: what this preserves is the
    shape of the production path (vectors stored, index built, hybrid fusion ranked),
    not the ranking a real model would give. A test that asserts on ranking supplies its
    own embedder.
    """

    def embed(self, texts: Sequence[str]) -> list[Sequence[float]]:
        """Embed ``texts`` into normalized vectors, as the real embedder does."""
        return [self._vector(text) for text in texts]

    def embed_spans(
        self, text: str, spans: Sequence[tuple[int, int]]
    ) -> list[Sequence[float]]:
        """One vector per span.

        Present because ``OnnxEmbedder`` satisfies the span protocol: an embedder
        without it sends the builder down its per-chunk fallback, so a stub missing
        this method would quietly test a path production never takes.
        """
        return [self._vector(text[start:end]) for start, end in spans]

    @staticmethod
    def _vector(text: str) -> Sequence[float]:
        digest = hashlib.sha256(text.encode()).digest()
        vector = [0.0] * EMBED_DIM
        for position, byte in enumerate(digest):
            vector[position % EMBED_DIM] += byte / 255.0
        norm = math.sqrt(sum(value * value for value in vector))
        return [value / norm for value in vector] if norm else vector


@pytest.fixture(autouse=True)
def _no_hub_embedder(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep booting a server off the Hugging Face Hub.

    ``serve.build`` gives every app a real ``OnnxEmbedder`` unless ``search`` is
    ``lexical``, and it defaults to ``hybrid``. Resolving that embedder downloads
    weights from the Hub, and the pooled TLS socket ``huggingface_hub``'s session
    opens is never closed. Collected mid-run it raised ResourceWarning, an error
    under ``filterwarnings``, failing whichever test was in teardown at the time.
    """
    from elbi import serve

    monkeypatch.setattr(serve, "OnnxEmbedder", _OfflineEmbedder)
