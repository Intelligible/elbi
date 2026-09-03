"""Self-registering connector registry.

Connectors decorate their class with ``@SourceRegistry.register`` and are imported
lazily on first registry use (via ``_load_all``), so a process that never touches the
warehouse doesn't import every vendor SDK at startup, which is what lets the catalog
grow to hundreds of connectors without paying for them all on import.
"""

from __future__ import annotations

import importlib
import threading
from typing import ClassVar

from .base import Source

_lock = threading.Lock()


class SourceRegistry:
    """The process-wide registry of available connectors, keyed by source type."""

    _sources: ClassVar[dict[str, Source]] = {}
    _loaded = False

    @classmethod
    def _ensure_loaded(cls) -> None:
        if cls._loaded:
            return
        with _lock:
            if cls._loaded:
                return
            # Importing the aggregator runs each connector's @register, populating the
            # registry. import_module (not `from . import`) avoids a bound-but-unused
            # name for a pure side-effect import.
            importlib.import_module(f"{__package__}._load_all")
            cls._loaded = True

    @classmethod
    def register(cls, source_class: type[Source]) -> type[Source]:
        """Register a connector class (used as a decorator); returns it unchanged."""
        instance = source_class()
        cls._sources[instance.source_type] = instance
        return source_class

    @classmethod
    def get(cls, source_type: str) -> Source:
        """The connector for ``source_type``, raising ``ValueError`` if unknown."""
        cls._ensure_loaded()
        if source_type not in cls._sources:
            raise ValueError(f"Unknown source type: {source_type}")
        return cls._sources[source_type]

    @classmethod
    def all(cls) -> dict[str, Source]:
        """Every registered connector, keyed by source type."""
        cls._ensure_loaded()
        return dict(cls._sources)

    @classmethod
    def is_registered(cls, source_type: str) -> bool:
        """Whether a connector is registered for ``source_type``."""
        cls._ensure_loaded()
        return source_type in cls._sources
