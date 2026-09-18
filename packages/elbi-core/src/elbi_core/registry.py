"""The derivation registry.

A :class:`Registry` collects derivations by name. The module-level
:data:`default_registry` is what ``@derivation`` writes to unless a specific
registry is passed; the CLI's discovery step imports a project's modules so their
decorators populate it.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextvars import ContextVar
from typing import TYPE_CHECKING

from .errors import DuplicateDerivationError, UnknownDerivationError
from .retrieval import Retriever, default_retriever

if TYPE_CHECKING:
    from .derivation import Derivation


class Registry:
    """A name-keyed collection of derivations.

    Search is delegated to a pluggable :class:`~elbi.retrieval.Retriever`
    (BM25 by default); pass one to use semantic or hybrid retrieval instead.
    """

    def __init__(self, *, retriever: Retriever | None = None) -> None:
        self._items: dict[str, Derivation] = {}
        self._retriever: Retriever = retriever or default_retriever()

    def register(self, derivation: Derivation) -> None:
        """Add a derivation.

        Raises:
            DuplicateDerivationError: if a different derivation is already
                registered under the same name.
        """
        existing = self._items.get(derivation.name)
        if existing is not None and existing is not derivation:
            raise DuplicateDerivationError(
                f"a derivation named {derivation.name!r} is already registered"
            )
        self._items[derivation.name] = derivation

    def replace(self, derivation: Derivation) -> None:
        """Insert ``derivation``, overwriting any existing entry of its name.

        Unlike :meth:`register`, this never raises on a name collision; it is the
        seam used to advance a derivation's lifecycle (e.g. certification) by
        swapping in a new immutable value under the same name.
        """
        self._items[derivation.name] = derivation

    def get(self, name: str) -> Derivation:
        """Return a derivation by name.

        Raises:
            UnknownDerivationError: if no derivation is registered under ``name``.
        """
        try:
            return self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "(none)"
            raise UnknownDerivationError(
                f"no derivation named {name!r}; registered: {known}"
            ) from None

    def remove(self, name: str) -> None:
        """Remove the derivation registered under ``name``.

        Raises:
            UnknownDerivationError: if no derivation is registered under ``name``.
        """
        try:
            del self._items[name]
        except KeyError:
            known = ", ".join(sorted(self._items)) or "(none)"
            raise UnknownDerivationError(
                f"no derivation named {name!r}; registered: {known}"
            ) from None

    @property
    def retriever(self) -> Retriever:
        """The retriever this registry ranks with."""
        return self._retriever

    def use_retriever(self, retriever: Retriever) -> None:
        """Swap in a different retriever (e.g. a hybrid one for the dev server)."""
        self._retriever = retriever

    def search(self, query: str, *, limit: int = 5) -> list[Derivation]:
        """Rank derivations by relevance to ``query`` over all registered entries.

        Delegates to this registry's :class:`~elbi.retrieval.Retriever`.
        An empty query returns nothing. This is the entry point an agent uses to
        find a few relevant derivations rather than scanning every tool. Callers
        that must rank only a subset (e.g. the served, certified ones) can call the
        :attr:`retriever` directly with that subset.
        """
        return self._retriever.search(query, self._items.values(), limit=limit)

    def names(self) -> list[str]:
        """All registered derivation names, sorted."""
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[Derivation]:
        return iter(self._items.values())

    def __len__(self) -> int:
        return len(self._items)

    def clear(self) -> None:
        """Remove all derivations. Primarily for tests."""
        self._items.clear()


#: The registry populated by ``@derivation`` when no registry is specified.
default_registry = Registry()

class NotebookRegistry(Registry):
    """A registry where re-declaring a derivation replaces it rather than colliding.

    Discovery wants a duplicate name to fail loudly: two files claiming one name is a
    project error, and the last import winning silently would be worse than stopping. A
    notebook inverts that. Re-running a cell is its normal motion — the reactive engine
    does it unprompted whenever an upstream cell changes — so a second run must redefine
    what the first one declared. Without this a derivation cell runs exactly once, which
    makes "open this derivation and iterate on it" impossible by construction.
    """

    def register(self, derivation: Derivation) -> None:
        """Insert ``derivation``, overwriting any existing entry of its name."""
        self.replace(derivation)


_active_registry: ContextVar[Registry] = ContextVar(
    "elbi_active_registry", default=default_registry
)


def active_registry() -> Registry:
    """The registry a bare ``@derivation`` writes to in the current context."""
    return _active_registry.get()


@contextlib.contextmanager
def use_registry(registry: Registry) -> Iterator[Registry]:
    """Temporarily route bare ``@derivation`` registrations into ``registry``.

    Used by discovery to populate a specific registry while importing a
    project's modules, without those modules naming the registry themselves.
    """
    token = _active_registry.set(registry)
    try:
        yield registry
    finally:
        _active_registry.reset(token)
