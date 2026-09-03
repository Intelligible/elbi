"""Optional extensions, discovered at startup.

An extension mounts additional surfaces onto a running app: routes and middleware.
Nothing here ships enabled, and the app is complete without any extension installed. A
package advertises itself by declaring an entry point in the ``elbi.extensions``
group::

    [project.entry-points."elbi.extensions"]
    my_extension = "my_package.extension"

The named module (or object) supplies ``install_extension(context)``, called once during
startup with everything already built: the app, the store, and the services. An
extension that its deployment has not configured should install nothing and return, so
that merely having the package present changes no behaviour.

It may also supply classes rather than surfaces: ``store_class`` for the store, and
``service_classes`` for the services. Both resolve before the thing they describe is
built, which is how an extension reaches what this package constructs rather than only
what it can mount afterwards.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from importlib.metadata import entry_points
from typing import TYPE_CHECKING, Protocol, TypeVar, cast

from elbi_core.errors import ConfigError

if TYPE_CHECKING:
    from fastapi import FastAPI

    from .db import Store
    from .metrics import MetricService
    from .monitoring import MonitorService
    from .notebooks import NotebookService
    from .warehouse.service import WarehouseService

logger = logging.getLogger("elbi")

#: The entry-point group an extension declares itself in.
GROUP = "elbi.extensions"

_S = TypeVar("_S")


@dataclass(frozen=True)
class ExtensionContext:
    """What an extension is handed at startup.

    Passing one object rather than a widening argument list means a later addition
    does not break an extension compiled against an earlier version.
    """

    #: The application, for registering routes and middleware. Extensions are installed
    #: before the single-page app's catch-all mount, so a route added here is reachable.
    app: FastAPI
    #: The persistence layer. ``None`` when the app runs storeless (a bare MCP server).
    store: Store | None
    #: The warehouse service, or ``None`` when the ``warehouse`` extra is not installed.
    warehouse: WarehouseService | None = None
    #: The notebook service, or ``None`` when notebooks are not configured.
    notebooks: NotebookService | None = None
    #: The metric service, or ``None`` when metrics are not configured.
    metrics: MetricService | None = None
    #: The monitor service, or ``None`` when monitoring is not configured.
    monitors: MonitorService | None = None


class Extension(Protocol):
    """A module or object that mounts a surface onto the app."""

    #: The class the store is built from, when this extension supplies one. Optional;
    #: an extension that adds no tables of its own does not declare it.
    store_class: type[Store]

    #: Subclasses to build the services from, in place of the shipped ones. Optional,
    #: and matched by what each one subclasses rather than by name.
    service_classes: Sequence[type]

    def install_extension(self, context: ExtensionContext) -> None:
        """Mount the surface, or return without doing anything.

        Called once at startup. An extension whose deployment has not configured it
        should return quietly rather than raise, so an unconfigured install is inert.
        """
        ...


def load_all() -> list[tuple[str, Extension]]:
    """Every installed extension, as ``(name, extension)``, sorted by name.

    Sorted so installation order depends on the names alone and not on the order the
    metadata happens to be read in, which would make a conflict between two extensions
    reproduce only intermittently.
    """
    found: list[tuple[str, Extension]] = []
    for entry in sorted(entry_points(group=GROUP), key=lambda e: e.name):
        found.append((entry.name, cast("Extension", entry.load())))
    return found


def store_class() -> type[Store] | None:
    """The class an installed extension supplies for the store, or ``None``.

    An extension that adds tables and queries of its own serves them from the one engine
    by supplying the class the store is built from, rather than opening a second engine
    against the same database.

    Two extensions supplying one is a conflict nothing here can settle, so it is refused
    rather than decided by the order the names sort in.
    """
    supplied = [
        (name, found)
        for name, extension in load_all()
        if (found := getattr(extension, "store_class", None)) is not None
    ]
    if not supplied:
        return None
    if len(supplied) > 1:
        raise ConfigError(
            "extensions "
            + ", ".join(sorted(name for name, _ in supplied))
            + " each supply a store class; only one can build the store"
        )
    return cast("type[Store]", supplied[0][1])


def service_class(base: type[_S]) -> type[_S]:
    """The class a service is built from: an extension's subclass, or ``base`` itself.

    An extension narrowing what a service does supplies a subclass of it, which is how
    it reaches a service this package constructs rather than one it could wrap after the
    fact. Matched on what the subclass inherits from, so nothing has to keep a name in
    step with the field it configures.

    Two extensions replacing the same service is a conflict nothing here can settle, so
    it is refused rather than decided by the order the names sort in.
    """
    supplied = [
        (name, candidate)
        for name, extension in load_all()
        for candidate in getattr(extension, "service_classes", ())
        if isinstance(candidate, type)
        and issubclass(candidate, base)
        and candidate is not base
    ]
    if not supplied:
        return base
    if len(supplied) > 1:
        raise ConfigError(
            "extensions "
            + ", ".join(sorted(name for name, _ in supplied))
            + f" each replace {base.__name__}; only one can build it"
        )
    return supplied[0][1]


def install_all(context: ExtensionContext) -> list[str]:
    """Install every extension against ``context``; return the names installed.

    One failing extension must not stop the app from serving, nor silently disappear:
    the failure is logged with its traceback and the remaining extensions still install.
    """
    installed: list[str] = []
    for name, extension in load_all():
        try:
            extension.install_extension(context)
        except Exception:
            logger.exception("extension %r failed to install", name)
            continue
        installed.append(name)
    return installed


def available() -> list[str]:
    """The names of every installed extension, for startup logging."""
    return sorted(entry.name for entry in entry_points(group=GROUP))
