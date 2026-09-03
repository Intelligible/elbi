"""Dataset references.

A :class:`Dataset` is a *logical* reference to a named data source. It carries no
credentials and no location; those are resolved at run time from local data
bindings (``elbi.dev.yaml``). This keeps the committed project logical and
the credentialed binding local-only.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Dataset:
    """A logical reference to a named dataset.

    Args:
        name: The dataset name, matching ``^[a-z][a-z0-9_]*$``. This is the key
            used to resolve a concrete source from the local data bindings.
    """

    name: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("Dataset name must be non-empty")
