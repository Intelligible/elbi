"""Shared component-id formatting for this example.

Kept in one place so `derivations/freight_components.py` and `publish_update.py`
(which needs to name a component it's about to supersede) can't drift apart on the
id scheme.
"""

from __future__ import annotations

NAMESPACE = "acme-freight"


def version_id(key: str, effective_date: str) -> str:
    """A version-qualified component id, e.g. `acme-freight/detention_fee.20240101`.

    Component names may only hold ``[a-z0-9_.]`` (no ``@`` or ``-``), so the
    version suffix is the date with its hyphens dropped.
    """
    return f"{NAMESPACE}/{key}.{effective_date.replace('-', '')}"
