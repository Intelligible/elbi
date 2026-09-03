"""The one door to dlt, with its outbound telemetry switched off before it loads.

The library is used narrowly here: `rest_api_source` handles pagination, auth and
incremental cursors for HTTP connectors. Records come back to us, and we batch them to
Arrow and write through delta-rs, so this is an extraction engine rather than a pipeline
framework.

It ships anonymous usage tracking **enabled by default**, posting environment metadata
to a dltHub endpoint on first use. That is upstream's default rather than a choice this
product makes, and it is wrong here twice over: a deployment inside a hospital or a bank
should open no outbound connection nobody asked for, and the deployment documentation
promises that the only outbound calls go to endpoints the operator configured. An
air-gapped install would see the attempt fail as well, and have to explain it.

Both call sites go through this module so that adding a third cannot miss the setting,
which a test enforces. Order is what makes it work: dlt resolves its runtime
configuration, and initialises the tracker, when the library is first imported, so the
variable has to be set before that import rather than before the first extraction.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - imported for the annotation only
    from dlt.sources.rest_api import RESTAPIConfig


def rest_api_source(config: RESTAPIConfig | dict[str, Any]) -> Any:
    """Build a declarative REST source with telemetry disabled.

    The variable is set unconditionally rather than with `setdefault`, so an inherited
    environment cannot switch it back on. `RUNTIME__DLTHUB_TELEMETRY` is dlt's own name
    for the setting: its configuration sections map to environment variables joined by a
    double underscore.
    """
    os.environ["RUNTIME__DLTHUB_TELEMETRY"] = "false"

    from dlt.sources.rest_api import rest_api_source as _rest_api_source

    return _rest_api_source(config)
