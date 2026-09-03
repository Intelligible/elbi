"""OpenTelemetry distributed tracing, exported over OTLP when configured.

Inert unless ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set. When it is (and the ``otel``
extra is installed), each HTTP request is traced and spans are exported to the
collector named by the standard ``OTEL_*`` environment variables
(``OTEL_SERVICE_NAME``, ``OTEL_EXPORTER_OTLP_ENDPOINT``, ...), which the SDK reads
itself. Kept out of the default install as an opt-in extra so the common deployment
carries none of the tracing dependencies.
"""

from __future__ import annotations

import logging
import os
from typing import Any

logger = logging.getLogger(__name__)


def setup_tracing(app: Any) -> None:
    """Instrument ``app`` for OTLP tracing when ``OTEL_EXPORTER_OTLP_ENDPOINT`` is set.

    A no-op when the endpoint is unset, or (with a warning) when the ``otel`` extra is
    missing, so enabling tracing is configuration plus the extra.
    """
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
            OTLPSpanExporter,
        )
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT is set but the 'otel' extra is not installed; "
            "install elbi-app[otel] to enable tracing"
        )
        return

    # Endpoint, service name, and resource attributes are read from the OTEL_* env by
    # the SDK: constructing these with no arguments is deliberate.
    provider = TracerProvider()
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app)
    logger.info("OpenTelemetry tracing enabled (OTLP export)")
