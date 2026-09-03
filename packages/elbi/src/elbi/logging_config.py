"""Logging config: plain text by default, structured JSON when ``LOG_FORMAT=json``.

Structured (one JSON object per line) logs are the norm for aggregation in Loki,
CloudWatch, or the ELK stack. This wires the root logger from the environment
(``LOG_LEVEL``, ``LOG_FORMAT``) with no dependency: a small JSON formatter over the
standard library. Call :func:`configure_logging` once at startup and run uvicorn with
``log_config=None`` so its access/error logs flow through the same handler and format.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone

# Attributes every LogRecord carries; anything else is a caller-supplied `extra`.
_RESERVED = frozenset(vars(logging.makeLogRecord({}))) | {
    "message",
    "asctime",
    "taskName",
}


class JsonFormatter(logging.Formatter):
    """Render each record as a single-line JSON object, including any extra fields."""

    def format(self, record: logging.LogRecord) -> str:
        """Render the record (and its extras) as a JSON line."""
        data: dict[str, object] = {
            "timestamp": datetime.fromtimestamp(
                record.created, tz=timezone.utc
            ).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            data["exception"] = self.formatException(record.exc_info)
        # Structured extras passed via logger.info(..., extra={...}).
        for key, value in record.__dict__.items():
            if key not in _RESERVED and not key.startswith("_"):
                data[key] = value
        return json.dumps(data, default=str)


class TextFormatter(logging.Formatter):
    """Render each record on one line, escaping newlines the message carries.

    Ids and names taken from a request reach the log, and a newline in one would
    otherwise start what reads as a second record.
    """

    def formatMessage(self, record: logging.LogRecord) -> str:
        """Format the record, with CR and LF in the message escaped."""
        record.message = record.message.replace("\r", "\\r").replace("\n", "\\n")
        return super().formatMessage(record)


def configure_logging() -> None:
    """Configure the root logger from ``LOG_LEVEL`` (INFO) and ``LOG_FORMAT``."""
    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    handler = logging.StreamHandler()
    if os.environ.get("LOG_FORMAT", "text").lower() == "json":
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            TextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
        )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
