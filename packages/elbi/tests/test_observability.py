"""Observability: split health probes, Prometheus metrics, JSON logging."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from elbi import create_app
from elbi.db import Store, open_store
from elbi.logging_config import JsonFormatter, configure_logging


class _NoClient:
    def step(
        self, transcript: object, tools: Sequence[object]
    ) -> object:  # pragma: no cover
        raise NotImplementedError


def _store(tmp_path: object) -> Store:
    return open_store(f"sqlite:{Path(str(tmp_path)) / 'obs.db'}")


def _app(store: Store) -> object:
    return create_app(load_datasets=lambda: {"d": []}, client=_NoClient(), store=store)


def test_health_live_and_ready(tmp_path: object) -> None:
    with TestClient(_app(_store(tmp_path))) as http:
        assert http.get("/health").json() == {"status": "ok"}  # alias
        assert http.get("/health/live").json() == {"status": "ok"}
        ready = http.get("/health/ready")
        assert ready.status_code == 200 and ready.json() == {"status": "ok"}


def test_readiness_503_when_db_unreachable(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _store(tmp_path)
    # Liveness must stay 200 (process is fine); only readiness flips to 503.
    monkeypatch.setattr(store, "ping", lambda: False)
    with TestClient(_app(store)) as http:
        assert http.get("/health/live").status_code == 200
        ready = http.get("/health/ready")
        assert ready.status_code == 503 and ready.json()["status"] == "unavailable"


def test_metrics_endpoint(tmp_path: object) -> None:
    with TestClient(_app(_store(tmp_path))) as http:
        http.get("/health")  # generate one request so counters exist
        # The scrape lives at /internal/metrics, not /metrics: /metrics is the SPA
        # Metrics-tab route and must not be shadowed by the exposition endpoint.
        body = http.get("/internal/metrics")
        assert body.status_code == 200
        assert "app_build_info" in body.text
        assert "http_request" in body.text  # instrumentator default metrics
        # /metrics is NOT the scrape endpoint (it belongs to the SPA); it must never
        # return the Prometheus exposition, or a hard refresh of the tab breaks.
        assert "app_build_info" not in http.get("/metrics").text


def test_operational_settings_report_their_source(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The environment pins an operational setting, and the API says so.

    The failure this guards against is silent: without ``source``/``editable`` an admin
    edits a field, the environment keeps overriding it, and nothing explains why.
    """
    monkeypatch.setenv("AUDIT_RETENTION_DAYS", "7")
    store = _store(tmp_path)
    with TestClient(_app(store)) as http:
        by_key = {
            s["key"]: s for s in http.get("/api/settings/runtime").json()["settings"]
        }
        assert by_key["audit_retention_days"]["value"] == "7"
        assert by_key["audit_retention_days"]["source"] == "env"
        assert by_key["audit_retention_days"]["editable"] is False

        # A stored value is kept but does not take effect while the env pins it.
        stored = http.put(
            "/api/settings/runtime/audit_retention_days", json={"value": "365"}
        ).json()
        assert stored["value"] == "7" and stored["source"] == "env"

        # Drop the variable and the stored value applies.
        monkeypatch.delenv("AUDIT_RETENTION_DAYS")
        after = http.get("/api/settings/runtime").json()["settings"]
        audit = next(s for s in after if s["key"] == "audit_retention_days")
        assert audit["value"] == "365" and audit["source"] == "setting"

        assert (
            http.put("/api/settings/runtime/nope", json={"value": "1"}).status_code
            == 404
        )


def test_build_info_reports_a_real_version(tmp_path: object) -> None:
    """The version label must be the installed version, not "unknown".

    It was "unknown" everywhere for a while: the lookup used the console-script name
    ("elbi-app") instead of the distribution name ("elbi"), so it always
    raised PackageNotFoundError. That silently defeats both the metric's purpose and the
    chart's version-skew alert, and nothing else notices.
    """
    with TestClient(_app(_store(tmp_path))) as http:
        body = http.get("/internal/metrics").text
    line = next(li for li in body.splitlines() if li.startswith("app_build_info{"))
    assert 'version="unknown"' not in line, line


def test_json_formatter_renders_fields() -> None:
    record = logging.LogRecord(
        "svc", logging.INFO, "path", 1, "hello %s", ("world",), None
    )
    record.widget = 7  # type: ignore[attr-defined]  # a structured extra
    out = json.loads(JsonFormatter().format(record))
    assert out["level"] == "INFO"
    assert out["logger"] == "svc"
    assert out["message"] == "hello world"
    assert out["widget"] == 7
    assert "timestamp" in out


def test_configure_logging_selects_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_FORMAT", "json")
    configure_logging()
    handler = logging.getLogger().handlers[0]
    assert isinstance(handler.formatter, JsonFormatter)
    # restore a plain formatter so later tests' logs are readable
    monkeypatch.setenv("LOG_FORMAT", "text")
    configure_logging()
