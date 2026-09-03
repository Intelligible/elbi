"""`elbi run` command wiring, driven with a stubbed HTTP client.

The commands are thin adapters over the app's HTTP API; these guard the request shape,
polling, and exit codes without a live app (the endpoints themselves are covered by the
app's own tests).
"""

from __future__ import annotations

from typing import Any

from typer.testing import CliRunner

from elbi_cli.app import app as cli_app
from elbi_cli.commands import run_cmd


class _Resp:
    def __init__(self, payload: Any) -> None:
        self._payload = payload

    def json(self) -> Any:
        return self._payload


def test_run_train_from_policy(monkeypatch: Any) -> None:
    posted: dict[str, Any] = {}

    class _Fake:
        def __enter__(self) -> _Fake:
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def get(self, path: str) -> _Resp:
            if path.endswith("/retrain"):
                return _Resp(
                    {
                        "configured": True,
                        "sourceKind": "derivation",
                        "dataset": "churn_features",
                        "target": "churned",
                        "features": ["ltv"],
                        "task": "classification",
                        "timeBudget": 5.0,
                    }
                )
            if "/api/jobs/" in path:
                return _Resp({"state": "succeeded", "result": "churn v1"})
            raise AssertionError(path)

        def post(self, path: str, json: dict[str, Any] | None = None) -> _Resp:
            assert path == "/api/registry/train"
            posted.update(json or {})
            return _Resp({"id": "job1"})

    monkeypatch.setattr(run_cmd, "_client", lambda url, token: _Fake())
    result = CliRunner().invoke(cli_app, ["run", "train", "churn"])
    assert result.exit_code == 0, result.output
    # The policy's source_kind became the payload's source field (derivation → name).
    assert posted["derivation"] == "churn_features"
    assert posted["target"] == "churned" and posted["name"] == "churn"
    assert "trained 'churn'" in result.output


def test_run_train_without_policy_exits_nonzero(monkeypatch: Any) -> None:
    class _Fake:
        def __enter__(self) -> _Fake:
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def get(self, path: str) -> _Resp:
            return _Resp({"configured": False})

    monkeypatch.setattr(run_cmd, "_client", lambda url, token: _Fake())
    result = CliRunner().invoke(cli_app, ["run", "train", "nope"])
    assert result.exit_code == 1


def test_a_dropped_run_stream_says_the_run_may_still_be_going(monkeypatch: Any) -> None:
    """The POST started the run; losing the stream does not stop it.

    Retrying here would start a second run, so the one useful thing is to say what
    happened and where to look -- not the traceback a bare transport error produced.
    """
    import httpx

    class _Fake:
        def __enter__(self) -> _Fake:
            return self

        def __exit__(self, *exc: Any) -> bool:
            return False

        def get(self, path: str) -> _Resp:
            return _Resp([{"id": "nb-1", "name": "readout"}])

        def stream(self, method: str, path: str, **kwargs: Any) -> Any:
            raise httpx.RemoteProtocolError("peer closed connection")

    monkeypatch.setattr(run_cmd, "_client", lambda url, token: _Fake())
    result = CliRunner().invoke(cli_app, ["run", "notebook", "readout"])
    assert result.exit_code == 1
    assert "lost the connection" in result.output
    assert "may still be going" in result.output
    assert "Traceback" not in result.output
