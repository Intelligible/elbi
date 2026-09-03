"""The persistent exploration REPL worker, at its behavior boundary: a namespace that
survives across requests, a popped result, captured stdout and errors, and the stdin
request loop. It runs as a subprocess in production; these tests call it in-process."""

from __future__ import annotations

import io
import json
import socket
import sys

import pytest

from elbi_core import _repl_worker


def test_run_once_persists_the_namespace_and_pops_result() -> None:
    namespace: dict[str, object] = {}
    first = _repl_worker._run_once(namespace, "x = 21")
    assert first["error"] is None and first["result"] is None
    # x survives into the next request; result is returned then popped, not carried on
    second = _repl_worker._run_once(namespace, "result = x * 2")
    assert second["result"] == "42"
    assert "result" not in namespace
    third = _repl_worker._run_once(namespace, "print('hi')")
    assert third["stdout"] == "hi\n"


def test_run_once_captures_an_error_and_survives() -> None:
    namespace: dict[str, object] = {}
    result = _repl_worker._run_once(namespace, "1 / 0")
    assert result["ok"] is True  # the session survives so the agent can iterate
    assert result["error"] is not None and "ZeroDivisionError" in result["error"]


def test_run_once_caps_stdout() -> None:
    result = _repl_worker._run_once({}, "print('a' * 20000)")
    assert len(result["stdout"]) <= _repl_worker._MAX_STDOUT


def test_main_runs_requests_until_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    stdin = io.StringIO(
        json.dumps({"code": "result = 1 + 1"})
        + "\n"
        + json.dumps({"shutdown": True})
        + "\n"
    )
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", out)
    # --allow-network skips the socket guard, so the test does not mutate global sockets
    assert _repl_worker.main(["--allow-network"]) == 0
    responses = [json.loads(line) for line in out.getvalue().splitlines() if line]
    assert responses[0]["result"] == "2"


def test_main_seeds_datasets_from_a_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: object
) -> None:
    from pathlib import Path

    data_file = Path(str(tmp_path)) / "data.json"
    data_file.write_text(json.dumps({"sales": [{"amount": 5}]}), encoding="utf-8")
    code = json.dumps({"code": "result = data['sales'][0]['amount']"})
    stdin = io.StringIO(code + "\n")
    out = io.StringIO()
    monkeypatch.setattr(sys, "stdin", stdin)
    monkeypatch.setattr(sys, "stdout", out)
    _repl_worker.main([str(data_file), "--allow-network"])
    responses = [json.loads(line) for line in out.getvalue().splitlines() if line]
    assert responses[0]["result"] == "5"


def test_deny_network_blocks_socket_creation() -> None:
    original = socket.socket
    try:
        _repl_worker._deny_network()
        with pytest.raises(OSError, match="disabled"):
            socket.socket()
    finally:
        socket.__dict__["socket"] = original  # restore for the rest of the suite
