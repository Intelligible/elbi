"""Tests for the stateful exploration session (persistent REPL).

The core behaviors run a real worker subprocess but need no dependencies, so they are
fast. The dependency-provisioning test uses uv and is marked ``slow``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from elbi_core import ReplSession
from elbi_core.errors import DerivationError


def test_session_persists_variables_across_calls() -> None:
    # The point of a session: a variable defined in one call is there in the next.
    s = ReplSession(timeout=30)
    try:
        assert s.run_code("x = 41").error is None
        assert s.run_code("result = x + 1").result == "42"
    finally:
        s.close()


def test_session_seeds_datasets_as_data() -> None:
    s = ReplSession(data={"d": [{"v": "1"}, {"v": "2"}, {"v": "3"}]}, timeout=30)
    try:
        out = s.run_code("result = sum(int(r['v']) for r in data['d'])")
        assert out.result == "6"
    finally:
        s.close()


def test_session_reports_error_and_stays_alive() -> None:
    # A code error comes back as a traceback and the session survives, so the agent can
    # fix it and keep its accumulated state.
    s = ReplSession(timeout=30)
    try:
        s.run_code("keep = 'here'")
        bad = s.run_code("1 / 0")
        assert bad.error is not None and "ZeroDivisionError" in bad.error
        assert s.run_code("result = keep").result == "'here'"
    finally:
        s.close()


def test_session_captures_stdout() -> None:
    s = ReplSession(timeout=30)
    try:
        assert "hello" in s.run_code("print('hello')").stdout
    finally:
        s.close()


def test_session_persists_files_in_the_workspace(tmp_path: Path) -> None:
    # The session's working directory is the workspace, so files carry across calls too.
    s = ReplSession(workspace=tmp_path, timeout=30)
    try:
        s.run_code("open('note.txt', 'w').write('kept')")
        assert s.run_code("result = open('note.txt').read()").result == "'kept'"
        assert (tmp_path / "note.txt").read_text() == "kept"
    finally:
        s.close()


def test_closed_session_reports_rather_than_raising() -> None:
    s = ReplSession(timeout=30)
    s.close()
    out = s.run_code("result = 1")
    assert out.error is not None and "closed" in out.error


def test_session_timeout_closes_the_session() -> None:
    # A call cannot be interrupted midway, so a timeout kills the child and closes the
    # session, reported rather than hanging.
    s = ReplSession(timeout=1)
    out = s.run_code("while True:\n    pass")
    assert out.error is not None and "timeout" in out.error
    # the session is now closed; a further call reports it
    assert "closed" in (s.run_code("result = 1").error or "")
    assert s.alive is False


def test_manager_recovers_after_a_timeout() -> None:
    # A timeout kills the session, but the manager must start a fresh one on the next
    # call so the conversation continues, rather than failing forever with "closed".
    from elbi_core import SessionManager

    m = SessionManager(datasets={})
    # Short timeout for the first session (so the busy loop times out fast), generous
    # for the recovered one; the manager builds each via _make.
    timeouts = [1.0, 30.0]
    m._make = lambda deps: ReplSession(deps=deps, timeout=timeouts.pop(0))  # type: ignore[method-assign]
    try:
        timed_out = m.run_code("while True:\n    pass")  # kills session 1
        assert "timeout" in (timed_out.error or "")
        recovered = m.run_code("result = 7")  # manager starts a fresh session 2
        assert recovered.result == "7"
        assert recovered.error is None
    finally:
        m.close()


def test_host_session_declines_bash() -> None:
    # A host session cannot run bash (that would touch the user's machine); bash is a
    # docker-backend capability. The refusal is reported, not raised.
    s = ReplSession(timeout=30)
    try:
        out = s.run_bash("echo hi")
        assert out.error is not None and "docker" in out.error
    finally:
        s.close()


def test_host_session_cannot_add_deps_in_place() -> None:
    # The host session's uv environment is fixed at start, so it reports it cannot add
    # deps; the manager then restarts it. (The docker session can add in place.)
    s = ReplSession(timeout=30)
    try:
        assert s.add_deps(["numpy"]) is False
    finally:
        s.close()


def test_session_rejects_invalid_dependency() -> None:
    with pytest.raises(DerivationError, match="invalid dependency"):
        ReplSession(deps=["--evil"])


def test_session_rejects_nonpositive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        ReplSession(timeout=0)


@pytest.mark.slow
def test_session_provisions_declared_deps() -> None:
    # Declared packages are provisioned into the session's environment and stay usable
    # across calls (the env is fixed for the session's life, like a kernel).
    s = ReplSession(deps=["numpy"], timeout=120)
    try:
        assert (
            s.run_code("import numpy as np; m = np.array([1.0, 2.0, 3.0]).mean()").error
            is None
        )
        assert s.run_code("result = float(m)").result == "2.0"
    finally:
        s.close()
