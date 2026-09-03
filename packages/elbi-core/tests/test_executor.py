"""Tests for the Executor seam: in-process and subprocess sandbox."""

from __future__ import annotations

import pickle
from pathlib import Path

import pytest

from elbi_core import (
    Artifact,
    Context,
    InProcessExecutor,
    Registry,
    RoutingExecutor,
    Runner,
    SubprocessExecutor,
    derivation,
    propose,
    serve,
)
from elbi_core._sandbox_child import main as child_main
from elbi_core.errors import DerivationError

_OK_SOURCE = """
def calc(ctx):
    "Sum a constant."
    return [{"n": 1}, {"n": 2}]
"""

_BOOM_SOURCE = """
def boom(ctx):
    raise ValueError("nope")
"""


def test_in_process_runs_human_compute() -> None:
    reg = Registry()

    @derivation(name="greet", serve=serve.text(), registry=reg)
    def greet(ctx: Context) -> Artifact:
        return Artifact.text("hi")

    assert InProcessExecutor().run(greet, Context({})).value == "hi"


def test_in_process_refuses_agent_derivation() -> None:
    reg = Registry()
    d = propose("calc", _OK_SOURCE, serve=serve.table(), registry=reg)
    with pytest.raises(DerivationError, match="cannot run in-process"):
        InProcessExecutor().run(d, Context({}))


def test_subprocess_runs_agent_source() -> None:
    reg = Registry()
    d = propose("calc", _OK_SOURCE, serve=serve.table(), registry=reg)
    artifact = SubprocessExecutor(timeout=30).run(d, Context({}))
    assert artifact.kind == "table"
    assert artifact.value == [{"n": 1}, {"n": 2}]


def test_subprocess_via_runner_caches() -> None:
    reg = Registry()
    propose("calc", _OK_SOURCE, serve=serve.table(), registry=reg)
    runner = Runner(reg, executor=SubprocessExecutor(timeout=30))
    first = runner.serve("calc")
    second = runner.serve("calc")  # session cache hit, no second subprocess
    assert first == second
    assert "| n |" in first


def test_subprocess_requires_source() -> None:
    reg = Registry()

    @derivation(name="human", serve=serve.text(), registry=reg)
    def human(ctx: Context) -> Artifact:
        return Artifact.text("x")

    with pytest.raises(DerivationError, match="has no source"):
        SubprocessExecutor().run(human, Context({}))


def test_subprocess_reports_child_failure() -> None:
    reg = Registry()
    d = propose("boom", _BOOM_SOURCE, serve=serve.text(), registry=reg)
    with pytest.raises(DerivationError, match="failed in sandbox: ValueError: nope"):
        SubprocessExecutor(timeout=30).run(d, Context({}))


def test_subprocess_timeout() -> None:
    reg = Registry()
    d = propose(
        "spin",
        "def spin(ctx):\n    while True:\n        pass\n",
        serve=serve.text(),
        registry=reg,
    )
    with pytest.raises(DerivationError, match="sandbox timeout"):
        SubprocessExecutor(timeout=1.0).run(d, Context({}))


def test_subprocess_rejects_nonpositive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        SubprocessExecutor(timeout=0)


def test_missing_result_is_reported() -> None:
    # A compute that hard-exits the child before it can write result.json: the
    # parent must report "produced no result" rather than hang or mis-parse.
    reg = Registry()
    source = "def boom(ctx):\n    import os\n    os._exit(0)\n"
    d = propose("boom", source, serve=serve.text(), registry=reg)
    with pytest.raises(DerivationError, match="produced no result"):
        SubprocessExecutor(timeout=30).run(d, Context({}))


def test_run_code_returns_stdout_and_result() -> None:
    # Exploratory code runs in the same sandbox, with datasets in `data`, and its
    # stdout + `result` come back for the agent to read.
    ex = SubprocessExecutor(timeout=30)
    r = ex.run_code(
        "vals = [int(x['v']) for x in data['xs']]\n"
        "print('count', len(vals))\n"
        "result = sum(vals)",
        data={"xs": [{"v": "2"}, {"v": "3"}]},
    )
    assert r.error is None
    assert "count 2" in r.stdout
    assert r.result == "5"


def test_run_code_captures_error_without_raising() -> None:
    # A code error is returned as a traceback, not raised, so the agent can iterate.
    r = SubprocessExecutor(timeout=30).run_code("result = 1 / 0")
    assert r.result is None
    assert r.error is not None and "ZeroDivisionError" in r.error


def test_child_main_success(tmp_path: Path) -> None:
    job = {"source": _OK_SOURCE, "fn_name": "calc", "inputs": {}, "params": {}}
    (tmp_path / "job.pkl").write_bytes(pickle.dumps(job))
    assert child_main(str(tmp_path)) == 0
    import json

    result = json.loads((tmp_path / "result.json").read_text())
    assert result == {"ok": True, "kind": "table", "value": [{"n": 1}, {"n": 2}]}


def test_routing_executor_dispatches_by_source() -> None:
    reg = Registry()

    @derivation(name="trusted", serve=serve.text(), registry=reg)
    def trusted(ctx: Context) -> Artifact:
        return Artifact.text("in-process")

    agent = propose("calc", _OK_SOURCE, serve=serve.table(), registry=reg)
    router = RoutingExecutor(sandbox=SubprocessExecutor(timeout=30))

    assert router.run(trusted, Context({})).value == "in-process"  # no source
    assert router.run(agent, Context({})).value == [{"n": 1}, {"n": 2}]  # sandbox


def test_child_main_captures_error(tmp_path: Path) -> None:
    job = {"source": _BOOM_SOURCE, "fn_name": "boom", "inputs": {}, "params": {}}
    (tmp_path / "job.pkl").write_bytes(pickle.dumps(job))
    assert child_main(str(tmp_path)) == 1

    import json

    result = json.loads((tmp_path / "result.json").read_text())
    assert result["ok"] is False
    assert "ValueError: nope" in result["error"]
    assert "traceback" in result


_NET_SOURCE = """
def reach_out(ctx):
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.close()
    return [{"ok": 1}]
"""


def test_child_main_denies_network_by_default(tmp_path: Path) -> None:
    import socket as _socket

    original = _socket.socket  # the guard must restore this for the test process
    job = {"source": _NET_SOURCE, "fn_name": "reach_out", "inputs": {}, "params": {}}
    (tmp_path / "job.pkl").write_bytes(pickle.dumps(job))
    assert child_main(str(tmp_path)) == 1

    import json

    result = json.loads((tmp_path / "result.json").read_text())
    assert result["ok"] is False
    assert "network access is disabled" in result["error"]
    # The in-process guard restored the real socket after the run.
    assert _socket.socket is original


def test_child_main_allows_network_when_opted_in(tmp_path: Path) -> None:
    # A socket is created but never connected, so this stays offline and fast;
    # it only proves the guard is *not* installed when allow_network is set.
    job = {
        "source": _NET_SOURCE,
        "fn_name": "reach_out",
        "inputs": {},
        "params": {},
        "allow_network": True,
    }
    (tmp_path / "job.pkl").write_bytes(pickle.dumps(job))
    assert child_main(str(tmp_path)) == 0


def test_subprocess_denies_network_end_to_end() -> None:
    reg = Registry()
    d = propose("reach_out", _NET_SOURCE, serve=serve.table(), registry=reg)
    with pytest.raises(DerivationError, match="network access is disabled"):
        SubprocessExecutor(timeout=30).run(d, Context({}))


def test_no_deps_runs_in_the_interpreter() -> None:
    # Without declared deps the child runs under the configured interpreter (no uv).
    argv = SubprocessExecutor()._argv((), "/io")
    assert argv[-3:] == ["-m", "elbi_core._sandbox_child", "/io"]
    assert "uv" not in argv


def test_declared_deps_provision_via_uv() -> None:
    # Declared packages are provisioned on demand with uv into an ephemeral env.
    argv = SubprocessExecutor()._argv(("pandas", "scipy>=1.14"), "/io")
    assert argv[:3] == ["uv", "run", "--no-project"]
    assert "pandas" in argv and "scipy>=1.14" in argv
    assert argv.count("--with") == 2 and argv[-1] == "/io"


def test_invalid_dependency_name_is_refused() -> None:
    # A malformed/option-injecting name is rejected before it reaches uv, and the
    # exploratory tool surfaces it as a readable error rather than raising.
    result = SubprocessExecutor().run_code("result = 1", deps=["--system"])
    assert result.error is not None and "invalid dependency" in result.error


def test_derivation_deps_install_the_package_editable_and_run_as_module() -> None:
    # A dependency-carrying derivation is provisioned with uv AND gets elbi_core
    # editable-installed, so the child can run as a module (its Context and pickled
    # inputs need elbi_core importable). Exploration code needs neither.
    argv = SubprocessExecutor()._argv(["numpy"], "/io", needs_package=True)
    assert argv[:3] == ["uv", "run", "--no-project"]
    assert "--with-editable" in argv
    assert "numpy" in argv and argv.count("--with") == 1
    # runs as `-m elbi_core._sandbox_child`, not by file path
    assert argv[-3:] == ["-m", "elbi_core._sandbox_child", "/io"]
    # the exploration (no-package) deps path stays by file path, no editable install
    explore = SubprocessExecutor()._argv(["numpy"], "/io", needs_package=False)
    assert "--with-editable" not in explore and "-m" not in explore
    assert explore[-1] == "/io" and explore[-2].endswith("_sandbox_child.py")


def test_derivation_missing_a_declared_dependency_fails_clearly() -> None:
    # The gap that shipped: a derivation importing a third-party package with no deps
    # declared cannot import it and fails in the sandbox (the exact error the model
    # must fix by declaring the package). Guarded so it holds whether or not the dev
    # environment happens to have the package installed.
    import importlib.util

    if importlib.util.find_spec("sklearn") is not None:
        pytest.skip("sklearn present here; cannot assert the missing-import failure")
    reg = Registry()
    src = "def fit(ctx):\n    import sklearn\n    return [{'ok': 1}]\n"
    d = propose("fit", src, serve=serve.table(), registry=reg)  # no deps declared
    with pytest.raises(DerivationError, match="No module named 'sklearn'"):
        SubprocessExecutor(timeout=60).run(d, Context({}))


@pytest.mark.slow
def test_derivation_provisions_a_declared_dependency() -> None:
    # The fix, end to end: an authored derivation that imports a third-party package
    # (a real model needs scikit-learn) is provisioned via uv, elbi stays
    # importable for its Context, and the compute runs and returns a value. This is the
    # path every model-building derivation takes; it had no test before.
    reg = Registry()
    src = (
        "def fit(ctx):\n"
        "    from sklearn.linear_model import LinearRegression\n"
        "    model = LinearRegression().fit([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0])\n"
        "    return [{'coef': round(float(model.coef_[0]), 3)}]\n"
    )
    d = propose("fit", src, serve=serve.table(), deps=["scikit-learn"], registry=reg)
    artifact = SubprocessExecutor(timeout=300).run(d, Context({}))
    assert artifact.value == [{"coef": 2.0}]


def test_run_code_workspace_persists_files_across_calls(tmp_path: Path) -> None:
    # A persistent workspace makes one call's output available to the next: the first
    # call writes a file (a stand-in for a fitted model), the second reads it back.
    ex = SubprocessExecutor(timeout=30)
    write = ex.run_code(
        "open('model.txt', 'w').write('trained:42')", workspace=tmp_path
    )
    assert write.error is None
    read = ex.run_code("result = open('model.txt').read()", workspace=tmp_path)
    assert read.error is None
    assert read.result == "'trained:42'"


def test_run_code_without_workspace_is_ephemeral(tmp_path: Path) -> None:
    # Without a workspace each call runs in its own throwaway directory, so a file one
    # call writes is gone for the next. This is the pre-persistence behavior, intact.
    ex = SubprocessExecutor(timeout=30)
    ex.run_code("open('model.txt', 'w').write('x')")
    seen = ex.run_code("import os\nresult = os.path.exists('model.txt')")
    assert seen.result == "False"


def test_run_code_workspace_does_not_persist_variables(tmp_path: Path) -> None:
    # Only the filesystem persists: each call is a fresh process, so an in-memory
    # variable does not carry over even with a shared workspace. Continuity is by file.
    ex = SubprocessExecutor(timeout=30)
    ex.run_code("trained_model = 99", workspace=tmp_path)
    later = ex.run_code("result = 'trained_model' in dir()", workspace=tmp_path)
    assert later.result == "False"


def test_derivation_cannot_see_run_code_workspace(tmp_path: Path) -> None:
    # The verification invariant: a certified derivation must stay a pure function of
    # its declared inputs, so it runs in a private, empty directory and cannot read a
    # file exploration left in the workspace. Exploration state never leaks into it.
    ex = SubprocessExecutor(timeout=30)
    ex.run_code("open('leak.txt', 'w').write('secret')", workspace=tmp_path)
    reg = Registry()
    source = (
        "def probe(ctx):\n"
        "    import os\n"
        '    return [{"sees_leak": os.path.exists("leak.txt")}]\n'
    )
    d = propose("probe", source, serve=serve.table(), registry=reg)
    artifact = ex.run(d, Context({}))
    assert artifact.value == [{"sees_leak": False}]
