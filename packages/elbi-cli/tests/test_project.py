"""Tests for loading a project into a registry + runner."""

from __future__ import annotations

import getpass
from collections.abc import Callable
from pathlib import Path

import pytest

from elbi_cli.authored import AuthoredStore
from elbi_cli.project import (
    STATE_DIR_ENV,
    default_issuer_identity,
    load_project,
    state_dir_for,
)
from elbi_core import Registry, certify, propose, serve
from elbi_core.errors import ConfigError


def test_load_project_reloads_authored_derivations(
    scaffold: Callable[[str], Path],
) -> None:
    project = scaffold("minimal")  # no derivations/ dir
    store = AuthoredStore(project / ".elbi" / "authored")
    built = Registry()
    proposed = propose(
        "agent_metric",
        'def agent_metric(ctx):\n    "Authored."\n    return [{"a": 1}]\n',
        serve=serve.json(),
        registry=built,
    )
    store.save(certify(proposed, registry=built))

    loaded = load_project(project)
    assert loaded.authored == ("agent_metric",)
    reloaded = loaded.registry.get("agent_metric")
    assert reloaded.is_agent_authored and reloaded.is_certified


def test_load_project_discovers_and_runs(scaffold: Callable[[str], Path]) -> None:
    project = scaffold("standard")
    loaded = load_project(project)
    assert loaded.config.project == "acme"
    assert "churn_risk" in loaded.discovered

    runner = loaded.make_runner()
    artifact = runner.run("churn_risk")
    assert len(artifact.value) == 3


def test_load_project_missing_config(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match=r"elbi\.yaml"):
        load_project(tmp_path)


def test_default_issuer_identity_prefers_the_os_user(
    scaffold: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    project = load_project(scaffold("minimal"))
    monkeypatch.setattr(getpass, "getuser", lambda: "sam")
    assert default_issuer_identity(project) == "sam"


def test_default_issuer_identity_falls_back_to_the_project_name_on_oserror(
    scaffold: Callable[[str], Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    # getpass.getuser() raises OSError with no matching login name and no passwd
    # entry (a minimal container or CI image running as an arbitrary UID); this
    # must not crash the caller, and must not silently invent a fake identity.
    project = load_project(scaffold("minimal"))

    def _raise() -> str:
        raise OSError("no such user")

    monkeypatch.setattr(getpass, "getuser", _raise)
    assert default_issuer_identity(project) == project.config.project


def test_make_runner_is_fresh_each_call(scaffold: Callable[[str], Path]) -> None:
    project = scaffold("standard")
    loaded = load_project(project)
    assert loaded.make_runner() is not loaded.make_runner()


def test_sandbox_executor_selects_backend() -> None:
    from elbi_cli.project import sandbox_executor
    from elbi_core import DockerExecutor, SubprocessExecutor

    assert isinstance(sandbox_executor("subprocess"), SubprocessExecutor)
    assert isinstance(sandbox_executor("docker"), DockerExecutor)


def test_sandbox_executor_passes_image_to_docker() -> None:
    from elbi_cli.project import sandbox_executor
    from elbi_core import DockerExecutor

    executor = sandbox_executor("docker", "elbi-sandbox")
    assert isinstance(executor, DockerExecutor)
    assert executor._image == "elbi-sandbox"


def test_load_minimal_project_has_no_derivations(
    scaffold: Callable[[str], Path],
) -> None:
    # The minimal template has no derivations/ directory.
    project = scaffold("minimal")
    loaded = load_project(project)
    assert loaded.discovered == ()
    assert len(loaded.registry) == 0


_PROPOSED = '''\
from elbi_core import Context, derivation, serve


@derivation(name="forecast", serve=serve.text(), origin="agent", status="proposed")
def forecast(ctx: Context) -> str:
    """A proposed forecast."""
    return "soon"
'''


def test_lifecycle_override_applies_on_load(
    scaffold: Callable[[str], Path],
) -> None:
    project = scaffold("standard")
    (project / "derivations" / "forecast.py").write_text(_PROPOSED, encoding="utf-8")

    # Authored as proposed.
    assert not load_project(project).registry.get("forecast").is_certified

    # Recording certification flips the effective status on the next load.
    load_project(project).lifecycle.set_status("forecast", "certified")
    assert load_project(project).registry.get("forecast").is_certified


def test_load_dataset_returns_table(scaffold: Callable[[str], Path]) -> None:
    from elbi_core.data import Table

    loaded = load_project(scaffold("standard"))
    table = loaded.load_dataset("sales")
    assert isinstance(table, Table)
    assert len(table) > 0


def test_lifecycle_override_ignores_unknown_and_noop(
    scaffold: Callable[[str], Path],
) -> None:
    project = scaffold("standard")  # churn_risk is human/certified
    store = load_project(project).lifecycle
    store.set_status("ghost", "certified")  # not in the registry → skipped
    store.set_status("churn_risk", "certified")  # already certified → no-op

    loaded = load_project(project)
    assert loaded.registry.get("churn_risk").is_certified
    assert "ghost" not in loaded.registry


def test_make_runner_routes_source_derivation_to_sandbox(
    scaffold: Callable[[str], Path],
) -> None:
    # A certified, source-carrying (agent-authored) derivation must run: it routes
    # to the sandbox, while a human one stays in-process. One runner.
    from elbi_core import certify, propose, serve

    loaded = load_project(scaffold("standard"))
    propose(
        "agent_fc",
        "def agent_fc(ctx):\n    return 'from sandbox'\n",
        serve=serve.text(),
        registry=loaded.registry,
    )
    certify(loaded.registry.get("agent_fc"), registry=loaded.registry)

    runner = loaded.make_runner()
    assert runner.serve("agent_fc") == "from sandbox"
    assert len(runner.run("churn_risk").value) == 3


def test_the_sandbox_executor_is_sized_by_a_profile() -> None:
    from elbi_cli.project import sandbox_executor
    from elbi_core import DockerExecutor
    from elbi_core.sandbox import ComputeProfile

    # Agent-written code should be bounded by the same menu a notebook picks from,
    # rather than by a number written into the library.
    executor = sandbox_executor(
        "docker", None, profile=ComputeProfile(name="big", cpu="4", memory="16Gi")
    )
    assert isinstance(executor, DockerExecutor)
    assert executor._memory == f"{16 * 1024**3}b"
    assert executor._cpus == "4"


def test_the_kubernetes_backend_runs_candidates_as_jobs() -> None:
    """Candidate code belongs on the session runner, which is what the plan says.

    Verified on a real cluster: a candidate ran as a Job and returned its artifact. The
    fallback this replaced was the dangerous one: a deployment that selected per-pod
    isolation for notebooks got agent-written code as a child of the app process.
    """
    from elbi_cli.project import sandbox_executor
    from elbi_core.k8s_executor import KubernetesExecutor

    executor = sandbox_executor(
        "kubernetes", "example.com/app:1.0", namespace="kernels"
    )
    assert isinstance(executor, KubernetesExecutor)

    # An unknown backend is still refused rather than silently downgraded.
    with pytest.raises(ConfigError, match="there is no 'vm' executor"):
        sandbox_executor("vm")


def test_state_dir_defaults_into_the_project(tmp_path: Path) -> None:
    assert state_dir_for(tmp_path) == tmp_path / ".elbi"


def test_state_dir_env_moves_everything_the_project_writes(
    scaffold: Callable[[str], Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A read-only project needs every writable path to land outside it.

    Enumerated rather than spot-checked: the failure this guards against is one path
    still pointing into the project, which does nothing on a laptop and makes the whole
    deployment fail on the read-only mount where it matters.
    """
    project_dir = scaffold("minimal")
    state = tmp_path / "state"
    monkeypatch.setenv(STATE_DIR_ENV, str(state))

    loaded = load_project(project_dir)

    assert loaded.state_dir == state
    writable = (
        loaded.cache_dir,
        loaded.workspaces_dir,
        loaded.audit_log_path,
        loaded.certificate_key_dir,
    )
    for path in writable:
        assert state in path.parents or path == state, path
        assert project_dir not in path.parents, f"{path} still writes into the project"


def test_a_read_only_project_directory_still_loads(
    scaffold: Callable[[str], Path], tmp_path: Path
) -> None:
    """The air-gapped and git-sync shapes both serve a project they cannot write to.

    Enforced by actually removing write permission rather than by asserting about paths,
    because the interesting failure is a path this test did not think to check.
    """
    project_dir = scaffold("minimal")
    state = tmp_path / "state"
    mode = project_dir.stat().st_mode
    project_dir.chmod(0o500)
    try:
        loaded = load_project(project_dir, state_dir=state)
        # Exercising the stores is the point: constructing them is not what writes.
        loaded.lifecycle.set_status("some_derivation", "certified")
        loaded.cache_dir.mkdir(parents=True, exist_ok=True)
        (loaded.cache_dir / "probe").write_text("x")
    finally:
        project_dir.chmod(mode)

    assert loaded.state_dir == state
    assert state.exists()
