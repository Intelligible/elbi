"""Tests for the Docker-backed executor.

The validation-and-error paths run anywhere (they fail before touching Docker). The
end-to-end runs need a working Docker daemon, so they skip cleanly when one is absent
and are marked ``slow`` because they pull an image and install packages.
"""

from __future__ import annotations

import subprocess

import pytest

from elbi_core import (
    Artifact,
    Context,
    DockerExecutor,
    Registry,
    derivation,
    propose,
    serve,
)
from elbi_core.errors import DerivationError


def _docker_available() -> bool:
    try:
        completed = subprocess.run(["docker", "info"], capture_output=True, check=False)
    except FileNotFoundError:
        return False
    return completed.returncode == 0


requires_docker = pytest.mark.skipif(
    not _docker_available(), reason="needs a running Docker daemon"
)


def test_docker_executor_image_defaults_to_slim() -> None:
    # image=None selects the slim base; a project can point at a data-science image.
    from elbi_core.container import _DEFAULT_IMAGE

    assert DockerExecutor(image=None)._image == _DEFAULT_IMAGE
    assert DockerExecutor(image="elbi-sandbox")._image == "elbi-sandbox"


def test_docker_requires_source() -> None:
    # A human derivation has no source; the docker executor runs generated source only.
    reg = Registry()

    @derivation(name="human", serve=serve.text(), registry=reg)
    def human(ctx: Context) -> Artifact:
        return Artifact.text("x")

    with pytest.raises(DerivationError, match="has no source"):
        DockerExecutor().run(human, Context({}))


def test_docker_rejects_invalid_dependency() -> None:
    # A malformed dependency name is refused before any container is created.
    reg = Registry()
    d = propose(
        "m",
        "def m(ctx):\n    return [{'v': 1}]\n",
        serve=serve.table(),
        deps=["--evil"],
        registry=reg,
    )
    with pytest.raises(DerivationError, match="invalid dependency"):
        DockerExecutor().run(d, Context({}))


def test_docker_missing_binary_is_reported() -> None:
    # Pointing at a non-existent docker binary fails with a readable error, not a crash.
    reg = Registry()
    d = propose(
        "m", "def m(ctx):\n    return [{'v': 1}]\n", serve=serve.table(), registry=reg
    )
    ex = DockerExecutor(docker_bin="definitely-not-docker")
    with pytest.raises(DerivationError, match="not found"):
        ex.run(d, Context({}))


def test_docker_rejects_nonpositive_timeout() -> None:
    with pytest.raises(ValueError, match="timeout must be positive"):
        DockerExecutor(timeout=0)


class _FakeDocker:
    """A minimal in-memory stand-in for the docker CLI, to test volume provisioning.

    Models each volume's existence and whether it carries the ready sentinel, and counts
    how many times a dependency install runs. ``install_hook`` lets a test inject a
    delay (to force concurrency) or a failure.
    """

    def __init__(self) -> None:
        self.volumes: dict[str, dict[str, bool]] = {}
        self.installs = 0
        self.install_hook = None

    @staticmethod
    def _mounted(args: list[str]) -> str:
        return next(a.split(":")[0] for a in args if a.endswith(":/deps"))

    def ok(self, docker_bin: str, args: list[str]) -> bool:
        if args[:2] == ["volume", "inspect"]:
            return args[2] in self.volumes
        if args[:2] == ["volume", "rm"]:
            self.volumes.pop(args[-1], None)
            return True
        if args[0] == "run" and "test" in args:  # the readiness probe
            return self.volumes.get(self._mounted(args), {}).get("ready", False)
        return True

    def run(
        self, docker_bin: str, args: list[str], what: str, timeout: float | None = None
    ) -> None:
        if args[:2] == ["volume", "create"]:
            self.volumes[args[2]] = {"ready": False}
        elif args[0] == "run" and "pip" in args:
            self.installs += 1
            if self.install_hook is not None:
                self.install_hook()
        elif args[0] == "run" and "touch" in args:
            self.volumes[self._mounted(args)]["ready"] = True


def test_deps_volume_discarded_when_install_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A failed install must not leave a volume a later run would trust: the volume is
    # removed, so a fresh attempt re-provisions instead of mounting an empty /deps.
    from elbi_core import container

    container._ready_volumes.clear()
    fake = _FakeDocker()

    def boom() -> None:
        raise DerivationError("pip exploded")

    fake.install_hook = boom
    monkeypatch.setattr(container, "_docker_ok", fake.ok)
    monkeypatch.setattr(container, "_docker_run_checked", fake.run)

    with pytest.raises(DerivationError, match="pip exploded"):
        container.ensure_deps_volume("img", ["numpy"])
    assert fake.volumes == {}  # nothing cached from the failed attempt

    fake.install_hook = None
    name = container.ensure_deps_volume("img", ["numpy"])
    assert fake.volumes[name]["ready"] is True


def test_deps_volume_provisions_once_under_concurrency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Six threads asking for the same dependency set must provision it exactly once; the
    # per-volume lock serializes them so no two `pip install --target` share a volume.
    import threading
    import time

    from elbi_core import container

    container._ready_volumes.clear()
    fake = _FakeDocker()
    fake.install_hook = lambda: time.sleep(0.02)  # hold the lock to force contention
    monkeypatch.setattr(container, "_docker_ok", fake.ok)
    monkeypatch.setattr(container, "_docker_run_checked", fake.run)

    threads = [
        threading.Thread(target=lambda: container.ensure_deps_volume("img", ["numpy"]))
        for _ in range(6)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert fake.installs == 1


@pytest.mark.slow
@requires_docker
def test_docker_runs_a_dependency_derivation() -> None:
    # The headline: a derivation that imports scikit-learn runs in a Linux container,
    # its dependency provisioned into a cached volume, and returns the right value.
    reg = Registry()
    src = (
        "def fit(ctx):\n"
        "    from sklearn.linear_model import LinearRegression\n"
        "    m = LinearRegression().fit([[1.0], [2.0], [3.0]], [2.0, 4.0, 6.0])\n"
        "    return [{'coef': round(float(m.coef_[0]), 3)}]\n"
    )
    d = propose("fit", src, serve=serve.table(), deps=["scikit-learn"], registry=reg)
    artifact = DockerExecutor(timeout=600).run(d, Context({}))
    assert artifact.value == [{"coef": 2.0}]


@pytest.mark.slow
@requires_docker
def test_docker_compute_has_no_network() -> None:
    # A certified derivation stays hermetic: the compute container has no network, so an
    # outbound connection fails rather than silently reaching out.
    reg = Registry()
    src = (
        "def reach(ctx):\n"
        "    import socket\n"
        "    socket.create_connection(('1.1.1.1', 53), timeout=3)\n"
        "    return [{'reached': 1}]\n"
    )
    d = propose("reach", src, serve=serve.table(), registry=reg)
    with pytest.raises(DerivationError):
        DockerExecutor(timeout=120).run(d, Context({}))


def test_docker_session_rejects_invalid_dependency() -> None:
    from elbi_core import DockerReplSession

    with pytest.raises(DerivationError, match="invalid dependency"):
        DockerReplSession(deps=["--evil"])


def test_docker_session_rejects_nonpositive_timeout() -> None:
    from elbi_core import DockerReplSession

    with pytest.raises(ValueError, match="timeout must be positive"):
        DockerReplSession(timeout=0)


@pytest.mark.slow
@requires_docker
def test_docker_session_shares_state_and_filesystem() -> None:
    # The isolated box for exploration: run_code keeps in-memory state across calls, and
    # run_code and bash share one container's filesystem, so files written by either are
    # visible to the other. This is what lets bash install a package Python then uses.
    from elbi_core import DockerReplSession

    s = DockerReplSession(timeout=300)
    try:
        assert s.run_code("x = 41").error is None
        assert s.run_code("result = x + 1").result == "42"  # namespace persists
        # a file Python writes is visible to bash (same container filesystem)
        s.run_code("open('a.txt', 'w').write('from-python')")
        assert "from-python" in s.run_bash("cat a.txt").stdout
        # a file bash writes is visible to Python
        s.run_bash("echo from-bash > b.txt")
        assert (
            s.run_code("result = open('b.txt').read().strip()").result == "'from-bash'"
        )
    finally:
        s.close()


def test_docker_session_rejects_invalid_egress_host() -> None:
    from elbi_core import DockerReplSession

    with pytest.raises(DerivationError, match="invalid egress host"):
        DockerReplSession(egress=["bad host!"])


@pytest.mark.slow
@requires_docker
def test_docker_session_egress_allowlist() -> None:
    # An egress allowlist lets the container reach only the named hosts (enforced by a
    # filtering proxy sidecar); everything else is refused, so data cannot leak out
    # to an arbitrary host.
    from elbi_core import DockerReplSession

    s = DockerReplSession(timeout=180, egress=["example.com"])
    try:
        allowed = s.run_code(
            "import urllib.request as u\n"
            "try:\n"
            "    u.urlopen('https://example.com', timeout=15); result = 'reached'\n"
            "except Exception as e:\n"
            "    result = 'failed'\n"
        )
        assert allowed.result == "'reached'"
        blocked = s.run_code(
            "import urllib.request as u\n"
            "try:\n"
            "    u.urlopen('https://pypi.org', timeout=12); result = 'reached'\n"
            "except Exception:\n"
            "    result = 'blocked'\n"
        )
        assert blocked.result == "'blocked'"
    finally:
        s.close()


@pytest.mark.slow
@requires_docker
def test_session_manager_adds_deps_without_losing_state() -> None:
    # The docker session installs a newly-needed package into the running container
    # instead of restarting, so an in-memory value from an earlier call survives (no
    # more retrain-from-scratch churn when the model reaches for a new import).
    from elbi_core import SessionManager

    m = SessionManager(datasets={}, backend="docker")
    try:
        m.run_code("marker = 'kept'")
        out = m.run_code(
            "import numpy as np\nresult = f'{marker}-{float(np.mean([1, 2, 3]))}'",
            deps=["numpy"],
        )
        assert out.result == "'kept-2.0'"  # marker survived AND numpy is usable
    finally:
        m.close()


@pytest.mark.slow
@requires_docker
def test_session_survives_a_failed_dep_install() -> None:
    # A dependency that cannot install (no such package) must NOT restart the session
    # and wipe its state; the failure surfaces as an import error the model adapts to,
    # while the in-memory value from before the failed install is still there.
    from elbi_core import SessionManager

    m = SessionManager(datasets={}, backend="docker")
    try:
        m.run_code("marker = 'kept'")
        m.run_code("pass", deps=["elbi-no-such-package-xyz"])
        out = m.run_code("result = marker")
        assert out.result == "'kept'"  # state survived the failed install
    finally:
        m.close()


def test_configure_egress_full_and_none_need_no_resources() -> None:
    # The common policies map straight to a docker network flag and create nothing to
    # tear down; only a host allowlist spins up the proxy sidecar (covered by the
    # docker-dependent suite).
    from elbi_core.container import configure_egress

    net, env, containers, networks = configure_egress("full", "python:3.12-slim", "s1")
    assert net == ["--network", "bridge"]
    assert env == [] and containers == [] and networks == []

    net, env, containers, networks = configure_egress("none", "python:3.12-slim", "s1")
    assert net == ["--network", "none"]
    assert containers == [] and networks == []


def test_configure_egress_rejects_a_bad_host() -> None:
    from elbi_core.container import configure_egress
    from elbi_core.errors import DerivationError

    with pytest.raises(DerivationError):
        configure_egress(["bad host!"], "python:3.12-slim", "s1")
