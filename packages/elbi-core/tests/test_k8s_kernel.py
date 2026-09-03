"""The Kubernetes runner, checked against a recording stand-in for ``kubectl``.

A real cluster is not available here, so what is verified is everything up to the API
call: the manifest a profile produces, the security properties that must survive a port,
and the argument lists. That is the part this code is responsible for: whether a pod
schedules is the cluster's business, and whether the protocol works over an attach is
already covered by the two backends that share the pump.
"""

from __future__ import annotations

import json
import os
import stat
import time
from pathlib import Path
from typing import Any

import pytest

from elbi_core.errors import DerivationError
from elbi_core.notebook.k8s_kernel import (
    KubernetesNotebookKernel,
    deps_digest,
    worker_bootstrap,
)
from elbi_core.sandbox import ComputeProfile


@pytest.fixture
def fake_kubectl(tmp_path: Path) -> Path:
    """A ``kubectl`` that records every invocation and succeeds at all of them.

    ``attach`` execs ``cat``, so the kernel gets a process whose stdin and stdout are
    real pipes; nothing speaks the protocol over it, but the object is constructible and
    its lifecycle runs.
    """
    log = tmp_path / "calls.jsonl"
    script = tmp_path / "kubectl"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> {log}\n'
        'for arg in "$@"; do\n'
        '  if [ "$arg" = "-f" ]; then cat > '
        + str(tmp_path / "manifest.json")
        + "; fi\n"
        "done\n"
        'case "$1" in\n'
        "  attach) exec cat ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _manifest(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))


def _calls(tmp_path: Path) -> list[str]:
    return (tmp_path / "calls.jsonl").read_text(encoding="utf-8").splitlines()


def _wait_for_call(tmp_path: Path, verb: str, timeout: float = 10.0) -> list[str]:
    """Calls recorded so far, once one of them starts with ``verb``.

    ``Popen`` returns before the process it started has run, so the attach may not have
    logged itself yet when the constructor returns. Polling for it is the difference
    between testing the ordering and testing the scheduler.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        calls = _calls(tmp_path) if (tmp_path / "calls.jsonl").exists() else []
        if any(call.startswith(verb) for call in calls):
            return calls
        time.sleep(0.05)
    raise AssertionError(f"no {verb!r} call was recorded within {timeout}s")


def _start(fake_kubectl: Path, **kwargs: Any) -> KubernetesNotebookKernel:
    kwargs.setdefault("profile", ComputeProfile(name="small"))
    # An image is required: the runner refuses one that cannot run the worker.
    kwargs.setdefault("image", "example.com/elbi/app:1.0")
    return KubernetesNotebookKernel(kubectl_bin=str(fake_kubectl), **kwargs)


def test_the_pod_is_created_waited_for_and_attached(
    fake_kubectl: Path, tmp_path: Path
) -> None:
    kernel = _start(fake_kubectl, namespace="analytics")
    try:
        calls = _wait_for_call(tmp_path, "attach")
        # Created, then waited for, then attached. Waiting is not optional: without it
        # the first cell would be billed for an image pull, which is the part that
        # actually takes minutes.
        assert [c.split()[0] for c in calls[:3]] == ["create", "wait", "attach"]
        assert calls[0].startswith("create -n analytics -f -")
        assert "wait -n analytics pod/elbi-nb-" in calls[1]
        assert "--for=condition=Ready" in calls[1]
        assert "-c kernel -i -q" in calls[2]
        assert kernel.alive
    finally:
        kernel.close()
    assert any("delete -n analytics pod/" in c for c in _calls(tmp_path))


def test_the_kernel_pod_gets_no_cluster_identity(
    fake_kubectl: Path, tmp_path: Path
) -> None:
    # The pod runs whatever a notebook contains. A service-account token mounted into it
    # would be the single most valuable thing in the sandbox.
    kernel = _start(fake_kubectl)
    kernel.close()
    spec = _manifest(tmp_path)["spec"]
    assert spec["automountServiceAccountToken"] is False
    assert spec["enableServiceLinks"] is False
    container = spec["containers"][0]
    assert container["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
        "readOnlyRootFilesystem": True,
    }
    assert spec["securityContext"]["runAsNonRoot"] is True
    assert spec["securityContext"]["seccompProfile"] == {"type": "RuntimeDefault"}
    # A leaked pod is bounded by the cluster rather than by the app remembering it.
    assert spec["activeDeadlineSeconds"] > 0
    assert spec["restartPolicy"] == "Never"


def test_the_protocol_gets_pipes_rather_than_a_terminal(
    fake_kubectl: Path, tmp_path: Path
) -> None:
    kernel = _start(fake_kubectl)
    kernel.close()
    container = _manifest(tmp_path)["spec"]["containers"][0]
    # A TTY would echo input and rewrite newlines; the protocol is bytes, not a session.
    assert container["tty"] is False
    assert container["stdin"] is True
    # stdinOnce is off on purpose: it leaves stdin *empty* until a client attaches, so
    # the worker reads EOF at startup and exits before the attach lands. Verified on a
    # real cluster, where that race showed up as "container kernel not found in pod".
    assert container["stdinOnce"] is False
    env = {e["name"]: e["value"] for e in container["env"]}
    # Block-buffered stdout would hold a cell's output until 8 KB accumulated.
    assert env["PYTHONUNBUFFERED"] == "1"
    # No PYTHONPATH without dependencies: setting it *replaces* whatever the image set,
    # and an image that exposes its own packages that way then cannot import them:
    # including elbi-core, which is what this runner needs. Found on a cluster.
    assert "PYTHONPATH" not in env


def test_a_profile_reaches_the_pod(fake_kubectl: Path, tmp_path: Path) -> None:
    profile = ComputeProfile(
        name="gpu",
        cpu="8",
        memory="32Gi",
        gpu=1,
        gpu_type="nvidia-l4",
        runtime_class="gvisor",
        egress="none",
    )
    kernel = _start(fake_kubectl, profile=profile)
    kernel.close()
    manifest = _manifest(tmp_path)
    spec = manifest["spec"]
    assert spec["runtimeClassName"] == "gvisor"
    assert spec["nodeSelector"] == {"intelligible.ai/accelerator": "nvidia-l4"}
    assert {"key": "nvidia.com/gpu", "operator": "Exists", "effect": "NoSchedule"} in (
        spec["tolerations"]
    )
    resources = spec["containers"][0]["resources"]
    assert resources["limits"] == {"cpu": "8", "memory": "32Gi", "nvidia.com/gpu": "1"}
    assert resources["requests"] == {"cpu": "8", "memory": "32Gi"}
    # Egress is enforced by a cluster NetworkPolicy selecting this label, not by the
    # sandbox itself, so a kernel cannot lift its own restriction.
    assert manifest["metadata"]["labels"]["intelligible.ai/egress"] == "none"
    assert manifest["metadata"]["labels"]["intelligible.ai/profile"] == "gpu"


def test_spot_capacity_is_selected_and_tolerated(
    fake_kubectl: Path, tmp_path: Path
) -> None:
    kernel = _start(fake_kubectl, profile=ComputeProfile(name="batch", spot=True))
    kernel.close()
    spec = _manifest(tmp_path)["spec"]
    assert spec["nodeSelector"]["intelligible.ai/capacity"] == "spot"
    keys = {t["key"] for t in spec["tolerations"]}
    assert "cloud.google.com/gke-spot" in keys


def test_an_image_that_cannot_run_the_worker_is_refused(fake_kubectl: Path) -> None:
    """A pod cannot mount the host source, so the image has to carry the package.

    Verified on a real cluster: a bare Python image starts, fails to import the worker,
    and exits, which reached the caller as an unexplained dead kernel.
    """
    with pytest.raises(DerivationError, match="needs an image with elbi-core"):
        KubernetesNotebookKernel(
            profile=ComputeProfile(name="small"), kubectl_bin=str(fake_kubectl)
        )
    with pytest.raises(DerivationError, match="cannot run the kernel worker"):
        _start(fake_kubectl, image="python:3.12-slim")


def test_dependencies_add_the_path_they_install_to(
    fake_kubectl: Path, tmp_path: Path
) -> None:
    kernel = _start(fake_kubectl, deps=["polars==1.0.0"])
    kernel.close()
    container = _manifest(tmp_path)["spec"]["containers"][0]
    env = {e["name"]: e["value"] for e in container["env"]}
    assert env["PYTHONPATH"] == "/deps"


def test_dependencies_install_once_and_skip_when_present(
    fake_kubectl: Path, tmp_path: Path
) -> None:
    kernel = _start(fake_kubectl, deps=["polars==1.0.0"])
    kernel.close()
    spec = _manifest(tmp_path)["spec"]
    (installer,) = spec["initContainers"]
    script = installer["command"][-1]
    # The sentinel is checked first and written last, which is what makes a reused
    # volume safe: an interrupted install leaves no sentinel, so the next session
    # reinstalls rather than importing a partial package tree.
    assert script.startswith("test -f /deps/.elbi-ready && exit 0;")
    assert script.rstrip().endswith("touch /deps/.elbi-ready")
    assert "polars==1.0.0" in script
    assert installer["securityContext"]["readOnlyRootFilesystem"] is True
    # Not the session's profile: this container downloads wheels, and giving it the
    # profile would reserve a GPU to run pip.
    assert "nvidia.com/gpu" not in installer["resources"]["limits"]


def test_no_dependencies_means_no_install_step(
    fake_kubectl: Path, tmp_path: Path
) -> None:
    kernel = _start(fake_kubectl)
    kernel.close()
    assert "initContainers" not in _manifest(tmp_path)["spec"]


def test_a_shared_deps_claim_is_used_when_a_storage_class_is_given(
    fake_kubectl: Path, tmp_path: Path
) -> None:
    kernel = _start(fake_kubectl, deps=["polars==1.0.0"], deps_storage_class="efs-sc")
    kernel.close()
    volumes = {v["name"]: v for v in _manifest(tmp_path)["spec"]["volumes"]}
    claim = volumes["deps"]["ephemeral"]["volumeClaimTemplate"]
    assert claim["spec"]["storageClassName"] == "efs-sc"
    assert claim["metadata"]["labels"]["intelligible.ai/deps"] == deps_digest(
        "example.com/elbi/app:1.0", ["polars==1.0.0"]
    )
    # Without one, every session installs into its own scratch: slower, but correct on
    # the block storage most clusters actually have.
    other = _start(fake_kubectl, deps=["polars==1.0.0"])
    other.close()
    volumes = {v["name"]: v for v in _manifest(tmp_path)["spec"]["volumes"]}
    assert volumes["deps"] == {"name": "deps", "emptyDir": {}}


def test_dataset_names_travel_in_the_environment(
    fake_kubectl: Path, tmp_path: Path
) -> None:
    # A pod cannot mount the file the docker backend passes them in.
    kernel = _start(fake_kubectl, dataset_names=["orders", "customers"])
    kernel.close()
    env = {
        e["name"]: e["value"]
        for e in _manifest(tmp_path)["spec"]["containers"][0]["env"]
    }
    assert json.loads(env["ELBI_DATASET_NAMES"]) == ["orders", "customers"]


def test_eagerly_seeded_data_is_refused_with_a_reason(fake_kubectl: Path) -> None:
    with pytest.raises(DerivationError, match="fetch data on demand"):
        _start(fake_kubectl, data={"orders": [{"id": 1}]})


def test_an_invalid_dependency_is_refused_before_any_pod_exists(
    fake_kubectl: Path, tmp_path: Path
) -> None:
    with pytest.raises(DerivationError, match="invalid dependency"):
        _start(fake_kubectl, deps=["polars; rm -rf /"])
    assert not (tmp_path / "calls.jsonl").exists()


def test_a_pod_that_never_becomes_ready_is_deleted(tmp_path: Path) -> None:
    # Otherwise a cluster that cannot schedule the pod accumulates one per attempt, each
    # still pulling an image.
    log = tmp_path / "calls.jsonl"
    script = tmp_path / "kubectl"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> {log}\n'
        'if [ "$1" = "wait" ]; then echo "timed out" >&2; exit 1; fi\n'
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    with pytest.raises(DerivationError, match="start the kernel pod"):
        KubernetesNotebookKernel(
            profile=ComputeProfile(name="small"),
            image="example.com/elbi/app:1.0",
            kubectl_bin=str(script),
        )
    assert any(c.startswith("delete pod/") for c in _calls(tmp_path))


def test_a_missing_kubectl_says_what_to_do(tmp_path: Path) -> None:
    with pytest.raises(DerivationError, match="install kubectl"):
        KubernetesNotebookKernel(
            profile=ComputeProfile(name="small"),
            image="example.com/elbi/app:1.0",
            kubectl_bin=str(tmp_path / "nothing-here"),
        )


def test_the_bootstrap_runs_the_installed_worker_by_path() -> None:
    """The one-liner the pod runs must actually find and start the worker.

    The worker's sibling modules import as top-level names, so it has to be launched
    as a script rather than with ``-m``, and the path it lives at inside an image is
    not the path it lives at here. Running it for real is the only way to know that
    the resolution works.
    """
    import subprocess
    import sys

    # Ask the worker what datasets it can see, which only answers correctly if the
    # bootstrap found it *and* it read the names from the environment -- the pod has no
    # file to pass them in.
    request = json.dumps({"code": "list(data)", "execution_count": 1}) + "\n"
    result = subprocess.run(
        [sys.executable, "-c", worker_bootstrap(), "--allow-network"],
        input=request,
        capture_output=True,
        text=True,
        timeout=60,
        env={**os.environ, "ELBI_DATASET_NAMES": '["orders", "customers"]'},
    )
    assert result.returncode == 0, result.stderr
    results = [
        json.loads(line)
        for line in result.stdout.splitlines()
        if line.strip().startswith("{")
    ]
    values = [
        m["data"]["text/plain"] for m in results if m.get("type") == "execute_result"
    ]
    assert values == ["['orders', 'customers']"]


def test_the_dependency_mount_is_read_only_in_the_kernel(
    fake_kubectl: Path, tmp_path: Path
) -> None:
    """A kernel may import its dependencies and must not be able to rewrite them.

    The dependency tree is content-addressed and shared between every session using the
    same package set, so a kernel that could write to it could change what another
    session imports. The docker backend keeps this by mounting its volume ``:ro``;
    losing it in a port is the silent regression the plan's invariants exist to catch.
    """
    kernel = _start(fake_kubectl, deps=["polars==1.0.0"])
    kernel.close()
    spec = _manifest(tmp_path)["spec"]

    deps_mount = next(
        m for m in spec["containers"][0]["volumeMounts"] if m["name"] == "deps"
    )
    assert deps_mount["readOnly"] is True

    # The init container is the one place it is writable, because provisioning is the
    # only step that writes -- and the only step with a network.
    installer_mount = next(
        m for m in spec["initContainers"][0]["volumeMounts"] if m["name"] == "deps"
    )
    assert not installer_mount.get("readOnly")


def test_a_host_scratch_path_never_becomes_a_claim(
    fake_kubectl: Path, tmp_path: Path
) -> None:
    """A workspace directory on the app's host means nothing to a pod elsewhere.

    Translating one into a ``claimName`` would name a claim nobody created and leave
    every pod unschedulable: a deployment broken by a default that looks harmless.
    """
    kernel = _start(fake_kubectl, workspace=tmp_path / "notebooks" / "nb-42")
    kernel.close()
    work = next(
        v for v in _manifest(tmp_path)["spec"]["volumes"] if v["name"] == "work"
    )
    assert work == {"name": "work", "emptyDir": {}}

    # A claim is used only when an operator names one.
    other = _start(fake_kubectl, workspace_claim="notebook-scratch")
    other.close()
    work = next(
        v for v in _manifest(tmp_path)["spec"]["volumes"] if v["name"] == "work"
    )
    assert work["persistentVolumeClaim"] == {"claimName": "notebook-scratch"}
