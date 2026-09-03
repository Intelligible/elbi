"""Agent-written code as a Job, checked against a recording stand-in for ``kubectl``.

The happy path was verified on a real cluster: a candidate ran as a Job and returned its
artifact, and a failing one surfaced its traceback. What is pinned here is the manifest
and the failure handling: the parts a cluster proves once, and a regression could undo.
"""

from __future__ import annotations

import json
import stat
from pathlib import Path
from typing import Any

import pytest

from elbi_core.context import Context
from elbi_core.derivation import Derivation
from elbi_core.errors import DerivationError
from elbi_core.k8s_executor import KubernetesExecutor


@pytest.fixture
def fake_kubectl(tmp_path: Path) -> Path:
    """A ``kubectl`` that records calls, captures the manifest, and returns a result."""
    result = json.dumps({"ok": True, "kind": "table", "value": [{"n": 42}]})
    script = tmp_path / "kubectl"
    script.write_text(
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> {tmp_path / "calls"}\n'
        'for a in "$@"; do\n'
        f'  if [ "$a" = "-f" ]; then cat > {tmp_path / "manifest.json"}; fi\n'
        "done\n"
        'case "$1" in\n'
        # A derivation that prints puts text on the same stream, so the stand-in emits
        # some: the reader has to take the *last* sentinel, not the first thing it sees.
        '  logs) printf "%s\\n%s\\n%s\\n" "chatty output" '
        f"'---elbi-result---' '{result}' ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    return script


def _manifest(tmp_path: Path) -> dict[str, Any]:
    return json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))


def _derivation(source: str, name: str = "total") -> Derivation:
    return Derivation(name=name, compute=None, source=source)


def test_a_candidate_runs_as_a_job_and_returns_its_artifact(
    fake_kubectl: Path, tmp_path: Path
) -> None:
    executor = KubernetesExecutor(
        image="example.com/app:1.0", kubectl_bin=str(fake_kubectl)
    )
    artifact = executor.run(
        _derivation("def total(ctx):\n    return []\n"), Context(inputs={}, params={})
    )
    assert artifact.kind == "table"
    assert artifact.value == [{"n": 42}]

    manifest = _manifest(tmp_path)
    assert manifest["kind"] == "Job"
    spec = manifest["spec"]
    # Kubernetes' guidance is a Job over a bare Pod even for one pod, because a bare
    # Pod on a failed node is simply gone.
    assert spec["template"]["spec"]["restartPolicy"] == "Never"
    # No retries: a candidate is deterministic, so a retry reproduces the same failure
    # at twice the cost, and the failure is what the agent reads to fix its code.
    assert spec["backoffLimit"] == 0
    assert spec["activeDeadlineSeconds"] > 0
    assert spec["ttlSecondsAfterFinished"] > 0

    calls = (tmp_path / "calls").read_text(encoding="utf-8").splitlines()
    assert calls[0].startswith("create -f -")
    # Deleted even on success: the TTL is a backstop for an app that crashed, not the
    # normal path.
    assert any(c.startswith("delete job/") for c in calls)


def test_the_job_pod_gets_no_identity_and_cannot_write_its_image(
    fake_kubectl: Path, tmp_path: Path
) -> None:
    # It runs code an agent wrote, which is the same threat model as a notebook cell.
    executor = KubernetesExecutor(
        image="example.com/app:1.0", kubectl_bin=str(fake_kubectl)
    )
    executor.run(_derivation("def total(ctx):\n    return []\n"), Context(inputs={}))
    pod = _manifest(tmp_path)["spec"]["template"]["spec"]
    assert pod["automountServiceAccountToken"] is False
    assert pod["enableServiceLinks"] is False
    assert pod["securityContext"]["runAsNonRoot"] is True
    container = pod["containers"][0]
    assert container["securityContext"] == {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
        "readOnlyRootFilesystem": True,
    }
    # Egress is a label the cluster's NetworkPolicy selects on, and a candidate has no
    # more reason to reach the network than a notebook cell does.
    assert _manifest(tmp_path)["metadata"]["labels"]["intelligible.ai/egress"] == "none"


def test_a_failure_inside_the_sandbox_is_reported_not_swallowed(
    tmp_path: Path,
) -> None:
    failed = json.dumps({"ok": False, "error": "ZeroDivisionError: division by zero"})
    script = tmp_path / "kubectl"
    script.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        f"  logs) printf '%s\\n%s\\n' '---elbi-result---' '{failed}' ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    executor = KubernetesExecutor(image="example.com/app:1.0", kubectl_bin=str(script))
    with pytest.raises(DerivationError, match="ZeroDivisionError"):
        executor.run(
            _derivation("def total(ctx):\n    return 1 / 0\n"), Context(inputs={})
        )


def test_a_job_that_produced_no_result_says_what_it_did_produce(
    tmp_path: Path,
) -> None:
    # Without this the caller gets "no result" and the evidence is in a Job the delete
    # already removed.
    script = tmp_path / "kubectl"
    script.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        "  logs) echo 'Killed: out of memory' ;;\n"
        "esac\n"
        "exit 0\n",
        encoding="utf-8",
    )
    script.chmod(script.stat().st_mode | stat.S_IEXEC)
    executor = KubernetesExecutor(image="example.com/app:1.0", kubectl_bin=str(script))
    with pytest.raises(DerivationError, match="out of memory"):
        executor.run(
            _derivation("def total(ctx):\n    return []\n"), Context(inputs={})
        )


def test_inputs_too_large_for_the_environment_are_refused_with_the_reason(
    fake_kubectl: Path,
) -> None:
    # A job travels in an environment variable, so there is a size past which this
    # transport cannot carry it. Refusing names the alternative; truncating would
    # corrupt the run silently.
    executor = KubernetesExecutor(
        image="example.com/app:1.0", kubectl_bin=str(fake_kubectl)
    )
    # Distinct values on purpose: pickle memoises identical strings, so a payload of one
    # repeated value compresses to nothing and would not exercise the bound at all.
    huge = Context(inputs={"rows": [{"v": f"{i:06d}" * 20} for i in range(1500)]})
    with pytest.raises(DerivationError, match="bind the data as a warehouse table"):
        executor.run(_derivation("def total(ctx):\n    return []\n"), huge)


def test_a_derivation_without_source_is_not_this_executors_job(
    fake_kubectl: Path,
) -> None:
    executor = KubernetesExecutor(
        image="example.com/app:1.0", kubectl_bin=str(fake_kubectl)
    )
    with pytest.raises(DerivationError, match="runs agent-authored derivations only"):
        executor.run(Derivation(name="trusted", compute=None), Context(inputs={}))


def test_an_invalid_dependency_is_refused_before_any_job_exists(
    fake_kubectl: Path, tmp_path: Path
) -> None:
    executor = KubernetesExecutor(
        image="example.com/app:1.0", kubectl_bin=str(fake_kubectl)
    )
    bad = Derivation(
        name="total",
        compute=None,
        source="def total(ctx):\n    return []\n",
        deps=("polars; rm -rf /",),
    )
    with pytest.raises(DerivationError, match="invalid dependency"):
        executor.run(bad, Context(inputs={}))
    assert not (tmp_path / "calls").exists()
