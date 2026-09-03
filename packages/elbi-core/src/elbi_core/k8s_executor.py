"""Run agent-written derivation code as a Kubernetes Job.

Notebooks and agent-written *candidate* code both belong on the session runner; only the
oracle and the trusted executor stay server-side. Selecting the Kubernetes runner
therefore gives candidate code a pod of its own, the same isolation the deployment chose
for notebooks, rather than running it as a child of the app process.

**A Job, not a bare Pod.** Kubernetes' own guidance is to "use a Job rather than a bare
Pod, even if your application requires only a single Pod", because a bare Pod on a
failed node is simply gone. The Job carries ``backoffLimit: 0`` all the same: a
derivation
candidate is deterministic, so re-running identical code produces the identical failure
at twice the cost, and the failure *is* the signal the agent reads to fix its code.
``ttlSecondsAfterFinished`` cleans up without a reaper.

**The child is unchanged.** :mod:`elbi_core._sandbox_child` reads a pickled job
from a directory and writes ``result.json`` beside it, and that contract is kept
exactly. The job arrives base64-encoded in the environment and is decoded into an
``emptyDir``
before the child runs; the result is read back from the pod's log, after a sentinel.
Redesigning the IO channel of the one process that executes untrusted code, in order to
gain a runner, would trade a large risk for a small convenience.
"""

from __future__ import annotations

import base64
import json
import pickle
import subprocess
import uuid
from typing import Any

from .artifact import Artifact
from .container import _CORE_DEPS, _DEFAULT_IMAGE
from .context import Context
from .derivation import Derivation
from .errors import DerivationError
from .executor import _DEP_RE

#: Where the job and its result live in the pod, mirroring the docker backend's mount.
_IO_PATH = "/io"

#: Marks the start of the result in the pod's log. The child folds a user's stdout into
#: its result for exploration jobs, but a derivation that prints writes to the real
#: stream, so the result is taken from after the *last* sentinel rather than assuming
#: the log holds nothing else.
_SENTINEL = "---elbi-result---"

#: A job travels to the pod in an environment variable, which bounds how large its
#: inputs may be. Kubernetes sets no limit of its own; Linux's ``MAX_ARG_STRLEN`` puts
#: it near 128 KiB per variable, so this is deliberately conservative, and refused
#: loudly rather than truncated.
_MAX_JOB_BYTES = 96 * 1024


class KubernetesExecutor:
    """Execute a derivation's source in a Job, on the same runner notebooks use."""

    def __init__(
        self,
        *,
        image: str | None = None,
        namespace: str | None = None,
        timeout: float = 300.0,
        service_account: str | None = None,
        kubectl_bin: str = "kubectl",
        allow_network: bool = False,
    ) -> None:
        self._image = image or _DEFAULT_IMAGE
        self._namespace = namespace
        self._timeout = timeout
        self._service_account = service_account
        self._kubectl = kubectl_bin
        self._allow_network = allow_network

    def run(self, derivation: Derivation, context: Context) -> Any:
        """Execute the derivation's source in a Job and return its artifact."""
        if derivation.source is None:
            raise DerivationError(
                f"derivation {derivation.name!r} has no source; the kubernetes "
                "executor runs agent-authored derivations only"
            )
        for dep in derivation.deps:
            if not _DEP_RE.match(dep):
                raise DerivationError(
                    f"refusing to provision invalid dependency {dep!r}"
                )
        job = {
            "source": derivation.source,
            "fn_name": derivation.name,
            "inputs": dict(context.inputs),
            "params": dict(context.params),
            "allow_network": self._allow_network,
        }
        result = self._run_job(job, tuple(derivation.deps), derivation.name)
        if not result.get("ok"):
            raise DerivationError(
                f"derivation {derivation.name!r} failed in sandbox: "
                f"{result.get('error')}"
            )
        return Artifact(kind=result["kind"], value=result["value"])

    # -- mechanics ---------------------------------------------------------------
    def _kube(self, *args: str) -> list[str]:
        namespace = ["-n", self._namespace] if self._namespace else []
        return [self._kubectl, *args[:1], *namespace, *args[1:]]

    def _run_job(
        self, job: dict[str, Any], deps: tuple[str, ...], label: str
    ) -> dict[str, Any]:
        """Create the Job, wait for it, and read the child's result from its log."""
        encoded = base64.b64encode(pickle.dumps(job)).decode("ascii")
        if len(encoded) > _MAX_JOB_BYTES:
            raise DerivationError(
                f"the job for {label!r} is {len(encoded) // 1024} KiB, over the "
                f"{_MAX_JOB_BYTES // 1024} KiB a pod's environment carries. Its inputs "
                "are too large to send this way: bind the data as a warehouse table so "
                "the derivation reads it rather than receiving it."
            )
        name = f"elbi-run-{uuid.uuid4().hex[:16]}"
        manifest = self._manifest(name, encoded, deps)
        created = subprocess.run(  # noqa: S603
            self._kube("create", "-f", "-"),
            input=json.dumps(manifest),
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
        )
        if created.returncode != 0:
            raise DerivationError(
                f"could not create the sandbox job for {label!r}: "
                f"{(created.stderr or '').strip()}"
            )
        try:
            return self._await_result(name, label)
        finally:
            # Deleted explicitly as well as by the TTL: a job left behind holds a
            # scheduling slot, and the TTL is a backstop for an app that crashed rather
            # than the normal path.
            subprocess.run(  # noqa: S603
                self._kube(
                    "delete", f"job/{name}", "--ignore-not-found", "--wait=false"
                ),
                capture_output=True,
                check=False,
                timeout=30.0,
            )

    def _await_result(self, name: str, label: str) -> dict[str, Any]:
        """Wait for the Job to finish, then parse the result out of its log."""
        waited = subprocess.run(  # noqa: S603
            self._kube(
                "wait",
                f"job/{name}",
                "--for=condition=Complete",
                f"--timeout={int(self._timeout)}s",
            ),
            capture_output=True,
            text=True,
            timeout=self._timeout + 30.0,
            check=False,
        )
        logs = subprocess.run(  # noqa: S603
            self._kube("logs", f"job/{name}", "--tail=-1"),
            capture_output=True,
            text=True,
            timeout=60.0,
            check=False,
        ).stdout
        if _SENTINEL in logs:
            # After the *last* sentinel: a derivation that prints writes to the same
            # stream, so an earlier occurrence in user output cannot win.
            payload = logs.rsplit(_SENTINEL, 1)[1].strip()
            try:
                parsed: dict[str, Any] = json.loads(payload)
            except ValueError as exc:
                raise DerivationError(
                    f"the sandbox job for {label!r} produced an unreadable result: "
                    f"{payload[:200]}"
                ) from exc
            return parsed
        # No sentinel: the child never got far enough to write one, so the log is the
        # only evidence and belongs in the error rather than in a cluster nobody checks.
        detail = (logs or waited.stderr or "").strip()[-2000:]
        raise DerivationError(
            f"the sandbox job for {label!r} produced no result. "
            f"{'Its output was: ' + detail if detail else 'It produced no output.'}"
        )

    def _manifest(
        self, name: str, encoded_job: str, deps: tuple[str, ...]
    ) -> dict[str, Any]:
        """The Job that runs one derivation."""
        packages = [*_CORE_DEPS, *deps]
        install = (
            f"pip install --no-cache-dir --target {_IO_PATH}/deps "
            + " ".join(packages)
            + " && "
            if deps
            else ""
        )
        # The child is invoked exactly as the docker backend invokes it, against a
        # directory holding the job, then the result is echoed after a sentinel so it
        # can be read back from the log.
        command = (
            f"{install}"
            f'printf %s "$ELBI_JOB" | base64 -d > {_IO_PATH}/job.pkl && '
            f"python -m elbi_core._sandbox_child {_IO_PATH}; "
            # The sentinel goes through %s rather than into the format string: it starts
            # with dashes, and printf reads a leading dash as an option.
            f"printf '\\n%s\\n' '{_SENTINEL}'; cat {_IO_PATH}/result.json"
        )
        spec: dict[str, Any] = {
            "restartPolicy": "Never",
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": 1000,
                "runAsGroup": 1000,
                "fsGroup": 1000,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "containers": [
                {
                    "name": "sandbox",
                    "image": self._image,
                    "command": ["sh", "-c", command],
                    "workingDir": _IO_PATH,
                    "env": [
                        {"name": "ELBI_JOB", "value": encoded_job},
                        {"name": "PYTHONUNBUFFERED", "value": "1"},
                        {"name": "HOME", "value": _IO_PATH},
                        *(
                            [{"name": "PYTHONPATH", "value": f"{_IO_PATH}/deps"}]
                            if deps
                            else []
                        ),
                    ],
                    "securityContext": {
                        "allowPrivilegeEscalation": False,
                        "capabilities": {"drop": ["ALL"]},
                        # The job writes only into the mounted scratch, so the image
                        # itself stays immutable.
                        "readOnlyRootFilesystem": True,
                    },
                    "volumeMounts": [{"name": "io", "mountPath": _IO_PATH}],
                }
            ],
            "volumes": [{"name": "io", "emptyDir": {}}],
        }
        if self._service_account:
            spec["serviceAccountName"] = self._service_account
        return {
            "apiVersion": "batch/v1",
            "kind": "Job",
            "metadata": {
                "name": name,
                "labels": {
                    "app.kubernetes.io/name": "elbi-sandbox",
                    "app.kubernetes.io/managed-by": "elbi",
                    # The same label the kernel pods carry, so one NetworkPolicy governs
                    # both: candidate code has no more reason to reach the network than
                    # a notebook cell does.
                    "intelligible.ai/egress": "full" if self._allow_network else "none",
                },
            },
            "spec": {
                # Zero retries. A derivation candidate is deterministic, so a retry
                # produces the same failure at twice the cost, and the failure is what
                # the agent reads in order to fix its code.
                "backoffLimit": 0,
                "activeDeadlineSeconds": int(self._timeout),
                # Cleans up without a reaper; the explicit delete is the normal path and
                # this covers an app that died holding the job.
                "ttlSecondsAfterFinished": 300,
                "template": {"metadata": spec.pop("_meta", {}), "spec": spec},
            },
        }
