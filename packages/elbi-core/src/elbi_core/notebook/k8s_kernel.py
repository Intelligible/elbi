"""A notebook kernel that runs as a pod, sized and isolated by a compute profile.

This is the runner a deployment is meant to use. The reason is the self-hosting one: a
customer's cluster already has capacity, an autoscaler, GPU node pools and, through
``runtimeClassName``, gVisor or Kata isolation, so the sandbox boundary can be as strong
as their platform allows without this product shipping a hypervisor. It is the same
shape JupyterHub's KubeSpawner uses, for the same reasons.

**How the protocol gets in and out.** The kernel speaks newline-delimited JSON on stdin
and stdout, and a pod's stdin is reachable exactly one way: attach. So a session is a
pod created with ``stdin`` open, waited for, then attached to; the attach's pipes are
the protocol. ``stdinOnce`` is deliberately *off*: it would let the container exit with
its one attach, but it also leaves stdin empty until a client attaches, so the worker
reads EOF at startup and exits before the attach arrives. A kernel's lifetime is bounded
by the delete in ``close`` and by ``activeDeadlineSeconds`` instead. A TTY is likewise
not allocated: a terminal would rewrite newlines and echo input, and the protocol is
bytes rather than a session for a human.

**What it does not do.** Eagerly seeded datasets are unsupported here, because they
travel as a file the docker backend bind-mounts from the host and a pod has no host to
mount from. That is not a gap so much as the point of the lazy data path: a kernel that
fetches what a cell asks for needs no such file. This runner requires it.

``kubectl`` is the client, rather than a Kubernetes API library. The docker backend
shells out to ``docker`` for the same reason: the binary is the interface an operator
already has configured, in-cluster and out, with their own auth plugins, contexts and
proxies working. It also keeps this package dependency-free.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import shlex
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ..container import _CORE_DEPS, _DEFAULT_IMAGE, _READY_SENTINEL
from ..errors import DerivationError
from ..executor import _DEP_RE
from ..sandbox import ComputeProfile, kubernetes_resources
from .kernel import MAX_OUTPUT_BYTES, _StreamingKernel

#: Where the deps volume mounts, matching the docker backend so the worker's environment
#: is identical on both.
_DEPS_PATH = "/deps"

#: A pod that outlives this is deleted by Kubernetes whatever the app believes. The
#: backstop for a leaked session: an app that crashed between creating a pod and
#: recording it would otherwise leave compute running until someone noticed the bill.
_DEFAULT_SESSION_DEADLINE = 12 * 3600

#: How long to wait for a pod to become ready before giving up. Generous because it
#: covers an image pull on a cold node, which is the part that actually takes minutes;
#: Kubernetes' own scheduling SLO explicitly excludes it.
_DEFAULT_START_TIMEOUT = 300.0


def worker_bootstrap() -> str:
    """Python that locates the installed kernel worker and runs it by path.

    The worker is written to be launched as a script (its sibling modules import as
    top-level names), so it cannot simply be ``-m``'d, and the path it lives at inside
    an image is not the path it lives at here. Resolving the spec in the pod covers
    both.
    """
    return (
        "import importlib.util,runpy,sys;"
        "s=importlib.util.find_spec('elbi_core.notebook._kernel_worker');"
        "sys.argv=[s.origin]+sys.argv[1:];"
        "runpy.run_path(s.origin,run_name='__main__')"
    )


def _kernel_image(image: str | None) -> str:
    """The image a kernel pod runs, refusing one that cannot run the worker.

    There is deliberately no default. The worker is launched from the installed
    ``elbi_core`` package, and a pod cannot bind-mount the source the way the docker
    backend does, so a plain Python image starts, fails to import, and exits, which
    reaches the user as an unexplained dead kernel. Refusing at construction turns that
    into one sentence naming the cause.
    """
    if not image:
        raise DerivationError(
            "the kubernetes runner needs an image with elbi-core installed "
            "and none was configured. A bare Python image cannot run the worker: "
            "unlike the docker backend, a pod has no host source to mount. Set "
            "compute.image (the app's own image works) or an image per profile."
        )
    if image == _DEFAULT_IMAGE:
        raise DerivationError(
            f"the kernel image is {image!r}, which has no elbi-core and so "
            "cannot run the kernel worker. Set compute.image to an image that does: "
            "the app's own image is the obvious choice."
        )
    return image


def deps_digest(image: str, deps: Sequence[str]) -> str:
    """Content address for a dependency set, matching the docker backend's volume name.

    The same image and the same packages give the same digest on both runners, which is
    what lets an environment be provisioned once and named the same way wherever it is
    reused.
    """
    packages = [*_CORE_DEPS, *deps]
    payload = "\n".join([image, *sorted(packages)])
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class KubernetesNotebookKernel(_StreamingKernel):
    """A notebook kernel running as a pod, attached over ``kubectl``."""

    def __init__(
        self,
        *,
        profile: ComputeProfile,
        deps: Sequence[str] = (),
        # Accepted and ignored: the common kernel arguments include a host scratch path,
        # which cannot mean anything to a pod. Use ``workspace_claim`` for a workspace
        # that persists.
        workspace: Path | None = None,
        data: dict[str, list[dict[str, Any]]] | None = None,
        dataset_names: Sequence[str] | None = None,
        cell_timeout: float = 120.0,
        max_output_bytes: int = MAX_OUTPUT_BYTES,
        image: str | None = None,
        namespace: str | None = None,
        service_account: str | None = None,
        deps_storage_class: str | None = None,
        workspace_claim: str | None = None,
        credentials: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        session_deadline: int = _DEFAULT_SESSION_DEADLINE,
        start_timeout: float = _DEFAULT_START_TIMEOUT,
        kubectl_bin: str = "kubectl",
    ) -> None:
        if cell_timeout <= 0:
            raise ValueError("cell timeout must be positive")
        for dep in deps:
            if not _DEP_RE.match(dep):
                raise DerivationError(
                    f"refusing to provision invalid dependency {dep!r}"
                )
        if data:
            raise DerivationError(
                "the kubernetes runner cannot seed a kernel with pre-loaded datasets: "
                "a pod has no host filesystem to read them from. Configure the "
                "warehouse query resolver so notebooks fetch data on demand."
            )
        self._cell_timeout = cell_timeout
        self._max_output_bytes = max_output_bytes
        self._alive = True
        self._kubectl = kubectl_bin
        self._namespace = namespace
        self._name = f"elbi-nb-{uuid.uuid4().hex[:16]}"
        self._profile = profile
        manifest = self._manifest(
            profile=profile,
            deps=tuple(deps),
            dataset_names=tuple(dataset_names or ()),
            image=_kernel_image(image or profile.image),
            workspace_claim=workspace_claim,
            service_account=service_account,
            deps_storage_class=deps_storage_class,
            credentials=credentials,
            env=env,
            session_deadline=session_deadline,
        )
        self._create(manifest, start_timeout)
        self._proc = self._attach()
        self._start_reader()

    # -- lifecycle ---------------------------------------------------------------
    def _kube(self, *args: str) -> list[str]:
        """A kubectl argument list with the namespace applied, if one was given."""
        namespace = ["-n", self._namespace] if self._namespace else []
        return [self._kubectl, *args[:1], *namespace, *args[1:]]

    def _run(self, argv: list[str], what: str, *, timeout: float = 60.0) -> str:
        """Run a kubectl command, raising with its stderr when it fails."""
        try:
            result = subprocess.run(  # noqa: S603
                argv, capture_output=True, text=True, timeout=timeout, check=False
            )
        except FileNotFoundError as exc:
            raise DerivationError(
                f"kubectl executable {self._kubectl!r} not found; install kubectl or "
                "select another sandbox backend"
            ) from exc
        except subprocess.TimeoutExpired as exc:
            raise DerivationError(f"timed out trying to {what}") from exc
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "").strip()
            raise DerivationError(f"could not {what}: {detail}")
        return result.stdout

    def _create(self, manifest: dict[str, Any], start_timeout: float) -> None:
        """Create the pod and wait for it to be ready, cleaning up if it never is."""
        argv = self._kube("create", "-f", "-")
        try:
            result = subprocess.run(  # noqa: S603
                argv,
                input=json.dumps(manifest),
                capture_output=True,
                text=True,
                timeout=60.0,
                check=False,
            )
        except FileNotFoundError as exc:
            self._alive = False
            raise DerivationError(
                f"kubectl executable {self._kubectl!r} not found; install kubectl or "
                "select another sandbox backend"
            ) from exc
        if result.returncode != 0:
            self._alive = False
            raise DerivationError(
                f"could not create the kernel pod: {(result.stderr or '').strip()}"
            )
        try:
            self._run(
                self._kube(
                    "wait",
                    f"pod/{self._name}",
                    "--for=condition=Ready",
                    f"--timeout={int(start_timeout)}s",
                ),
                "start the kernel pod",
                timeout=start_timeout + 30.0,
            )
        except DerivationError as exc:
            # Read why *before* deleting. The delete is right (a pod that never became
            # ready is still holding a scheduling slot and may still be pulling an
            # image), but doing it first destroys the only evidence, leaving an operator
            # with "timed out waiting for the condition" and nothing to act on.
            detail = self._not_ready_reason()
            self._delete_pod()
            self._alive = False
            raise DerivationError(f"{exc}{detail}") from exc

    def _not_ready_reason(self) -> str:
        """Why the pod is not ready: its container states and the events about it.

        Container state names the common cases outright (``ImagePullBackOff`` for an
        image the node cannot get, ``CrashLoopBackOff`` for one that starts and dies)
        and events cover causes outside the pod, like no node having the resources the
        profile asked for.

        Read as JSON and picked apart here rather than with a ``jsonpath`` template.
        Templates need their own escaping, and a diagnostic that fails to parse its own
        query reports nothing at the moment it is most needed.
        """
        raw = self._run_quietly(self._kube("get", f"pod/{self._name}", "-o", "json"))
        parts = []
        if raw:
            try:
                pod = json.loads(raw)
            except ValueError:  # pragma: no cover - kubectl emits valid JSON or nothing
                pod = {}
            status = pod.get("status", {})
            for key in ("initContainerStatuses", "containerStatuses"):
                for container in status.get(key) or []:
                    waiting = (container.get("state") or {}).get("waiting") or {}
                    terminated = (container.get("state") or {}).get("terminated") or {}
                    detail = waiting or terminated
                    if detail:
                        reason = detail.get("reason") or "unknown"
                        message = (detail.get("message") or "").strip()
                        parts.append(
                            f"{container.get('name', '?')}: {reason}"
                            + (f": {message}" if message else "")
                        )
        events = self._run_quietly(
            self._kube(
                "get",
                "events",
                "--field-selector",
                f"involvedObject.name={self._name}",
                "-o",
                "json",
            )
        )
        if events:
            try:
                items = json.loads(events).get("items") or []
            except ValueError:  # pragma: no cover - as above
                items = []
            for item in items[-3:]:
                if item.get("type") == "Normal":
                    continue  # scheduling chatter; only the problems are useful here
                parts.append(
                    f"{item.get('reason')}: {(item.get('message') or '').strip()}"
                )
        return ("\n  " + "\n  ".join(parts)) if parts else ""

    def _run_quietly(self, argv: list[str]) -> str:
        """Run a kubectl read, returning its output or empty on any failure.

        Used only to enrich an error already being raised, so a second failure here must
        not replace the first one with something less useful.
        """
        try:
            result = subprocess.run(  # noqa: S603
                argv, capture_output=True, text=True, timeout=20.0, check=False
            )
        except (OSError, subprocess.SubprocessError):
            return ""
        return (result.stdout or "").strip()

    def _attach(self) -> subprocess.Popen[str]:
        """Attach to the pod, giving the protocol its stdin and stdout.

        ``-q`` suppresses kubectl's own prompt messages, which would otherwise land in
        the middle of the JSON stream. The reader ignores lines that do not parse, so a
        stray message from the client is survivable rather than fatal, but not printing
        it in the first place is better.
        """
        argv = self._kube("attach", f"pod/{self._name}", "-c", "kernel", "-i", "-q")
        try:
            return subprocess.Popen(  # noqa: S603
                argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True
            )
        except FileNotFoundError as exc:  # pragma: no cover - _create would have failed
            self._delete_pod()
            self._alive = False
            raise DerivationError(f"could not attach to the kernel pod: {exc}") from exc

    @property
    def alive(self) -> bool:
        """Whether the attached session is still running."""
        return self._alive and self._proc.poll() is None

    def _death_reason(self) -> str:
        """Why the pod stopped, from its own logs.

        Without this a kernel that died on startup reports only ``dead`` and the
        explanation sits in ``kubectl logs`` that nobody reads. The commonest cause
        is an image that cannot run the worker, and the traceback saying so is there.
        """
        result = subprocess.run(  # noqa: S603
            self._kube("logs", f"pod/{self._name}", "-c", "kernel", "--tail=20"),
            capture_output=True,
            text=True,
            check=False,
            timeout=30.0,
        )
        return (result.stdout or result.stderr or "").strip()

    def interrupt(self) -> None:
        """Signal the worker to abandon its running cell, keeping the namespace.

        Delivered with Python rather than ``kill`` because Python is the one binary the
        image is guaranteed to have: a slim image may carry no shell utilities at all.
        """
        if not self.alive:
            return
        subprocess.run(  # noqa: S603
            self._kube(
                "exec",
                self._name,
                "-c",
                "kernel",
                "--",
                "python",
                "-c",
                "import os,signal; os.kill(1, signal.SIGINT)",
            ),
            capture_output=True,
            check=False,
            timeout=30.0,
        )

    def close(self, *, kill: bool = False) -> None:
        """End the attach and delete the pod.

        Closing stdin is the graceful path: the worker reads EOF and exits, and because
        the container was created with ``stdinOnce`` the pod ends with it. The delete is
        still issued, because a pod that fails to notice is a pod someone is paying for.
        """
        if self._alive and not kill and self._proc.stdin is not None:
            with contextlib.suppress(OSError, ValueError):
                self._proc.stdin.write(json.dumps({"shutdown": True}) + "\n")
                self._proc.stdin.flush()
        self._alive = False
        with contextlib.suppress(OSError, ValueError, subprocess.TimeoutExpired):
            self._proc.terminate()
            self._proc.wait(timeout=5)
        for stream in (self._proc.stdin, self._proc.stdout):
            if stream is not None:
                with contextlib.suppress(OSError, ValueError):
                    stream.close()
        self._delete_pod(grace=0 if kill else None)

    def _delete_pod(self, *, grace: int | None = None) -> None:
        """Delete the pod, not waiting for it and not failing if it is already gone."""
        args = ["delete", f"pod/{self._name}", "--ignore-not-found", "--wait=false"]
        if grace is not None:
            args.append(f"--grace-period={grace}")
        subprocess.run(  # noqa: S603
            self._kube(*args), capture_output=True, check=False, timeout=30.0
        )

    # -- the manifest ------------------------------------------------------------
    def _manifest(
        self,
        *,
        profile: ComputeProfile,
        deps: tuple[str, ...],
        dataset_names: tuple[str, ...],
        image: str,
        workspace_claim: str | None,
        service_account: str | None,
        deps_storage_class: str | None,
        credentials: Mapping[str, str] | None,
        env: Mapping[str, str] | None,
        session_deadline: int,
    ) -> dict[str, Any]:
        """The pod this session runs as."""
        digest = deps_digest(image, deps)
        pod_env = [
            # Without this the worker's stdout is block-buffered when it is a pipe, and
            # a cell's output arrives only when 8 KB have accumulated or the cell ends.
            {"name": "PYTHONUNBUFFERED", "value": "1"},
            {"name": "MPLBACKEND", "value": "Agg"},
            {"name": "HOME", "value": "/work"},
        ]
        if deps:
            # Only when there is something at /deps to import. Setting it always
            # *replaces* whatever PYTHONPATH the image had, and an image that exposes
            # its own packages that way then cannot import them -- including
            # elbi-core, which is exactly what this runner needs.
            pod_env.append({"name": "PYTHONPATH", "value": _DEPS_PATH})
        if credentials:
            # Scoped to this session's prefixes and short-lived, so the kernel reads the
            # warehouse itself rather than every byte crossing the app. Passed as
            # environment rather than a Secret because a Secret would outlive the pod
            # and need its own deletion, and this credential expires on its own.
            pod_env.extend(
                {"name": key, "value": value}
                for key, value in sorted(credentials.items())
            )
        if env:
            # Plain configuration, unlike `credentials`: nothing here expires.
            pod_env.extend(
                {"name": key, "value": value} for key, value in sorted(env.items())
            )
        if dataset_names:
            pod_env.append(
                {
                    "name": "ELBI_DATASET_NAMES",
                    "value": json.dumps(list(dataset_names)),
                }
            )
        volumes: list[dict[str, Any]] = [
            {"name": "work", "emptyDir": {}},
            {"name": "tmp", "emptyDir": {}},
            _deps_volume(digest, deps_storage_class),
        ]
        mounts = [
            {"name": "work", "mountPath": "/work"},
            {"name": "tmp", "mountPath": "/tmp"},  # noqa: S108
            # Read-only in the kernel; writable only in the init container that
            # fills it. Provisioning is the one step with a network, and a kernel that
            # could write here could rewrite a shared, content-addressed dependency
            # tree other sessions import -- the property the docker backend keeps by
            # mounting its volume `:ro`.
            {"name": "deps", "mountPath": _DEPS_PATH, "readOnly": True},
        ]
        spec: dict[str, Any] = {
            "restartPolicy": "Never",
            # The kernel runs whatever a notebook contains. A mounted service-account
            # token would hand that code an identity in the cluster, which is the single
            # most valuable thing in the pod.
            "automountServiceAccountToken": False,
            "enableServiceLinks": False,
            "activeDeadlineSeconds": session_deadline,
            "terminationGracePeriodSeconds": 5,
            "securityContext": {
                "runAsNonRoot": True,
                "runAsUser": 1000,
                "runAsGroup": 1000,
                "fsGroup": 1000,
                "seccompProfile": {"type": "RuntimeDefault"},
            },
            "containers": [
                {
                    "name": "kernel",
                    "image": image,
                    "command": ["python", "-c", worker_bootstrap(), "--allow-network"],
                    "workingDir": "/work",
                    # stdin stays open across attaches. `stdinOnce: true` reads more
                    # tidily (the container would exit when its one attach ended), but
                    # it means stdin is *empty* until a client attaches, so the worker
                    # reads EOF the moment it starts and exits before the attach lands.
                    # That race is lost often enough on a real cluster to make the
                    # runner unusable, and when lost it surfaces as "container kernel
                    # not found in pod". The session is bounded instead by the explicit
                    # delete in `close` and by `activeDeadlineSeconds`.
                    "stdin": True,
                    "stdinOnce": False,
                    "tty": False,
                    "env": pod_env,
                    "resources": kubernetes_resources(profile),
                    "securityContext": _container_security_context(),
                    "volumeMounts": mounts,
                }
            ],
            "volumes": volumes,
        }
        if deps:
            spec["initContainers"] = [_deps_installer(image, deps, digest)]
        if service_account:
            spec["serviceAccountName"] = service_account
        if profile.runtime_class:
            spec["runtimeClassName"] = profile.runtime_class
        selector, tolerations = _placement(profile)
        if selector:
            spec["nodeSelector"] = selector
        if tolerations:
            spec["tolerations"] = tolerations
        if workspace_claim:
            # An operator-provisioned claim, named explicitly. A host scratch path means
            # nothing to a pod on another node, so one is never translated into a claim
            # here: doing that would name a claim nobody created and leave every pod
            # unschedulable. Without a claim the workspace is an emptyDir, which is
            # honest about not surviving the session.
            volumes[0] = {
                "name": "work",
                "persistentVolumeClaim": {"claimName": workspace_claim},
            }
        return {
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": self._name,
                "labels": {
                    "app.kubernetes.io/name": "elbi-kernel",
                    "app.kubernetes.io/managed-by": "elbi",
                    "intelligible.ai/profile": profile.name,
                    # The label a NetworkPolicy selects on. Egress is enforced by
                    # the cluster rather than by the sandbox, so a kernel cannot lift
                    # its own restriction the way an in-process guard might be
                    # persuaded to.
                    "intelligible.ai/egress": _egress_label(profile.egress),
                },
            },
            "spec": spec,
        }


def _container_security_context() -> dict[str, Any]:
    """Least privilege for the container that runs a notebook's code."""
    return {
        "allowPrivilegeEscalation": False,
        "capabilities": {"drop": ["ALL"]},
        # Everything the kernel writes goes to a mounted emptyDir, so the image itself
        # can be immutable. This is the control that turns "it dropped a binary in
        # /usr/local/bin" into "it could not".
        "readOnlyRootFilesystem": True,
    }


def _egress_label(egress: str | Sequence[str]) -> str:
    """The value a NetworkPolicy matches on for this egress policy.

    An allowlist is a single label value (``restricted``) rather than a rendered list:
    label values cannot hold a hostname list, and the policy that implements the
    allowlist is a cluster object the chart ships. Losing the distinction between
    ``full`` and ``restricted`` in the port would silently open a sandbox, so the three
    cases are named explicitly here rather than defaulted.
    """
    if egress == "full":
        return "full"
    if egress == "none":
        return "none"
    return "restricted"


def _placement(profile: ComputeProfile) -> tuple[dict[str, str], list[dict[str, Any]]]:
    """Node selectors and tolerations implied by a profile's shape.

    A GPU pod has to tolerate the taint the device plugin puts on accelerator nodes, and
    interruptible capacity is labelled differently by every cloud, so the label is the
    one Kubernetes itself standardised (``node.kubernetes.io/instance-type`` has no spot
    equivalent, but the ``spot`` taint is near-universal), and an operator who needs
    another can express it through their own scheduling policy.
    """
    selector: dict[str, str] = {}
    tolerations: list[dict[str, Any]] = []
    if profile.gpu:
        tolerations.append(
            {
                "key": "nvidia.com/gpu",
                "operator": "Exists",
                "effect": "NoSchedule",
            }
        )
        if profile.gpu_type:
            selector["intelligible.ai/accelerator"] = profile.gpu_type
    if profile.spot:
        selector["intelligible.ai/capacity"] = "spot"
        for key in (
            "cloud.google.com/gke-spot",
            "kubernetes.azure.com/scalesetpriority",
        ):
            tolerations.append(
                {"key": key, "operator": "Exists", "effect": "NoSchedule"}
            )
    return selector, tolerations


def _deps_volume(digest: str, storage_class: str | None) -> dict[str, Any]:
    """The volume dependencies are installed into.

    With a storage class that supports ``ReadWriteMany`` the same claim is shared by
    every session using that dependency set, so the install happens once. Without one it
    is an ``emptyDir`` and each session pays the install: slower, but correct on every
    cluster, and the alternative would be a default that fails outright on the block
    storage most clusters actually have.
    """
    if storage_class is None:
        return {"name": "deps", "emptyDir": {}}
    return {
        "name": "deps",
        "ephemeral": {
            "volumeClaimTemplate": {
                "metadata": {"labels": {"intelligible.ai/deps": digest}},
                "spec": {
                    "accessModes": ["ReadWriteOnce"],
                    "storageClassName": storage_class,
                    "resources": {"requests": {"storage": "4Gi"}},
                },
            }
        },
    }


def _deps_installer(image: str, deps: Sequence[str], digest: str) -> dict[str, Any]:
    """An init container that populates the deps volume, skipping an already-full one.

    The sentinel is written last and checked first, which is what makes a shared volume
    safe to reuse: a half-finished install from an interrupted pod has no sentinel, so
    the next session installs again rather than importing a partial package tree.
    """
    packages = [*_CORE_DEPS, *deps]
    quoted = " ".join(shlex.quote(p) for p in packages)
    script = (
        f"test -f {_READY_SENTINEL} && exit 0; "
        f"pip install --no-cache-dir --target {_DEPS_PATH} {quoted} && "
        f"touch {_READY_SENTINEL}"
    )
    return {
        "name": "deps",
        "image": image,
        "command": ["sh", "-c", script],
        "env": [{"name": "ELBI_DEPS_DIGEST", "value": digest}],
        "securityContext": _container_security_context(),
        "volumeMounts": [
            {"name": "deps", "mountPath": _DEPS_PATH},
            {"name": "tmp", "mountPath": "/tmp"},  # noqa: S108
        ],
        # Small and fixed: this container only downloads and unpacks wheels, and giving
        # it the session's profile would reserve a GPU to run pip.
        "resources": {
            "requests": {"cpu": "500m", "memory": "1Gi"},
            "limits": {"cpu": "2", "memory": "4Gi"},
        },
    }
