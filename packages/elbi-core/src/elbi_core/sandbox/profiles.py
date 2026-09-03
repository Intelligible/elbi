"""The compute profile: a named, sized, governed shape a sandbox runs as."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ..errors import ElbiError


class ComputeProfileError(ElbiError):
    """A compute profile is invalid, unknown, or not available to this caller."""


#: Kubernetes CPU quantities: whole or fractional cores (``2``, ``0.5``) or millicores
#: (``500m``). Kubernetes refuses finer than ``1m``, and so do we.
_CPU_RE = re.compile(r"^(\d+(?:\.\d+)?)(m?)$")

#: Kubernetes memory quantities: bytes, with a binary (``Gi``) or decimal (``G``)
#: suffix. Note the case trap Kubernetes itself carries (``M`` is megabytes and ``m`` is
#: millibytes), so a lowercase suffix is rejected here rather than silently meaning a
#: fraction of one byte.
_MEMORY_RE = re.compile(r"^(\d+(?:\.\d+)?)(Ki|Mi|Gi|Ti|Pi|Ei|k|M|G|T|P|E)?$")

_MEMORY_FACTORS = {
    "": 1,
    "k": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
    "P": 1000**5,
    "E": 1000**6,
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "Pi": 1024**5,
    "Ei": 1024**6,
}

#: A profile name reaches a container name, a Kubernetes label value and a URL, so it is
#: held to the strictest of the three (RFC 1123 label) rather than sanitised per use.
_NAME_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")

#: A GPU class names a node label value (``nvidia-l4``, ``nvidia-a100-80gb``), so it is
#: held to the same shape as a name.
_GPU_TYPE_RE = _NAME_RE


def parse_cpu(value: str) -> float:
    """Cores as a float, from a Kubernetes CPU quantity (``"2"``, ``"500m"``)."""
    match = _CPU_RE.match(value.strip())
    if match is None:
        raise ComputeProfileError(
            f"cpu {value!r} is not a Kubernetes CPU quantity (e.g. '2' or '500m')"
        )
    cores = float(match.group(1))
    if match.group(2) == "m":
        cores /= 1000
    if cores <= 0:
        raise ComputeProfileError("cpu must be greater than zero")
    return cores


def parse_memory(value: str) -> int:
    """Bytes, from a Kubernetes memory quantity (``"8Gi"``, ``"512Mi"``)."""
    match = _MEMORY_RE.match(value.strip())
    if match is None:
        raise ComputeProfileError(
            f"memory {value!r} is not a Kubernetes memory quantity (e.g. '8Gi'). "
            "Note that a lowercase 'm' means millibytes to Kubernetes, not megabytes."
        )
    size = int(float(match.group(1)) * _MEMORY_FACTORS[match.group(2) or ""])
    if size <= 0:
        raise ComputeProfileError("memory must be greater than zero")
    return size


@dataclass(frozen=True, slots=True)
class CostRates:
    """What an hour of each resource is charged at, for governance arithmetic.

    An estimate used to compare profiles and to enforce a ceiling, not a bill.
    Databricks does the same thing with a synthetic unit (a DBU) rather than currency,
    and their policy reference is explicit that the equivalent attribute is a
    "calculated attribute representing the maximum DBUs a resource can use on an hourly
    basis": a governance number. Defaults are rough on-demand list prices; an operator
    running reserved instances or another region should set their own.
    """

    cpu_core_hour: float = 0.04
    memory_gib_hour: float = 0.005
    gpu_hour: float = 1.0
    #: Multiplier applied when a profile runs on interruptible capacity. Spot discounts
    #: vary by instance and region; this is the shape of the discount, not a quote.
    spot_multiplier: float = 0.35

    def per_hour(
        self, *, cores: float, memory_bytes: int, gpu: int, spot: bool
    ) -> float:
        """The estimated hourly cost of a shape."""
        total = (
            cores * self.cpu_core_hour
            + memory_bytes / 1024**3 * self.memory_gib_hour
            + gpu * self.gpu_hour
        )
        return round(total * (self.spot_multiplier if spot else 1.0), 4)


@dataclass(frozen=True, slots=True)
class ComputeProfile:
    """One selectable compute shape and its limits.

    Constructed through :meth:`from_mapping` when it comes from configuration, which
    validates; constructing directly is for tests and for the synthesized default.
    """

    name: str
    #: Kubernetes quantities, kept in that notation all the way to the runner.
    cpu: str = "2"
    memory: str = "4Gi"
    #: Whole GPUs. Single-GPU is the mainstream position rather than a compromise:
    #: Databricks' AI Runtime accelerators "provision a single node", Hex ships two
    #: profiles of one GPU each, and on Databricks a GPU forces dedicated access mode,
    #: so a GPU independently implies a single-user machine. Multi-GPU and distributed
    #: training are out of scope, and the docs say so.
    gpu: int = 0
    #: Node label value selecting the accelerator (``nvidia-l4``). Meaningful on the
    #: Kubernetes runner; ignored elsewhere.
    gpu_type: str | None = None
    #: Overrides the deployment's sandbox image for this profile, so a GPU profile can
    #: carry a CUDA userland without every profile paying for it.
    image: str | None = None
    #: Seconds of no execution and no attached client before the kernel is reaped. Hex's
    #: precedent is a per-project setting defaulting to an hour; 30 minutes is the
    #: tighter default here, raisable per profile.
    idle_timeout: float = 1800.0
    #: Seconds a single cell may run. A separate limit from the idle timeout because
    #: they answer different failure modes: one bounds a forgotten kernel, the other a
    #: runaway cell.
    max_runtime: float = 120.0
    #: ``"full"``, ``"none"``, or an allowlist of hosts. Never widens what the project
    #: allows; see :meth:`narrowed_to`.
    egress: str | tuple[str, ...] = "full"
    #: Interruptible capacity: cheap, and an eviction loses the kernel. Right for a
    #: large batch profile, wrong for an interactive default, so it is opt-in per
    #: profile.
    spot: bool = False
    #: Pre-started sandboxes held ready. Zero by default, because idle capacity is
    #: paid for whether or not anything runs on it.
    warm_pool_size: int = 0
    #: Kubernetes ``runtimeClassName``: ``gvisor`` on GKE Sandbox,
    #: ``kata-vm-isolation`` on AKS. The cluster supplies microVM-grade isolation, so
    #: nothing here has to ship a hypervisor.
    runtime_class: str | None = None
    #: Processes the sandbox may create, as a fork-bomb bound.
    pids: int = 512
    #: Share one kernel between every notebook on this profile, rather than giving each
    #: its own. Off by default and it should stay that way for anything holding governed
    #: data: one process holds one identity, so per-notebook scoped credentials become
    #: impossible and whatever is in that kernel is effectively available to everyone
    #: using it. Databricks says the same of their equivalent ("data or internal
    #: credentials provisioned to that environment might be accessible to any code
    #: running within that environment") and has since made that mode legacy and off by
    #: default. Offered for the case it is actually good for: a cheap shared scratch
    #: tier where the cost of a process per notebook is not worth paying.
    shared: bool = False
    #: Fields to leave out of what the app presents. Governance as interface
    #: simplification rather than only restriction: Databricks lists "simplify the user
    #: interface" among the *purposes* of a policy, not just its constraints, and a menu
    #: showing eight numbers per size is a menu nobody reads. Hiding a field changes
    #: what is shown, never what is enforced.
    hidden: tuple[str, ...] = ()
    #: Set when the profile came from configuration rather than being synthesized, so
    #: the UI can say a profile is defined by infrastructure and not editable in
    #: Settings.
    managed: bool = True

    def __post_init__(self) -> None:
        if not _NAME_RE.match(self.name):
            raise ComputeProfileError(
                f"profile name {self.name!r} must be lowercase alphanumeric with "
                "hyphens (it becomes a container name and a Kubernetes label)"
            )
        parse_cpu(self.cpu)
        parse_memory(self.memory)
        if self.gpu < 0:
            raise ComputeProfileError("gpu must not be negative")
        if self.gpu_type is not None and not _GPU_TYPE_RE.match(self.gpu_type):
            raise ComputeProfileError(f"gpu_type {self.gpu_type!r} is not a node label")
        if self.gpu_type is not None and self.gpu == 0:
            raise ComputeProfileError(
                f"profile {self.name!r} names a gpu_type but asks for 0 GPUs"
            )
        for label, seconds in (
            ("idle_timeout", self.idle_timeout),
            ("max_runtime", self.max_runtime),
        ):
            if seconds <= 0:
                raise ComputeProfileError(f"{label} must be positive")
        if self.warm_pool_size < 0:
            raise ComputeProfileError("warm_pool_size must not be negative")
        if self.pids <= 0:
            raise ComputeProfileError("pids must be positive")
        shown = set(self.to_dict()) - {"name"}
        unknown_hidden = sorted(set(self.hidden) - shown)
        if unknown_hidden:
            raise ComputeProfileError(
                f"profile {self.name!r} hides unknown field(s) "
                f"{', '.join(unknown_hidden)}; hideable: {', '.join(sorted(shown))}"
            )
        _validate_egress(self.egress, self.name)

    @property
    def cores(self) -> float:
        """CPU as a number of cores."""
        return parse_cpu(self.cpu)

    @property
    def memory_bytes(self) -> int:
        """Memory as bytes."""
        return parse_memory(self.memory)

    @property
    def version(self) -> str:
        """A content-addressed version of this definition.

        Changing any field changes the version, which is what makes profile drift
        detectable: a session records the version it started with, and a mismatch means
        the definition moved underneath it. Derived rather than hand-maintained, because
        a version an admin has to remember to bump is a version that silently does not.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]

    def cost_per_hour(self, rates: CostRates | None = None) -> float:
        """Estimated hourly cost of this shape under ``rates``."""
        return (rates or CostRates()).per_hour(
            cores=self.cores,
            memory_bytes=self.memory_bytes,
            gpu=self.gpu,
            spot=self.spot,
        )

    def narrowed_to(self, egress: str | Sequence[str]) -> ComputeProfile:
        """This profile with its egress intersected against a project's policy.

        Egress narrows and never widens: a project that denies the network cannot have a
        profile hand it back. The intersection is computed rather than the stricter of
        the two being picked, so two allowlists yield only the hosts both permit.
        """
        return replace(self, egress=_intersect_egress(self.egress, egress))

    def to_dict(self) -> dict[str, Any]:
        """The profile as plain data, for the API, the UI and the version hash."""
        return {
            "name": self.name,
            "cpu": self.cpu,
            "memory": self.memory,
            "gpu": self.gpu,
            "gpu_type": self.gpu_type,
            "image": self.image,
            "idle_timeout": self.idle_timeout,
            "max_runtime": self.max_runtime,
            "egress": (
                self.egress if isinstance(self.egress, str) else sorted(self.egress)
            ),
            "spot": self.spot,
            "warm_pool_size": self.warm_pool_size,
            "runtime_class": self.runtime_class,
            "pids": self.pids,
        }

    def presented(self, rates: CostRates | None = None) -> dict[str, Any]:
        """The profile as the app shows it: without hidden fields, with its version.

        Separate from :meth:`to_dict` because that one feeds the version hash and has to
        stay the whole definition: hiding a field must not silently change a profile's
        identity, or an admin simplifying the UI would look like an admin resizing the
        compute.
        """
        shown = {k: v for k, v in self.to_dict().items() if k not in self.hidden}
        return {
            **shown,
            "version": self.version,
            "cost_per_hour": self.cost_per_hour(rates),
        }

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any], *, context: str) -> ComputeProfile:
        """Build a profile from configuration, rejecting unknown keys.

        Both spellings of a field name are accepted, because a profile is written in two
        places whose conventions differ: ``idle_timeout`` in a YAML project file and
        ``idleTimeout`` in Helm values. Making an operator remember which is which would
        be a trap, and the failure would look like a rejected typo.

        Unknown keys are an error rather than ignored: a misspelled ``runtimeClas``
        silently dropped means an operator believes a sandbox is isolated when it is
        not, which is the worst way for a typo to fail.
        """
        allowed = {f.name for f in cls.__dataclass_fields__.values()} - {"managed"}
        raw = {_snake(key): value for key, value in raw.items()}
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ComputeProfileError(
                f"{context}: unknown profile field(s) {', '.join(unknown)}; "
                f"allowed: {', '.join(sorted(allowed))}"
            )
        if "name" not in raw:
            raise ComputeProfileError(f"{context}: a profile needs a name")
        values: dict[str, Any] = dict(raw)
        for key in ("hidden",):
            if key in values:
                values[key] = tuple(_string_list(values[key], f"{context}.{key}"))
        if isinstance(values.get("egress"), list):
            values["egress"] = tuple(
                _string_list(values["egress"], f"{context}.egress")
            )
        for key in ("idle_timeout", "max_runtime"):
            if key in values:
                values[key] = float(values[key])
        for key in ("gpu", "warm_pool_size", "pids"):
            if key in values:
                values[key] = int(values[key])
        for key in ("cpu", "memory"):
            if key in values:
                # Accept the YAML-natural `cpu: 2` as well as `cpu: "2"`, since a
                # quantity that happens to be an integer is what a person writes.
                values[key] = str(values[key])
        try:
            return cls(**values)
        except TypeError as exc:  # pragma: no cover - `allowed` already gates this
            raise ComputeProfileError(f"{context}: {exc}") from exc


def _snake(name: str) -> str:
    """``idleTimeout`` as ``idle_timeout``, leaving an already-snake name alone."""
    return re.sub(r"(?<=[a-z0-9])([A-Z])", r"_\1", name).lower()


def _string_list(raw: object, context: str) -> list[str]:
    """A list of non-empty strings, or an error naming where it came from."""
    if isinstance(raw, str) or not isinstance(raw, Iterable):
        raise ComputeProfileError(f"{context} must be a list of strings")
    items = [str(item).strip() for item in raw]
    if not all(items):
        raise ComputeProfileError(f"{context} must not contain empty strings")
    return items


def _validate_egress(egress: object, name: str) -> None:
    """Reject an egress policy that is neither a known word nor a host allowlist."""
    if isinstance(egress, str):
        if egress not in ("full", "none"):
            raise ComputeProfileError(
                f"profile {name!r}: egress must be 'full', 'none', or a list of hosts"
            )
        return
    if not isinstance(egress, tuple) or not egress:
        raise ComputeProfileError(
            f"profile {name!r}: an egress allowlist must be a non-empty list of hosts"
        )


def _intersect_egress(
    profile: str | tuple[str, ...], project: str | Sequence[str]
) -> str | tuple[str, ...]:
    """The narrower of two egress policies, intersecting two allowlists."""
    if profile == "none" or project == "none":
        return "none"
    if project == "full":
        return profile
    if profile == "full":
        return tuple(project) if not isinstance(project, str) else project
    shared = tuple(sorted(set(profile) & set(project)))
    # Two allowlists that share nothing permit nothing, which is 'none' rather than an
    # empty allowlist -- an empty list would read as "unset" to every runner downstream.
    return shared or "none"


@dataclass(frozen=True, slots=True)
class ComputeProfiles:
    """The menu of profiles a deployment offers, and the ceiling it enforces."""

    profiles: tuple[ComputeProfile, ...]
    default: str
    #: Refuse to offer a profile estimated above this hourly cost. Databricks'
    #: equivalent is enforced at creation as a range with a maxValue, and it is the
    #: single number their admins look for: "a direct way to control cost at the
    #: individual compute level".
    max_cost_per_hour: float | None = None
    rates: CostRates = field(default_factory=CostRates)

    def __post_init__(self) -> None:
        if not self.profiles:
            raise ComputeProfileError("at least one compute profile is required")
        names = [p.name for p in self.profiles]
        duplicates = sorted({n for n in names if names.count(n) > 1})
        if duplicates:
            raise ComputeProfileError(
                f"duplicate profile name(s): {', '.join(duplicates)}"
            )
        if self.default not in names:
            raise ComputeProfileError(
                f"default profile {self.default!r} is not defined; "
                f"available: {', '.join(sorted(names))}"
            )
        if self.max_cost_per_hour is not None:
            over = [
                f"{p.name} (~{p.cost_per_hour(self.rates)}/h)"
                for p in self.profiles
                if p.cost_per_hour(self.rates) > self.max_cost_per_hour
            ]
            if over:
                raise ComputeProfileError(
                    f"profile(s) exceed max_cost_per_hour={self.max_cost_per_hour}: "
                    f"{', '.join(over)}"
                )

    def __iter__(self) -> Any:
        return iter(self.profiles)

    def get(self, name: str) -> ComputeProfile:
        """The named profile, or an error listing what is defined."""
        for profile in self.profiles:
            if profile.name == name:
                return profile
        raise ComputeProfileError(
            f"unknown compute profile {name!r}; available: "
            f"{', '.join(p.name for p in self.profiles)}"
        )

    def select(self, name: str | None) -> ComputeProfile:
        """Resolve a requested profile, or raise.

        ``None`` selects the deployment default.
        """
        if name is None:
            return self.get(self.default)
        return self.get(name)

    @classmethod
    def from_config(
        cls,
        raw: object,
        *,
        context: str = "compute_profiles",
        default: str | None = None,
        max_cost_per_hour: float | None = None,
        rates: CostRates | None = None,
        profile_class: type[ComputeProfile] = ComputeProfile,
    ) -> ComputeProfiles:
        """Parse the profile menu from configuration (a list of mappings).

        ``profile_class`` builds each entry, so a deployment that adds a field of its
        own is parsed and validated by the same path rather than around it.
        """
        if not isinstance(raw, (list, tuple)) or not raw:
            raise ComputeProfileError(f"{context} must be a non-empty list of profiles")
        profiles = []
        for index, item in enumerate(raw):
            if not isinstance(item, Mapping):
                raise ComputeProfileError(f"{context}[{index}] must be a mapping")
            profiles.append(
                profile_class.from_mapping(item, context=f"{context}[{index}]")
            )
        return cls(
            profiles=tuple(profiles),
            default=default or profiles[0].name,
            max_cost_per_hour=max_cost_per_hour,
            rates=rates or CostRates(),
        )


def docker_resource_args(profile: ComputeProfile) -> dict[str, Any]:
    """Docker's flags for a profile's shape.

    Bytes rather than a re-rendered suffix, because Docker and Kubernetes disagree about
    what ``2g`` means (decimal versus binary) and a silent 7% difference in a memory
    limit is the kind of thing nobody finds.
    """
    args: dict[str, Any] = {
        "memory": f"{profile.memory_bytes}b",
        "cpus": f"{profile.cores:g}",
        "pids_limit": profile.pids,
    }
    if profile.gpu:
        # `--gpus` needs the NVIDIA container toolkit; a profile asking for a GPU on a
        # host without it fails at `docker run` with the runtime's own message, which is
        # more useful than anything guessed here.
        args["gpus"] = str(profile.gpu)
    return args


def kubernetes_resources(profile: ComputeProfile) -> dict[str, dict[str, str]]:
    """A pod container's ``resources`` block for a profile.

    Requests equal limits. For memory that is the only way to get the Guaranteed QoS
    class, which is what stops the kubelet evicting an analyst's kernel to make room;
    for CPU it means a profile's cores are reserved rather than borrowed, so a cell's
    runtime does not depend on what else landed on the node. A GPU is an extended
    resource, which Kubernetes requires be specified as a limit and only in integers.
    """
    limits = {"cpu": profile.cpu, "memory": profile.memory}
    requests = dict(limits)
    if profile.gpu:
        limits["nvidia.com/gpu"] = str(profile.gpu)
    return {"requests": requests, "limits": limits}
