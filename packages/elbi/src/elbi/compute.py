"""Where the compute menu comes from, and what a session on it cost.

**Resolution.** The profile menu is infrastructure, not product configuration: it comes
from the environment (Helm renders it) or from the project file, and it is read-only in
the app. That is the opposite of the LLM profiles, where the store wins because naming
a set of them in one variable is not possible.

**Attribution.** Databricks is candid that spend limits for compute are
notification-only and that they do "not proactively terminate resources to maintain the
limit", so that is what is promised here too: a usage record per session and an alert
when a window's spend passes a threshold. They are equally clear about the trap that
makes attribution urgent (missing tags "can't be added to past events"), so every
session is attributed from the first one rather than when someone asks.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy.exc import SQLAlchemyError

from elbi_core.config import ProjectConfig
from elbi_core.sandbox import (
    ComputeProfile,
    ComputeProfileError,
    ComputeProfiles,
    CostRates,
    CredentialError,
    broker_for,
    scope_for,
)

if TYPE_CHECKING:  # pragma: no cover - import cycle at runtime, fine for typing
    from .db import Store

logger = logging.getLogger(__name__)

#: JSON list of profile definitions, rendered by the chart from ``compute.profiles``.
PROFILES_ENV = "COMPUTE_PROFILES"
#: Name of the profile a session gets when it does not ask for one.
DEFAULT_PROFILE_ENV = "COMPUTE_DEFAULT_PROFILE"
#: Refuse to offer any profile estimated above this many currency units per hour.
MAX_COST_ENV = "COMPUTE_MAX_COST_PER_HOUR"
#: Per-resource hourly rates for the cost estimate, as a JSON object. Defaults are rough
#: on-demand list prices, which are wrong for anyone on reserved capacity.
RATES_ENV = "COMPUTE_COST_RATES"

#: The menu a deployment gets when it defines none. Three sizes, because the alternative
#: to a small menu is not "no menu" but every notebook silently taking the same box, and
#: because Databricks' own well-architected guidance names this shape directly: "define
#: T-shirt size policies (that is, Small, Medium, or Large)".  The floor is 8 GiB rather
#: than something tidier because that is where the field actually sits: Hex's default
#: kernel is 8 GB, and Databricks' agent sandbox ships fixed at 16 GB. A 4 GiB default
#: would have been a smaller box than anything a person arriving from either product has
#: used.
BUILTIN_PROFILES: tuple[Mapping[str, Any], ...] = (
    {"name": "small", "cpu": "2", "memory": "8Gi"},
    {"name": "medium", "cpu": "4", "memory": "16Gi"},
    {"name": "large", "cpu": "8", "memory": "32Gi"},
)


def resolve_profiles(config: ProjectConfig | None = None) -> ComputeProfiles:
    """The compute menu for this deployment.

    Environment first, then the project file, then the built-in sizes. Environment wins
    because the menu is a property of the infrastructure the app was deployed onto: an
    operator who sets it in Helm should not have it overridden by a file inside the
    image.
    """
    raw = os.environ.get(PROFILES_ENV, "").strip()
    source = PROFILES_ENV
    definitions: Sequence[Mapping[str, Any]]
    if raw:
        try:
            parsed = json.loads(raw)
        except ValueError as exc:
            raise ComputeProfileError(
                f"{PROFILES_ENV} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(parsed, list):
            raise ComputeProfileError(f"{PROFILES_ENV} must be a JSON list of profiles")
        definitions = parsed
    elif config is not None and config.compute_profiles:
        definitions, source = config.compute_profiles, "elbi.yaml"
    else:
        definitions, source = BUILTIN_PROFILES, "built-in defaults"

    default = os.environ.get(DEFAULT_PROFILE_ENV) or (
        config.default_compute_profile if config else None
    )
    max_cost = _float_env(MAX_COST_ENV)
    profiles = ComputeProfiles.from_config(
        definitions,
        context=source,
        default=default,
        max_cost_per_hour=max_cost,
        rates=_rates_from_env(),
    )
    logger.info(
        "compute profiles from %s: %s (default %s)",
        source,
        ", ".join(p.name for p in profiles),
        profiles.default,
    )
    return profiles


def _float_env(name: str) -> float | None:
    """A float from the environment, or None. A bad value is an error, not a zero."""
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ComputeProfileError(f"{name} must be a number, not {raw!r}") from exc


def _rates_from_env() -> CostRates:
    """Cost rates from ``COMPUTE_COST_RATES``, falling back to the defaults."""
    raw = os.environ.get(RATES_ENV, "").strip()
    if not raw:
        return CostRates()
    try:
        parsed = json.loads(raw)
    except ValueError as exc:
        raise ComputeProfileError(f"{RATES_ENV} is not valid JSON: {exc}") from exc
    if not isinstance(parsed, Mapping):
        raise ComputeProfileError(f"{RATES_ENV} must be a JSON object")
    allowed = set(CostRates.__dataclass_fields__)
    unknown = sorted(set(parsed) - allowed)
    if unknown:
        raise ComputeProfileError(
            f"{RATES_ENV}: unknown rate(s) {', '.join(unknown)}; "
            f"allowed: {', '.join(sorted(allowed))}"
        )
    return CostRates(**{k: float(v) for k, v in parsed.items()})


#: Which runner starts a kernel. An environment override, because where compute runs is
#: a property of the infrastructure the app was deployed onto rather than of the project
#: committed into it: the same reason the profile menu resolves this way.
RUNNER_ENV = "SANDBOX_BACKEND"
NAMESPACE_ENV = "COMPUTE_NAMESPACE"
DEPS_STORAGE_CLASS_ENV = "COMPUTE_DEPS_STORAGE_CLASS"
SESSION_DEADLINE_ENV = "COMPUTE_SESSION_DEADLINE_SECONDS"

#: Where *agent-written candidate code* runs, when that must differ from where notebook
#: kernels run. It must on Kubernetes: notebook kernels get a pod each, and a derivation
#: candidate has no pod-backed executor yet, so an operator picks between a container
#: per run and a hardened child of the app rather than getting the weaker one by
#: default. Databricks ships the same shape (their agent sandbox is a separate
#: environment from notebook compute, fixed in size), so the split is the norm rather
#: than a shortcut; the silent fallback was the problem.
EXECUTOR_BACKEND_ENV = "SANDBOX_EXECUTOR_BACKEND"

#: The image a kernel pod or container runs. Deployment-shaped rather than
#: project-shaped, for the same reason as the runner above: a chart knows the registry
#: it pushed to and the project committed into git does not, and the two disagree the
#: moment the same project is deployed twice.
IMAGE_ENV = "SANDBOX_IMAGE"

_RUNNERS = ("subprocess", "docker", "kubernetes")


def resolve_runner(config: ProjectConfig | None = None) -> str:
    """Which backend runs a kernel: the environment's answer, else the project's."""
    runner = os.environ.get(RUNNER_ENV, "").strip() or (
        config.sandbox if config else "subprocess"
    )
    if runner not in _RUNNERS:
        raise ComputeProfileError(
            f"{RUNNER_ENV}={runner!r} is not a runner; expected one of "
            f"{', '.join(_RUNNERS)}"
        )
    return runner


def resolve_image(config: ProjectConfig | None = None) -> str | None:
    """The image a kernel runs: the environment's answer, else the project's.

    ``None`` where neither says, which is correct for the subprocess runner and fatal
    for the other two: they raise with an explanation rather than guessing at a base
    image that could not run the worker anyway.
    """
    return os.environ.get(IMAGE_ENV, "").strip() or (
        config.sandbox_image if config else None
    )


#: Kubernetes namespaces are RFC 1123 labels, so a name that is not one cannot become
#: a namespace and quietly mangling it would land kernels somewhere unintended.
_NAMESPACE_RE = re.compile(r"^[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?$")


def namespace_for() -> str | None:
    """The kernel namespace, from ``COMPUTE_NAMESPACE``; ``None`` when unset.

    The namespace must already exist. Creating one would need cluster-scoped rights the
    app deliberately does not hold; its Role is namespaced, covering pods and nothing
    else.
    """
    template = os.environ.get(NAMESPACE_ENV, "").strip()
    if not template:
        return None
    if not _NAMESPACE_RE.match(template):
        raise ComputeProfileError(
            f"{template!r} is not a valid Kubernetes namespace. Namespaces are "
            "lowercase alphanumeric with hyphens."
        )
    return template


def executor_backend(config: ProjectConfig | None = None) -> str:
    """Which backend runs agent-written candidate code.

    Follows the notebook runner unless that runner has no executor for a one-shot
    derivation, which is the Kubernetes case. There the operator must choose, because
    both remaining options are weaker than what they selected for notebooks and picking
    for them would hide that.
    """
    explicit = os.environ.get(EXECUTOR_BACKEND_ENV, "").strip()
    if explicit:
        if explicit not in ("subprocess", "docker"):
            raise ComputeProfileError(
                f"{EXECUTOR_BACKEND_ENV}={explicit!r} must be 'docker' or 'subprocess'"
            )
        return explicit
    return resolve_runner(config)


def runner_options() -> dict[str, Any]:
    """Cluster settings the Kubernetes runner needs, from the environment.

    Only what an operator sets is returned, so the runner's own defaults stand for the
    rest and an unset value never arrives as an empty string a cluster would reject.
    """
    options: dict[str, Any] = {}
    namespace = namespace_for()
    if namespace:
        options["namespace"] = namespace
    value = os.environ.get(DEPS_STORAGE_CLASS_ENV, "").strip()
    if value:
        options["deps_storage_class"] = value
    deadline = os.environ.get(SESSION_DEADLINE_ENV, "").strip()
    if deadline:
        try:
            options["session_deadline"] = int(deadline)
        except ValueError as exc:
            raise ComputeProfileError(
                f"{SESSION_DEADLINE_ENV} must be a whole number of seconds, "
                f"not {deadline!r}"
            ) from exc
    return options


#: Role the broker assumes to mint a sandbox credential. Unset means no vending, and
#: warehouse reads reach a cell through the app instead.
SANDBOX_ROLE_ENV = "SANDBOX_ROLE_ARN"
#: Opt in to vending on GCS and Azure, where there is no role ARN to serve as the
#: switch.
#: Explicit because handing a sandbox a storage credential is a decision, not a default.
VEND_CREDENTIALS_ENV = "SANDBOX_VEND_CREDENTIALS"
STORAGE_URI_ENV = "STORAGE_URI"


def credential_vendor() -> Callable[[], Mapping[str, str]] | None:
    """A callable minting this deployment's sandbox storage credentials, or None.

    The scope is the warehouse root, read-only, rather than the tables one notebook
    happens to have opened: a per-table credential would be reminted on every query and
    would still cover everything a cell could ask for by the end of a session. Narrowing
    it to the warehouse, not the bucket, and to reads, for fifteen minutes, is the bound
    that actually holds.
    """
    storage_uri = os.environ.get(STORAGE_URI_ENV, "").strip()
    if not storage_uri:
        return None
    role_arn = os.environ.get(SANDBOX_ROLE_ENV, "").strip()
    scheme = storage_uri.split("://", 1)[0].lower()
    # AWS is the only broker that needs to be told an identity: it assumes a role, while
    # GCS downscopes the credentials the pod already has and Azure signs with a
    # delegation
    # key derived from them. So an unset role means "no vending" on AWS and nothing at
    # all
    # elsewhere -- and vending must not be switched on by accident, so the other two are
    # opted into explicitly.
    if scheme in ("s3", "s3a"):
        if not role_arn:
            return None
    elif os.environ.get(VEND_CREDENTIALS_ENV, "").strip().lower() not in (
        "1",
        "true",
        "yes",
    ):
        return None
    try:
        scope = scope_for(storage_uri, prefixes=[""], read_only=True)
    except CredentialError as exc:
        # A URI with no path has nothing to scope to, and a credential for a whole
        # bucket is not a scoped credential. Refuse rather than quietly widen.
        raise CredentialError(
            f"{STORAGE_URI_ENV}={storage_uri!r} names a bucket with no prefix, so a "
            "sandbox credential for it would cover the whole bucket. Put the warehouse "
            f"under a prefix (s3://bucket/warehouse), or unset {SANDBOX_ROLE_ENV} "
            f"to proxy reads through the app instead. ({exc})"
        ) from exc
    broker = broker_for(
        storage_uri, **({"role_arn": role_arn} if scheme in ("s3", "s3a") else {})
    )

    def vend() -> Mapping[str, str]:
        return broker.vend(scope).env

    logger.info("vending sandbox credentials scoped to %s", storage_uri)
    return vend


@dataclass(frozen=True)
class SessionCost:
    """What one compute session is attributed with."""

    kind: str
    notebook_id: str
    profile: str
    profile_version: str
    seconds: float
    cost: float

    def as_row(self) -> dict[str, Any]:
        """The record as the store and the API want it."""
        return {
            "kind": self.kind,
            "notebook_id": self.notebook_id,
            "profile": self.profile,
            "profile_version": self.profile_version,
            "seconds": round(self.seconds, 1),
            "cost": round(self.cost, 4),
        }


def session_cost(
    profile: ComputeProfile,
    seconds: float,
    notebook_id: str,
    kind: str = "interactive",
    rates: CostRates | None = None,
) -> SessionCost:
    """Attribute a finished session to ``(notebook, profile, duration)``."""
    return SessionCost(
        kind=kind,
        notebook_id=notebook_id,
        profile=profile.name,
        profile_version=profile.version,
        seconds=seconds,
        cost=profile.cost_per_hour(rates) * seconds / 3600.0,
    )


#: Alert when a window's spend passes this. Hex's surface is a spend cap with alerting,
#: which is the shape copied here.
SPEND_LIMIT_ENV = "COMPUTE_SPEND_LIMIT"


def spend_limit() -> float | None:
    """The alert threshold, or ``None`` when none is set."""
    return _float_env(SPEND_LIMIT_ENV)


def over_budget(spent: float, limit: float | None) -> str | None:
    """The alert text when a window's spend has passed ``limit``, else None.

    An alert and not a kill. Terminating someone's session to enforce a budget destroys
    work in progress to save a few cents of the overage, and the vendor whose behaviour
    this follows does not do it either.
    """
    if limit is None or spent <= limit:
        return None
    return (
        f"Compute spend for this window is {spent:.2f}, over the {limit:.2f} limit. "
        "Sessions keep running: this is an alert, not a cap."
    )


def record_session(store: Store, cost: SessionCost) -> None:
    """Persist one session's attribution, surviving a database failure.

    Called as a kernel shuts down, from the reaper thread. A database that is briefly
    unreachable should cost one accounting row and a warning, not every later reap: an
    exception here would propagate out of the sweep and leave kernels running forever.
    Narrow to database errors, so a bug in the arithmetic still surfaces as a crash.
    """
    try:
        store.record_compute_usage(**cost.as_row())
    except SQLAlchemyError:
        logger.warning(
            "could not record compute usage for notebook %s", cost.notebook_id
        )
