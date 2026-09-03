"""How a deployment says where compute runs, and what it may run as.

The chart is exercised for real here rather than asserted about: the failure this
catches is the chart and the parser drifting apart, and only running both finds it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from elbi.compute import (
    BUILTIN_PROFILES,
    resolve_profiles,
    resolve_runner,
    runner_options,
)
from elbi_core.config import ProjectConfig
from elbi_core.sandbox import ComputeProfileError, ComputeProfiles

CHART = Path(__file__).resolve().parents[3] / "deploy" / "helm" / "elbi"


def test_a_deployment_that_says_nothing_gets_t_shirt_sizes() -> None:
    profiles = resolve_profiles(None)
    assert [p.name for p in profiles] == ["small", "medium", "large"]
    assert profiles.default == "small"
    assert len(BUILTIN_PROFILES) == 3


def test_the_environment_outranks_the_project_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The menu is a property of the infrastructure the app was deployed onto, so an
    # operator setting it in Helm must not be overridden by a file inside the image.
    path = tmp_path / "elbi.yaml"
    path.write_text(
        "project: t\ncompute_profiles:\n  - name: from-file\n", encoding="utf-8"
    )
    config = ProjectConfig.load(path)
    assert [p.name for p in resolve_profiles(config)] == ["from-file"]

    monkeypatch.setenv("COMPUTE_PROFILES", json.dumps([{"name": "from-env"}]))
    assert [p.name for p in resolve_profiles(config)] == ["from-env"]


def test_a_malformed_menu_fails_at_boot(monkeypatch: pytest.MonkeyPatch) -> None:
    # Rather than the first time somebody opens a notebook.
    monkeypatch.setenv("COMPUTE_PROFILES", "not json")
    with pytest.raises(ComputeProfileError, match="not valid JSON"):
        resolve_profiles(None)
    monkeypatch.setenv("COMPUTE_PROFILES", json.dumps({"name": "x"}))
    with pytest.raises(ComputeProfileError, match="JSON list"):
        resolve_profiles(None)


def test_the_cost_ceiling_and_rates_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("COMPUTE_COST_RATES", json.dumps({"cpu_core_hour": 1.0}))
    monkeypatch.setenv("COMPUTE_MAX_COST_PER_HOUR", "100")
    profiles = resolve_profiles(None)
    assert profiles.max_cost_per_hour == 100.0
    assert profiles.get("small").cost_per_hour(profiles.rates) > 2.0

    # A ceiling below the menu is a configuration error, caught where it is set.
    monkeypatch.setenv("COMPUTE_MAX_COST_PER_HOUR", "0.5")
    with pytest.raises(ComputeProfileError, match="exceed max_cost_per_hour"):
        resolve_profiles(None)

    monkeypatch.setenv("COMPUTE_MAX_COST_PER_HOUR", "cheap")
    with pytest.raises(ComputeProfileError, match="must be a number"):
        resolve_profiles(None)


def test_an_unknown_rate_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    # Silently ignoring it would leave an operator believing they had repriced GPUs.
    monkeypatch.setenv("COMPUTE_COST_RATES", json.dumps({"gpu_hour_": 2.0}))
    with pytest.raises(ComputeProfileError, match="unknown rate"):
        resolve_profiles(None)


def test_the_runner_and_its_cluster_settings_come_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "elbi.yaml"
    path.write_text("project: t\nsandbox: docker\n", encoding="utf-8")
    config = ProjectConfig.load(path)
    assert resolve_runner(config) == "docker"

    monkeypatch.setenv("SANDBOX_BACKEND", "kubernetes")
    assert resolve_runner(config) == "kubernetes"
    monkeypatch.setenv("SANDBOX_BACKEND", "vm")
    with pytest.raises(ComputeProfileError, match="is not a runner"):
        resolve_runner(config)

    # Only what is set is passed on, so an unset value never arrives as an empty string
    # the cluster would reject.
    assert runner_options() == {}
    monkeypatch.setenv("COMPUTE_NAMESPACE", "kernels")
    monkeypatch.setenv("COMPUTE_SESSION_DEADLINE_SECONDS", "3600")
    assert runner_options() == {"namespace": "kernels", "session_deadline": 3600}
    monkeypatch.setenv("COMPUTE_SESSION_DEADLINE_SECONDS", "a while")
    with pytest.raises(ComputeProfileError, match="whole number of seconds"):
        runner_options()


def test_no_broker_means_no_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    from elbi.compute import credential_vendor

    monkeypatch.delenv("SANDBOX_ROLE_ARN", raising=False)
    monkeypatch.setenv("STORAGE_URI", "s3://bucket/warehouse")
    # Nothing is vended, and reads reach a cell through the app instead. Silence rather
    # than an error: proxying is a correct configuration, just a slower one.
    assert credential_vendor() is None


def test_a_credential_that_would_cover_a_whole_bucket_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elbi.compute import credential_vendor
    from elbi_core.sandbox import CredentialError

    monkeypatch.setenv("SANDBOX_ROLE_ARN", "arn:aws:iam::1:role/sandbox")
    monkeypatch.setenv("STORAGE_URI", "s3://bucket")
    with pytest.raises(CredentialError, match="whole bucket"):
        credential_vendor()

    monkeypatch.setenv("STORAGE_URI", "s3://bucket/warehouse")
    assert credential_vendor() is not None


def test_the_executor_backend_follows_the_runner_until_it_cannot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elbi.compute import executor_backend

    monkeypatch.delenv("SANDBOX_EXECUTOR_BACKEND", raising=False)
    monkeypatch.setenv("SANDBOX_BACKEND", "docker")
    assert executor_backend(None) == "docker"

    # On Kubernetes the notebook runner has no executor for a one-shot derivation, so
    # the operator must choose rather than silently getting the weaker option.
    monkeypatch.setenv("SANDBOX_BACKEND", "kubernetes")
    assert executor_backend(None) == "kubernetes"  # resolved here, refused when built
    monkeypatch.setenv("SANDBOX_EXECUTOR_BACKEND", "docker")
    assert executor_backend(None) == "docker"
    monkeypatch.setenv("SANDBOX_EXECUTOR_BACKEND", "kubernetes")
    with pytest.raises(ComputeProfileError, match="must be 'docker' or 'subprocess'"):
        executor_backend(None)


def test_recorded_usage_totals_over_the_window(tmp_path: Path) -> None:
    """A session is billed from the row it wrote, so the window bounds the total."""
    from datetime import datetime, timedelta, timezone

    from elbi.db import open_store

    store = open_store(f"sqlite:{tmp_path / 'usage.db'}")
    for cost in (3.0, 1.5, 10.0, 0.5):
        store.record_compute_usage(
            notebook_id="nb", profile="small", seconds=60, cost=cost
        )
    since = datetime.now(timezone.utc) - timedelta(days=1)
    assert store.compute_spend(since=since) == pytest.approx(15.0)
    assert len(store.compute_usage(since=since)) == 4

    # A window that starts after every row sums to nothing rather than to everything.
    later = datetime.now(timezone.utc) + timedelta(days=1)
    assert store.compute_spend(since=later) == pytest.approx(0.0)


def test_vending_on_gcs_and_azure_is_opted_into_explicitly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Handing a sandbox a storage credential is a decision, not a default.

    On AWS the role ARN is the switch, because the broker cannot assume a role it was
    not given. GCS downscopes the credentials the pod already has and Azure signs with a
    key derived from them, so both would work with no configuration at all, which is
    exactly why they need an explicit opt-in rather than starting to vend the moment the
    app runs on a cloud that allows it.
    """
    from elbi.compute import credential_vendor

    monkeypatch.delenv("SANDBOX_ROLE_ARN", raising=False)
    monkeypatch.delenv("SANDBOX_VEND_CREDENTIALS", raising=False)

    monkeypatch.setenv("STORAGE_URI", "gs://bucket/warehouse")
    assert credential_vendor() is None
    monkeypatch.setenv("SANDBOX_VEND_CREDENTIALS", "true")
    assert credential_vendor() is not None

    monkeypatch.setenv(
        "STORAGE_URI", "abfss://warehouse@acct.dfs.core.windows.net/lake"
    )
    assert credential_vendor() is not None
    monkeypatch.delenv("SANDBOX_VEND_CREDENTIALS")
    assert credential_vendor() is None

    # AWS still keys off the role, since the broker needs one to assume.
    monkeypatch.setenv("STORAGE_URI", "s3://bucket/warehouse")
    assert credential_vendor() is None
    monkeypatch.setenv("SANDBOX_ROLE_ARN", "arn:aws:iam::1:role/sandbox")
    assert credential_vendor() is not None


def test_the_subprocess_kernel_honours_the_profile_egress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A profile saying ``full`` must open the network on this backend too.

    It used to be denied unconditionally here, so a profile that said otherwise was
    quietly overruled and anything the kernel was meant to reach -- the deployment's own
    tracking server -- failed as a connection error minutes later. Only ``full`` opens
    it: a host allowlist needs the proxy the container backends get.
    """
    from elbi.notebooks import NotebookService
    from elbi_core.notebook import SubprocessKernel
    from elbi_core.sandbox import ComputeProfile

    seen: dict[str, object] = {}

    class _Recorded(SubprocessKernel):
        def __init__(self, **kwargs: object) -> None:
            seen.update(kwargs)
            # Never actually start a child: the decision under test is the argument.
            raise RuntimeError("not started")

    monkeypatch.setattr("elbi.notebooks.SubprocessKernel", _Recorded)
    service = NotebookService(
        store=None,  # type: ignore[arg-type]
        load_datasets=lambda: {},
        profiles=ComputeProfiles(
            profiles=(
                ComputeProfile(name="open", egress="full", managed=False),
                ComputeProfile(name="shut", egress="none", managed=False),
                ComputeProfile(name="listed", egress=("pypi.org",), managed=False),
            ),
            default="open",
        ),
    )
    for profile, expected in (("open", True), ("shut", False), ("listed", False)):
        seen.clear()
        with pytest.raises(RuntimeError, match="not started"):
            service._make_kernel((), None, service.profiles.get(profile))
        got = seen["allow_network"]
        assert got is expected, f"{profile} egress gave allow_network={got}"
