"""Compute profiles: validation, governance, and the translations to each runner."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from elbi_core.sandbox import (
    ComputeProfile,
    ComputeProfileError,
    ComputeProfiles,
    CostRates,
    docker_resource_args,
    kubernetes_resources,
    parse_cpu,
    parse_memory,
)


def test_kubernetes_quantities_parse_including_the_case_trap() -> None:
    assert parse_cpu("2") == 2.0
    assert parse_cpu("500m") == 0.5
    assert parse_memory("8Gi") == 8 * 1024**3
    assert parse_memory("8G") == 8 * 1000**3
    assert parse_memory("512Mi") == 512 * 1024**2

    # Kubernetes reads a lowercase 'm' on memory as millibytes, so `memory: 512m` is a
    # request for half a byte. Rejecting it beats honouring it.
    with pytest.raises(ComputeProfileError, match="millibytes"):
        parse_memory("512m")
    for bad in ("", "lots", "-1", "2 cores"):
        with pytest.raises(ComputeProfileError):
            parse_cpu(bad)


def test_a_profile_rejects_what_would_break_a_runner() -> None:
    with pytest.raises(ComputeProfileError, match="lowercase alphanumeric"):
        ComputeProfile(name="Big Box")
    with pytest.raises(ComputeProfileError, match="asks for 0 GPUs"):
        ComputeProfile(name="gpu", gpu_type="nvidia-l4")
    with pytest.raises(ComputeProfileError, match="idle_timeout must be positive"):
        ComputeProfile(name="x", idle_timeout=0)
    with pytest.raises(ComputeProfileError, match="egress must be"):
        ComputeProfile(name="x", egress="sometimes")


def test_unknown_profile_fields_are_rejected_rather_than_ignored() -> None:
    # A misspelled isolation field that is silently dropped means an operator believes a
    # sandbox is sandboxed when it is not.
    with pytest.raises(ComputeProfileError, match="runtime_clas"):
        ComputeProfile.from_mapping(
            {"name": "small", "runtimeClas": "gvisor"}, context="profiles[0]"
        )


def test_both_spellings_of_a_field_are_accepted() -> None:
    # A profile is written in a YAML project file (snake_case) and in Helm values
    # (camelCase, as the rest of that chart is). Making an operator remember which is
    # which would be a trap whose failure looks like a rejected typo.
    from_helm = ComputeProfile.from_mapping(
        {
            "name": "gpu",
            "gpuType": "nvidia-l4",
            "gpu": 1,
            "idleTimeout": 600,
            "runtimeClass": "gvisor",
        },
        context="values.compute.profiles[0]",
    )
    from_yaml = ComputeProfile.from_mapping(
        {
            "name": "gpu",
            "gpu_type": "nvidia-l4",
            "gpu": 1,
            "idle_timeout": 600,
            "runtime_class": "gvisor",
        },
        context="elbi.yaml",
    )
    assert from_helm == from_yaml
    assert from_helm.version == from_yaml.version


def test_yaml_natural_scalars_are_accepted() -> None:
    profile = ComputeProfile.from_mapping(
        {"name": "small", "cpu": 2, "memory": "4Gi", "gpu": 0}, context="p"
    )
    assert profile.cpu == "2"
    assert profile.cores == 2.0


def test_the_version_moves_when_the_definition_does() -> None:
    # The whole point of a derived version: an admin who tightens a limit gets a new
    # version whether or not they remember to say so, which is what makes drift visible.
    before = ComputeProfile(name="small", memory="4Gi")
    after = ComputeProfile(name="small", memory="2Gi")
    assert before.version != after.version
    assert before.version == ComputeProfile(name="small", memory="4Gi").version
    # Not a field of the definition, so it cannot itself be drifted.
    assert "version" not in before.to_dict()


def test_the_cost_ceiling_is_enforced_when_the_menu_is_defined() -> None:
    # Databricks enforces its equivalent at creation rather than at run, so a profile
    # nobody may launch is never offered in the first place.
    with pytest.raises(ComputeProfileError, match="exceed max_cost_per_hour"):
        ComputeProfiles(
            profiles=(ComputeProfile(name="huge", cpu="64", memory="512Gi"),),
            default="huge",
            max_cost_per_hour=1.0,
        )
    cheap = ComputeProfile(name="small", cpu="2", memory="4Gi")
    assert cheap.cost_per_hour(CostRates()) == pytest.approx(0.1, abs=0.01)
    # Spot is the same shape at a discount, which is the only claim made for it.
    spot = ComputeProfile(name="small", cpu="2", memory="4Gi", spot=True)
    assert spot.cost_per_hour() < cheap.cost_per_hour()


def test_a_duplicate_or_missing_default_is_a_configuration_error() -> None:
    with pytest.raises(ComputeProfileError, match="duplicate"):
        ComputeProfiles(
            profiles=(ComputeProfile(name="a"), ComputeProfile(name="a")), default="a"
        )
    with pytest.raises(ComputeProfileError, match="not defined"):
        ComputeProfiles(profiles=(ComputeProfile(name="a"),), default="b")


def test_docker_and_kubernetes_get_the_same_size_in_their_own_notation() -> None:
    profile = ComputeProfile(name="m", cpu="500m", memory="8Gi", pids=256)
    # Bytes, not a re-rendered suffix: docker's `8g` is decimal and Kubernetes' `8Gi` is
    # binary, a 7% difference that would otherwise pass silently.
    assert docker_resource_args(profile) == {
        "memory": f"{8 * 1024**3}b",
        "cpus": "0.5",
        "pids_limit": 256,
    }
    assert kubernetes_resources(profile) == {
        "requests": {"cpu": "500m", "memory": "8Gi"},
        "limits": {"cpu": "500m", "memory": "8Gi"},
    }


def test_a_gpu_profile_asks_each_runtime_the_way_it_expects() -> None:
    profile = ComputeProfile(name="gpu", gpu=1, gpu_type="nvidia-l4")
    assert docker_resource_args(profile)["gpus"] == "1"
    limits = kubernetes_resources(profile)["limits"]
    # An extended resource is a limit only, and never a request.
    assert limits["nvidia.com/gpu"] == "1"
    assert "nvidia.com/gpu" not in kubernetes_resources(profile)["requests"]


_EGRESS = st.one_of(
    st.just("full"),
    st.just("none"),
    st.lists(st.sampled_from(["a.example", "b.example", "c.example"]), min_size=1).map(
        tuple
    ),
)


@given(profile_egress=_EGRESS, project_egress=_EGRESS)
def test_egress_never_widens(
    profile_egress: str | tuple[str, ...], project_egress: str | tuple[str, ...]
) -> None:
    """A profile can only narrow the project's network policy, never open it up.

    Stated as a property because it is a security invariant with three shapes on each
    side, and the interesting cases (two allowlists that barely overlap, an allowlist
    against ``none``) are exactly the ones nobody writes by hand.
    """
    result = ComputeProfile(name="p", egress=profile_egress).narrowed_to(project_egress)

    def permitted(policy: str | tuple[str, ...]) -> set[str] | None:
        """Hosts the policy allows; None means 'everything'."""
        if policy == "full":
            return None
        if policy == "none":
            return set()
        return set(policy)

    allowed, by_project = permitted(result.egress), permitted(project_egress)
    if by_project is None:
        return  # the project permits everything, so nothing can be wider
    assert allowed is not None, "a profile must not open a project's restricted network"
    assert allowed <= by_project


def test_egress_narrowing_worked_examples() -> None:
    profile = ComputeProfile(name="p", egress=("a.example", "b.example"))
    assert profile.narrowed_to("none").egress == "none"
    assert profile.narrowed_to("full").egress == ("a.example", "b.example")
    assert profile.narrowed_to(["b.example", "c.example"]).egress == ("b.example",)
    # Two allowlists sharing nothing permit nothing, and that has to be spelled 'none':
    # an empty list reads as "unset" to every runner downstream.
    assert profile.narrowed_to(["z.example"]).egress == "none"
    assert ComputeProfile(name="p", egress="full").narrowed_to("none").egress == "none"


def test_the_menu_parses_from_configuration() -> None:
    profiles = ComputeProfiles.from_config(
        [
            {"name": "small", "cpu": "2", "memory": "4Gi"},
            {
                "name": "gpu",
                "cpu": "8",
                "memory": "32Gi",
                "gpu": 1,
                "gpu_type": "nvidia-l4",
                "runtime_class": "gvisor",
                "egress": ["pypi.org"],
            },
        ],
        default="small",
    )
    assert [p.name for p in profiles] == ["small", "gpu"]
    gpu = profiles.get("gpu")
    assert gpu.egress == ("pypi.org",)
    assert gpu.runtime_class == "gvisor"
    with pytest.raises(ComputeProfileError, match="non-empty list"):
        ComputeProfiles.from_config([], context="compute.profiles")
    with pytest.raises(ComputeProfileError, match=r"\[0\] must be a mapping"):
        ComputeProfiles.from_config(["small"], context="compute.profiles")


def test_verification_never_runs_on_user_selected_compute() -> None:
    """The oracle and the trusted executor must not depend on compute profiles.

    Certification is the product's guarantee, so it runs server-side on the app's own
    resources. If verification ever ran on compute a user picks, a user could starve the
    guardrail: by choosing a tiny profile, by exhausting a quota, by occupying the
    kernel. Asserted structurally because the way this breaks is somebody threading a
    profile through "just for consistency", and by then it reads as intentional.
    """
    import ast
    from pathlib import Path

    root = Path(__file__).resolve().parents[1] / "src" / "elbi_core"
    trusted = [
        root / "executor.py",
        root / "runner.py",
        *(root / "verification").rglob("*.py"),
    ]
    offenders = []
    for path in trusted:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and "sandbox" in (node.module or ""):
                offenders.append(f"{path.name}:{node.lineno}")
            if isinstance(node, ast.Import):
                offenders += [
                    f"{path.name}:{node.lineno}"
                    for alias in node.names
                    if "sandbox" in alias.name
                ]
    assert not offenders, (
        "the verification path imported the compute profile module at "
        f"{', '.join(offenders)}; certification must not run on compute a user chooses"
    )


def test_hiding_a_field_changes_what_is_shown_not_what_is_enforced() -> None:
    # Databricks lists "simplify the user interface" among the purposes of a policy, not
    # just its constraints. Hiding must not become a second way to change a profile.
    plain = ComputeProfile(name="small", pids=256)
    tidy = ComputeProfile(name="small", pids=256, hidden=("pids", "warm_pool_size"))
    assert tidy.version == plain.version
    assert tidy.pids == 256
    shown = tidy.presented()
    assert "pids" not in shown and "warm_pool_size" not in shown
    assert shown["memory"] == "4Gi"
    # Its version and cost travel with it, since that is what a choice is made against.
    assert shown["version"] == tidy.version
    assert shown["cost_per_hour"] > 0
    # `name` is never hideable: a profile you cannot name is one you cannot pick.
    with pytest.raises(ComputeProfileError, match="hides unknown field"):
        ComputeProfile(name="small", hidden=("name",))
    with pytest.raises(ComputeProfileError, match="hides unknown field"):
        ComputeProfile(name="small", hidden=("memry",))
