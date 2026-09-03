"""Certified result history: the run record, the log, the diff, and the MLflow emit.

Tested at the public boundary: real versions recorded to a real log and diffed, plus an
end-to-end author() to prove the content-addressed version and its components are
surfaced and that a certified outcome becomes a version. The MLflow emit is exercised
against a fake client so its logging shape is covered without a live server.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from elbi_core import Registry, Runner, SubprocessExecutor, author
from elbi_core.authoring import AuthorOutcome, VerificationResult, propose
from elbi_core.tracking import (
    CertifiedRun,
    JsonlRunLog,
    changed_dimensions,
    run_from_author,
    with_changes,
)
from elbi_core.tracking import mlflow as mlflow_emit

# A clean dose-response effect (n=300) the oracle certifies sound with an estimate.
_CLEAN = """
def clean_effect(ctx):
    "Dose-response rows with a real effect."
    import random
    rng = random.Random(0)
    out = []
    for _ in range(300):
        x = rng.gauss(0, 1)
        y = 1.2 * x + rng.gauss(0, 0.8)
        out.append({"dose": round(x, 3), "response": round(y, 3)})
    return out
"""


def _run(
    version: str = "v1",
    estimate: float | None = 0.42,
    *,
    name: str = "m",
    verdict: str = "sound",
    created_at: str = "2026-07-03T00:00:00+00:00",
    adjusted_for: tuple[str, ...] = ("grade",),
    code_version: str = "code1",
    input_versions: dict[str, str] | None = None,
) -> CertifiedRun:
    return CertifiedRun(
        name=name,
        derivation_version=version,
        verdict=verdict,
        created_at=created_at,
        estimate=estimate,
        estimate_label="pct",
        data_hash="dh",
        adjusted_for=adjusted_for,
        claim={"x": "edu", "y": "inc"},
        checks=(("collider", "sound", "no collider"),),
        code_version=code_version,
        input_versions=input_versions if input_versions is not None else {"d": "d1"},
    )


def test_run_round_trips_through_its_mapping() -> None:
    run = _run()
    assert CertifiedRun.from_dict(run.to_dict()) == run
    assert run.short_version == "v1"


def test_skipped_round_trips_and_defaults_for_legacy_records() -> None:
    run = _run()
    with_skipped = CertifiedRun.from_dict({**run.to_dict(), "skipped": ["did", "iv"]})
    assert with_skipped.skipped == ("did", "iv")
    assert with_skipped.to_dict()["skipped"] == ["did", "iv"]
    # A record written before the field existed omits it and defaults to empty.
    legacy = {k: v for k, v in run.to_dict().items() if k != "skipped"}
    assert CertifiedRun.from_dict(legacy).skipped == ()


def test_metrics_expose_the_certified_estimate() -> None:
    assert _run(estimate=0.5).metrics() == {"estimate": 0.5}
    assert _run(estimate=None).metrics() == {}


def test_log_collapses_identical_versions_and_orders_newest_first(
    tmp_path: Path,
) -> None:
    log = JsonlRunLog(tmp_path / "runs.jsonl")
    assert log.append(_run(version="v1", created_at="2026-07-03T00:00:00+00:00"))
    # Same (name, version) is a deterministic re-run: it must not append again.
    assert not log.append(_run(version="v1", created_at="2026-07-03T09:00:00+00:00"))
    assert log.append(_run(version="v2", created_at="2026-07-03T10:00:00+00:00"))
    history = log.runs(name="m")
    assert [r.derivation_version for r in history] == ["v2", "v1"]
    assert log.runs(name="other") == []


def test_changed_dimensions_attributes_a_moved_estimate() -> None:
    base = _run(version="v1", estimate=0.42)
    # data refresh: an input version changed
    assert changed_dimensions(_run(version="v2", input_versions={"d": "d2"}), base) == (
        "data",
    )
    # a new control
    assert changed_dimensions(
        _run(version="v3", adjusted_for=("grade", "bedrooms")), base
    ) == ("controls",)
    # a code edit
    assert changed_dimensions(_run(version="v4", code_version="code2"), base) == (
        "code",
    )
    # several at once, reported in a stable order
    moved = _run(
        version="v5",
        code_version="code2",
        adjusted_for=("x",),
        input_versions={"d": "d2"},
    )
    assert changed_dimensions(moved, base) == ("data", "code", "controls")


def test_changed_dimensions_ignores_uncaptured_components() -> None:
    # A backfilled run with no code_version cannot attribute a code change, so it is not
    # reported (the diff degrades rather than over-claiming).
    a = _run(version="v2", code_version=None)
    b = _run(version="v1", code_version=None)
    assert "code" not in changed_dimensions(a, b)


def test_with_changes_pairs_each_version_with_what_moved() -> None:
    runs = [
        # newest: data changed from v2 (same code, new input version)
        _run(version="v3", code_version="code1", input_versions={"d": "d2"}),
        # code changed from v1 (same input)
        _run(version="v2", code_version="code1", input_versions={"d": "d1"}),
        # oldest
        _run(version="v1", code_version="code0", input_versions={"d": "d1"}),
    ]
    paired = with_changes(runs)
    assert [changed for _, changed in paired] == [("data",), ("code",), ()]


@given(
    rows=st.lists(
        st.tuples(
            st.sampled_from(["sound", "unsound", "inconclusive"]),
            st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        ),
        min_size=1,
        max_size=30,
    )
)
@settings(max_examples=50, deadline=None)
def test_with_changes_is_total_and_oldest_has_no_change(
    rows: list[tuple[str, float]],
) -> None:
    runs = [
        _run(version=f"v{i}", verdict=v, estimate=e) for i, (v, e) in enumerate(rows)
    ]
    paired = with_changes(runs)
    assert len(paired) == len(runs)
    assert paired[-1][1] == ()  # the oldest version has nothing before it


# -- end-to-end: a real author() surfaces the version and its components -----------


def _runner(registry: Registry) -> Runner:
    return Runner(registry, executor=SubprocessExecutor(timeout=60))


def test_author_surfaces_a_stable_content_version_and_parts(registry: Registry) -> None:
    out = author(
        "clean_effect",
        _CLEAN,
        runner=_runner(registry),
        claim={"x": "dose", "y": "response"},
        registry=registry,
    )
    assert out.certified and out.result.oracle_verdict == "sound"
    version = out.result.data_version
    assert version and len(version) == 64  # a full sha-256 hex content hash
    assert out.result.code_version  # the code component is surfaced too
    # The version is a pure function of code+params+inputs: recomputing agrees.
    composite, code, _inputs = _runner(registry).version_parts("clean_effect")
    assert composite == version and code == out.result.code_version


def test_run_from_author_captures_the_certified_estimate(
    registry: Registry, tmp_path: Path
) -> None:
    out = author(
        "clean_effect",
        _CLEAN,
        runner=_runner(registry),
        claim={"x": "dose", "y": "response"},
        registry=registry,
    )
    run = run_from_author(out)
    assert run is not None
    assert run.verdict == "sound"
    assert run.estimate is not None and run.estimate_label
    assert run.derivation_version == out.result.data_version
    assert run.code_version == out.result.code_version
    assert run.claim == {"x": "dose", "y": "response"}
    assert run.checks  # the gates that ran ride along
    log = JsonlRunLog(tmp_path / "runs.jsonl")
    assert log.append(run) is True
    assert log.append(run) is False  # the same certified version collapses


def test_run_from_author_marks_a_claimless_certification_unverified(
    registry: Registry,
) -> None:
    # AutoCertifyOnVerify (the default policy) certifies on a clean run alone, with
    # no claim declared: oracle_verdict stays None, so the run must not read "sound".
    out = author(
        "revenue",
        'def revenue(ctx):\n    return [{"total": 42}]\n',
        runner=_runner(registry),
        registry=registry,
    )
    assert out.certified is True
    assert out.result.oracle_verdict is None
    run = run_from_author(out)
    assert run is not None
    assert run.verdict == "unverified"


def test_run_from_author_returns_none_when_uncertified(registry: Registry) -> None:
    derivation = propose("held", "def held(ctx):\n    return []\n", registry=registry)
    outcome = AuthorOutcome(
        derivation=derivation, result=VerificationResult(ok=False), certified=False
    )
    assert run_from_author(outcome) is None


# -- MLflow emit: fail-soft without a client, correct shape with a fake one --------


def test_emit_is_a_no_op_when_the_client_is_unavailable(monkeypatch: Any) -> None:
    # The skinny client ships by default; defensively, a broken/absent one must return
    # False rather than raise, so a certification is never taken down by the export.
    monkeypatch.setitem(sys.modules, "mlflow", None)
    assert mlflow_emit.emit_run(_run(), tracking_uri="http://x") is False


class _FakeRunInfo:
    run_id = "r1"


class _FakeRun:
    info = _FakeRunInfo()


class _FakeExperiment:
    experiment_id = "e1"


class _FakeClient:
    def __init__(self, tracking_uri: str) -> None:
        self.tracking_uri = tracking_uri
        self.created: dict[str, Any] = {}
        self.batch: dict[str, Any] = {}
        self.status: str | None = None

    def get_experiment_by_name(self, name: str) -> _FakeExperiment | None:
        return None

    def create_experiment(self, name: str) -> str:
        self.created["experiment"] = name
        return "e1"

    def create_run(self, experiment_id: str, **kwargs: Any) -> _FakeRun:
        self.created["run"] = {"experiment_id": experiment_id, **kwargs}
        return _FakeRun()

    def log_batch(self, run_id: str, **kwargs: Any) -> None:
        self.batch = {"run_id": run_id, **kwargs}

    def set_terminated(self, run_id: str, status: str) -> None:
        self.status = status


def test_emit_logs_the_attestation_to_a_fake_mlflow(monkeypatch: Any) -> None:
    clients: list[_FakeClient] = []

    def _client(tracking_uri: str) -> _FakeClient:
        client = _FakeClient(tracking_uri)
        clients.append(client)
        return client

    monkeypatch.setattr(
        mlflow_emit,
        "_mlflow_entities",
        lambda: (_client, _Metric, _Param, _Tag),
    )
    run = _run(estimate=0.42)
    assert mlflow_emit.emit_run(run, tracking_uri="http://mlflow") is True
    client = clients[0]
    # The derivation maps to an MLflow experiment named after it.
    assert client.created["experiment"] == "m"
    tags = {t.key: t.value for t in client.batch["tags"]}
    assert tags["elbi.verdict"] == "sound"
    assert tags["elbi.data_hash"] == "dh"
    params = {p.key: p.value for p in client.batch["params"]}
    # Config keys are namespaced with ':' sanitized to '.'; controls ride as a param.
    assert "claim.x" in params and "claim:x" not in params
    assert params["adjusted_for"] == "grade"
    metrics = {m.key: m.value for m in client.batch["metrics"]}
    assert metrics["estimate"] == 0.42
    assert client.status == "FINISHED"


def test_emit_reuses_an_existing_experiment_and_omits_absent_fields(
    monkeypatch: Any,
) -> None:
    clients: list[_FakeClient] = []

    class _ExistingClient(_FakeClient):
        def get_experiment_by_name(self, name: str) -> _FakeExperiment:
            return _FakeExperiment()

    def _client(tracking_uri: str) -> _ExistingClient:
        client = _ExistingClient(tracking_uri)
        clients.append(client)
        return client

    monkeypatch.setattr(
        mlflow_emit, "_mlflow_entities", lambda: (_client, _Metric, _Param, _Tag)
    )
    # A minimal version: no data_hash, checks, estimate, controls, or deps to log.
    minimal = CertifiedRun(
        name="m",
        derivation_version="v",
        verdict="sound",
        created_at="2026-07-03T00:00:00+00:00",
    )
    assert mlflow_emit.emit_run(minimal, tracking_uri="http://x") is True
    assert "experiment" not in clients[0].created  # existing one was reused
    tags = {t.key: t.value for t in clients[0].batch["tags"]}
    assert "elbi.data_hash" not in tags
    assert "elbi.checks" not in tags
    params = {p.key: p.value for p in clients[0].batch["params"]}
    assert "adjusted_for" not in params and "deps" not in params
    assert not clients[0].batch["metrics"]  # no numeric estimate to log


def test_mlflow_entities_resolve_the_client_classes(monkeypatch: Any) -> None:
    # Inject a stand-in mlflow so the import's success path is exercised against known
    # classes, deterministically and independent of the installed client's internals.
    import types

    fake = types.ModuleType("mlflow")
    fake.MlflowClient = _FakeClient  # type: ignore[attr-defined]
    entities = types.ModuleType("mlflow.entities")
    entities.Metric = _Metric  # type: ignore[attr-defined]
    entities.Param = _Param  # type: ignore[attr-defined]
    entities.RunTag = _Tag  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", fake)
    monkeypatch.setitem(sys.modules, "mlflow.entities", entities)
    assert mlflow_emit._mlflow_entities() == (_FakeClient, _Metric, _Param, _Tag)


def test_emit_is_fail_soft_when_the_client_raises(monkeypatch: Any) -> None:
    class _BoomClient(_FakeClient):
        def create_run(self, experiment_id: str, **kwargs: Any) -> _FakeRun:
            raise RuntimeError("boom")

    monkeypatch.setattr(
        mlflow_emit,
        "_mlflow_entities",
        lambda: (lambda uri: _BoomClient(uri), _Metric, _Param, _Tag),
    )
    # The emit warns rather than raising, so certification is never broken by it.
    with pytest.warns(UserWarning, match="MLflow emit"):
        assert mlflow_emit.emit_run(_run(), tracking_uri="http://x") is False


class _Metric:
    def __init__(self, key: str, value: float, timestamp: int, step: int) -> None:
        self.key, self.value, self.timestamp, self.step = key, value, timestamp, step


class _Param:
    def __init__(self, key: str, value: str) -> None:
        self.key, self.value = key, value


class _Tag:
    def __init__(self, key: str, value: str) -> None:
        self.key, self.value = key, value
