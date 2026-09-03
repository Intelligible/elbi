"""Certified result history in the app: the run store, the bridge, the endpoint.

Exercised at the public boundary: versions written through the store and the real
authoring bridge, then read back through the HTTP history API. The bridge test authors a
genuine claim so a certified version is recorded exactly as the chat path records it.
"""

from __future__ import annotations

import threading
import types
from collections.abc import Sequence
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from elbi import create_app
from elbi.authoring import make_derive_factory, make_draft_factory
from elbi.db import Derivation, Store, open_store
from elbi.tracking import emit_configured_run, tracking_uri
from elbi_core import Registry, Runner, SubprocessExecutor
from elbi_core.tracking import CertifiedRun

# A clean dose-response effect the oracle certifies sound with an estimate.
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

# Independent x and y: a claimed effect the oracle must refuse to support.
_NOISE = """
def noise_effect(ctx):
    "Pure noise; dose and response are unrelated."
    import random
    rng = random.Random(0)
    return [
        {"dose": round(rng.gauss(0, 1), 3), "response": round(rng.gauss(0, 1), 3)}
        for _ in range(300)
    ]
"""

# Unseeded randomness: runs, but can never reproduce.
_FLAKY = """
def flaky(ctx):
    "A result that changes on every run."
    import random
    return [{"n": random.random()} for _ in range(20)]
"""

# Crashes in the sandbox.
_BROKEN = """
def broken(ctx):
    "Raises at run time."
    raise ValueError("boom")
"""


class _NoClient:
    def step(self, transcript: object, tools: Sequence[object]) -> object:
        raise NotImplementedError


@pytest.fixture
def store(tmp_path: Path) -> Store:
    return open_store(f"sqlite:{tmp_path / 'app.db'}")


def _run(
    name: str = "m",
    version: str = "v1",
    estimate: float | None = 0.42,
    *,
    verdict: str = "sound",
    created_at: str = "2026-07-03T00:00:00+00:00",
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
        claim={"x": "edu", "y": "inc"},
        checks=(("collider", "sound", "no collider"),),
        code_version="code1",
        input_versions=input_versions if input_versions is not None else {"d": "d1"},
    )


# -- the store ---------------------------------------------------------------------


def test_append_collapses_and_orders_history(store: Store) -> None:
    assert store.append_run(_run(version="v1", created_at="2026-07-03T00:00:00+00:00"))
    # A deterministic re-derive of the same version must not append a second row.
    assert not store.append_run(
        _run(version="v1", created_at="2026-07-03T09:00:00+00:00")
    )
    assert store.append_run(_run(version="v2", created_at="2026-07-03T10:00:00+00:00"))
    history = store.runs_for_derivation("m")
    assert [r.derivation_version for r in history] == ["v2", "v1"]


def test_run_fields_round_trip_through_the_row(store: Store) -> None:
    store.append_run(_run(name="a", version="a1", input_versions={"d": "d7"}))
    got = store.runs_for_derivation("a")[0]
    # The structured attestation and version components survive the JSON round trip.
    assert got.claim == {"x": "edu", "y": "inc"}
    assert got.checks == (("collider", "sound", "no collider"),)
    assert got.code_version == "code1"
    assert got.input_versions == {"d": "d7"}


# -- the authoring bridge records a run on certification ---------------------------


def test_bridge_records_a_certified_run(store: Store) -> None:
    registry = Registry()
    project = types.SimpleNamespace(
        registry=registry,
        config=types.SimpleNamespace(datasets=[]),
        make_runner=lambda: Runner(registry, executor=SubprocessExecutor(timeout=60)),
    )
    derive = make_derive_factory(
        project,
        store,
        make_runner=project.make_runner,
        dataset_names=lambda: [d.name for d in project.config.datasets],
    )("c1", "does dose affect response?")
    outcome = derive(
        "clean_effect", _CLEAN, {"x": "dose", "y": "response"}, None, "table", [], []
    )
    assert outcome.certified and outcome.verdict == "sound"
    assert outcome.derivation_version  # surfaced back to the runtime
    runs = store.runs_for_derivation("clean_effect")
    assert len(runs) == 1
    assert runs[0].verdict == "sound"
    assert runs[0].estimate is not None
    assert runs[0].derivation_version == outcome.derivation_version


def test_draft_bridge_verifies_without_certifying_or_registering(store: Store) -> None:
    """The real draft bridge: sandbox-verifies through the project registry, leaves
    no trace, and never clobbers an existing derivation of the same name."""
    registry = Registry()
    project = types.SimpleNamespace(
        registry=registry,
        config=types.SimpleNamespace(datasets=[]),
        make_runner=lambda: Runner(registry, executor=SubprocessExecutor(timeout=60)),
    )
    draft = make_draft_factory(
        project,
        make_runner=project.make_runner,
        dataset_names=lambda: [],
    )("explore", "does dose affect response?")

    result = draft(
        "clean_effect", _CLEAN, {"x": "dose", "y": "response"}, None, "table", []
    )
    assert result["ok"] is True
    assert result["verdict"] == "sound"
    assert result["certified"] is False
    # A draft leaves no trace: nothing stays registered, so nothing can serve.
    assert "clean_effect" not in registry

    # Certify for real (the promote path), then draft the same name again: the
    # certified derivation must survive the preview untouched.
    derive = make_derive_factory(
        project,
        store,
        make_runner=project.make_runner,
        dataset_names=lambda: [],
    )("c1", "does dose affect response?")
    outcome = derive(
        "clean_effect", _CLEAN, {"x": "dose", "y": "response"}, None, "table", [], []
    )
    assert outcome.certified
    again = draft(
        "clean_effect", _CLEAN, {"x": "dose", "y": "response"}, None, "table", []
    )
    assert again["ok"] is True and again["certified"] is False
    assert registry.get("clean_effect").is_certified  # prior holder restored


def _draft_project(store: Store):  # type: ignore[no-untyped-def]
    """A real registry + sandbox runner and the draft fn bound to them."""
    registry = Registry()
    project = types.SimpleNamespace(
        registry=registry,
        config=types.SimpleNamespace(datasets=[]),
        make_runner=lambda: Runner(registry, executor=SubprocessExecutor(timeout=60)),
    )
    draft = make_draft_factory(
        project, make_runner=project.make_runner, dataset_names=lambda: []
    )("explore", "q")
    return registry, draft


def test_draft_bridge_blocks_an_unsupported_claim(store: Store) -> None:
    """A claim the oracle cannot support never becomes certifiable or servable."""
    registry, draft = _draft_project(store)
    result = draft(
        "noise_effect", _NOISE, {"x": "dose", "y": "response"}, None, "table", []
    )
    assert result["ok"] is False  # the certify gate would refuse this draft
    assert result["verdict"] in ("inconclusive", "unsound")
    assert "noise_effect" not in registry  # nothing registered, nothing to serve


def test_draft_bridge_blocks_an_irreproducible_result(store: Store) -> None:
    """A result that cannot be reproduced cannot be certified."""
    registry, draft = _draft_project(store)
    result = draft("flaky", _FLAKY, None, None, "table", [])
    assert result["ok"] is False
    assert "flaky" not in registry


def test_draft_bridge_cleans_up_when_the_source_crashes(store: Store) -> None:
    """A crashing draft surfaces its error and leaves the registry untouched."""
    registry, draft = _draft_project(store)
    result = draft("broken", _BROKEN, None, None, "table", [])
    assert result["ok"] is False
    assert result.get("error") or result.get("detail")
    assert "broken" not in registry


def test_draft_bridge_claimless_is_computed_not_verified(store: Store) -> None:
    """No claim declared: certifiable on computation alone, verdict ``None``."""
    registry, draft = _draft_project(store)
    result = draft("clean_effect", _CLEAN, None, None, "table", [])
    assert result["ok"] is True
    assert result["verdict"] is None  # nothing inferred, honestly reported
    assert "clean_effect" not in registry


def test_concurrent_draft_and_derive_keep_the_certified_result(store: Store) -> None:
    """Same-name authoring serializes: a draft never clobbers a concurrent certify.

    Whichever order the per-name lock settles on, the certified derivation must hold
    the name at the end -- either the draft ran first, or it ran second and restored
    the certified entry it held aside.
    """
    registry = Registry()
    project = types.SimpleNamespace(
        registry=registry,
        config=types.SimpleNamespace(datasets=[]),
        make_runner=lambda: Runner(registry, executor=SubprocessExecutor(timeout=60)),
    )
    derive = make_derive_factory(
        project, store, make_runner=project.make_runner, dataset_names=lambda: []
    )("c1", "q")
    draft = make_draft_factory(
        project, make_runner=project.make_runner, dataset_names=lambda: []
    )("explore", "q")
    claim = {"x": "dose", "y": "response"}
    threads = [
        threading.Thread(
            target=lambda: derive("clean_effect", _CLEAN, claim, None, "table", [], [])
        ),
        threading.Thread(
            target=lambda: draft("clean_effect", _CLEAN, claim, None, "table", [])
        ),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert registry.get("clean_effect").is_certified


# -- the HTTP endpoints ------------------------------------------------------------


def _app_with_history(store: Store) -> TestClient:
    store.save_derivation(Derivation(name="m", question="q"))
    # v1 -> v2 refreshes the data; v2 -> v3 adds no change but a new version hash via a
    # code edit, so the history should attribute each move to its cause.
    store.append_run(
        _run(version="v1", estimate=0.40, created_at="2026-07-03T01:00:00+00:00")
    )
    store.append_run(
        _run(
            version="v2",
            estimate=0.42,
            created_at="2026-07-03T02:00:00+00:00",
            input_versions={"d": "d2"},
        )
    )
    app = create_app(load_datasets=lambda: {"d": []}, client=_NoClient(), store=store)
    return TestClient(app)


def test_history_endpoint_serves_versions_with_what_changed(store: Store) -> None:
    with _app_with_history(store) as http:
        history = http.get("/api/derivations/m/history").json()
        assert [r["shortVersion"] for r in history] == ["v2", "v1"]
        # The newest version attributes its move to the refreshed data; the oldest has
        # nothing before it, so nothing changed.
        assert history[0]["changed"] == ["data"]
        assert history[1]["changed"] == []
        assert history[0]["metrics"] == {"estimate": 0.42}


def test_history_404_for_unknown_name(store: Store) -> None:
    with _app_with_history(store) as http:
        assert http.get("/api/derivations/nope/history").status_code == 404


# -- the MLflow export wiring ------------------------------------------------------


def test_tracking_uri_prefers_env_over_setting(
    store: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert tracking_uri(store) is None
    store.set_config("mlflow_tracking_uri", "http://stored")
    assert tracking_uri(store) == "http://stored"
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://env")
    assert tracking_uri(store) == "http://env"


def test_emit_is_skipped_without_a_configured_uri(store: Store) -> None:
    # No URI configured: the export is a no-op (the run is still stored either way).
    assert emit_configured_run(store, _run()) is False


def test_mlflow_settings_endpoint_round_trips(store: Store) -> None:
    app = create_app(load_datasets=lambda: {"d": []}, client=_NoClient(), store=store)
    with TestClient(app) as http:
        # `editable` accompanies `source` so a UI can grey out a field the environment
        # pins; nothing here sets MLFLOW_TRACKING_URI, so the setting is editable.
        assert http.get("/api/settings/mlflow").json() == {
            "trackingUri": "",
            "source": "none",
            "editable": True,
        }
        http.put("/api/settings/mlflow", json={"tracking_uri": "http://mlflow"})
        body = http.get("/api/settings/mlflow").json()
        assert body == {
            "trackingUri": "http://mlflow",
            "source": "setting",
            "editable": True,
        }


def test_bridge_persists_to_the_authored_sidecar(store: Store, tmp_path: Path) -> None:
    # A chat-certified derivation must survive a restart as a registry entry, or
    # feature pipelines silently stop being training sources and scoring
    # transforms. The sidecar record plus load_into is that durability.
    from elbi_cli.authored import AuthoredStore

    registry = Registry()
    sidecar = AuthoredStore(tmp_path / "authored")
    project = types.SimpleNamespace(
        registry=registry,
        config=types.SimpleNamespace(datasets=[]),
        make_runner=lambda: Runner(registry, executor=SubprocessExecutor(timeout=60)),
        authored_store=sidecar,
    )
    derive = make_derive_factory(
        project,
        store,
        make_runner=project.make_runner,
        dataset_names=lambda: [d.name for d in project.config.datasets],
    )("c1", "engineer the features")
    outcome = derive(
        "clean_effect", _CLEAN, {"x": "dose", "y": "response"}, None, "table", [], []
    )
    assert outcome.certified

    fresh = Registry()
    restored = sidecar.load_into(fresh)
    assert "clean_effect" in restored
    assert fresh.get("clean_effect").is_certified
