"""The record of the last pull, and what it lets sync refuse.

Two states can only say "these differ". Three say who changed, which is what separates a
change to push from somebody else's edit to leave alone, and from a collision that must
not be applied silently. These drive that matrix through the real engine against a fake
app, one case per outcome.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from elbi_cli.config_sync import (
    STATE_PATH,
    Change,
    Surface,
    SyncError,
    plan,
    pull,
    read_state,
    sync,
)


class _FakeApp:
    """One surface's worth of app state, driven directly rather than over HTTP.

    The engine reaches the app only through a ``Surface``'s callables, so a fake app and
    a surface bound to it exercise the real classification, refusal and recording.
    """

    def __init__(self) -> None:
        self.objects: dict[str, dict[str, Any]] = {}
        self.pushed: list[str] = []
        self.deleted: list[str] = []

    def surface(self, name: str = "metrics") -> Surface:
        def push(
            client: Any, obj: str, body: dict[str, Any], existing: str | None
        ) -> None:
            self.objects[obj] = dict(body)
            self.pushed.append(obj)

        def delete(client: Any, obj_id: str) -> None:
            self.objects.pop(obj_id, None)
            self.deleted.append(obj_id)

        return Surface(
            name=name,
            folder=name,
            ext=".json",
            remote=lambda client: {k: dict(v) for k, v in self.objects.items()},
            remote_ids=lambda client: {k: k for k in self.objects},
            serialize=lambda body: json.dumps(body, indent=2, sort_keys=True) + "\n",
            deserialize=json.loads,
            push=push,
            delete=delete,
        )


@pytest.fixture
def app() -> _FakeApp:
    return _FakeApp()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    return tmp_path


def _write(repo: Path, surface: str, name: str, body: dict[str, Any]) -> None:
    directory = repo / surface
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _read(repo: Path, surface: str, name: str) -> dict[str, Any]:
    return json.loads((repo / surface / f"{name}.json").read_text())


def _actions(changes: list[Change]) -> dict[str, str]:
    return {c.name: c.action for c in changes}


# -- the baseline --------------------------------------------------------------
def test_pull_records_what_it_saw(app: _FakeApp, repo: Path) -> None:
    app.objects["revenue"] = {"sql": "select 1"}
    pull(None, repo, [app.surface()])  # type: ignore[arg-type]
    assert (repo / STATE_PATH).exists()
    recorded = read_state(repo)
    assert set(recorded["metrics"]) == {"revenue"}
    # a digest, not the body: the file is bookkeeping and must not become a second copy
    assert "select 1" not in (repo / STATE_PATH).read_text()


def test_an_absent_or_corrupt_baseline_is_an_empty_one(repo: Path) -> None:
    """A repo predating this, or one assembled by hand, still has to work."""
    assert read_state(repo) == {}
    (repo / STATE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (repo / STATE_PATH).write_text("{ not json")
    assert read_state(repo) == {}


# -- the matrix ---------------------------------------------------------------
def test_nothing_changed(app: _FakeApp, repo: Path) -> None:
    app.objects["revenue"] = {"sql": "select 1"}
    pull(None, repo, [app.surface()])  # type: ignore[arg-type]
    assert _actions(plan(None, repo, [app.surface()])) == {"revenue": "unchanged"}  # type: ignore[arg-type]


def test_changed_only_here_is_an_update(app: _FakeApp, repo: Path) -> None:
    app.objects["revenue"] = {"sql": "select 1"}
    pull(None, repo, [app.surface()])  # type: ignore[arg-type]
    _write(repo, "metrics", "revenue", {"sql": "select 2"})
    assert _actions(plan(None, repo, [app.surface()])) == {"revenue": "update"}  # type: ignore[arg-type]
    sync(None, repo, surfaces_=[app.surface()])  # type: ignore[arg-type]
    assert app.objects["revenue"] == {"sql": "select 2"}


def test_changed_only_in_the_app_is_drift_and_is_left_alone(
    app: _FakeApp, repo: Path
) -> None:
    """Pushing this repo's older copy would undo somebody's edit for no reason."""
    app.objects["revenue"] = {"sql": "select 1"}
    pull(None, repo, [app.surface()])  # type: ignore[arg-type]
    app.objects["revenue"] = {"sql": "select 99"}  # edited in the app
    assert _actions(plan(None, repo, [app.surface()])) == {"revenue": "drift"}  # type: ignore[arg-type]
    sync(None, repo, surfaces_=[app.surface()])  # type: ignore[arg-type]
    assert app.objects["revenue"] == {"sql": "select 99"}
    assert app.pushed == []


def test_changed_in_both_is_a_conflict_and_sync_refuses(
    app: _FakeApp, repo: Path
) -> None:
    """The lost-update problem: one of the two changes would simply be gone."""
    app.objects["revenue"] = {"sql": "select 1"}
    pull(None, repo, [app.surface()])  # type: ignore[arg-type]
    _write(repo, "metrics", "revenue", {"sql": "select 2"})
    app.objects["revenue"] = {"sql": "select 99"}
    assert _actions(plan(None, repo, [app.surface()])) == {"revenue": "conflict"}  # type: ignore[arg-type]

    with pytest.raises(SyncError) as raised:
        sync(None, repo, surfaces_=[app.surface()])  # type: ignore[arg-type]
    # the message has to name the object, or the reader cannot act on it
    assert "metrics/revenue" in str(raised.value)
    assert app.objects["revenue"] == {"sql": "select 99"}, "must not have been applied"


def test_force_overwrites_a_conflict(app: _FakeApp, repo: Path) -> None:
    """For somebody who has looked at both and decided."""
    app.objects["revenue"] = {"sql": "select 1"}
    pull(None, repo, [app.surface()])  # type: ignore[arg-type]
    _write(repo, "metrics", "revenue", {"sql": "select 2"})
    app.objects["revenue"] = {"sql": "select 99"}
    sync(None, repo, force=True, surfaces_=[app.surface()])  # type: ignore[arg-type]
    assert app.objects["revenue"] == {"sql": "select 2"}


def test_nothing_is_applied_when_any_surface_conflicts(
    app: _FakeApp, repo: Path
) -> None:
    """A partial sync leaves a state neither the repo nor the app describes."""
    app.objects["revenue"] = {"sql": "select 1"}
    other = _FakeApp()
    other.objects["signups"] = {"sql": "select 1"}
    surfaces = [app.surface("metrics"), other.surface("dashboards")]
    pull(None, repo, surfaces)  # type: ignore[arg-type]

    _write(repo, "dashboards", "signups", {"sql": "select 2"})  # a clean change
    _write(repo, "metrics", "revenue", {"sql": "select 2"})  # and a conflict
    app.objects["revenue"] = {"sql": "select 99"}

    with pytest.raises(SyncError):
        sync(None, repo, surfaces_=surfaces)  # type: ignore[arg-type]
    assert other.pushed == [], "the clean change must not have been applied either"


def test_with_no_baseline_a_difference_is_an_update(app: _FakeApp, repo: Path) -> None:
    """Degrades to the two-way comparison rather than refusing to work."""
    app.objects["revenue"] = {"sql": "select 1"}
    _write(repo, "metrics", "revenue", {"sql": "select 2"})
    assert _actions(plan(None, repo, [app.surface()])) == {"revenue": "update"}  # type: ignore[arg-type]
    sync(None, repo, surfaces_=[app.surface()])  # type: ignore[arg-type]
    assert app.objects["revenue"] == {"sql": "select 2"}


def test_an_object_only_here_is_a_create(app: _FakeApp, repo: Path) -> None:
    _write(repo, "metrics", "new", {"sql": "select 1"})
    assert _actions(plan(None, repo, [app.surface()])) == {"new": "create"}  # type: ignore[arg-type]


def test_an_object_only_in_the_app_is_a_delete_when_pruning(
    app: _FakeApp, repo: Path
) -> None:
    """And is left unmentioned otherwise, because only a pruning sync removes it."""
    app.objects["theirs"] = {"sql": "select 1"}
    (repo / "metrics").mkdir()  # the repo tracks metrics; it just has none right now
    assert _actions(plan(None, repo, [app.surface()], prune=True)) == {  # type: ignore[arg-type]
        "theirs": "delete"
    }
    assert _actions(plan(None, repo, [app.surface()])) == {}  # type: ignore[arg-type]


# -- deletion, in both directions --------------------------------------------
def test_pull_removes_a_file_for_an_object_deleted_in_the_app(
    app: _FakeApp, repo: Path
) -> None:
    """Without the baseline the file survived and the next sync put the object back."""
    app.objects["retired"] = {"sql": "select 1"}
    pull(None, repo, [app.surface()])  # type: ignore[arg-type]
    assert (repo / "metrics" / "retired.json").exists()

    del app.objects["retired"]
    written = pull(None, repo, [app.surface()])  # type: ignore[arg-type]
    assert Change("metrics", "retired", "delete") in written
    assert not (repo / "metrics" / "retired.json").exists()
    # and the object stays gone: a following sync has nothing to resurrect
    sync(None, repo, surfaces_=[app.surface()])  # type: ignore[arg-type]
    assert app.objects == {}


def test_pull_keeps_a_local_file_that_was_never_pushed(
    app: _FakeApp, repo: Path
) -> None:
    """Authored here and not yet synced is not the same as deleted upstream."""
    app.objects["known"] = {"sql": "select 1"}
    pull(None, repo, [app.surface()])  # type: ignore[arg-type]
    _write(repo, "metrics", "mine", {"sql": "select 2"})
    pull(None, repo, [app.surface()])  # type: ignore[arg-type]
    assert (repo / "metrics" / "mine.json").exists()
    assert _read(repo, "metrics", "mine") == {"sql": "select 2"}


# -- the baseline moves ------------------------------------------------------
def test_a_sync_moves_the_baseline_so_the_next_one_is_a_no_op(
    app: _FakeApp, repo: Path
) -> None:
    app.objects["revenue"] = {"sql": "select 1"}
    pull(None, repo, [app.surface()])  # type: ignore[arg-type]
    _write(repo, "metrics", "revenue", {"sql": "select 2"})
    sync(None, repo, surfaces_=[app.surface()])  # type: ignore[arg-type]
    assert _actions(plan(None, repo, [app.surface()])) == {"revenue": "unchanged"}  # type: ignore[arg-type]


def test_a_partial_run_does_not_forget_the_other_surfaces(
    app: _FakeApp, repo: Path
) -> None:
    """A `--surface metrics` pull must not claim the rest were seen."""
    other = _FakeApp()
    other.objects["signups"] = {"sql": "select 1"}
    app.objects["revenue"] = {"sql": "select 1"}
    pull(None, repo, [app.surface("metrics"), other.surface("dashboards")])  # type: ignore[arg-type]
    assert set(read_state(repo)) == {"metrics", "dashboards"}

    pull(None, repo, [app.surface("metrics")])  # type: ignore[arg-type]
    recorded = read_state(repo)
    assert set(recorded) == {"metrics", "dashboards"}
    assert set(recorded["dashboards"]) == {"signups"}


def test_the_digest_is_taken_from_the_written_form(app: _FakeApp, repo: Path) -> None:
    """Key order in transit must not read as a change; the file is what matters."""
    app.objects["revenue"] = {"a": 1, "b": 2}
    pull(None, repo, [app.surface()])  # type: ignore[arg-type]
    app.objects["revenue"] = {"b": 2, "a": 1}  # same object, different order
    assert _actions(plan(None, repo, [app.surface()])) == {"revenue": "unchanged"}  # type: ignore[arg-type]


# -- the plane boundary --------------------------------------------------------
#: Every surface `sync` may push, pinned. Adding one has to change this list, which is
#: the point: it forces a decision about which plane the new artifact belongs to before
#: an ordinary user's checkout can push it.
PUSHABLE_SURFACES = frozenset(
    {
        "certificates",
        "checks",
        "dashboards",
        "features",
        "metrics",
        "models",
        "monitors",
        "notebooks",
        "queries",
        "schedules",
        "workflows",
    }
)

#: Settings that decide what a deployment costs and what it can reach. They are the
#: platform's, applied by whoever runs it, and must never become something a checkout
#: can push: declaring a GPU profile in a repo and syncing it would be an escalation,
#: and so would redirecting a data connection.
PLATFORM_ONLY = frozenset(
    {
        "sources",
        "compute_profiles",
        "default_compute_profile",
        "egress",
        "sandbox",
        "sandbox_image",
        "targets",
    }
)

#: The rest: naming, wiring and retrieval preferences, which one practitioner authors
#: and pushing costs nothing. Kept so the two sets must account for every field between
#: them, which is what makes an unclassified addition fail.
PROJECT_KEYS = frozenset(
    {
        "project",
        "derivations_dir",
        "datasets",
        "ai_context",
        "search",
    }
)


def test_the_pushable_surfaces_are_the_ones_we_expect() -> None:
    """A guard on the boundary rather than on behaviour.

    `sync` authenticates as an ordinary user, so anything reachable through a surface is
    reachable by anybody with a checkout. That is right for content and wrong for
    platform settings, and the difference is invisible at the point somebody adds a
    surface -- so this fails then, rather than after it ships.
    """
    from elbi_cli.config_sync import surfaces

    assert {surface.name for surface in surfaces()} == PUSHABLE_SURFACES


def test_no_surface_is_named_after_a_platform_setting() -> None:
    """The specific mistake worth catching: a `compute_profiles` surface."""
    from elbi_cli.config_sync import surfaces

    for surface in surfaces():
        assert surface.name not in PLATFORM_ONLY
        assert surface.folder not in PLATFORM_ONLY


def test_every_project_key_is_classified_by_plane() -> None:
    """So the guard cannot rot, in either direction.

    A renamed setting must be renamed here too, and a *new* one must be classified
    before it can ship -- which a `fields >= PLATFORM_ONLY` check cannot ask, since no
    addition can fail it. Partitioning the fields instead puts the plane question where
    `PUSHABLE_SURFACES` already puts it for surfaces: at the moment somebody writes one.
    """
    from elbi_core.config import ProjectConfig

    fields = set(ProjectConfig.__dataclass_fields__)
    classified = PLATFORM_ONLY | PROJECT_KEYS
    assert PLATFORM_ONLY.isdisjoint(PROJECT_KEYS), PLATFORM_ONLY & PROJECT_KEYS
    assert fields == classified, fields ^ classified


# -- what a repo declares, and in what order -----------------------------------
def test_prune_leaves_a_type_the_repo_says_nothing_about(
    app: _FakeApp, repo: Path
) -> None:
    """An absent folder is not an empty one.

    A repo that tracks metrics and nothing else must not lose its certificates the first
    time somebody prunes; only a folder that exists claims its type, and an empty one is
    how a repo asks for none.
    """
    app.objects["revenue"] = {"sql": "select 1"}
    untracked = app.surface("certificates")
    assert sync(None, repo, prune=True, surfaces_=[untracked]) == []  # type: ignore[arg-type]
    assert app.deleted == [] and "revenue" in app.objects

    (repo / "certificates").mkdir()  # now the repo claims the type, and asks for none
    sync(None, repo, prune=True, surfaces_=[untracked])  # type: ignore[arg-type]
    assert app.deleted == ["revenue"]


def test_a_metric_is_pushed_after_the_metrics_it_is_built_from(
    app: _FakeApp, repo: Path
) -> None:
    """The app rejects a ratio whose inputs it has not seen, and files sort by name.

    ``checkout_conversion_rate`` sorts before both metrics it divides, so filename order
    is exactly wrong here -- which is the common case, not a contrived one.
    """
    surface = app.surface("metrics")
    surface.references = lambda body: {  # the real surface's rule, in miniature
        str(body[k]) for k in ("numerator", "denominator") if body.get(k)
    }
    _write(repo, "metrics", "checkout_users", {"type": "simple"})
    _write(repo, "metrics", "checkout_conversions", {"type": "simple"})
    _write(
        repo,
        "metrics",
        "checkout_conversion_rate",
        {
            "type": "ratio",
            "numerator": "checkout_conversions",
            "denominator": "checkout_users",
        },
    )
    sync(None, repo, surfaces_=[surface])  # type: ignore[arg-type]
    assert app.pushed.index("checkout_conversion_rate") > app.pushed.index(
        "checkout_conversions"
    )
    assert app.pushed.index("checkout_conversion_rate") > app.pushed.index(
        "checkout_users"
    )


def test_a_cycle_still_pushes_rather_than_hanging(app: _FakeApp, repo: Path) -> None:
    """Rejecting a cycle is the app's job; it explains one better than a sort could."""
    surface = app.surface("metrics")
    surface.references = lambda body: set(body.get("refs") or ())
    _write(repo, "metrics", "a", {"refs": ["b"]})
    _write(repo, "metrics", "b", {"refs": ["a"]})
    sync(None, repo, surfaces_=[surface])  # type: ignore[arg-type]
    assert sorted(app.pushed) == ["a", "b"]


def test_one_file_may_define_several_metrics(app: _FakeApp, repo: Path) -> None:
    """The shape every semantic layer writes, and the one this project's docs show."""
    from elbi_cli.config_sync import _explode_metrics

    surface = app.surface("metrics")
    surface.explode = _explode_metrics
    _write(
        repo,
        "metrics",
        "checkout",
        {
            "metrics": [
                {"name": "signups", "type": "simple"},
                {"name": "activations", "type": "simple"},
            ]
        },
    )
    sync(None, repo, surfaces_=[surface])  # type: ignore[arg-type]
    assert sorted(app.pushed) == ["activations", "signups"]
    assert "name" not in app.objects["signups"]  # the key is the name, not a field


def test_a_grouped_metric_without_a_name_is_a_clear_error(
    app: _FakeApp, repo: Path
) -> None:
    from elbi_cli.config_sync import _explode_metrics

    surface = app.surface("metrics")
    surface.explode = _explode_metrics
    _write(repo, "metrics", "checkout", {"metrics": [{"type": "simple"}]})
    with pytest.raises(SyncError, match="needs a name"):
        sync(None, repo, surfaces_=[surface])  # type: ignore[arg-type]
