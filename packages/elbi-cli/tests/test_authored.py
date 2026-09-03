"""Tests for the agent-authored derivation sidecar store."""

from __future__ import annotations

from pathlib import Path

import pytest

from elbi_cli.authored import AuthoredStore
from elbi_core import (
    Artifact,
    Context,
    Dataset,
    DerivationError,
    Registry,
    certify,
    derivation,
    propose,
    serve,
)
from elbi_core.errors import ConfigError


def _src(name: str) -> str:
    return f'def {name}(ctx):\n    "A derivation."\n    return [{{"a": 1}}]\n'


def _agent(reg: Registry, name: str, *, certified: bool = True, **kw: object):
    proposed = propose(name, _src(name), registry=reg, **kw)  # type: ignore[arg-type]
    return certify(proposed, registry=reg) if certified else proposed


def test_save_then_reload_certified(tmp_path: Path) -> None:
    reg = Registry()
    d = _agent(
        reg, "score", serve=serve.table(title="S"), inputs={"sales": Dataset("sales")}
    )
    store = AuthoredStore(tmp_path / "authored")
    store.save(d)

    fresh = Registry()
    assert store.load_into(fresh) == ("score",)
    loaded = fresh.get("score")
    assert loaded.is_agent_authored and loaded.is_certified
    assert loaded.serve is not None and loaded.serve.title == "S"
    assert "sales" in loaded.dataset_inputs()
    assert loaded.source is not None  # source kept; it runs only via the sandbox


def test_reload_preserves_proposed_status(tmp_path: Path) -> None:
    reg = Registry()
    store = AuthoredStore(tmp_path / "authored")
    store.save(_agent(reg, "p", certified=False, serve=serve.json()))
    fresh = Registry()
    store.load_into(fresh)
    assert fresh.get("p").status == "proposed"


def test_reloaded_derivation_refuses_in_process(tmp_path: Path) -> None:
    reg = Registry()
    store = AuthoredStore(tmp_path / "authored")
    store.save(_agent(reg, "d", serve=serve.json()))
    fresh = Registry()
    store.load_into(fresh)
    with pytest.raises(DerivationError, match="in-process"):
        fresh.get("d").compute(Context({}))  # never runs in the host process


def test_internal_agent_derivation_roundtrips_without_serve(tmp_path: Path) -> None:
    reg = Registry()
    store = AuthoredStore(tmp_path / "authored")
    store.save(_agent(reg, "m", serve=None))
    fresh = Registry()
    store.load_into(fresh)
    assert fresh.get("m").serve is None


def test_remove_deletes_record(tmp_path: Path) -> None:
    reg = Registry()
    store = AuthoredStore(tmp_path / "authored")
    store.save(_agent(reg, "d", serve=serve.json()))
    store.remove("d")
    assert store.load_into(Registry()) == ()


def test_save_rejects_human_authored(tmp_path: Path) -> None:
    reg = Registry()

    @derivation(name="human", serve=serve.json(), registry=reg)
    def human(ctx: Context) -> Artifact:
        return Artifact.json(1)

    with pytest.raises(ConfigError, match="agent-authored"):
        AuthoredStore(tmp_path / "authored").save(human)


def test_load_into_skips_name_already_present(tmp_path: Path) -> None:
    store = AuthoredStore(tmp_path / "authored")
    store.save(_agent(Registry(), "d", serve=serve.json()))
    target = Registry()
    propose("d", _src("d"), serve=serve.json(), registry=target)  # already present
    assert store.load_into(target) == ()  # file-authored takes precedence


def test_load_into_missing_dir_returns_empty(tmp_path: Path) -> None:
    assert AuthoredStore(tmp_path / "nope").load_into(Registry()) == ()


def test_read_returns_record_or_none(tmp_path: Path) -> None:
    store = AuthoredStore(tmp_path / "authored")
    assert store.read("missing") is None
    store.save(_agent(Registry(), "d", serve=serve.json()))
    record = store.read("d")
    assert record is not None
    assert record["name"] == "d" and record["status"] == "certified"


def test_trash_hides_from_load_into(tmp_path: Path) -> None:
    store = AuthoredStore(tmp_path / "authored")
    store.save(_agent(Registry(), "probe", serve=serve.json()))
    assert store.trash("probe") is True
    # The same glob load_into always uses -- a trashed sidecar is simply not
    # there for it to find, no loader-side check needed.
    assert store.load_into(Registry()) == ()
    assert store.read("probe") is None  # read() also only looks at the live path


def test_restore_brings_it_back_for_load_into(tmp_path: Path) -> None:
    store = AuthoredStore(tmp_path / "authored")
    store.save(_agent(Registry(), "probe", serve=serve.json()))
    store.trash("probe")
    assert store.restore("probe") is True
    fresh = Registry()
    assert store.load_into(fresh) == ("probe",)
    assert fresh.get("probe").is_agent_authored


def test_load_one_restores_a_single_derivation_without_rewalking_disk(
    tmp_path: Path,
) -> None:
    store = AuthoredStore(tmp_path / "authored")
    store.save(_agent(Registry(), "a", serve=serve.json()))
    store.save(_agent(Registry(), "b", serve=serve.json()))
    store.trash("a")
    store.restore("a")
    fresh = Registry()
    # load_one brings back exactly the one name, unlike load_into's full sweep.
    assert store.load_one("a", fresh) is True
    assert "a" in fresh and "b" not in fresh


def test_trash_of_a_missing_name_returns_false(tmp_path: Path) -> None:
    store = AuthoredStore(tmp_path / "authored")
    assert store.trash("nope") is False


def test_restore_of_something_never_trashed_returns_false(tmp_path: Path) -> None:
    store = AuthoredStore(tmp_path / "authored")
    store.save(_agent(Registry(), "live", serve=serve.json()))
    assert store.restore("live") is False  # it was never moved to .trash/


def test_remove_cleans_up_a_trashed_sidecar_too(tmp_path: Path) -> None:
    store = AuthoredStore(tmp_path / "authored")
    store.save(_agent(Registry(), "probe", serve=serve.json()))
    store.trash("probe")
    store.remove("probe")
    assert store.restore("probe") is False  # nothing left to restore
    assert store.load_into(Registry()) == ()


def test_render_module_with_inputs_and_serve() -> None:
    import ast

    from elbi_cli.authored import render_module

    module = render_module(
        {
            "name": "score",
            "source": 'def score(ctx):\n    "Doc."\n    return [{"a": 1}]\n',
            "inputs": {"sales": "sales"},
            "serve": {"format": "table", "title": "S"},
        },
        promoted_on="2026-01-01",
    )
    ast.parse(module)  # valid Python
    assert "AI-generated" in module and "human-maintained" in module
    assert "DO NOT EDIT" not in module and "@generated" not in module
    assert "from elbi_core import Dataset, derivation, serve" in module
    assert 'inputs={"sales": Dataset("sales")}' in module
    assert 'serve=serve.table(title="S")' in module
    assert "def score(ctx):" in module


def test_render_module_minimal() -> None:
    import ast

    from elbi_cli.authored import render_module

    module = render_module(
        {"name": "m", "source": "def m(ctx):\n    return 1\n"}, promoted_on="2026-01-01"
    )
    ast.parse(module)
    assert "from elbi_core import derivation" in module
    assert "@derivation\n" in module  # no decorator args
