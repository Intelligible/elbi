"""Which deployment a repo belongs to, and what it takes to act on a different one.

One environment variable otherwise serves a personal install and a company deployment
alike, so a stale value pushes personal work into production and nothing in either repo
can notice. A project that declares its targets can be checked instead of trusted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from elbi_cli.commands.config_cmd import resolve_host
from elbi_cli.config_sync import SyncError
from elbi_core.config import ConfigError, ProjectConfig

COMPANY = "https://elbi.acme.com"
PERSONAL = "http://127.0.0.1:7700"


def _project(tmp_path: Path, body: str) -> Path:
    (tmp_path / "elbi.yaml").write_text(body, encoding="utf-8")
    (tmp_path / "derivations").mkdir(exist_ok=True)
    return tmp_path


def _with_targets(tmp_path: Path) -> Path:
    return _project(
        tmp_path,
        f"""
project: acme
targets:
  dev:
    default: true
    host: {COMPANY}
  prod:
    host: https://prod.acme.com
""",
    )


# -- parsing and validation ----------------------------------------------------
def test_targets_are_parsed(tmp_path: Path) -> None:
    config = ProjectConfig.load(_with_targets(tmp_path) / "elbi.yaml")
    by_name = {t.name: t for t in config.targets}
    assert by_name["dev"].host == COMPANY
    assert by_name["dev"].default is True
    assert by_name["prod"].default is False


def test_a_project_may_declare_none(tmp_path: Path) -> None:
    """Every repo predating this, and every laptop, looks like this."""
    config = ProjectConfig.load(_project(tmp_path, "project: acme") / "elbi.yaml")
    assert config.targets == ()


def test_a_trailing_slash_on_a_host_is_not_a_different_host(tmp_path: Path) -> None:
    config = ProjectConfig.load(
        _project(tmp_path, f"project: acme\ntargets:\n  dev:\n    host: {COMPANY}/\n")
        / "elbi.yaml"
    )
    assert config.targets[0].host == COMPANY


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("project: acme\ntargets: [dev]", "must be a mapping of name to settings"),
        ("project: acme\ntargets:\n  dev: nope", "must be a mapping"),
        ("project: acme\ntargets:\n  dev: {}", "needs a 'host'"),
        ("project: acme\ntargets:\n  dev:\n    host: ''", "needs a 'host'"),
        (
            "project: acme\ntargets:\n  dev:\n    host: x\n    default: yes-please",
            "must be true or false",
        ),
        (
            "project: acme\ntargets:\n"
            "  a:\n    host: x\n    default: true\n"
            "  b:\n    host: y\n    default: true\n",
            "only one target may be the default",
        ),
    ],
)
def test_a_malformed_targets_block_is_refused(
    tmp_path: Path, body: str, expected: str
) -> None:
    """Named at load time: a target that will not parse must not be a silent default.

    Two defaults is the case worth refusing rather than picking. With two, what a bare
    command means depends on iteration order, and naming a host in the file is pointless
    if which host cannot be determined.
    """
    with pytest.raises(ConfigError) as raised:
        ProjectConfig.load(_project(tmp_path, body) / "elbi.yaml")
    assert expected in str(raised.value)


# -- resolution ----------------------------------------------------------------
def test_the_default_target_is_used_when_none_is_named(tmp_path: Path) -> None:
    assert resolve_host(_with_targets(tmp_path), None, None) == COMPANY


def test_a_named_target_wins(tmp_path: Path) -> None:
    root = _with_targets(tmp_path)
    assert resolve_host(root, "prod", None) == "https://prod.acme.com"


def test_a_url_that_disagrees_with_the_target_is_refused(tmp_path: Path) -> None:
    """The whole point: a stale variable must not quietly redirect a push.

    Refusing rather than preferring one is deliberate. Either could be what was meant,
    and the cost of guessing is somebody's work landing on another company's deployment.
    """
    root = _with_targets(tmp_path)
    with pytest.raises(SyncError) as raised:
        resolve_host(root, None, PERSONAL)
    message = str(raised.value)
    assert COMPANY in message and PERSONAL in message


def test_a_url_agreeing_with_the_target_is_fine(tmp_path: Path) -> None:
    root = _with_targets(tmp_path)
    assert resolve_host(root, "dev", COMPANY) == COMPANY
    # and a trailing slash is the same host, not a mismatch
    assert resolve_host(root, "dev", COMPANY + "/") == COMPANY


def test_the_environment_is_checked_too(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A variable left from another project is the likeliest way this goes wrong."""
    root = _with_targets(tmp_path)
    monkeypatch.setenv("ELBI_URL", PERSONAL)
    with pytest.raises(SyncError):
        resolve_host(root, None, None)
    monkeypatch.setenv("ELBI_URL", COMPANY)
    assert resolve_host(root, None, None) == COMPANY


def test_an_unknown_target_lists_the_ones_there_are(tmp_path: Path) -> None:
    root = _with_targets(tmp_path)
    with pytest.raises(SyncError) as raised:
        resolve_host(root, "staging", None)
    assert "dev" in str(raised.value) and "prod" in str(raised.value)


def test_targets_with_no_default_refuse_a_bare_command(tmp_path: Path) -> None:
    """Better than picking one: with none marked, nothing is meant by default."""
    root = _project(
        tmp_path,
        f"project: acme\ntargets:\n  dev:\n    host: {COMPANY}\n"
        "  prod:\n    host: https://prod.acme.com\n",
    )
    with pytest.raises(SyncError) as raised:
        resolve_host(root, None, None)
    assert "none is the default" in str(raised.value)
    # naming one still works
    assert resolve_host(root, "dev", None) == COMPANY


# -- the unconstrained case ----------------------------------------------------
def test_without_targets_the_url_is_taken_as_given(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _project(tmp_path, "project: acme")
    assert resolve_host(root, None, COMPANY) == COMPANY
    monkeypatch.setenv("ELBI_URL", PERSONAL)
    assert resolve_host(root, None, None) == PERSONAL


def test_without_targets_the_default_is_localhost(tmp_path: Path) -> None:
    root = _project(tmp_path, "project: acme")
    assert resolve_host(root, None, None) == PERSONAL


def test_naming_a_target_where_none_are_declared_says_so(tmp_path: Path) -> None:
    """Silently ignoring -t would look like it worked and act on the wrong host."""
    root = _project(tmp_path, "project: acme")
    with pytest.raises(SyncError) as raised:
        resolve_host(root, "prod", None)
    assert "declares none" in str(raised.value)


def test_a_directory_with_no_project_is_unconstrained(tmp_path: Path) -> None:
    """`schema --json` and the like run outside a project; that is not this error."""
    assert resolve_host(tmp_path, None, COMPANY) == COMPANY
