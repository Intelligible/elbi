"""``elbi update``: the only thing here that reaches the network, and only when asked.

No test in this file makes a real request. That is not only about speed -- a test suite
for a tool whose selling point is that it makes no unrequested connections should not
itself connect to PyPI on every run. The one function that would is replaced throughout.
"""

from __future__ import annotations

import json
import os
from typing import Any

import pytest
from typer.testing import CliRunner

from elbi_cli.app import app
from elbi_cli.update import (
    INSTALL_ENV,
    Install,
    Report,
    UpdateCheckError,
    check,
    detect_install,
    distribution,
    latest_release,
)

runner = CliRunner()


class _Response:
    """Enough of httpx's response surface for the code under test."""

    def __init__(self, payload: Any, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"status {self.status_code}")

    def json(self) -> Any:
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


def _releases(*versions: str, yanked: tuple[str, ...] = ()) -> dict[str, Any]:
    """A PyPI payload naming each version, with the named ones fully yanked."""
    return {
        "releases": {
            v: [{"filename": f"x-{v}.whl", "yanked": v in yanked}] for v in versions
        }
    }


@pytest.fixture
def pypi(monkeypatch: pytest.MonkeyPatch) -> Any:
    """Replace the one network call, and record what it was asked for."""
    import httpx

    calls: list[dict[str, Any]] = []
    box: dict[str, Any] = {"response": _Response(_releases("1.0.0"))}

    def fake_get(url: str, **kwargs: Any) -> Any:
        calls.append({"url": url, **kwargs})
        response = box["response"]
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(httpx, "get", fake_get)
    box["calls"] = calls
    return box


# --- what the request contains ----------------------------------------------------


def test_the_request_names_the_tool_and_nothing_about_the_machine(pypi: Any) -> None:
    """The claim the privacy page makes, pinned so a later change cannot break it.

    pip's User-Agent on the same errand carries CPU architecture, OS name and version,
    Python and OpenSSL versions. This one carries the tool and its version.
    """
    latest_release("elbi")
    headers = pypi["calls"][0]["headers"]
    agent = headers["User-Agent"]
    assert agent.startswith("elbi/")
    assert "github.com/Intelligible/elbi" in agent
    for leak in ("Darwin", "Linux", "Windows", "arm64", "x86", "CPython", "openssl"):
        assert leak.lower() not in agent.lower()


def test_only_a_public_json_document_is_requested(pypi: Any) -> None:
    latest_release("elbi")
    assert pypi["calls"][0]["url"] == "https://pypi.org/pypi/elbi/json"


def test_no_cookie_no_identifier_and_no_body_is_sent(pypi: Any) -> None:
    latest_release("elbi")
    call = pypi["calls"][0]
    assert set(call["headers"]) == {"User-Agent", "Accept"}
    assert "cookies" not in call
    assert "data" not in call and "json" not in call


def test_the_request_is_bounded_so_a_wedged_network_is_not_a_hang(pypi: Any) -> None:
    latest_release("elbi")
    assert pypi["calls"][0]["timeout"] > 0


# --- picking the release to recommend ----------------------------------------------


def test_the_newest_release_wins_not_the_last_one_listed(pypi: Any) -> None:
    """Dictionary order is not version order, and 1.10 is not less than 1.9."""
    pypi["response"] = _Response(_releases("1.0.0", "1.10.0", "1.9.0", "0.2.0"))
    assert latest_release("elbi") == "1.10.0"


def test_a_prerelease_is_not_offered_to_a_stable_install(pypi: Any) -> None:
    """Being told to upgrade from 1.0 onto 2.0rc1 is not an upgrade anyone asked for."""
    pypi["response"] = _Response(_releases("1.0.0", "2.0.0rc1", "2.0.0b2"))
    assert latest_release("elbi") == "1.0.0"


def test_a_prerelease_is_offered_to_somebody_already_running_one(pypi: Any) -> None:
    pypi["response"] = _Response(_releases("1.0.0", "2.0.0rc1"))
    assert latest_release("elbi", allow_prerelease=True) == "2.0.0rc1"


def test_a_fully_yanked_release_is_skipped(pypi: Any) -> None:
    """Yanking exists to stop people upgrading onto it, so recommending it is wrong."""
    pypi["response"] = _Response(_releases("1.0.0", "1.1.0", yanked=("1.1.0",)))
    assert latest_release("elbi") == "1.0.0"


def test_a_partly_yanked_release_is_still_installable(pypi: Any) -> None:
    """One bad wheel among several is not a withdrawn release."""
    payload = {
        "releases": {
            "1.1.0": [
                {"filename": "a.whl", "yanked": True},
                {"filename": "b.whl", "yanked": False},
            ]
        }
    }
    pypi["response"] = _Response(payload)
    assert latest_release("elbi") == "1.1.0"


def test_an_unparseable_version_is_ignored_rather_than_fatal(pypi: Any) -> None:
    pypi["response"] = _Response(_releases("1.0.0", "not-a-version"))
    assert latest_release("elbi") == "1.0.0"


# --- when there is no answer -------------------------------------------------------


def test_a_name_pypi_does_not_know_says_so(pypi: Any) -> None:
    """A different problem from an unreachable network, and a different fix."""
    pypi["response"] = _Response({}, status_code=404)
    with pytest.raises(UpdateCheckError, match="no project called"):
        latest_release("nosuchthing")


def test_an_unreachable_network_says_that_instead(pypi: Any) -> None:
    pypi["response"] = OSError("nodename nor servname provided")
    with pytest.raises(UpdateCheckError, match="could not reach PyPI"):
        latest_release("elbi")


def test_a_project_with_no_usable_release_says_so(pypi: Any) -> None:
    pypi["response"] = _Response(_releases("1.0.0", yanked=("1.0.0",)))
    with pytest.raises(UpdateCheckError, match="no published release"):
        latest_release("elbi")


def test_not_knowing_is_never_reported_as_being_up_to_date() -> None:
    """The failure that would matter: a check that failed must not look like a pass."""
    report = Report(
        package="elbi",
        installed="1.0.0",
        latest=None,
        install=Install(kind="pip", command="x"),
        error="could not reach PyPI",
    )
    assert report.outdated is False
    assert report.error


# --- which command to print --------------------------------------------------------


@pytest.mark.parametrize(
    ("prefix", "kind", "command"),
    [
        ("/home/u/.local/share/uv/tools/elbi", "uv-tool", "uv tool upgrade elbi"),
        ("/home/u/.local/pipx/venvs/elbi", "pipx", "pipx upgrade elbi"),
    ],
)
def test_the_install_method_is_read_from_where_the_interpreter_lives(
    monkeypatch: pytest.MonkeyPatch, prefix: str, kind: str, command: str
) -> None:
    """PATH says what would run next; sys.prefix says what is running now."""
    monkeypatch.delenv(INSTALL_ENV, raising=False)
    monkeypatch.setattr("sys.prefix", prefix)
    monkeypatch.setattr("sys.base_prefix", "/usr")
    install = detect_install("elbi")
    assert install.kind == kind
    assert install.command == command


def test_a_directory_merely_called_tools_is_not_a_uv_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`uv/tools` is the signal, not `tools` anywhere in the path."""
    monkeypatch.delenv(INSTALL_ENV, raising=False)
    monkeypatch.setattr("sys.prefix", "/srv/tools/elbi/venv")
    monkeypatch.setattr("sys.base_prefix", "/usr")
    assert detect_install("elbi").kind == "venv"


def test_a_virtualenv_is_told_apart_from_a_system_install(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(INSTALL_ENV, raising=False)
    monkeypatch.setattr("sys.prefix", "/srv/app/.venv")
    monkeypatch.setattr("sys.base_prefix", "/usr")
    assert detect_install("elbi").kind == "venv"

    monkeypatch.setattr("sys.prefix", "/usr")
    assert detect_install("elbi").kind == "pip"


def test_the_environment_can_say_what_the_filesystem_cannot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A container can be built any number of ways; the image states its own answer."""
    monkeypatch.setenv(INSTALL_ENV, "docker")
    install = detect_install("elbi")
    assert install.kind == "docker"
    assert "docker compose pull" in install.command
    assert "backup" in install.note


def test_an_unknown_declared_method_admits_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Better than printing a confident command that does not apply here."""
    monkeypatch.setenv(INSTALL_ENV, "nixpkgs")
    install = detect_install("elbi")
    assert install.command == ""
    assert "nixpkgs" in install.note


def test_the_distribution_actually_installed_is_the_one_named() -> None:
    """Telling somebody to upgrade a package they do not have wastes their time."""
    assert distribution() in {"elbi", "elbi-cli"}


# --- the command ------------------------------------------------------------------


def test_up_to_date_exits_zero_and_says_so(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("elbi_cli.update.current_version", lambda _: "1.0.0")
    pypi["response"] = _Response(_releases("1.0.0"))
    result = runner.invoke(app, ["update", "--package", "elbi"])
    assert result.exit_code == 0
    assert "latest release" in result.output


def test_an_available_update_prints_the_command_and_exits_non_zero(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-zero so a script can act on it; the wording is what a person reads."""
    monkeypatch.setattr("elbi_cli.update.current_version", lambda _: "1.0.0")
    monkeypatch.setenv(INSTALL_ENV, "uv-tool")
    pypi["response"] = _Response(_releases("1.0.0", "2.0.0"))
    result = runner.invoke(app, ["update", "--package", "elbi"])
    assert result.exit_code == 1
    assert "1.0.0 -> 2.0.0" in result.output
    assert "uv tool upgrade elbi" in result.output


def test_quiet_says_nothing_when_current(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("elbi_cli.update.current_version", lambda _: "1.0.0")
    pypi["response"] = _Response(_releases("1.0.0"))
    result = runner.invoke(app, ["update", "--package", "elbi", "--quiet"])
    assert result.exit_code == 0
    assert result.output.strip() == ""


def test_quiet_is_one_parseable_line_when_behind(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("elbi_cli.update.current_version", lambda _: "1.0.0")
    pypi["response"] = _Response(_releases("1.0.0", "2.0.0"))
    result = runner.invoke(app, ["update", "--package", "elbi", "--quiet"])
    assert result.exit_code == 1
    assert result.output.strip() == "elbi 1.0.0 -> 2.0.0"


def test_a_failed_check_reports_the_reason_and_the_version_in_hand(pypi: Any) -> None:
    pypi["response"] = _Response({}, status_code=404)
    result = runner.invoke(app, ["update", "--package", "nosuchthing"])
    assert result.exit_code == 1
    assert "no project called" in result.output


def test_nothing_else_in_the_cli_reaches_the_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The design, asserted: no startup check, no timer, nothing folded into a command.

    `--version` and `--help` are the two paths every invocation passes through, so a
    check smuggled into either would fire for everybody. Both must be silent.
    """
    import httpx

    def refuse(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("the CLI made a network request without being asked")

    monkeypatch.setattr(httpx, "get", refuse)
    monkeypatch.setattr(httpx, "post", refuse)

    assert runner.invoke(app, ["--version"]).exit_code == 0
    assert runner.invoke(app, ["--help"]).exit_code == 0


# --- the report, end to end --------------------------------------------------------


def test_check_composes_the_whole_answer(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("elbi_cli.update.current_version", lambda _: "1.0.0")
    monkeypatch.setenv(INSTALL_ENV, "pipx")
    pypi["response"] = _Response(_releases("1.0.0", "1.2.0"))
    report = check("elbi")
    assert (report.installed, report.latest, report.outdated) == (
        "1.0.0",
        "1.2.0",
        True,
    )
    assert report.install.command == "pipx upgrade elbi"
    assert report.error == ""


def test_a_prerelease_install_is_offered_prereleases(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Somebody running 2.0.0rc1 wants to hear about rc2."""
    monkeypatch.setattr("elbi_cli.update.current_version", lambda _: "2.0.0rc1")
    pypi["response"] = _Response(_releases("1.0.0", "2.0.0rc1", "2.0.0rc2"))
    assert check("elbi").latest == "2.0.0rc2"


def test_a_source_checkout_with_no_metadata_still_reports_the_latest(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Running from a clone is normal during development and must not be an error."""
    from importlib.metadata import PackageNotFoundError

    def missing(_: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("elbi_cli.update.installed_version", missing)
    pypi["response"] = _Response(_releases("1.0.0"))
    report = check("elbi")
    assert report.installed == "0+unknown"
    assert report.latest == "1.0.0"
    assert report.outdated is True


def test_the_payload_shape_matches_what_pypi_actually_returns() -> None:
    """Guards the parsing against a fixture that drifted from the real document.

    The recorded shape is the one PyPI served for a real project; if the reader stops
    understanding it, this fails without anyone needing to be online to find out.
    """
    recorded = json.loads(
        '{"info": {"version": "0.28.1"}, '
        '"releases": {"0.28.0": [{"filename": "a.whl", "yanked": false}], '
        '"0.28.1": [{"filename": "b.whl", "yanked": false}]}}'
    )
    assert "releases" in recorded
    first = next(iter(recorded["releases"].values()))
    assert "yanked" in first[0]


def test_a_pod_is_not_told_to_run_docker_compose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pod is a container, so the image's own answer is true and still not useful.

    Nobody upgrades a Deployment with `docker compose pull`, and printing it to an
    operator who is looking at kubectl sends them somewhere that cannot work.
    """
    from elbi_cli.update import KUBERNETES_ENV

    monkeypatch.setenv(INSTALL_ENV, "docker")
    monkeypatch.setenv(KUBERNETES_ENV, "10.0.0.1")
    install = detect_install("elbi")
    assert install.kind == "kubernetes"
    assert "docker compose" not in install.note
    assert "image tag" in install.note


def test_a_plain_container_still_gets_the_compose_command(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elbi_cli.update import KUBERNETES_ENV

    monkeypatch.setenv(INSTALL_ENV, "docker")
    monkeypatch.delenv(KUBERNETES_ENV, raising=False)
    assert "docker compose pull" in detect_install("elbi").command


def test_a_method_with_no_single_command_says_so_rather_than_inventing_one() -> None:
    """Kubernetes has no one-liner, and a made-up one is worse than none."""
    from elbi_cli.update import _kubernetes

    install = _kubernetes()
    assert install.command == ""
    assert install.note


# --- the branches the happy path never reaches ------------------------------------


def test_neither_distribution_installed_still_names_one(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Running from a clone, where nothing is installed; it still has to work."""
    from importlib.metadata import PackageNotFoundError

    def missing(_: str) -> str:
        raise PackageNotFoundError

    monkeypatch.setattr("elbi_cli.update.installed_version", missing)
    assert distribution() == "elbi"


def test_a_dockerenv_file_is_recognised_without_the_variable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """An image built by someone else will not have set ELBI_INSTALL."""
    import elbi_cli.update as module

    monkeypatch.delenv(INSTALL_ENV, raising=False)
    monkeypatch.delenv("KUBERNETES_SERVICE_HOST", raising=False)
    marker = tmp_path / ".dockerenv"
    marker.write_text("")

    real = module.Path

    class _Path:
        def __init__(self, value: str) -> None:
            self._value = marker if value == "/.dockerenv" else real(value)

        def exists(self) -> bool:
            return bool(self._value.exists())

        def resolve(self) -> Any:
            return real(self._value).resolve()

    monkeypatch.setattr(module, "Path", _Path)
    assert detect_install("elbi").kind == "docker"


def test_kubernetes_can_also_be_declared_rather_than_detected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from elbi_cli.update import KUBERNETES_ENV

    monkeypatch.delenv(KUBERNETES_ENV, raising=False)
    monkeypatch.setenv(INSTALL_ENV, "kubernetes")
    assert detect_install("elbi").kind == "kubernetes"


def test_a_server_error_is_reported_as_the_status_it_was(pypi: Any) -> None:
    """Not a 404 and not a dead network: PyPI answered, badly."""
    pypi["response"] = _Response({}, status_code=503)
    with pytest.raises(UpdateCheckError, match="503"):
        latest_release("elbi")


def test_an_unreadable_installed_version_does_not_stop_the_check(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A local build can carry a version string PEP 440 will not parse.

    That is a reason to skip the pre-release question, not a reason to refuse to say
    what the latest release is.
    """
    monkeypatch.setattr("elbi_cli.update.current_version", lambda _: "not-a-version")
    pypi["response"] = _Response(_releases("1.0.0", "2.0.0rc1"))
    report = check("elbi")
    assert report.latest == "1.0.0"


def test_a_method_with_no_command_tells_the_user_to_use_their_own(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Kubernetes and an unknown method both land here; a made-up command is worse."""
    monkeypatch.setattr("elbi_cli.update.current_version", lambda _: "1.0.0")
    monkeypatch.setenv(INSTALL_ENV, "kubernetes")
    pypi["response"] = _Response(_releases("1.0.0", "2.0.0"))
    result = runner.invoke(app, ["update", "--package", "elbi"])
    assert result.exit_code == 1
    assert "the way you installed it" in result.output
    # And the note is what carries the actual instruction in that case.
    assert "image tag" in result.output


# --- release notes -----------------------------------------------------------------


def _with_urls(**urls: str) -> dict[str, Any]:
    payload = _releases("1.0.0", "2.0.0")
    payload["info"] = {"name": "elbi", "project_urls": urls}
    return payload


@pytest.mark.parametrize(
    "label", ["Changelog", "Change Log", "Release Notes", "Releases", "History"]
)
def test_release_notes_are_found_under_any_name_a_maintainer_used(label: str) -> None:
    """PyPI stores whatever was written; the label is not standardised."""
    from elbi_cli.update import changelog_url

    assert changelog_url(_with_urls(**{label: "https://x/notes"})) == "https://x/notes"


def test_the_preferred_label_wins_when_a_project_publishes_several() -> None:
    from elbi_cli.update import changelog_url

    payload = _with_urls(History="https://x/history", Changelog="https://x/changes")
    assert changelog_url(payload) == "https://x/changes"


def test_a_project_with_no_notes_link_is_not_an_error() -> None:
    from elbi_cli.update import changelog_url

    assert changelog_url(_with_urls(Homepage="https://x")) == ""
    assert changelog_url({}) == ""


def test_the_notes_link_is_shown_with_the_new_version(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two version numbers say something changed, not what."""
    monkeypatch.setattr("elbi_cli.update.current_version", lambda _: "1.0.0")
    pypi["response"] = _Response(_with_urls(Changelog="https://x/CHANGELOG.md"))
    result = runner.invoke(app, ["update", "--package", "elbi"])
    assert "https://x/CHANGELOG.md" in result.output


# --- --pre and --apply -------------------------------------------------------------


def test_pre_asks_for_prereleases_a_stable_install_would_not_be_offered(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("elbi_cli.update.current_version", lambda _: "1.0.0")
    pypi["response"] = _Response(_releases("1.0.0", "2.0.0rc1"))

    assert runner.invoke(app, ["update", "--package", "elbi"]).exit_code == 0
    asked = runner.invoke(app, ["update", "--package", "elbi", "--pre"])
    assert asked.exit_code == 1
    assert "2.0.0rc1" in asked.output


def test_apply_hands_the_process_to_the_installer(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """exec, not a subprocess: nothing of Elbi may be running while its files change."""
    handed: dict[str, Any] = {}

    def fake_exec(program: str, argv: list[str]) -> None:
        handed["program"], handed["argv"] = program, argv
        raise SystemExit(0)

    monkeypatch.setattr("elbi_cli.update.current_version", lambda _: "1.0.0")
    monkeypatch.setenv(INSTALL_ENV, "uv-tool")
    monkeypatch.setattr(os, "execvp", fake_exec)
    pypi["response"] = _Response(_releases("1.0.0", "2.0.0"))

    runner.invoke(app, ["update", "--package", "elbi", "--apply"])
    assert handed["program"] == "uv"
    assert handed["argv"] == ["uv", "tool", "upgrade", "elbi"]


def test_apply_refuses_a_command_that_needs_a_shell(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Docker's is two commands joined by `&&`; running it means invoking a shell."""
    monkeypatch.setattr("elbi_cli.update.current_version", lambda _: "1.0.0")
    monkeypatch.setenv(INSTALL_ENV, "docker")
    monkeypatch.setattr(
        os, "execvp", lambda *a: pytest.fail("a shell command was executed")
    )
    pypi["response"] = _Response(_releases("1.0.0", "2.0.0"))

    result = runner.invoke(app, ["update", "--package", "elbi", "--apply"])
    assert result.exit_code == 1
    assert "not run for you" in result.output
    assert "docker compose pull" in result.output


def test_apply_refuses_when_there_is_no_command_at_all(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("elbi_cli.update.current_version", lambda _: "1.0.0")
    monkeypatch.setenv(INSTALL_ENV, "kubernetes")
    monkeypatch.setattr(
        os, "execvp", lambda *a: pytest.fail("there was nothing to execute")
    )
    pypi["response"] = _Response(_releases("1.0.0", "2.0.0"))

    result = runner.invoke(app, ["update", "--package", "elbi", "--apply"])
    assert result.exit_code == 1
    assert "no single command" in result.output


def test_apply_that_cannot_start_the_installer_prints_it_instead(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """uv missing from PATH is a reason to show the command, not to fail quietly."""

    def missing(*args: Any) -> None:
        raise OSError("No such file or directory: 'uv'")

    monkeypatch.setattr("elbi_cli.update.current_version", lambda _: "1.0.0")
    monkeypatch.setenv(INSTALL_ENV, "uv-tool")
    monkeypatch.setattr(os, "execvp", missing)
    pypi["response"] = _Response(_releases("1.0.0", "2.0.0"))

    result = runner.invoke(app, ["update", "--package", "elbi", "--apply"])
    assert result.exit_code == 1
    assert "uv tool upgrade elbi" in result.output


def test_apply_does_nothing_when_already_up_to_date(
    pypi: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check comes first; --apply is not a reason to reinstall what you have."""
    monkeypatch.setattr("elbi_cli.update.current_version", lambda _: "1.0.0")
    monkeypatch.setattr(
        os, "execvp", lambda *a: pytest.fail("it upgraded an up-to-date install")
    )
    pypi["response"] = _Response(_releases("1.0.0"))
    assert runner.invoke(app, ["update", "--package", "elbi", "--apply"]).exit_code == 0


def test_apply_with_neither_a_command_nor_a_note_still_exits_cleanly() -> None:
    """Belt and braces: every install method today has one or the other."""
    import typer

    from elbi_cli.commands.update_cmd import _apply

    with pytest.raises(typer.Exit):
        _apply("", "")
