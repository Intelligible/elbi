"""Finding out whether a newer release exists, when the user asks and only then.

Nothing here runs on its own. There is no check at startup, none on a timer, and none
folded into another command -- the only way any of this executes is ``elbi update``,
typed deliberately. That is the whole design, and it is what keeps the promise in
``docs/privacy.md`` true as written: there is nothing to opt out of, because there is
nothing that happens without being asked for.

The alternative was the common one. pip checks periodically unless told not to, and
sends a User-Agent carrying the CPU architecture, the operating system and its version,
and the Python and OpenSSL versions; dbt does the same on a one-second budget. Both
predate the expectation that a self-hosted tool makes no unrequested connections. uv,
which is newer, has no automatic check at all -- ``uv self update`` and nothing else --
and that is the model followed here.

What the request contains, exactly, so the documentation can state it without hedging:
the URL of a public JSON file on PyPI, and a User-Agent naming this tool and its
version. No identifier, no counter, nothing about the machine, and nothing telling one
install from another beyond the address the request came from.

Upgrading is left to whatever installed this. Running an installer from inside the
program it is installing means replacing files that are currently open, and the tools
that manage that -- uv, pipx, pip, the container runtime -- already do it correctly. So
this reports the command rather than running it, which is also what makes the output
useful when the answer is "you are up to date".
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Any

#: The public JSON document describing a project's releases.
PYPI_JSON = "https://pypi.org/pypi/{package}/json"

#: Long enough for a slow network, short enough that a wedged connection does not look
#: like a hang. The user is waiting on this command, unlike a background check, so the
#: budget is generous rather than the sub-second one a passive check needs.
TIMEOUT_SECONDS = 10.0

#: An operator's override, and what the container image sets for itself. Named rather
#: than sniffed because a container can be built any number of ways and a wrong guess
#: prints a command that does not apply.
INSTALL_ENV = "ELBI_INSTALL"

#: Set by the kubelet in every pod. A pod is a container, so the image's own
#: ``ELBI_INSTALL=docker`` is true and still not useful there: nobody upgrades a
#: Deployment with `docker compose`.
KUBERNETES_ENV = "KUBERNETES_SERVICE_HOST"

#: The two distributions a person installs. ``elbi`` brings the app and depends on the
#: CLI; ``elbi-cli`` is the MCP-only install. Whichever is present is the one to upgrade
#: -- telling somebody to upgrade ``elbi`` when they installed ``elbi-cli`` names a
#: package they do not have.
DISTRIBUTIONS = ("elbi", "elbi-cli")

#: Where a project points at its own release notes. PyPI stores whatever the maintainer
#: wrote, and the label is not standardised, so several spellings are tried in the order
#: a reader would prefer them.
CHANGELOG_KEYS = ("Changelog", "Change Log", "Release Notes", "Releases", "History")


class UpdateCheckError(Exception):
    """The check could not produce an answer, with a reason worth showing the user."""


@dataclass(frozen=True)
class Install:
    """How this copy was installed, and what upgrades it."""

    #: A stable identifier for the method, for tests and for the API.
    kind: str
    #: The command to run, ready to copy. Empty when the method has no single command.
    command: str
    #: Anything true about upgrading this way that the command alone does not say.
    note: str = ""


@dataclass(frozen=True)
class Report:
    """The answer to "is there a newer version", and what to do about it."""

    package: str
    installed: str
    latest: str | None
    install: Install
    #: Why there is no answer, when there is none. Empty when the check succeeded.
    error: str = ""
    #: Where the release notes live, if the project publishes a link to them. Printing
    #: two version numbers tells somebody that something changed but not what, which is
    #: the question they actually have before upgrading.
    changelog: str = ""

    @property
    def outdated(self) -> bool:
        """Whether a newer release exists.

        ``False`` when the check could not reach PyPI, because not knowing is not the
        same as being current and this must not report an upgrade it did not find.
        """
        if self.latest is None:
            return False
        from packaging.version import Version

        return Version(self.latest) > Version(self.installed)


def distribution() -> str:
    """The distribution this install is upgraded through.

    ``elbi`` wins when both are present, because it is the one a person asked for and
    the CLI came along with it.
    """
    for name in DISTRIBUTIONS:
        try:
            installed_version(name)
        except PackageNotFoundError:
            continue
        return name
    return DISTRIBUTIONS[0]


def current_version(package: str) -> str:
    """The installed version of ``package``, or ``0+unknown`` if it is not installed.

    Not an error: a checkout run from source has no distribution metadata, and the
    command should still be able to say what the latest release is.
    """
    try:
        return installed_version(package)
    except PackageNotFoundError:
        return "0+unknown"


def detect_install(package: str) -> Install:
    """Work out how this copy got here, and what upgrades it.

    Ordered most specific first. Every branch is decided by where the interpreter lives
    rather than by what is on PATH, because PATH says what would run next and this needs
    to know what is running now.
    """
    # Before the declared method, because the image declares itself a container and is
    # right about that -- it just does not know it is running under an orchestrator that
    # replaces it differently.
    if os.environ.get(KUBERNETES_ENV):
        return _kubernetes()

    named = os.environ.get(INSTALL_ENV, "").strip().lower()
    if named:
        return _named(named, package)

    if Path("/.dockerenv").exists():
        return _docker()

    prefix = Path(sys.prefix).resolve()
    parts = prefix.parts

    # uv keeps each tool in its own environment under `uv tool dir`, whose tail is
    # always uv/tools/<name>; matching the pair avoids a false positive on a project
    # that merely happens to have a directory called tools.
    if _has_pair(parts, "uv", "tools"):
        return Install(
            kind="uv-tool",
            command=f"uv tool upgrade {package}",
            note="`uv tool upgrade --all` upgrades everything installed this way.",
        )

    if _has_pair(parts, "pipx", "venvs"):
        return Install(kind="pipx", command=f"pipx upgrade {package}")

    if sys.prefix != sys.base_prefix:
        return Install(
            kind="venv",
            command=f"pip install --upgrade {package}",
            note="Run it with this environment active, or the upgrade lands elsewhere.",
        )

    return Install(
        kind="pip",
        command=f"pip install --upgrade {package}",
        note="Add --user if that is how it was installed.",
    )


def _named(kind: str, package: str) -> Install:
    """An install method the environment declared, rather than one inferred."""
    if kind == "docker":
        return _docker()
    if kind == "kubernetes":
        return _kubernetes()
    known = {
        "uv-tool": f"uv tool upgrade {package}",
        "pipx": f"pipx upgrade {package}",
        "pip": f"pip install --upgrade {package}",
        "venv": f"pip install --upgrade {package}",
    }
    if kind in known:
        return Install(kind=kind, command=known[kind])
    return Install(
        kind=kind,
        command="",
        note=f"{INSTALL_ENV} is set to {kind!r}, which this version does not know how "
        "to upgrade. Upgrade it the way you installed it.",
    )


def _kubernetes() -> Install:
    """A pod, which is replaced by changing the image its Deployment names."""
    return Install(
        kind="kubernetes",
        command="",
        note="Change the image tag in your values file or manifest and roll the "
        "deployment. The schema migrates forward on start; take a backup first, and "
        "run one replica through the migration before scaling back up.",
    )


def _docker() -> Install:
    """The container image, which is replaced rather than upgraded in place."""
    return Install(
        kind="docker",
        command="docker compose pull && docker compose up -d",
        note="Pin APP_VERSION to the new tag first. The schema migrates forward on "
        "start; take a backup before upgrading anything you care about.",
    )


def _has_pair(parts: tuple[str, ...], first: str, second: str) -> bool:
    """Whether ``parts`` contains ``first`` immediately followed by ``second``."""
    from itertools import pairwise

    return any(a == first and b == second for a, b in pairwise(parts))


def latest_release(package: str, *, allow_prerelease: bool = False) -> str:
    """The newest release of ``package`` on PyPI, or ``None`` if it cannot be reached.

    Read from the full release list rather than from ``info.version``, so that two
    things can be excluded that a person upgrading should not be sent to: a release
    whose files have all been yanked, and -- unless they are already running one -- a
    pre-release.

    Raises :class:`UpdateCheckError` when there is no answer, carrying which kind of
    no-answer it was: a name PyPI does not know is a different problem from a network
    that cannot be reached, and telling someone to check their connection when they
    mistyped a package name sends them the wrong way.
    """
    return _pick(_fetch(package), allow_prerelease=allow_prerelease)


def _fetch(package: str) -> dict[str, Any]:
    """The project's public JSON document, or an error saying why there is none."""
    import httpx

    from . import __version__

    try:
        response = httpx.get(
            PYPI_JSON.format(package=package),
            timeout=TIMEOUT_SECONDS,
            follow_redirects=True,
            headers={
                # Names the tool and its version and nothing else. PyPI asks automated
                # clients to identify themselves; this is the least that satisfies that.
                "User-Agent": f"elbi/{__version__} (+https://github.com/Intelligible/elbi)",
                "Accept": "application/json",
            },
        )
    except Exception as e:
        raise UpdateCheckError(f"could not reach PyPI: {e}") from e

    if response.status_code == 404:
        raise UpdateCheckError(f"PyPI has no project called {package!r}")
    try:
        response.raise_for_status()
        payload: dict[str, Any] = response.json()
    except Exception as e:
        raise UpdateCheckError(f"PyPI answered with {response.status_code}") from e
    return payload


def changelog_url(payload: dict[str, Any]) -> str:
    """The project's release-notes link, if it publishes one under any usual name."""
    urls = (payload.get("info") or {}).get("project_urls") or {}
    for key in CHANGELOG_KEYS:
        for label, url in urls.items():
            if label.strip().lower() == key.lower() and url:
                return str(url)
    return ""


def _pick(payload: dict[str, Any], *, allow_prerelease: bool) -> str:
    """The newest release in ``payload`` a person should be sent to."""
    from packaging.version import InvalidVersion, Version

    package = (payload.get("info") or {}).get("name") or "this project"
    candidates: list[Version] = []

    for raw, files in (payload.get("releases") or {}).items():
        try:
            parsed = Version(raw)
        except InvalidVersion:
            continue
        if parsed.is_prerelease and not allow_prerelease:
            continue
        if files and all(f.get("yanked") for f in files):
            continue
        candidates.append(parsed)
    if not candidates:
        raise UpdateCheckError(f"{package} has no published release to upgrade to")
    return str(max(candidates))


def check(
    package: str | None = None, *, allow_prerelease: bool | None = None
) -> Report:
    """Ask PyPI what the newest release is, and work out how to install it.

    The only function here that touches the network, and it is reached only from the
    ``update`` command.

    ``allow_prerelease`` left as ``None`` follows the installed version: somebody
    running a release candidate wants to hear about the next one, and somebody on a
    stable does not. Pass ``True`` to ask for them regardless, which is what ``--pre``
    does.
    """
    name = package or distribution()
    installed = current_version(name)
    from packaging.version import InvalidVersion, Version

    if allow_prerelease is None:
        try:
            allow_prerelease = Version(installed).is_prerelease
        except InvalidVersion:
            allow_prerelease = False
    latest: str | None
    notes = ""
    try:
        payload = _fetch(name)
        latest = _pick(payload, allow_prerelease=allow_prerelease)
        notes = changelog_url(payload)
        reason = ""
    except UpdateCheckError as e:
        latest, reason = None, str(e)
    return Report(
        package=name,
        installed=installed,
        latest=latest,
        install=detect_install(name),
        error=reason,
        changelog=notes,
    )
