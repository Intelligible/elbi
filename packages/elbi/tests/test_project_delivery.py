"""How a project reaches a deployed app, and what happens when it does not.

Three delivery shapes, because the customers are genuinely different: a laptop takes the
bundled starter project, an air-gapped hospital bakes its own into an image, and a bank
with an internal GitLab has a sidecar pull it. The first can be defaulted and the other
two cannot.

The behaviour worth testing hardest is the refusal. When something else is supposed to
deliver the project and has not, seeding the starter would leave the app serving example
data under the customer's project name: plausible answers, confidently wrong, with
nothing in the logs that reads as a failure. These assert it stops instead.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from elbi.serve import project_revision, project_root

REPO = Path(__file__).resolve().parents[3]
ENTRYPOINT = REPO / "deploy" / "docker-entrypoint.sh"
CHART = REPO / "deploy" / "helm" / "elbi"


def _run_entrypoint(tmp_path: Path, **env: str) -> subprocess.CompletedProcess[str]:
    """Run the entrypoint with a stub `elbi-app` so it never really serves."""
    stub_dir = tmp_path / "bin"
    stub_dir.mkdir(exist_ok=True)
    stub = stub_dir / "elbi-app"
    # Echoes DB_URI as well as the argv, so a test can assert on what the entrypoint
    # assembled without the app ever connecting to anything.
    stub.write_text('#!/bin/sh\necho "SERVED $*"\necho "DB_URI=${DB_URI:-}"\n')
    stub.chmod(0o755)
    return subprocess.run(
        ["sh", str(ENTRYPOINT)],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "PATH": f"{stub_dir}:{os.environ['PATH']}",
            "DB_URI": "sqlite:test.db",
            **env,
        },
    )


def test_a_missing_project_is_fatal_when_git_was_supposed_to_deliver_it(
    tmp_path: Path,
) -> None:
    project = tmp_path / "git-current"
    project.mkdir()

    result = _run_entrypoint(
        tmp_path,
        ELBI_PROJECT_DIR=str(project),
        ELBI_PROJECT_SOURCE="git",
        ELBI_STATE_DIR=str(tmp_path / "state"),
    )

    assert result.returncode == 1
    assert "SERVED" not in result.stdout
    # The message has to name the cause, because the symptom an operator sees is a pod
    # that will not start and the cause is two containers away.
    assert "git" in result.stderr
    assert "git-sync" in result.stderr
    # And it must not have quietly scaffolded anything into the directory.
    assert not (project / "elbi.yaml").exists()


def test_a_missing_project_is_fatal_when_it_should_have_been_in_the_image(
    tmp_path: Path,
) -> None:
    project = tmp_path / "baked"
    project.mkdir()

    result = _run_entrypoint(
        tmp_path,
        ELBI_PROJECT_DIR=str(project),
        ELBI_PROJECT_SOURCE="image",
        ELBI_STATE_DIR=str(tmp_path / "state"),
    )

    assert result.returncode == 1
    assert "SERVED" not in result.stdout


def test_a_delivered_project_is_served_and_state_goes_elsewhere(tmp_path: Path) -> None:
    """The read-only-source shape: serve the project, write state somewhere else."""
    project = tmp_path / "baked"
    project.mkdir()
    (project / "elbi.yaml").write_text("project: theirs\n")
    state = tmp_path / "state"

    result = _run_entrypoint(
        tmp_path,
        ELBI_PROJECT_DIR=str(project),
        ELBI_PROJECT_SOURCE="image",
        ELBI_STATE_DIR=str(state),
    )

    assert result.returncode == 0, result.stderr
    assert f"--directory {project}" in result.stdout
    # State went outside the project, which is what lets the project be read-only.
    assert state.is_dir()
    assert not (project / ".elbi").exists()


def test_the_bundled_default_still_seeds_and_serves(tmp_path: Path) -> None:
    """The laptop case, unchanged: an empty directory gets the starter project."""
    project = tmp_path / "data"

    result = _run_entrypoint(
        tmp_path,
        ELBI_PROJECT_DIR=str(project),
        ELBI_STATE_DIR=str(project / ".elbi"),
    )

    assert result.returncode == 0, result.stderr
    assert "seeding the starter project" in result.stdout
    assert f"--directory {project}" in result.stdout


# -- the chart ---------------------------------------------------------------------


def _to_json(rendered: str) -> str:
    """The rendered manifest as JSON, so assertions read as data not as substrings."""
    import yaml

    docs = [d for d in yaml.safe_load_all(rendered) if d]
    assert len(docs) == 1, f"expected one manifest, got {len(docs)}"
    return json.dumps(docs[0])


# -- noticing a new commit ---------------------------------------------------------


def test_a_plain_project_directory_has_no_revision_to_watch(tmp_path: Path) -> None:
    """Watching has to cost nothing for the two shapes that never change underneath.

    A project baked into an image or sitting on a laptop is a real directory, so there
    is no revision and the watcher stops at one ``readlink``.
    """
    plain = tmp_path / "project"
    plain.mkdir()
    assert project_revision(plain) is None
    assert project_revision(tmp_path / "missing") is None


def test_a_git_delivered_project_reports_the_revision_it_points_at(
    tmp_path: Path,
) -> None:
    """git-sync retargets a symlink, so the link's target names the served revision."""
    first = tmp_path / "rev-aaaaaaa"
    second = tmp_path / "rev-bbbbbbb"
    first.mkdir()
    second.mkdir()
    link = tmp_path / "current"
    link.symlink_to(first)

    assert project_revision(link) == str(first)

    # What the sidecar does on a new commit: swap the link, atomically.
    replacement = tmp_path / "current.tmp"
    replacement.symlink_to(second)
    replacement.replace(link)

    assert project_revision(link) == str(second)


def test_a_dangling_link_still_reports_a_revision(tmp_path: Path) -> None:
    """Mid-swap the target can be absent for an instant, and that is not "no revision".

    Returning None there would read as "this deployment does not use git" and silently
    stop watching for the life of the pod.
    """
    link = tmp_path / "current"
    link.symlink_to(tmp_path / "gone")
    assert project_revision(link) == str(tmp_path / "gone")


def test_a_symlinked_project_is_not_resolved_away(tmp_path: Path) -> None:
    """Resolving the project path silently defeats git delivery, twice over.

    git-sync hands the app `current -> .worktrees/<sha>` and retargets it each commit.
    Resolve that at startup and the app pins itself to one worktree (which git-sync
    later deletes), while the link it was meant to watch is nowhere in the loaded
    project. Nothing errors: the deployment comes up, serves the first commit forever,
    and reports success. Which is why this is asserted rather than assumed.
    """
    worktree = tmp_path / ".worktrees" / "abc123"
    worktree.mkdir(parents=True)
    link = tmp_path / "current"
    link.symlink_to(worktree)

    root = project_root(link)

    assert root == link, "the symlink was resolved away"
    assert project_revision(root) == str(worktree)
    # A plain directory is still normalised, which is what every other deployment gets.
    plain = tmp_path / "plain"
    plain.mkdir()
    assert project_root(plain) == plain.resolve()
    assert project_root(str(plain)).is_absolute()


# -- reaching the app by name ------------------------------------------------------


def test_a_database_password_can_arrive_as_a_mounted_file(tmp_path: Path) -> None:
    """`DB_PASSWORD_FILE` has to work on the path that assembles the URI.

    The app resolves `<NAME>_FILE` when it reads a variable, but the URI is assembled
    in the entrypoint before the app runs, so that mechanism cannot reach it. Left
    unhandled, the documented convention produces a URI with an empty password, which
    surfaces as an authentication failure a long way from its cause -- and it fails in
    exactly the case the docs recommend it for: a rotated secret mounted as a file.
    """
    secret = tmp_path / "db-password"
    secret.write_text("pk<ZiIKVTtOSF")  # RDS-managed passwords look like this

    result = _run_entrypoint(
        tmp_path,
        DB_URI="",  # the harness pins one; clear it so the parts are used
        DB_HOST="elbi-pg.example.rds.amazonaws.com",
        DB_USER="elbi",
        DB_NAME="elbi",
        DB_PASSWORD_FILE=str(secret),
        ELBI_PROJECT_DIR=str(tmp_path / "data"),
        ELBI_STATE_DIR=str(tmp_path / "state"),
    )

    assert result.returncode == 0, result.stderr
    assert (
        "DB_URI=postgres://elbi:pk<ZiIKVTtOSF"
        "@elbi-pg.example.rds.amazonaws.com:5432/elbi" in result.stdout
    )


def test_setting_both_a_password_and_a_password_file_is_refused(
    tmp_path: Path,
) -> None:
    """Two sources for one secret is a configuration error, not a precedence question.

    This is the line the Docker official images' `file_env` takes for this same
    convention ("but are exclusive"), and `env()` takes it too -- so a secret behaves
    the same whether the container consumes it or the app does. Choosing either one
    would honour at most one of two stated intentions, silently.
    """
    secret = tmp_path / "db-password"
    secret.write_text("from-the-file\n")

    result = _run_entrypoint(
        tmp_path,
        DB_URI="",
        DB_HOST="db.example.com",
        DB_PASSWORD="from-the-variable",
        DB_PASSWORD_FILE=str(secret),
        ELBI_PROJECT_DIR=str(tmp_path / "data"),
        ELBI_STATE_DIR=str(tmp_path / "state"),
    )

    assert result.returncode == 1
    assert "exclusive" in result.stderr
    assert "SERVED" not in result.stdout


def test_a_password_containing_a_space_survives_being_read_from_a_file(
    tmp_path: Path,
) -> None:
    """Strip the surrounding whitespace, never the whitespace inside the secret.

    Deleting every space would turn a valid credential into an authentication failure --
    the same class of fault this whole path exists to avoid.
    """
    secret = tmp_path / "db-password"
    secret.write_text("two words\n")

    result = _run_entrypoint(
        tmp_path,
        DB_URI="",
        DB_HOST="db.example.com",
        DB_PASSWORD_FILE=str(secret),
        ELBI_PROJECT_DIR=str(tmp_path / "data"),
        ELBI_STATE_DIR=str(tmp_path / "state"),
    )

    assert result.returncode == 0, result.stderr
    assert "two words" in result.stdout


def test_an_unreadable_password_file_stops_instead_of_connecting_without_one(
    tmp_path: Path,
) -> None:
    """A mounted secret that is not there must not degrade to an empty password.

    Silently assembling `postgres://user:@host/db` turns a missing-mount problem into
    an authentication error, which sends whoever is on call to the wrong system.
    """
    result = _run_entrypoint(
        tmp_path,
        DB_URI="",
        DB_HOST="db.example.com",
        DB_PASSWORD_FILE=str(tmp_path / "never-mounted"),
    )

    assert result.returncode == 1
    assert "DB_PASSWORD_FILE" in result.stderr
    assert "SERVED" not in result.stdout


def test_an_explicit_database_url_is_never_rebuilt_from_parts(tmp_path: Path) -> None:
    """`DB_URI` wins outright, so a URL carrying options survives untouched."""
    result = _run_entrypoint(
        tmp_path,
        DB_URI="postgres://given:pw@elsewhere:6543/db?sslmode=require",
        DB_HOST="ignored.example.com",
        DB_PASSWORD_FILE=str(tmp_path / "also-ignored"),
        ELBI_PROJECT_DIR=str(tmp_path / "data"),
        ELBI_STATE_DIR=str(tmp_path / "state"),
    )

    assert result.returncode == 0, result.stderr
    assert (
        "DB_URI=postgres://given:pw@elsewhere:6543/db?sslmode=require" in result.stdout
    )
