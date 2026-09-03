"""``elbi search`` at the command boundary.

The recovery path is the one a person reaches for when the index and the store already
disagree, so the things that can only break here are the things worth covering: that the
commands are registered at all, that their options parse, that a failure exits non-zero
with the wording the module deliberately substitutes for DuckDB's, and that ``status``
reports without rebuilding -- the mistake that once made it destructive.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest
from typer.testing import CliRunner

from elbi_cli.app import app

pytest.importorskip("duckdb")


def _cache(project: Path) -> Path:
    return project / ".elbi" / "cache"


def _build(runner: CliRunner, project: Path) -> None:
    result = runner.invoke(app, ["search", "reindex", "-C", str(project)])
    assert result.exit_code == 0, result.output


def test_status_reports_what_the_index_holds(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    _build(runner, project)

    result = runner.invoke(app, ["search", "status", "-C", str(project)])

    assert result.exit_code == 0, result.output
    assert "documents:" in result.output
    assert "current enough to serve" in result.output


def test_status_does_not_rebuild_the_index_it_reports_on(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    """It opens with ``recreate=False``, and that is the whole of the guarantee.

    An index whose recorded model does not match is dropped and rebuilt on open, so a
    ``status`` that opened the ordinary way destroyed what it was asked about. Asserted
    on the file rather than on the flag: the flag is the mechanism, not the promise.
    """
    project = scaffold("standard")
    _build(runner, project)
    index = _cache(project) / "search.duckdb"
    before = index.stat().st_mtime_ns, index.stat().st_size

    assert runner.invoke(app, ["search", "status", "-C", str(project)]).exit_code == 0

    assert (index.stat().st_mtime_ns, index.stat().st_size) == before


def test_reindex_rebuilds_and_says_what_it_wrote(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")

    result = runner.invoke(app, ["search", "reindex", "-C", str(project)])

    assert result.exit_code == 0, result.output
    assert "documents over" in result.output
    assert "written" in result.output
    assert (_cache(project) / "search.duckdb").exists()


def test_reindex_vectors_only_takes_the_flag_and_re_embeds_nothing(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    """The flag exists to rebuild the experimental index over vectors that never moved.

    Checked at the boundary because the option is the surface: a rebuild that quietly
    re-embedded would still exit zero and still print the same line.
    """
    project = scaffold("standard")
    _build(runner, project)

    result = runner.invoke(
        app, ["search", "reindex", "-C", str(project), "--vectors-only"]
    )

    assert result.exit_code == 0, result.output
    assert "vector index rebuilt" in result.output
    assert "documents over" not in result.output, "it rebuilt the corpus as well"


def test_a_locked_index_fails_with_the_wording_a_person_can_act_on(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    """DuckDB's own message reads like corruption; the substitution is the feature.

    One process holds the file, so the recovery command has to say the server is the
    thing to stop rather than passing a lock error through.
    """
    import subprocess
    import sys

    project = scaffold("standard")
    _build(runner, project)
    index = _cache(project) / "search.duckdb"
    # A separate process, because DuckDB lets one process open a file as often as it
    # likes -- an in-process second connection takes no lock and proves nothing.
    holder = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import sys, time, duckdb\n"
            "con = duckdb.connect(sys.argv[1])\n"
            "con.execute('CREATE TABLE IF NOT EXISTS _held(x INTEGER)')\n"
            "print('held', flush=True)\n"
            "time.sleep(60)\n",
            str(index),
        ],
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert holder.stdout is not None
        assert holder.stdout.readline().strip() == "held", "the holder never opened it"
        result = runner.invoke(app, ["search", "status", "-C", str(project)])
    finally:
        holder.kill()
        holder.wait()
        holder.stdout.close()

    assert result.exit_code == 1, result.output
    assert "another process" in result.output
    assert "server" in result.output.lower()
