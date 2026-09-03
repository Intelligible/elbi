"""Tests for `elbi test`: run derivations and re-verify their claims."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from typer.testing import CliRunner

from elbi_cli.app import app

# A derivation that generates a genuinely confounded effect (z drives both x and y).
# Adjusting for z recovers the real +2 effect (sound); omitting it is
# confounded (unsound).
_SOUND = '''\
import random
from elbi_core import Artifact, Context, derivation, serve


@derivation(
    serve=serve.table(title="adjusted", columns=["x", "y", "z"], max_rows=10),
    claim={"x": "x", "y": "y", "controls": ["z"]},
)
def adjusted_effect(ctx: Context) -> Artifact:
    """Effect of x on y adjusting for the confounder z."""
    r = random.Random(0)
    rows = []
    for _ in range(300):
        z = r.gauss(0, 1)
        x = z + r.gauss(0, 0.5)
        y = 2.0 * x + 1.5 * z + r.gauss(0, 0.5)
        rows.append({"x": x, "y": y, "z": z})
    return Artifact.table(rows)
'''

_UNSOUND = '''\
import random
from elbi_core import Artifact, Context, derivation, serve


@derivation(
    serve=serve.table(title="naive", columns=["x", "y", "z"], max_rows=10),
    claim={"x": "x", "y": "y"},
)
def naive_effect(ctx: Context) -> Artifact:
    """Claims the effect without adjusting for z: confounded, must not certify."""
    r = random.Random(0)
    rows = []
    for _ in range(300):
        z = r.gauss(0, 1)
        x = z + r.gauss(0, 0.5)
        y = -0.2 * x + 1.5 * z + r.gauss(0, 0.5)
        rows.append({"x": x, "y": y, "z": z})
    return Artifact.table(rows)
'''

# A claim on prose output: the oracle needs rows, so it cannot certify, and must report
# that rather than crash on a value it cannot index.
_NOT_ROWS = '''\
from elbi_core import Artifact, Context, derivation, serve


@derivation(
    serve=serve.markdown(title="prose"),
    claim={"x": "x", "y": "y"},
)
def prose_claim(ctx: Context) -> Artifact:
    """Claims an effect but returns prose, not rows."""
    return Artifact.markdown("x drives y.")
'''


def _write(project: Path, name: str, src: str) -> None:
    (project / "derivations" / f"{name}.py").write_text(src, encoding="utf-8")


def test_test_passes_when_claims_stay_sound(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    _write(project, "adjusted_effect", _SOUND)
    result = runner.invoke(app, ["test", "-C", str(project)])
    assert result.exit_code == 0, result.output
    assert "sound" in result.output
    assert "churn_risk: ran" in result.output  # no-claim derivations just run


def test_test_flags_a_soundness_regression(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    _write(project, "naive_effect", _UNSOUND)
    result = runner.invoke(app, ["test", "-C", str(project)])
    assert result.exit_code == 1, result.output
    assert "SOUNDNESS REGRESSION" in result.output
    assert "naive_effect" in result.output


def test_test_reports_a_claim_it_cannot_check(
    scaffold: Callable[[str], Path], runner: CliRunner
) -> None:
    project = scaffold("standard")
    _write(project, "prose_claim", _NOT_ROWS)
    result = runner.invoke(app, ["test", "-C", str(project)])
    assert result.exit_code == 1, result.output
    assert "claim cannot be checked" in result.output
    assert "not row data" in result.output
