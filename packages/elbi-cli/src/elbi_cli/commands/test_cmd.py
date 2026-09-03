"""``elbi test``: run every derivation and re-verify its claim with the oracle.

The developer-time bookend to monitoring: `validate` checks that manifests conform to
the spec; `test` actually runs each derivation against the project's bindings and, for
any derivation carrying a claim, re-runs the verification oracle. A derivation whose
claim no longer certifies `sound` (because its code, its data binding, or the oracle
changed) is a soundness regression and fails the command, so it is caught before merge
rather than in production.
"""

from __future__ import annotations

from pathlib import Path

import typer

from elbi_core import verify_all
from elbi_core.errors import ElbiError

from .._console import arrow, fail, ok
from ..project import load_project


def _all_dicts(value: list[object]) -> bool:
    return all(isinstance(row, dict) for row in value)


def test(
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Project directory.", exists=True
    ),
) -> None:
    """Run every derivation and re-verify each claim against the oracle.

    Exits non-zero if a derivation fails to run or a claimed effect no longer
    certifies ``sound`` (a soundness regression).
    """
    base_dir = directory.resolve()
    try:
        project = load_project(base_dir)
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc

    arrow(f"project {project.config.project!r}")
    runner = project.make_runner()
    ran = verified = 0
    failures: list[str] = []

    for derivation in project.registry:
        try:
            artifact = runner.run(derivation.name)
        except (
            Exception
        ) as exc:  # a failing derivation is a reported test failure, not a crash
            failures.append(f"{derivation.name}: failed to run, {exc}")
            fail(f"{derivation.name}: failed to run, {exc}")
            continue
        ran += 1

        value = artifact.value
        rows = value if isinstance(value, list) and _all_dicts(value) else None

        claim = derivation.claim
        if not claim:
            ok(
                f"{derivation.name}: ran ({len(rows)} rows)"
                if rows is not None
                else f"{derivation.name}: ran"
            )
            continue

        # An output that is not row data cannot support a conclusion, so it cannot
        # certify: the same bar as the authoring path, reported rather than raised.
        if rows is None:
            failures.append(f"{derivation.name}: claim cannot be checked")
            fail(
                f"{derivation.name}: claim cannot be checked; "
                f"the output is not row data"
            )
            continue

        report = verify_all(rows, **dict(claim))
        if report.verdict == "sound":
            verified += 1
            label = report.estimate_label or (
                f"{report.estimate:+.3g}" if report.estimate is not None else ""
            )
            ok(
                f"{derivation.name}: sound, {label}"
                if label
                else f"{derivation.name}: sound"
            )
        else:
            broke = next((g for g in report.ran if g.verdict != "sound"), None)
            detail = broke.detail if broke else "no applicable check"
            failures.append(
                f"{derivation.name}: claim no longer sound ({report.verdict})"
            )
            fail(
                f"{derivation.name}: SOUNDNESS REGRESSION, "
                f"verdict={report.verdict}: {detail}"
            )

    arrow(f"ran {ran} derivation(s); {verified} claim(s) re-verified sound")
    if failures:
        fail(f"{len(failures)} failure(s)")
        raise typer.Exit(code=1)
    ok("all derivations ran and every claim still certifies sound")
