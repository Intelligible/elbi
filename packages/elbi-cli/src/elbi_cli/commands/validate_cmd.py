"""``elbi validate``: check the project and derivations against the spec."""

from __future__ import annotations

from pathlib import Path

import typer

from elbi_core import SpecValidationError, validate_manifest
from elbi_core.errors import ElbiError

from .._console import arrow, fail, ok, warn
from ..project import load_project


def validate(
    directory: Path = typer.Option(
        Path(), "--directory", "-C", help="Project directory.", exists=True
    ),
) -> None:
    """Validate the project wiring and every derivation manifest.

    Exits non-zero if anything fails to load or fails spec validation.
    """
    base_dir = directory.resolve()
    try:
        project = load_project(base_dir)
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc

    arrow(f"project {project.config.project!r}")
    arrow(
        f"discovered {len(project.discovered)} derivation(s) in "
        f"{project.config.derivations_dir}/"
    )

    declared = set(project.config.dataset_names)
    problems = 0

    for derivation in project.registry:
        try:
            validate_manifest(derivation.to_manifest())
        except SpecValidationError as exc:
            problems += 1
            fail(f"{derivation.name}: {exc}")
            continue

        undeclared = {
            dataset.name
            for dataset in derivation.dataset_inputs().values()
            if dataset.name not in declared
        }
        if undeclared:
            warn(
                f"{derivation.name}: reads dataset(s) not declared in "
                f"elbi.yaml: {', '.join(sorted(undeclared))}"
            )
        ok(f"{derivation.name}")

    if problems:
        fail(f"{problems} derivation(s) failed validation")
        raise typer.Exit(code=1)
    ok("all derivations conform to the spec")
