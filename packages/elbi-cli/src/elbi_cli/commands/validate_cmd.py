"""``elbi validate``: check the project and derivations against the spec."""

from __future__ import annotations

from pathlib import Path

import typer

from elbi_core import SpecValidationError, validate_manifest
from elbi_core.config import DEV_BINDINGS_FILENAME, PROJECT_FILENAME, DataBindings
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
        # `exists()`: an absent file loads as empty bindings, which would report
        # every declared dataset as unbound.
        bindings = (
            DataBindings.load(project.bindings_path)
            if project.bindings_path.exists()
            else None
        )
    except ElbiError as exc:
        fail(str(exc))
        raise typer.Exit(code=1) from exc

    arrow(f"project {project.config.project!r}")
    arrow(
        f"discovered {len(project.discovered)} derivation(s) in "
        f"{project.config.derivations_dir}/"
    )

    declared = set(project.config.dataset_names)
    failed_derivation_count = 0

    for derivation in project.registry:
        try:
            validate_manifest(derivation.to_manifest())
        except SpecValidationError as exc:
            failed_derivation_count += 1
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

    wiring_problems: tuple[str, ...] = ()
    if bindings is None:
        arrow(f"wiring check skipped: no {DEV_BINDINGS_FILENAME} found")
    else:
        wiring_problems = bindings.wiring_problems(
            project.config.dataset_names, project.base_dir
        )
        for problem in wiring_problems:
            fail(problem)

    if failed_derivation_count:
        fail(f"{failed_derivation_count} derivation(s) failed validation")
    else:
        ok("all derivations conform to the spec")
    if wiring_problems:
        fail(
            f"{len(wiring_problems)} wiring problem(s) between "
            f"{PROJECT_FILENAME} and {DEV_BINDINGS_FILENAME}"
        )
    if failed_derivation_count or wiring_problems:
        raise typer.Exit(code=1)
