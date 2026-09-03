"""File templates for ``elbi init``.

Templates are plain strings with a ``__PROJECT__`` placeholder rather than
``str.format`` so that the Python/YAML braces in the bodies need no escaping.
Scaffolding is fully offline: no login, no cloud call.
"""

from __future__ import annotations

from pathlib import Path

Template = str

_PLACEHOLDER = "__PROJECT__"

# The workspace packages a scaffolded project resolves. Ordinarily they come from
# the registry; ``--editable`` points them at a checkout instead.
WORKSPACE_PACKAGES = ("elbi-core", "elbi-cli")

_PROJECT_YAML = """\
# Committed project wiring: logical only, no credentials.
project: __PROJECT__
derivations_dir: derivations

datasets:
  - sales

# How agents search your derivations. Default "hybrid" fuses BM25 with semantic
# matching, so a derivation is found by meaning as well as by shared words. Set
# "lexical" for a purely lexical, offline BM25 ranker.
# search: lexical
"""

_DEV_YAML = """\
# Local-only data bindings. This file is gitignored; never commit credentials.
# Bind each dataset to a local file, or to env(VAR) for a connection string you
# are authorized to use.
data:
  sales: ./fixtures/sales.csv
"""

_PYPROJECT = """\
[project]
name = "__PROJECT__"
version = "0.1.0"
description = "Data context for __PROJECT__."
requires-python = ">=3.10"
dependencies = [
    "elbi-core",
]

[dependency-groups]
dev = [
    "elbi-core[cli]",
    "pytest>=8.0",
]
"""

_CHURN_RISK = '''\
"""Example derivation. Replace with your own."""

from __future__ import annotations

from elbi_core import Artifact, Context, Dataset, derivation, serve


@derivation(
    inputs={"sales": Dataset("sales")},
    serve=serve.table(title="Churn risk", columns=["customer_id", "risk"], max_rows=50),
)
def churn_risk(ctx: Context) -> Artifact:
    """Per-customer churn-risk scores derived from recent sales activity."""
    rows = ctx.input("sales").rows
    scored = [
        {
            "customer_id": row["customer_id"],
            "risk": round(1.0 / (1.0 + float(row["amount"])), 4),
        }
        for row in rows
    ]
    scored.sort(key=lambda r: r["risk"], reverse=True)
    return Artifact.table(scored)
'''

_FIXTURE_CSV = """\
customer_id,amount
c1,100
c2,5
c3,40
"""

_TEST = '''\
"""Tests for this project's derivations."""

from __future__ import annotations

from pathlib import Path

from elbi_core import Registry, Runner
from elbi_core.config import DataBindings
from elbi_core.discovery import discover

ROOT = Path(__file__).resolve().parent.parent


def _runner() -> Runner:
    registry = Registry()
    discover(ROOT / "derivations", registry=registry)
    bindings = DataBindings.load(ROOT / "elbi.dev.yaml")
    return Runner(registry, bindings=bindings, base_dir=ROOT)


def test_churn_risk_scores_every_customer() -> None:
    artifact = _runner().run("churn_risk")
    assert len(artifact.value) == 3
    assert {"customer_id", "risk"} <= artifact.value[0].keys()
'''

_README = """\
# __PROJECT__

Data context for __PROJECT__, built with
[elbi](https://github.com/Intelligible/elbi).

```bash
elbi serve       # chat UI + MCP at http://localhost:7700, opens a browser
elbi validate    # check the project against the spec
elbi test        # run every derivation and re-verify each claim
```

Those three run from the installed CLI and need no virtual environment here.
`uv run pytest` runs this project's own tests and does need one: `uv sync` first.

`elbi serve` needs the `elbi` app package installed. For a
lighter, MCP-only server with no chat UI, use `elbi mcp` instead.
"""

_AGENTS_MD = """\
# __PROJECT__: agent guide

An [elbi](https://github.com/Intelligible/elbi) data-context project:
versioned, cached, oracle-certified derivations served to agents over MCP. This is the
source of truth for AI coding agents (AGENTS.md standard); `CLAUDE.md` imports it.

## Layout
- `derivations/*.py` - the core artifacts: Python computations decorated with
  `@derivation`. **Loaded from disk, not pushed by `sync`**: after editing, run
  `elbi sync` to reload them into a running app.
- `metrics/` `dashboards/` `features/` `monitors/` `schedules/` `workflows/` `checks/`
  `models/` `queries/` `notebooks/` `certificates/` - declarative artifacts, **applied
  to a running app with `elbi sync`** (one file each). `certificates/` is
  derived, signed proof: `pull` writes it and `sync` re-verifies it, rejecting a tamper.
- `elbi.yaml` - project wiring (sources, datasets). `elbi.dev.yaml` -
  local-only data bindings and credentials: **gitignored, never commit, never read
  secrets from it into code**.

## Workflow
- `elbi validate`: parse + schema-check every file. Run before committing.
- `elbi test`: run every derivation and re-verify each claim with the oracle;
  fails on a soundness regression (a claimed effect that no longer certifies).
- `elbi plan`: preview what `sync` would change (repo vs app).
- `elbi sync`: push the repo to a running app; `pull` writes it back.
- `uv run pytest`: run the derivation tests.

## Running things
Trigger runs **through the product, never by POSTing `/api/*` directly**:
- `elbi run materialize|workflow|notebook|backfill|train|status`
- or the MCP tools at the app's `/mcp` endpoint (`materialize`, `run_workflow`,
  `run_notebook`, `run_<derivation>`, `asset_status`, …).

## Certification
The oracle certifies derivations on the agent-authoring path: it gates *agents*, not
humans. Never fabricate a result or claim a run passed when it did not; report the real
verdict. A model with no skill stays `inconclusive` and is not promoted to champion.

## Add a derivation
Create `derivations/<name>.py` with an `@derivation` function that reads inputs via
`ctx.input(...)` and returns an `Artifact`; add a test; then `elbi validate`.

## More
`elbi <command> --help` and https://github.com/Intelligible/elbi.
"""

_CLAUDE_MD = """\
@AGENTS.md
"""

_GITIGNORE = """\
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
.mypy_cache/

# Local derivation cache.
.elbi/

# Local-only credentials & data bindings: never commit.
elbi.dev.yaml
.env
.env.*
"""

_TEST_WORKFLOW = """\
name: test

on:
  push:
    branches: [main]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          enable-cache: true
      - run: uv sync
      - run: uvx ruff check .            # lint
      - run: uvx ruff format --check .   # formatting
      - run: uv run elbi validate  # project + manifests conform to the spec
      - run: uv run elbi test      # re-verify every claim is still sound
      - run: uv run pytest                 # the project's own tests
"""


def render(
    project: str, *, minimal: bool = False, editable_root: Path | None = None
) -> dict[str, str]:
    """Return a map of relative path -> file content for a new project.

    ``editable_root`` is a checkout to depend on instead of the registry, for
    working on a derivation project and the framework together.
    """
    files: dict[str, str] = {
        "elbi.yaml": _sub(_PROJECT_YAML, project),
        "README.md": _sub(_README, project),
        "AGENTS.md": _sub(_AGENTS_MD, project),
        "CLAUDE.md": _CLAUDE_MD,
        ".gitignore": _GITIGNORE,
        ".github/workflows/test.yml": _TEST_WORKFLOW,
    }
    if minimal:
        return files

    pyproject = _sub(_PYPROJECT, project)
    if editable_root is not None:
        pyproject += _uv_sources(editable_root)
    files.update(
        {
            "pyproject.toml": pyproject,
            "elbi.dev.yaml": _DEV_YAML,
            "derivations/churn_risk.py": _CHURN_RISK,
            "fixtures/sales.csv": _FIXTURE_CSV,
            "tests/test_derivations.py": _TEST,
        }
    )
    return files


def _uv_sources(root: Path) -> str:
    """A ``[tool.uv.sources]`` table redirecting the framework to a checkout.

    The paths are absolute, so the generated table is machine-local: it belongs in
    a scratch project, not in one whose ``pyproject.toml`` is shared.
    """
    entries = "\n".join(
        f'{name} = {{ path = "{root / "packages" / name}", editable = true }}'
        for name in WORKSPACE_PACKAGES
    )
    return f"\n[tool.uv.sources]\n{entries}\n"


def _sub(template: Template, project: str) -> str:
    return template.replace(_PLACEHOLDER, project)
