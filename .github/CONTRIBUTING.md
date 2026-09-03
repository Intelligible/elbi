# Contributing to elbi

Thanks for helping build the open layer. This guide gets you from clone to PR.

## Development setup

The repo is a [uv](https://docs.astral.sh/uv/) workspace. With `uv` installed:

```bash
git clone https://github.com/Intelligible/elbi
cd elbi
uv sync --all-extras   # installs every workspace package + dev tools
```

That single environment contains both packages (`elbi`, `elbi-cli`)
in editable mode plus the test, lint, and docs tooling.

Install the git hooks so lint/format/type checks run on every commit:

```bash
uv run pre-commit install
```

## The inner loop

With [`just`](https://github.com/casey/just): `just check` runs the Python checks.
Or directly:

```bash
uv run ruff check .            # lint
uv run ruff format --check .   # formatting
uv run mypy                    # static types (strict)
uv run pytest                  # the full test suite
```

All four must pass before a PR is merged; CI runs them across Python 3.10–3.13 on
Linux and macOS.

Changes under `packages/elbi/web` are also gated, and `just check` does not
reach them. From that directory:

```bash
npm run lint         # Biome
npm run typecheck    # tsc
npm test             # Vitest
npm run build        # tsc -b && vite build
npm run test:e2e     # Playwright
```

### Coverage

`pytest` is configured to fail under 90% coverage. New code needs tests.

### Working on one package

```bash
uv run --package elbi pytest packages/elbi-core/tests
```

## Changing the spec

The derivation spec lives in [`spec/`](../spec). If you change
`derivation.schema.json`:

1. Update [`spec/derivation.md`](../spec/derivation.md) and the conformance
   fixtures under `spec/tests/` in the same PR.
2. Copy the schema into the SDK's bundled location
   (`packages/elbi-core/src/elbi/spec/derivation.schema.json`). The
   `test_bundled_schema_matches_canonical_spec` test enforces they stay identical.
3. Bump `specVersion` references per [`spec/Versioning.md`](../spec/Versioning.md).

## News fragments

Every user-facing change needs a [towncrier](https://towncrier.readthedocs.io/)
fragment under [`changelog.d/`](../changelog.d). See that directory's README for
the naming convention.

## Commit & PR conventions

- Imperative-mood commit subjects under 72 characters
  (e.g. `Validate inputs against the derivation spec`).
- Small, focused commits over large ones.
- Fill in the pull request template, including the news-fragment checkbox.
- Never use `--no-verify`.

## Code style

- TypeScript-style strictness in Python: full type annotations, `mypy --strict`.
- Prefer early returns and small, single-purpose functions.
- Match the style of the surrounding code.
- No `print` in library code; the CLI uses the shared console helpers.

## Licensing of contributions

Everything here is Apache-2.0, and a contribution is licensed the same way.

Sign off each commit with `git commit -s`, which certifies the
[Developer Certificate of Origin](https://developercertificate.org/): you keep your
copyright and assign nothing. There is no contributor licence agreement to sign.
