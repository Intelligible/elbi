# Common development tasks. Run `just` (or `just --list`) to see them.

# List available recipes.
default:
    @just --list

# Install all workspace packages + dev tooling.
sync:
    uv sync --all-extras --dev

# Build the web SPA into the app package (run before packaging the app wheel).
web-build:
    cd packages/elbi/web && npm ci && npm run build

# Lint with ruff.
lint:
    uv run ruff check .

# Auto-format with ruff.
fmt:
    uv run ruff format .

# Static type-check (strict).
typecheck:
    uv run mypy

# Run the full test suite with coverage.
test:
    uv run pytest --cov

# Everything CI enforces: lint, format check, types, tests.
check:
    uv run ruff check .
    uv run ruff format --check .
    uv run mypy
    uv run pytest --cov

# Mutation-test the core module: change the code, expect a test to fail.
# The real test-quality gate (coverage proves code ran, not that it is checked).
mutation:
    rm -f .mutation.sqlite
    uv run cosmic-ray init mutation.toml .mutation.sqlite
    uv run cosmic-ray exec mutation.toml .mutation.sqlite
    @echo "--- survival rate (lower is better; survivors = unchecked behavior) ---"
    uv run cr-rate .mutation.sqlite

# Run the statistical-pitfall benchmark against the cached LLM baseline and write the report.
benchmark:
    uv run python -m benchmark.run

# Materialize every trap's dataset to CSV under benchmark/report/data for inspection.
benchmark-dump-data:
    uv run python -m benchmark.run --dump-data

# Regenerate the logo, wordmark, favicon and social card from the mark and Geist.
# Nothing under public/brand or docs/assets should be edited by hand.
brand:
    uv run python tools/brand.py
    # The generator writes valid JSX but not biome's formatting of it, so the one
    # generated source file is formatted here rather than left dirty for the next check.
    cd packages/elbi/web && npx biome check --write src/components/Logo.tsx

# Regenerate each synthetic trap's committed data.jsonl from its generator (then commit).
benchmark-regen-data:
    uv run python -m benchmark.run --regenerate-data

# Refresh the cached bare-LLM baseline (opt-in; needs ANTHROPIC_API_KEY). Review the diff.
benchmark-llm-refresh:
    uv run python -m benchmark.run --refresh-llm

# Prove the traps bite: mutate each trap's gate module (Cosmic Ray) and require the trap
# to kill >=1 mutant. Exits non-zero if any trap passes regardless of its gate. Add
# `--module effect.py` or `--trap <id>` to scope it (see `python -m benchmark.bite -h`).
benchmark-bite:
    uv run python -m benchmark.bite

# Serve the documentation locally.
docs:
    uv run mkdocs serve

# Build the derivation-spec conformance suite.
conformance:
    uv run pytest spec/tests -v

# Build the data-science sandbox image for `sandbox: docker` (see docker/sandbox.Dockerfile).
# Tags it elbi-sandbox:latest; set `sandbox_image: elbi-sandbox` to use it.
sandbox-image:
    docker build -f docker/sandbox.Dockerfile -t elbi-sandbox:latest .
