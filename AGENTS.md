# Working in this repository

Read this before changing code. It is the contract for agents and humans alike.
For what the project *is*, see `README.md` and `docs/`.

## Layout

- `packages/elbi-core/`: the SDK (domain-agnostic; the published library).
- `packages/elbi-cli/`: the `elbi` CLI and MCP dev server.
- `packages/elbi-agent/`: the verifying analysis loop.
- `packages/elbi/`: the FastAPI app and the web UI it serves.
- `spec/`: the Open Derivation Spec and its conformance suite.
- Workspace managed by `uv`; shared tool config lives in the root `pyproject.toml`.

## Extension seams

These are a published contract rather than internal shape, because a package outside
this repository can attach to them. `packages/elbi/tests/test_extension_seams.py` holds
them to it; if you change one, that suite is what tells you.

- The `elbi.extensions` entry-point group and `extensions.install_extension(context)`,
  which mount routes and middleware onto a built app.
- An extension module's `store_class`, read by `open_store`, and its `service_classes`,
  read as `serve.build` constructs each service. Both resolve *before* the thing they
  describe exists, which is the only moment it can be replaced: a service handed over
  already built can be wrapped, but every route holding the original goes on calling it.
- `app.state.withhold_rendering`, asked before a derivation's stored rendering is served.
  Read off the app per request, so something installed after the app is built can answer
  it; the `create_app` argument of the same name seeds it and still wins.
- `Store.housekeeping()`, which contributes cutoff-based cleanups to the maintenance
  pass under the same contract as the built-in ones: report how many rows went, and be
  safe to re-run.
- `Store.reading()` / `Store.writing()`, which hand out this store's own sessions, so an
  extension shares the pool and the write lock rather than opening a second engine.
- The `elbi.cli` entry-point group, which adds commands to the CLI.
- `elbi_cli.http.set_auth_provider`, which supplies a credential per host for a CLI
  talking to a remote app.

## Commands

`just` is the command surface. Prefer it over remembering raw invocations.

- `just sync`: install all packages + dev tooling.
- `just check`: everything CI enforces, namely ruff lint, format check, mypy (strict), tests + coverage. Run before declaring work done.
- `just test` / `just lint` / `just typecheck` / `just fmt`: the individual steps.
- `just mutation`: mutation-test the core module (see Testing).
- `just conformance`: the derivation-spec conformance suite.

## Testing standard

Coverage is necessary but not sufficient. The repo sits at 100% coverage, yet that
only proves code *ran*, not that behavior is *checked*. Hold to these:

1. **Test behavior at the public boundary**, not private internals. The strongest
   tests drive the real surface (the in-memory and over-the-wire MCP server, a
   retriever's `search`), so they survive refactors and catch real breakage.
2. **Property-based tests (Hypothesis) for invariants**: ordering, monotonicity,
   floors, idempotence. They constrain behavior across inputs an example cannot.
3. **Mutation testing is the quality gate.** `just mutation` changes the code and
   expects a test to fail; a surviving mutant is an unchecked behavior, not a
   coverage gap. When you add core logic, run it and close real gaps. Some
   survivors are equivalent mutants (e.g. a tuning constant that preserves
   ordering); do not contort tests to kill those.
4. **Prove a test catches its bug.** For a non-trivial test, confirm it fails when
   the behavior is broken (revert the fix, or mutate the line) before trusting it.
   A test that passes against broken code is worse than none.

## Hard rules

- **Never weaken, delete, `skip`, or `xfail` a test to make a change pass.** The
  tests are the specification. If a test is genuinely wrong, fix it deliberately
  and say why in the same change; do not quietly loosen an assertion.
- **Do not target a coverage number.** Chasing 100% breeds assertion-free tests.
  Optimize for behavior checked (mutation score), not lines hit.
- **No new dependencies without asking.** Optional features go behind an extra.
- **Keep changes scoped.** Do not refactor unrelated code; note it instead.

## Code conventions

- mypy runs `--strict`; ruff lint + format are enforced. `just check` is the gate.
- No em-dashes anywhere in prose, docstrings, or comments.
- Comments and docstrings describe the *active* code and *why* it is the way it
  is. Do not narrate history ("was X", "improved over Y") or speculate about
  future changes. Keep inline comments to a line or two, explaining intent.
- Match the style, naming, and comment density of the surrounding code.
- Docstrings follow PEP 257; module and class docstrings carry the design
  rationale, inline comments stay terse.

## Definition of done

`just check` is green, and for new core logic, `just mutation` shows no *real*
surviving mutants. State test results plainly; if something is skipped or fails,
say so.
