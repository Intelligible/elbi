<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="./assets/ods-logo-dark.svg">
    <img alt="Open Derivation Spec" src="./assets/ods-logo.svg" width="320">
  </picture>
</p>

# The Open Derivation Spec (ODS)

This directory is the **Open Derivation Spec (ODS)**, the open standard for
elbi derivations. It is co-located with the reference SDK on purpose: the
SDK loads
[`derivation.schema.json`](./derivation.schema.json) as data and validates against
it, so the spec is never an implementation-defined afterthought.

| File | Purpose |
| --- | --- |
| [`derivation.schema.json`](./derivation.schema.json) | The machine-readable contract (JSON Schema 2020-12). **Authoritative.** |
| [`data_contract.schema.json`](./data_contract.schema.json) | The Data Contract standard: the quality bar a derivation's output must satisfy (JSON Schema 2020-12). **Authoritative.** |
| [`derivation.md`](./derivation.md) | Normative prose (MUST / SHOULD). |
| [`Versioning.md`](./Versioning.md) | How the spec is versioned, independent of the SDK. |
| [`CHANGELOG.md`](./CHANGELOG.md) | Spec-only changelog. |
| [`examples/`](./examples/) | Hand-written valid and invalid manifests. |
| [`tests/`](./tests/) | Conformance fixtures: `{description, data, valid}` cases run by CI (JSON Schema Test Suite vocabulary). |

## Conformance

A conformant implementation agrees with every fixture in `tests/` when validating
against the schema. Run the suite with:

```bash
uv run pytest spec/tests
```

## When (and when not) to split this out

The spec stays here until there is a second-language SDK, external implementers
filing spec issues, or a foundation home. Splitting earlier signals theater, not
openness.
