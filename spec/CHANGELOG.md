# Open Derivation Spec (ODS) Changelog

This changelog tracks the Open Derivation Spec (ODS) only. It is independent of the SDK and
CLI changelog at the repository root.

## 1.1 (unreleased)

Adds provenance and lifecycle, supporting agent-authored derivations.

- New optional `origin` field (`human` | `agent`, default `human`).
- New optional `status` field (`proposed` | `certified`, default `certified`).
- Runners MUST NOT serve a `proposed` derivation; `agent`-origin derivations
  SHOULD run under isolation. Defaults are backward compatible: a 1.0 manifest is
  a human-authored, certified derivation.
- New optional `maxCells` field on the `table` serve contract (default `2000`): a
  cell budget for the inline preview a runner returns to the model, so a large
  table is previewed inline with the full result offered out of band rather than
  dumped into the model's context. Backward compatible: omitting it keeps the
  default.
- Two new parameter types, `object` (a record of named values) and `array` (a
  list, whose element type is an optional `items`). Nesting is bounded to one
  level. They let a derivation take structured input, such as a feature record to
  score. Backward compatible: existing scalar params are unaffected.
- New input kind `semantic_model`, binding a governed semantic model (for example
  an Open Semantic Interchange document) so a derivation can compute over metric
  definitions rather than only over rows. Such an input implies no `dependsOn`
  edge, and runners SHOULD version the derivation by the document's content.
  Backward compatible: existing `dataset` and `derivation` inputs are unaffected.
- New serve contract format `components`: the artifact is served as a list of
  OpenReasoningComponents (ORC) components, each a self-contained natural-language
  statement about the data with optional structure/evidence/relations/provenance.
  A runner SHOULD stamp `provenance.derivation`/`provenance.derivation_version`
  onto a component that doesn't already declare its own provenance, so freshness
  is checkable against the same content-hash versioning the spec already uses.
  Backward compatible: existing formats are unaffected.

## 1.0 (unreleased)

Initial specification.

- `Derivation` manifest with `specVersion`, `kind`, `name`, `description`,
  `inputs`, `dependsOn`, and `serve`.
- Input descriptors of kind `dataset` and `derivation`.
- Serve contract formats: `table`, `markdown`, `json`, `text`.
- Conformance fixtures under `tests/`.
