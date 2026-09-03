# Spec Versioning

The Open Derivation Spec (ODS) is versioned independently of any SDK, on its own
[Semantic Versioning](https://semver.org/) line. The `specVersion` field in a
manifest is `MAJOR.MINOR` (patch-level changes never alter the contract).

## What each bump means

- **MAJOR**: a backward-incompatible change, such as removing or renaming a field,
  tightening a constraint such that a previously valid manifest becomes invalid,
  or changing the meaning of an existing field.
- **MINOR**: a backward-compatible addition, such as a new optional field, a new
  `serve.format`, or a relaxed constraint such that a previously invalid manifest
  becomes valid.
- **PATCH**: editorial fixes to prose or examples that do not change validation
  behavior. Not reflected in `specVersion`.

## SDK ↔ spec relationship

Each SDK release declares the spec version it implements (see
`elbi.SPEC_VERSION`). An SDK MUST reject a manifest whose `specVersion`
has a different MAJOR than the version it implements. An SDK SHOULD accept a
manifest whose MINOR is less than or equal to the version it implements.

## Process

1. Propose the change as a spec issue, with motivation and migration notes.
2. Update `derivation.schema.json`, `derivation.md`, and the `tests/` fixtures
   in the same change.
3. Record the change in `CHANGELOG.md` and bump `specVersion` references.
