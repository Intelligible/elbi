# The Open Derivation Spec (ODS)

**Version 1.1**

This document is the normative specification of an elbi *derivation*. It
uses the keywords MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY as defined in
[RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

The machine-readable contract is [`derivation.schema.json`](./derivation.schema.json),
a [JSON Schema 2020-12](https://json-schema.org/draft/2020-12) document. Where this
prose and the schema disagree, **the schema is authoritative**. Conforming
implementations MUST validate manifests against the schema as data; they MUST NOT
hand-mirror its constraints in code that can drift.

## 1. Overview

A *derivation* is a named, versioned transformation that reads zero or more
*inputs* and produces a single *artifact*, together with a *serve contract*
describing how that artifact is presented to an agent.

A *manifest* is the declarative serialization of a derivation. An SDK that
authors derivations in a host language (e.g. Python's `@derivation`) MUST be able
to emit a manifest that validates against the schema.

## 2. Top-level fields

| Field | Required | Description |
| --- | --- | --- |
| `specVersion` | yes | The spec version the manifest targets, as `MAJOR.MINOR` (e.g. `"1.0"`). |
| `kind` | yes | MUST be the literal `"Derivation"`. |
| `name` | yes | A stable identifier, matching `^[a-z][a-z0-9_]*$`. MUST be unique within a project. |
| `origin` | no | Who authored the derivation: `"human"` or `"agent"`. Defaults to `"human"`. See §6. |
| `status` | no | Lifecycle/trust state: `"proposed"` or `"certified"`. Defaults to `"certified"`. See §6. |
| `description` | no | Human- and agent-readable summary. |
| `inputs` | no | A map of input name → input descriptor. Defaults to `{}`. |
| `params` | no | A map of parameter name → parameter descriptor. Defaults to `{}`. See §3.1. |
| `dependsOn` | no | Explicit derivation dependencies. Defaults to `[]`. |
| `serve` | no | The serve contract. See §4. When absent, the derivation is *internal*: usable as an input to other derivations but not exposed to agents. |

A manifest MUST NOT contain properties beyond those defined here
(`additionalProperties: false`).

## 3. Inputs

Each entry in `inputs` is keyed by an input name matching `^[a-z][a-z0-9_]*$` and
is an object with:

- `kind` (required): `"dataset"`, `"derivation"`, or `"semantic_model"`.
- `ref` (required): the name of the referenced dataset, derivation, or semantic
  model.
- `required` (optional, default `true`): whether the input must be bound for the
  derivation to run.

Every input whose `kind` is `"derivation"` SHOULD also appear in `dependsOn`.
Implementations SHOULD infer `dependsOn` from such inputs; `dependsOn` MAY
also express ordering-only edges that are not data inputs.

An input whose `kind` is `"semantic_model"` binds a governed semantic model, a
document defining metrics and the dimensions they may be sliced by, so that a
derivation can compute over the definitions themselves. The document format is
outside this spec; an implementation MAY accept any semantic-model standard, such
as an Open Semantic Interchange document. How the document is located, parsed, and
validated is the implementation's concern. Because a semantic model is static
content rather than a computation, such an input MUST NOT imply a `dependsOn`
edge. An implementation SHOULD version the derivation by the document's content,
so that editing a definition invalidates the results derived from it.

The dependency graph implied by `dependsOn` MUST be acyclic. Cycle detection is
the responsibility of the runner, not the schema.

### 3.1 Parameters

Each entry in `params` is keyed by a parameter name matching `^[a-z][a-z0-9_]*$`
and is an object with:

- `type` (required): one of `"string"`, `"integer"`, `"number"`, `"boolean"`,
  `"object"` (a record of named values), or `"array"` (a list).
- `items` (optional): for an `"array"`, the element type, one of `"string"`,
  `"integer"`, `"number"`, `"boolean"`, or `"object"`. Nesting is bounded to one
  level (an array of scalars, or an array of records), because agents fill flat
  arguments more reliably than deeply nested ones. Meaningful only when `type` is
  `"array"`.
- `description` (optional): human- and agent-readable guidance.
- `required` (optional, default `true`): whether the agent must supply a value.
- `default` (optional): the value used when the parameter is not supplied.

Parameters are the values an agent supplies when running a derivation. When a
derivation is served over MCP, its parameters become the run tool's input schema,
a JSON Schema in which `object` and `array` parameters become the corresponding
nested types. A required parameter with no supplied value is an error; an optional
parameter falls back to its `default`. The structured types let a derivation take
the kind of payload a prediction needs (a record of named features, or a batch of
them), not only flat filters.

## 4. The serve contract

`serve` describes how the artifact is rendered for an agent. It MUST contain a
`format`, one of:

### 4.1 `table`

Tabular data. Optional fields:

- `columns`: an explicit list of column names, controlling projection and order.
  When omitted, the artifact's natural columns are used.
- `maxRows`: the maximum number of rows exposed (default `100`). Runners MUST NOT
  serve more than `maxRows` rows.
- `maxCells`: a cell budget (rows × columns, default `2000`) for the *inline
  preview* a runner returns to the model, a portable proxy for a token budget,
  since model accuracy degrades on large inline tables. When the served result
  exceeds the budget, a runner SHOULD return a trimmed preview inline and make the
  full result available out of band (e.g. MCP `structuredContent` and/or a
  resource), rather than inlining every row. `maxCells` bounds the preview only;
  it never reduces the authoritative result below `maxRows`.

### 4.2 `markdown`

The artifact is served as a Markdown document.

### 4.3 `json`

The artifact is served as JSON. Optional `indent` (0–8, default `2`) controls
pretty-printing.

### 4.4 `text`

The artifact is served as plain text.

### 4.5 `components`

The artifact is served as a list of OpenReasoningComponents (ORC) components: a
self-contained natural-language `statement` about the data per item, each optionally
carrying `structure`, `evidence`, `relations` to other components, and `provenance`.
A component's `statement` is composed directly into an agent's reasoning, the way a
`text` artifact is; unlike `text`, each item also carries the machine-checkable fields
(`structure`/`evidence`/`relations`/`provenance`) that support retrieval and validation,
exposed to an agent as structured content alongside the rendered statements.

The component payload's shape is defined by ORC's own schema, not this one — this spec
takes no position on it beyond "an array of components," the same way §3 treats a
`semantic_model` input's document format as external. A runner SHOULD attach
`provenance.derivation` and `provenance.derivation_version` (this derivation's `name`
and its current content-hash version) to each component that doesn't already declare
its own provenance, so a consumer can decide whether a component is still current by
comparing that version against the derivation's current one — the same versioning this
spec already uses for caching, without any component-level refresh machinery of its own.

Each format permits an optional `title`. No other fields are permitted: the
`serve` object declares its shared fields (`format`, `title`) and its
format-specific fields across a `oneOf`, and closes the object with
`unevaluatedProperties: false`, so a field that belongs to a different format
(e.g. `maxRows` on a `markdown` contract) is rejected.

## 5. Provenance and lifecycle

Two optional fields record where a derivation came from and how far it has
progressed toward being trusted. They support agent-authored derivations: an
agent may *propose* a derivation, but only a *certified* one is served.

- `origin` (default `"human"`): `"human"` for a derivation authored as code in
  the project, or `"agent"` for one produced by the authoring loop from
  generated source.
- `status` (default `"certified"`): `"proposed"` for a derivation awaiting
  review, or `"certified"` for one approved for serving.

A runner MUST NOT serve a derivation whose `status` is `"proposed"` to an agent
over MCP. A derivation whose `origin` is `"agent"` SHOULD be treated as untrusted
code: a runner SHOULD execute it under isolation rather than in the host process,
and SHOULD NOT advance it to `"certified"` without an explicit review step.
The defaults preserve existing behavior: a manifest that omits both fields is a
human-authored, certified derivation.

How certification is governed (who may certify, signatures, multi-party review)
is out of scope for this spec; it is a deployment concern.

## 6. Versioning

The spec is versioned on its own SemVer line, independent of any SDK. See
[`Versioning.md`](./Versioning.md). Each SDK release states which spec version it
implements.

## 7. Conformance

An implementation is *conformant* if, for every fixture under
[`tests/`](./tests/), it agrees with the fixture's `valid` verdict when
validating the fixture's `data` against `derivation.schema.json`. Each fixture is
a `{description, data, valid}` object, matching the vocabulary of the
[JSON Schema Test Suite](https://github.com/json-schema-org/JSON-Schema-Test-Suite)
(the implicit schema for every case is `derivation.schema.json`). The fixtures are
the shared, executable definition of conformance and are run by this repository's
CI.
