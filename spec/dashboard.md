# The Dashboard Spec

**Version 1.0**

This document is the normative specification of an elbi *dashboard*. It uses
the keywords MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY as defined in
[RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

The machine-readable contract is [`dashboard.schema.json`](./dashboard.schema.json), a
[JSON Schema 2020-12](https://json-schema.org/draft/2020-12) document. Where this prose
and the schema disagree, **the schema is authoritative**. Conforming implementations
MUST validate manifests against the schema as data; they MUST NOT hand-mirror its
constraints in code that can drift.

## 1. Overview

A *dashboard* is a named presentation surface composed of *pages*, each a grid of
*widgets*. A widget either *binds* to a derivation, naming it and supplying parameters,
or renders a *filter* control or *text*. A dashboard carries no computation of its
own: the derivations it binds are where computation and its verification live.

A *manifest* is the declarative serialization of a dashboard. An implementation that
authors dashboards MUST be able to emit a manifest that validates against the schema.

## 2. Top-level fields

| Field | Required | Description |
| --- | --- | --- |
| `specVersion` | yes | The spec version the manifest targets, as `MAJOR.MINOR`. |
| `kind` | yes | MUST be the literal `"Dashboard"`. |
| `name` | yes | A stable identifier matching `^[a-z][a-z0-9_]*$`, unique within a project. |
| `title` | no | Human-readable title. |
| `description` | no | Human- and agent-readable summary. |
| `theme` | no | `"auto"` (default), `"light"`, or `"dark"`. |
| `variables` | no | Dashboard-scoped values a viewer supplies. Defaults to `[]`. |
| `pages` | yes | The dashboard's pages; MUST contain at least one. |
| `refresh` | no | `{ "interval": ... }`, how often a viewer re-checks widget staleness. |

A manifest MUST NOT contain properties beyond those defined here
(`additionalProperties: false`).

## 3. Variables

Each variable is keyed by a `name` matching `^[a-z][a-z0-9_]*$` and has a `type` (one
of `string`, `integer`, `number`, `boolean`, `date`) and a `control` (`dropdown`,
`multiselect`, `search`, `date`, `daterange`, `range`, or `toggle`). A variable MAY
declare a `default`, an `options` source, a `scope`, and cross-filter targeting
(`appliesTo`, `immune`).

`options` is either a static `values` list or `{ "derivation", "column", "labelColumn" }`,
in which case the referenced derivation's distinct column values populate the control.

`scope` is `"global"` (the default) or the name of a page; a page-scoped variable MUST
name an existing page. `appliesTo` and `immune` are advisory to the renderer: they
govern which widgets a cross-filter selection propagates to. They do not affect whether
an explicit `$name` reference resolves, which it always does.

## 4. Pages and widgets

Each page has a `name` (matching `^[a-z][a-z0-9_]*$`, unique within the dashboard), an
optional `title`, an optional `columns` count (1–24, default 24), and a list of
`widgets`.

Each widget has an `id` (matching `^[a-z][a-z0-9_]*$`, **unique across the whole
dashboard**), a `type`, and a `gridPos`. `gridPos` is `{ x, y, w, h }` in grid units,
where `x` and `w` are columns (`0 ≤ x ≤ 23`, `1 ≤ w ≤ 24`) and `y`/`h` are rows.

The `type` determines the required shape:

- `metric`, `chart`, `map`, `table`: a *data* widget; it MUST have a `bind` and MAY
  have a `viz`.
- `text`: MUST have exactly one of `content` (Markdown) or a `bind` naming a
  derivation that returns markdown.
- `filter`: MUST name a `variable` that exists, and MUST NOT have a `bind`.

## 5. Binding

`bind` is `{ "derivation", "params" }`. `derivation` names the bound derivation.
`params` maps parameter names to values. A value is a literal, or the string `"$name"`
referencing variable `name`; a string beginning `"$$"` is a literal whose leading
dollar sign is escaped. A `$name` reference MUST resolve to a declared variable.

A runner resolves each `$name` to the viewer's current selection, or the variable's
`default` when unset, and runs the derivation with the resulting parameters.

## 6. Interactions

A widget MAY declare `interactions`:

- `crossFilter.emit` maps variable names to fields of a selected datum; a selection sets
  those variables and re-resolves the non-immune widgets that reference them. Each named
  variable MUST exist.
- `drillThrough.target` is `page:<name>` or `dashboard:<name>`; a `page:` target MUST
  name an existing page. `carry` lists variables whose values travel to the target.
- `drillDown` steps through a `hierarchy` of columns by rebinding `param`, which MUST be
  a key of the widget's `bind.params`.

## 7. Lifecycle

A dashboard has a draft state and, once published, a frozen published snapshot. An
implementation MUST NOT serve a widget bound to a derivation that is not `certified`
(per the Open Derivation Spec) in a *published* dashboard. How publishing is governed
(who may publish, review) is a deployment concern out of scope for this spec.

## 8. Conformance

An implementation is *conformant* if, for every fixture under [`tests/`](./tests/), it
agrees with the fixture's `valid` verdict when validating the fixture's `data` against
`dashboard.schema.json`. The invariants the schema cannot express (unique widget ids,
resolvable variable and page references, and the per-widget-type shape) are part of
conformance and are checked by the reference implementation.
