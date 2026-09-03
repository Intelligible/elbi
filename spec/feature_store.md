# The Feature Store Spec

**Version 1.0**

This document is the normative specification of an elbi *feature store*. It uses
the keywords MUST, MUST NOT, SHOULD, SHOULD NOT, and MAY as defined in
[RFC 2119](https://www.rfc-editor.org/rfc/rfc2119).

The machine-readable contract is
[`feature_store.schema.json`](./feature_store.schema.json), a
[JSON Schema 2020-12](https://json-schema.org/draft/2020-12) document. Where this prose
and the schema disagree, **the schema is authoritative**. Conforming implementations
MUST validate manifests against the schema as data; they MUST NOT hand-mirror its
constraints in code that can drift.

## 1. Overview

A *feature store* is a registry of *entities* and *feature views*. An entity names a
join key. A feature view groups feature columns produced by a certified derivation,
keyed by one or more entities and, optionally, an event-time column that enables
point-in-time joins. The derivation is the feature pipeline; the store adds retrieval.

## 2. Top-level fields

| Field | Required | Description |
| --- | --- | --- |
| `specVersion` | yes | The spec version the manifest targets, as `MAJOR.MINOR`. |
| `kind` | yes | MUST be the literal `"FeatureStore"`. |
| `entities` | no | The objects features describe. Defaults to `[]`. |
| `featureViews` | no | Named groups of features. Defaults to `[]`. |

A manifest MUST NOT contain properties beyond those defined here
(`additionalProperties: false`).

## 3. Entities

Each entity has a `name` (matching `^[a-z][a-z0-9_]*$`, unique within the store), a
`joinKey` (the column identifying the entity in feature and entity data), an optional
`valueType` (`string` default, `integer`, `float`, or `boolean`), and an optional
`description`.

## 4. Feature views

Each feature view has:

- `name` (matching `^[a-z][a-z0-9_]*$`, unique within the store).
- `entities`: the names of the entities it is keyed by; each MUST resolve to a defined
  entity. The view's join keys are those entities' join keys.
- `source`: the certified derivation whose rows supply the feature values. The rows
  carry the join keys, the optional timestamp field, and the feature columns.
- `timestampField` (optional): the event-time column in the source rows. When present,
  retrieval is point-in-time; when absent, the view holds a single current value per
  entity.
- `ttlSeconds` (optional): how far back a point-in-time join may look from each entity
  timestamp. Absent means unbounded.
- `features` (optional): the feature columns exposed, each `{ name, dtype?, description? }`.
  When absent, every source column that is not a join key or the timestamp field is
  exposed.
- `description` (optional).

## 5. Retrieval semantics

- **Historical (point-in-time).** Given an entity dataframe whose rows carry the join
  keys and an `event_timestamp`, the value joined for a timestamped view MUST be the
  latest source row for that key whose event time is at or before the entity timestamp,
  and, when `ttlSeconds` is set, no older than that TTL relative to the entity
  timestamp. A source row with an event time after the entity timestamp MUST NOT be
  joined. This is the guarantee that keeps future values out of training data.
- **Online.** The latest materialized value per entity key, produced by keeping the
  row with the greatest event time (or the last-seen row for an untimestamped view).

The same source derivation feeds both paths, so offline and online values agree at the
present moment.

## 6. Certification

A view's `source` SHOULD be a `certified` derivation (see the Open Derivation Spec). An
implementation SHOULD refuse to materialize or serve a view whose source is not
certified, so that every served feature value is verified.

## 7. Conformance

An implementation is *conformant* if, for every fixture under [`tests/`](./tests/), it
agrees with the fixture's `valid` verdict when validating the fixture's `data` against
`feature_store.schema.json`. The invariants the schema cannot express (unique names and
feature views referencing defined entities) are part of conformance and are checked by
the reference implementation.
