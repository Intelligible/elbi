# Feature store

A feature store turns your certified derivations into reusable, retrievable features.
It adds two things derivations alone don't have: a **point-in-time-correct** historical
join for building training data without leakage, and a **low-latency online store** for
serving the latest feature values at inference. Because a feature is just a certified
derivation, every feature value is verified and versioned, and the same derivation
feeds training and serving, so there is no train/serve skew.

The API follows [Feast](https://docs.feast.dev/): entities, feature views,
`get_historical_features`, `get_online_features`, and `materialize`.

## Entities and feature views

An **entity** names a join key. A **feature view** groups feature columns produced by a
certified derivation, keyed by entities and (optionally) an event-time column:

```json
{
  "specVersion": "1.0",
  "kind": "FeatureStore",
  "entities": [
    { "name": "user", "joinKey": "user_id" }
  ],
  "featureViews": [
    {
      "name": "user_stats",
      "entities": ["user"],
      "source": "user_activity_features",
      "timestampField": "event_timestamp",
      "ttlSeconds": 604800,
      "features": [{ "name": "clicks_7d" }, { "name": "sessions_7d" }]
    }
  ]
}
```

`source` is a certified derivation whose rows carry the join key (`user_id`), the event
time (`event_timestamp`), and the feature columns. If `features` is omitted, every
source column that is not a join key or the timestamp is exposed.

## Point-in-time historical retrieval

To build training data, you join features onto an *entity dataframe*: rows of join
keys plus the timestamp each label was observed:

```python
get_historical_features(
    store,
    entity_df=[{"user_id": "u1", "event_timestamp": "2024-03-15"}],
    features=["user_stats:clicks_7d"],
    run=run_derivation,
)
```

For each entity row, the value joined is the latest feature snapshot **at or before**
that row's timestamp, within the view's TTL. A snapshot recorded *after* the label's
timestamp is never joined, so future information stays out of training data. The
[prediction gate](models.md) enforces the same leakage guarantee.

## Training sets

A point-in-time join is how you build training data, but a join you have to re-run
identically to reproduce a model is not itself an artifact. A **training set** freezes
the join output into a named, reusable training source:

```
POST /api/features/training-sets
{ "name": "churn_2024", "features": ["user_stats:clicks_7d"],
  "entity_df": [{"user_id": "u1", "event_timestamp": "2024-03-15", "label": 1}],
  "label": "label" }
```

The result is materialized and persisted, then offered alongside bound datasets and
certified derivations as a model training source (`kind: "training_set"`): pick it in
the training form, name it in the `train_model` tool, or POST `{"training_set": "..."}`
to `/api/registry/train`. Because the join is leakage-free, so is every model trained on
the set. `GET /api/features/training-sets` lists them; the agent builds one with
`create_training_set`.

## Online retrieval

`materialize` snapshots the newest value per entity into the online store; inference
reads it back in a single lookup:

```python
materialize(store, online, run=run_derivation)
get_online_features(store, [{"user_id": "u1"}], ["user_stats:clicks_7d"], online=online)
```

The online value equals the historical value joined at the present moment, because both
come from the same derivation.

## Monitoring: statistics and drift

Certification proves a feature view's derivation is *sound*; it says nothing about
whether this week's values have *moved*. Monitoring closes that gap. A **statistics
snapshot** profiles a view's feature columns (completeness, distinct count, and range
per feature) and records it, so you can watch a distribution over time:

```
POST /api/features/views/user_stats/statistics      # snapshot now
POST /api/features/views/user_stats/statistics {"set_baseline": true}
GET  /api/features/views/user_stats/statistics       # history, newest first
```

Setting a baseline also keeps a capped row sample as the reference a **drift
check** compares against. The check reuses the same [Evidently](https://evidentlyai.com/)
data-drift engine the [model monitor](models.md) uses, so a feature view and a model
report drift the same way, and records which features moved and whether the view as a
whole drifted:

```
POST /api/features/views/user_stats/drift-check      # compare current vs baseline
GET  /api/features/views/user_stats/drift            # drift history, newest first
```

A dataset-drift outcome fires the `feature.drift_detected` webhook. The maintenance
scheduler re-checks every view that has a baseline once per
`FEATURE_DRIFT_INTERVAL_HOURS` (default 24), so monitoring is opt-in per view and runs
unattended once a baseline is set. Drift monitoring rides on the ML stack, which the app
ships installed; in the SDK it needs the `ml` extra (`pip install 'elbi[ml]'`).

## Expectations: a data contract on a view

Drift asks whether values *moved*; a **data contract** asks whether they meet a declared
bar at all: types, ranges, non-null, uniqueness. A view can carry one contract (the
platform's own [DataContract](spec.md), the same quality bar derivations use), and its
feature values are checked against it with the same three-valued verdict the oracle
speaks:

```
POST /api/features/views/user_stats/contract/suggest   # profile → proposed contract
PUT  /api/features/views/user_stats/contract           # attach { "contract": {...} }
POST /api/features/views/user_stats/expectations-check  # sound / unsound / inconclusive
GET  /api/features/views/user_stats/expectations        # check history
```

An `unsound` outcome fires the `feature.expectations_failed` webhook. The agent verifies
a view with `check_feature_expectations`; a human attaches the contract (the UI can
suggest one from a profile and attach it in a click). This reuses the contract engine
wholesale, with no separate expectations system to learn.

## Serving and governance

The app exposes the store under `/api/features`: register entities and feature views,
`materialize`, retrieve online and point-in-time historical features, and browse the
registry for discovery. The registry is a searchable list; each view opens a detail page
showing its schema (feature name, type, description), materialization freshness (how many
entity keys are materialized and when), statistics, drift, expectations, a point-in-time
or online lookup, and, through the [lineage graph](lineage.md), the models that consume
it via its training sets. A **certification gate** refuses to materialize or serve a view
whose source derivation is not certified, so a served feature value is always
verified.

### Serving and train/serve skew

A model served here does **not** read the online store to enrich a request. Instead it
re-applies the *same certified derivation* that engineered its training features to the
raw request rows (see [models](models.md)). That is a stronger skew guarantee than an
online feature store gives: training and serving share the derivation itself, not merely
a store both sides hope was written consistently. The online store's job is low-latency
lookup of precomputed values for callers that want it, not a hidden dependency of the
model server.

## Deliberately out of scope

Two capabilities other feature stores advertise are intentionally absent, because the
platform meets the same need elsewhere:

- **Request-time (on-demand) transformations.** Rather than compute a feature from
  request data inside the store, the model server re-applies the feature's certified
  derivation to the request rows, as described above. A feature is the derivation, so
  there is no separate request-time transform to define or keep in sync.
- **Streaming ingestion.** `materialize` refreshes the online store on demand, and the
  maintenance scheduler re-runs monitoring unattended; there is no always-on streaming
  materializer.

Feature views are also not independently versioned: a view names a derivation, and the
derivation is already content-addressed and versioned, so the view inherits that history
rather than keeping a parallel one.

See the [Feature Store Spec](spec.md) for the normative schema.
