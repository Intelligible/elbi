# What platform search indexes

<!-- Generated from `elbi.search.specs`. Do not edit by hand: a test
     asserts this file matches the declarations the indexer reads. -->

Every entity is either mapped below or excluded with a reason. The
mapping is the contract the indexer implements; nothing reflects over a
model, so a column added later cannot reach the index by existing.

## Entities

A field is worth 2 as a title and 1 as body, which is where the mapping's
judgement lives: `email`→title on `user` is the decision that finding
a person by address ranks with finding them by name. A warehouse table's
columns weigh 1, as body does.

| Entity | Fields | Chunking | Facets | Route |
| --- | --- | --- | --- | --- |
| `asset_check` | `name`→title, `expr`→body, `asset`→body | whole | - | `/orchestration?asset={asset}` |
| `asset_run` *(only `state=failed`)* | `asset`→title, `error`→body, `logs`→body | split 2000/200 | status, verdict, created_at | `/orchestration?run={parent}` |
| `audit_event` | `action`→title, `target_id`→body, `target_type`→body | whole | verdict | `/settings/audit` |
| `conversation` | `title`→title, `summary`→body | whole | created_at | `/conversations/{id}` |
| `dashboard` *(only `deleted_at=None`)* | `title`→title, `name`→title | whole | status | `/dashboards/{id}` |
| `data_source` | `name`→title, `description`→body, `source_type`→body | whole | status | `/warehouse/sources/{id}` |
| `derivation` *(only `deleted_at=None`)* | `name`→title, `question`→body, `source`→body, `narrative`→body, `assumptions_json`→body (json), `attestation_json`→body (json) | field | verdict, created_at, data_hash | `/derivations/{id}` |
| `feature_entity` | `name`→title, `description`→body | whole | - | `/features` |
| `feature_view` *(only `deleted_at=None`)* | `name`→title, `description`→body, `source`→body, `contract_json`→body (json) | whole | - | `/features/{id}` |
| `llm_profile` | `name`→title | whole | - | `/settings/models` |
| `materialization_schedule` | `name`→title, `selection`→body, `dataset`→body | whole | - | `/orchestration` |
| `message` | `content`→body | split 2000/200, late | created_at | `/conversations/{parent}?message={id}` |
| `metric` *(only `deleted_at=None`)* | `name`→title, `manifest_json`→body (json) | whole | created_at | `/metrics?metric={id}` |
| `metric_monitor` | `name`→title, `target`→body, `method`→body | whole | created_at | `/monitors?monitor={id}` |
| `monitor_incident` | `reason`→body | whole | incident_state, created_at | `/monitors?incident={id}` |
| `notebook` *(only `deleted_at=None`)* | `name`→title | whole | created_at | `/notebooks/{id}` |
| `notebook_cell` | `source`→body, `outputs_json`→body (json) | split 2000/200, late | - | `/notebooks/{parent}` |
| `notebook_folder` *(only `deleted_at=None`)* | `name`→title | whole | - | `/notebooks?folder={id}` |
| `orchestration_run` | `cause`→title | whole | status, created_at | `/orchestration?run={id}` |
| `saved_query` *(only `deleted_at=None`)* | `name`→title, `sql`→body | whole | - | `/explore?query={id}` |
| `setting` | `key`→title | whole | - | `/settings` |
| `training_set` | `name`→title, `label`→body | whole | - | `/features` |
| `workflow` | `name`→title | whole | - | `/orchestration?workflow={id}` |

### Why each is chunked that way

*late* means each chunk is embedded in the context of its whole row rather
than on its own, so a fragment keeps what surrounded it. It needs a
`SpanEmbedder`; the static fallback embedder degrades to per-chunk.

- `asset_check`: whole, a name and a short description
- `asset_run`: split 2000/200, a failed run's log is long, and the error is anywhere in it
- `audit_event`: whole, a name and a short description
- `conversation`: whole, a name and a short description
- `dashboard`: whole, a name and a short description
- `data_source`: whole, a name and a short description
- `derivation`: field, prose, code and structured assumptions score separately
- `feature_entity`: whole, a name and a short description
- `feature_view`: whole, a name and a short description
- `llm_profile`: whole, a name and a short description
- `materialization_schedule`: whole, a name and a short description
- `message`: split 2000/200, late, a message can be arbitrarily long prose
- `metric`: whole, a name and a short description
- `metric_monitor`: whole, a name and a short description
- `monitor_incident`: whole, a name and a short description
- `notebook`: whole, a name and a short description
- `notebook_cell`: split 2000/200, late, cell source and its printed output are both unbounded
- `notebook_folder`: whole, a name and a short description
- `orchestration_run`: whole, a name and a short description
- `saved_query`: whole, a name and a short description
- `setting`: whole, a name and a short description
- `training_set`: whole, a name and a short description
- `workflow`: whole, a name and a short description

## Sources with no table behind them

| Entity | Read from | Title | Other text | Facets |
| --- | --- | --- | --- | --- |
| `warehouse_table` | WarehouseService.tables() joined to WarehouseService.columns() | the warehouse table name | its column names, from the persisted catalog (IP-17) | created_at |
| `model` | the MLflow model registry, via ModelService.registry() | the registered model name | its description and the tags a person set | verdict, created_at |

## Excluded, and why

| Table | Reason |
| --- | --- |
| `Budget` | a numeric limit, filtered on rather than searched |
| `ComputeUsage` | usage accounting, no searchable text |
| `DashboardSubscription` | a delivery rule with no searchable text |
| `DashboardVersion` | its label is only ever 'saved' or 'published', never written |
| `DataSource` | superseded by ExternalDataSource for connections |
| `DerivationRun` | a run of an indexed derivation |
| `ExternalDataSchema` | the warehouse_table document is assembled from it |
| `FeatureDriftRow` | computed over the data |
| `FeatureExpectationRow` | computed over the data |
| `FeatureStatisticsRow` | profiles and samples of the data |
| `InferenceEvent` | production request and prediction payloads |
| `JobRow` | its label is an indexed derivation's name, and nothing more |
| `MetricSnapshot` | a numeric observation, filtered on rather than searched |
| `MetricVersion` | a version of an indexed metric |
| `Notification` | a delivery of an event whose artifact is indexed |
| `NotificationPreference` | per-event-type switches, with no prose of their own |
| `OnlineFeatureRow` | the feature values themselves |
| `PromotedQuery` | promoting authors a derivation, and that gets indexed |
| `RetrainPolicy` | settings on an indexed model, with no prose of its own |
| `Secret` | holds an encrypted credential |
| `WarehouseColumn` | the warehouse_table document is assembled from it |

## Size at a realistic corpus

A mature single-team project. These row counts and text lengths are
stated assumptions, not measurements: argue with a number rather than
with the conclusion.

| Entity | Rows | Chunks/row | Chunks | Assumption |
| --- | ---: | ---: | ---: | --- |
| `audit_event` | 50,000 | 1 | 50,000 | the highest-count entity in the corpus |
| `derivation` | 1,000 | 5 | 5,000 | question, source, narrative, two JSON |
| `message` | 5,000 | 1 | 5,000 | a chat turn, occasionally much longer |
| `orchestration_run` | 5,000 | 1 | 5,000 | one per materialization run |
| `notebook_cell` | 2,400 | 1 | 2,400 | ~12 cells a notebook, source + outputs |
| `asset_run` | 250 | 5 | 1,250 | 5% of ~5k runs fail; only those are indexed |
| `conversation` | 500 | 1 | 500 | - |
| `saved_query` | 300 | 1 | 300 | SQL text, short enough to stay whole |
| `notebook` | 200 | 1 | 200 | - |
| `monitor_incident` | 200 | 1 | 200 | - |
| `metric` | 100 | 1 | 100 | - |
| `asset_check` | 100 | 1 | 100 | - |
| `warehouse_table` | 100 | 1 | 100 | columns are a field, not documents |
| `dashboard` | 50 | 1 | 50 | - |
| `metric_monitor` | 50 | 1 | 50 | - |
| `setting` | 50 | 1 | 50 | - |
| `feature_view` | 40 | 1 | 40 | - |
| `notebook_folder` | 30 | 1 | 30 | - |
| `training_set` | 30 | 1 | 30 | - |
| `materialization_schedule` | 30 | 1 | 30 | - |
| `model` | 30 | 1 | 30 | - |
| `feature_entity` | 20 | 1 | 20 | - |
| `workflow` | 20 | 1 | 20 | - |
| `data_source` | 20 | 1 | 20 | - |
| `llm_profile` | 5 | 1 | 5 | - |
| **total** | | | **70,525** | |

At 384 dimensions and 4 bytes a float, 70,525 chunks is **108 MB** of vectors.

An exact `array_cosine_similarity` scan over 200k documents at this width
measures 127 ms, and 14 ms once a filter narrows it. A corpus this
size is therefore answered by scanning a column: a dedicated vector store
would buy latency that is already there. The parent's no-vector-store
decision assumed as much, and this validates it.

### One number decides it

`audit_event` is 50,000 of those 70,525 chunks, 71% of the
whole corpus, and the only entity with no natural ceiling. It is also the one
whose text is mostly identifiers rather than prose, so it gains least from
being embedded at all.

Excluding it, or indexing it lexically without a vector, leaves
20,525 chunks and 32 MB.
That is the decision worth taking deliberately before the indexer is
built, rather than discovering it when the audit log has grown.
