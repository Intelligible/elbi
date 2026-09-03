# Models: AutoML, the registry, and serving

Derivations answer questions; sometimes the deliverable is not an answer but a
**model**: something that scores tomorrow's records, keeps versions, and serves
predictions to other systems. The `ml` extra adds that lifecycle with deliberately
standard parts: a choice of AutoML engines (FLAML, AutoGluon, an Optuna tuner, the
open TabICL tabular foundation model, and a cross-model ensemble),
MLflow for tracking, the MLflow Model Registry for versions and promotion, and
MLflow's scoring protocol for serving. Nothing here is a house invention; the point is
that a data scientist's existing tools and habits work unchanged, with the platform's
verification oracle woven in where it earns its keep.

```bash
pip install 'elbi[ml]'        # the SDK: train, register, load, score
pip install elbi-app          # the app: chat tools, registry API, serving (ml included)
```

Everything below is drivable three ways, like any data-science platform: from the
**web app** (a training page, one-click promotion, a query pane, and the real
MLflow UI embedded at `/mlflow/`), from **code** (the SDK and REST API), or by
**asking the chat** - the chat is one client of the same lifecycle, not the entry
point.

## The MLflow UI, embedded

The app mounts MLflow's own server application at `/mlflow/`, bound to the same
tracking store the platform trains into. That is the full MLflow experience
(experiment tables, run charts and comparisons, the registry pages, artifact
browsing), not a re-implementation; the platform's Models pages link into it per
model, version, and run. The mount is served like the rest of the
API.

## Training

Use the Models page's *Train model* button (pick a dataset, target, budget; the
run executes as a background job you can watch in the jobs bar), ask the chat, or
call `train_automl` directly:

```python
from elbi.ml import train_automl

report = train_automl(
    rows,  # the dataset's records
    name="churn",  # the registry name; reuse it to add a version
    target="churned",
    time_budget=60,  # seconds of AutoML search
    tracking_uri="sqlite:///.elbi/mlflow.db",
)
```

Six engines share one lifecycle, chosen with the `engine` field. Everything around
the engine (the oracle gate, the registry and champion policy, lineage, evidence
artifacts, drift, batch and raw scoring) is engine-neutral: whichever engine trains,
the model serves through the same protocol and is judged by the same gate.

| `engine`     | What it does | Reach for it when |
| ------------ | ------------ | ----------------- |
| `flaml` (default) | Budget-aware AutoML over the standard tabular learners (LightGBM, XGBoost, random forest, linear models). Light and fast. | Any first pass; interactive turns; forecasting (the only engine that supports `ts_forecast`). |
| `autogluon`  | Accuracy-first multi-layer stack ensembling that tops the tabular benchmarks. `ensemble: true` selects its `best` preset. | You want maximum accuracy and can spend a longer budget. |
| `optuna`     | An explicit Bayesian tuner: a TPE study with median pruning tunes one gradient booster (XGBoost by default, or `estimator_list: [lgbm]`) by cross-validation. Every trial is a child run. | You want a legible, single-model hyperparameter search rather than a black-box AutoML. |
| `tabicl`     | The TabICL tabular foundation model, a pretrained transformer that predicts in one forward pass, no per-dataset training. Currently #1 on the TabArena benchmark; open (BSD-3) weights, no token. | Small-to-mid data (hundreds to tens of thousands of rows) where a foundation model shines and loop latency matters. |
| `ensemble`   | Fits a diverse base set (LightGBM, XGBoost, CatBoost, random forest, extra-trees, and TabICL when available) and blends them by greedy weighted selection, the ensembling the living benchmarks show beats any single model. | You want the strongest model and don't need to know which family won. |

The foundation model, the tuner, and the blend do not forecast; `ts_forecast` runs on
FLAML. TabICL predicts in a single pass, so the search budget and the `ensemble` flag
do not apply to it; its open (BSD-3) pretrained weights download from Hugging Face on
first use with no token or license step, so it is safe to ship and serve in a product.
(TabPFN and Google's TabFM are the same idea but ship non-commercial weights, so they
are deliberately not wired in as engines here.) Budgets are
real-world scale: a training job (the UI's path, or a chat request past the inline
allowance, which the chat automatically hands to the job runner) may run up to 24 hours;
only an inline chat tool call is clamped to minutes, because it blocks the
conversation's turn. Before anything trains, the verification oracle runs the same
`prediction` gate a `derive` with a `{target, features}` claim uses, checking that
leakage-free held-out signal exists in the data at all, regardless of engine. The
report's metrics are computed on a held-out split the search never saw, never the
search's own validation score, which is optimistically biased by the search itself.

The trained model is logged to MLflow with its signature, an input example, and
exact dependency pins, then registered as a new version of `name`.

## The registry and the champion alias

The registry is MLflow's, so versions, aliases, and lineage behave exactly as MLflow
documents them. Promotion follows MLflow 3's alias convention: `@champion` is the
version a bare model name serves.

The one platform-specific rule is how the champion is first assigned: the **first
version whose signal the oracle certified as sound** gets the alias automatically;
everything after that is an explicit act. A model trained over data the oracle could
not certify stays unaliased until a person promotes it deliberately, and a new
version never displaces a champion silently: compare its held-out metrics and
promote when they earn it.

```python
from elbi.ml import ModelRegistry

registry = ModelRegistry("sqlite:///.elbi/mlflow.db")
registry.models()  # every registered model, champion included
registry.versions("churn")  # versions with their runs' metrics and params
registry.promote("churn", 3)  # move @champion to version 3
model = registry.load("churn")  # the champion (or newest), as a pyfunc model
```

## Serving

The app serves every registered model over MLflow's scoring protocol, so anything
that can call an `mlflow models serve` endpoint can call this one:

```bash
curl -X POST http://localhost:7700/api/serving/churn/invocations \
  -H 'Content-Type: application/json' \
  -d '{"dataframe_records": [{"recency": 90, "frequency": 1}]}'
# {"predictions": [1]}
```

All four MLflow payload framings work (`dataframe_split`, `dataframe_records`,
`instances`, `inputs`, plus optional `params`), and the response is
`{"predictions": ...}`. A `?version=` query pins a version or alias; the default is
the champion.

The same lifecycle is a JSON API: `GET/DELETE /api/registry/models[/{name}]`,
`POST /api/registry/train` (returns a job to poll), promotion, per-version
deletion, and `GET .../versions/{v}/schema` (the signature and the input example
logged at training time). The model detail page uses that schema for its query
pane: pick a version, *Show example* fills the request with the logged example,
*Send request* scores it, exactly the workflow an MLflow or Databricks serving
page offers.

## Where the runs live

Everything above writes to one MLflow store, resolved the same way as the
certified-run export: the `MLFLOW_TRACKING_URI` environment variable wins, then the
app's `mlflow_tracking_uri` setting, then a zero-configuration SQLite store under
the project's `.elbi/` directory. Point it at a team MLflow server and the
runs, registry, and models appear there; point nothing and it works locally out of
the box.

Give it a Postgres URI without naming a driver. MLflow's registry compares a version
string against an integer column, so it needs psycopg2, which the `ml` extra installs and
the app selects for this store alone; its own tables stay on psycopg 3. Naming a driver
yourself is respected and, if that driver types its parameters, aliases will fail.

A notebook kernel reaches that same store, but through the app's `/mlflow` mount rather
than the store itself: it is handed a URL and no credentials, and artifacts are proxied
through the server. Set `NOTEBOOK_MLFLOW_TRACKING_URI` to that mount (in a cluster, the
app's in-cluster address, because a kernel is a pod) and optionally
`NOTEBOOK_MLFLOW_TRACKING_TOKEN` if the mount is behind a bearer credential. See
[Notebooks](./notebooks.md).

## What serving resolves

Serving resolves the `champion` alias, and only that alias. A version earns it from the
prediction gate, so a model with versions but no champion is listed and loadable by
explicit version while nothing serves it, including one logged straight from a
notebook.
Promote deliberately; that is the second act, and MLflow and Unity Catalog both treat it
that way.

## Feature engineering: derivations as the feature pipeline

Feature engineering is a derivation, the platform's own primitive: durable
compute-as-code that is certified (optionally against a data contract), cached,
content-addressed, and chained. Both trainers accept a **certified derivation as
the training source** alongside raw datasets, which closes the loop three ways:

- **Lineage**: the model's version records the feature derivation
  (`elbi.feature_derivation`) and the content hash of the exact rows it
  trained on, so "which feature code built this model" is a query.
- **Continuous training over the pipeline**: a retrain policy on a derivation
  source keys on the derivation's *output* hash. Upstream data refreshes, the
  derivation recomputes (incrementally, through the cache), its output identity
  moves, and the model retrains: a real feature-to-model DAG with no
  orchestrator to configure.
- **No training-serving skew**: `POST /api/serving/{name}/raw-invocations`
  scores raw, source-shaped records by first running them through the *same
  certified feature code* that built the training data (the model page's query
  pane has a Raw mode; the chat's `predict` takes `raw: true`). The engineered
  rows are echoed back, and the standard `/invocations` endpoint still serves
  engineered features for callers that have them.

Chat-authored derivations persist to the project's authored sidecar, so a
feature pipeline built in conversation survives restarts as a registry entry,
a training source, and a scoring transform.

## Every training run's evidence

A training run is not one opaque row. It carries, as MLflow artifacts on the run:
every search **trial as a nested child run** (the leaderboard in the MLflow UI),
the standard **evaluation plots** (ROC, precision-recall, lift, calibration,
confusion matrix) computed on the held-out split, **feature importances** and a
**SHAP beeswarm**, a **data profile** of the training columns with reviewer
warnings, an editable **training script** reproducing the winning configuration,
the **reference sample** later drift checks compare against, and a **model card**
that doubles as the registered version's description. Runs and versions are
tagged with the training data's content hash and dataset name, so lineage from a
model back to the exact rows is a query, not archaeology. Forecasting is a task
(`ts_forecast` with a time column and horizon): the split is temporal and the
oracle's check is its forecast gate (does the model beat naive baselines on the
held-out tail).

## Batch scoring

Score a whole dataset with a registered version (the `ai_query` analog): from the
model page, the chat (`batch_score`), or
`POST /api/registry/models/{name}/batch-score`. The full predictions land as a
`predictions.csv` artifact on a scoring run in the model's experiment; the caller
gets summary statistics and a sample.

## Production monitoring

Every `/invocations` call is logged to the **inference table** (payload,
predictions, latency, failures; pruned on a retention window). Drift checks
compare that traffic against the training reference sample using Evidently's
data-drift statistics: on demand per model, and automatically on the maintenance
schedule once enough traffic accumulates. Results are recorded as drift-check
runs in the model's experiment (summary metrics plus the full HTML report), and a
dataset-drift outcome fires the registry **webhook** (`drift.detected`).

## Continuous training and webhooks

A per-model **retraining policy** retrains either on an interval or when the
dataset's content hash changes (the platform already knows every dataset's
content identity), running the same training job a person would submit. Registry
events (`model_version.created`, `alias.updated`, deletions, `drift.detected`)
post to a configurable webhook with a GitHub-style HMAC signature
(`X-Elbi-Signature`), and every model operation lands in the audit log.

## In the chat

The chat model has four tools: `train_model`, `list_models`, `predict`, and
`promote_model`. It is instructed to reach for them when you want a reusable model,
and to keep using `derive` when the question is only whether predictive signal
exists: training is for a deliverable, certification is for a claim. Reported
numbers in a training answer are the held-out metrics from the report, and the
oracle's verdict rides along in the run and version tags
(`elbi.oracle_verdict`), so an audit can always separate certified signal
from registered-but-unproven models.
