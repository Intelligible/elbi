# Orchestration

Orchestration here is **asset-centric**, in the [Dagster](https://dagster.io/) sense: a
derivation is a software-defined asset, and you *materialize assets* rather than run a
task DAG. The system runs only what's needed, because derivations are content-addressed
and cached, an unchanged asset is a cache hit, and materializing is idempotent, so
re-runs and backfills are safe by construction.

## Staleness

An asset is **stale** when the content version of its inputs or code has moved since it
was last materialized; **materialized** when the versions match; **never** when it has
no materialization on record. This falls out of the same `data_version` that drives the
cache, with no separate freshness bookkeeping. A change to a source dataset marks every
asset downstream of it stale, which the Orchestration page shows at a glance.

## Materialization runs

A run materializes a selection in dependency order (from the lineage graph):

- **stale**: only the assets that need it (the common case).
- **all**: every asset.
- **named assets**: a chosen set, optionally with their downstream.

Each asset is materialized through the project runner; fresh ones are skipped, and a
transient failure is retried with backoff. The run records a per-asset result (status
(`succeeded` / `failed` / `skipped`), attempts, duration, and the oracle verdict) so you
can see exactly what ran and what failed.

The retry count is configurable (Schedules tab, or `/api/orchestration/settings`); a
failing run can be **retried from failure** (re-materializing only the failed assets and
their downstream) or **cancelled** at the next asset boundary.

## Backfills

A **backfill** materializes a parameterized derivation once per partition value, so each
value runs the asset with `{param: value}` and records its own step, the way Dagster
backfills an asset across partition keys. Unchanged partitions are cache hits at the
runtime layer, so re-running a backfill is cheap. `POST /api/orchestration/backfill`, or
the Backfill form on an asset.

## Data-quality checks

A check is a boolean SQL expression over a derivation's output columns, evaluated with
DuckDB at materialization, the same idea as a Delta Live Tables expectation or a Dagster
asset check. Each check has a name, an `expr` (e.g. `amount >= 0`), and a severity:

- **warn**: record how many rows fail the constraint; the run still succeeds.
- **error**: a failing row turns the asset's step `failed` (and so the run), the way
  `expect_or_fail` prevents a bad update.

Checks run on every materialization, including a skipped (fresh) asset, so a newly-added
check takes effect without forcing a rebuild and current data is always re-validated. The
per-row pass/fail counts show on each run's steps. Manage checks per asset on the Assets
tab or through `/api/orchestration/checks`.

## Notifications

A run that **fails** (after per-asset retries are exhausted) or runs **slow** (more than
2× its recent median duration) is delivered to the configured webhook, using the same endpoint
and signature as model and monitor events, and recorded in the audit log. This mirrors
Databricks' job Failure and Duration-Warning notifications; transient blips are muted
because a step only reaches `failed` once its retries are spent.

## Workflows (procedural control flow)

A **workflow** is a named DAG of steps, where each step materializes a selection and is
gated by a **trigger rule** over its upstreams, the Databricks/Airflow control-flow
model. Rules: `all_success` (default), `none_failed`, `at_least_one_success`, `all_done`,
`at_least_one_failed`, `all_failed`; a step whose rule isn't met is *excluded* (skipped),
and an excluded upstream counts as a success. This lets a workflow **branch**, e.g.
`build → publish (all_success)` alongside `build → cleanup (at_least_one_failed)`, rather
than only fan assets out by data dependency. All steps record into one run, so the run
detail shows every asset that ran plus the per-step outcomes. `/api/orchestration/workflows`
and the Workflows tab.

## Compute

Derivations run through a pluggable **Executor** protocol: human-authored ones in-process,
agent-authored ones in the configured **sandbox** backend (`subprocess` or `docker`),
routed automatically. The same protocol is the seam for a remote/distributed backend
(a networked worker or microVM). Incrementality comes from the content-addressed cache
(an unchanged asset is skipped) and partitioned backfills (per-partition cache hits); the
compute backend in effect is shown on the Schedules tab and `/api/orchestration/settings`.

## Schedules and sensors

- **Schedules** fire a materialization on a **cron** expression (parsed with the
  standard `croniter`), e.g. `0 6 * * *` to rebuild stale assets every morning.
- **Sensors** fire when a dataset's content hash changes: event-driven
  materialization, the same mechanism the retrain and notebook schedules use.

Both run on the in-process maintenance scheduler; a schedule targets a selection (all,
stale, or named assets).

## In the app and to the agent

The Orchestration page is organized into tabs: **Overview** (health KPIs and the
dependency graph), **Assets** (freshness, last-materialized time, and per-asset
materialize, and data-quality checks), **Runs** (a runs-by-assets history matrix with
each run's steps, Gantt timings, verdict, checks, and logs in a side panel), and
**Schedules**. The API is under `/api/orchestration` (`status`, `graph`, `history`,
`materialize`, `runs`, `schedules`, `checks`).

The chat agent has the same control through two tools, `asset_status` (what's stale)
and `materialize_assets` (recompute in dependency order), so it can keep the project's
assets fresh the way a human operator would.

Backfills over partitioned parameters and a staleness overlay on the
[lineage graph](lineage.md) are natural next steps.
