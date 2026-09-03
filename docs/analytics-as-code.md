# Analytics as code

Everything the app can configure is also a file in your project repo, so you can build
it with your own coding agent (Claude Code, Cursor) and sync it live, with no need to author
through the app's chat. The app UI stays fully usable; the repo is just a second, version-
controlled way onto the same objects.

## How it works

- **You author files** in fixed folders (below): declarative YAML/SQL for the config
  artifacts, Python for derivations.
- **`elbi sync`** pushes the whole repo to a running app: it sends the
  declarative artifacts over the HTTP API (the app's own routes validate and gate every
  one: a metric's source must be certified, a derivation re-runs the oracle), then asks
  the app to re-read the `derivations/*.py` source and the declared source data from
  disk. A changed derivation or data file goes live with no restart.
- **`elbi pull`** writes the app's objects back as canonical files, so edits made
  in the UI round-trip into git.
- **`elbi plan`** previews the diff; **`elbi schema`** prints the
  warehouse schema so your agent knows what to build against, and
  **`elbi schema --json`** writes a JSON Schema per artifact type so your agent
  authors valid files (and your editor can validate them).

It is meant to feel like git: **`pull` to bring the app's state into the repo, `sync` to
push your changes out.** Sync is repo-authoritative and idempotent: re-running it on an
unchanged repo does nothing, and objects are matched by name so a re-sync updates in
place rather than duplicating.

## Two halves, and only one of them travels over the API

`sync` pushes the declarative artifacts to the app and then asks it to re-read
`derivations/*.py` and the declared source data **from its own disk**. So the two halves
reach a deployment differently, and it is worth being blunt about which is which:

| | How it arrives | Works against a remote app? |
| --- | --- | --- |
| The folders below | pushed over the HTTP API by `sync` | yes, from anywhere that can reach it |
| `derivations/*.py`, source data | read from the app's filesystem | only if the app can see them |

On a laptop the app *is* the repo, so both halves are the same thing and there is nothing
to think about. On a deployment they are not, and the derivations (the certified artifact,
the thing the rest of this points at) have to get onto the app's disk some other way. See
[running in Docker](docker.md) for how a long-lived instance picks up your project.

The short version: **the repo does not live in the cluster.** `sync` is a client you run
from your machine or from CI, and the project the app serves is delivered separately:
baked into an image, or pulled from git by a sidecar.

## Repo layout

```
elbi.yaml          # project + warehouse sources: (data is added here)
derivations/*.py           # Python: the certified-compute artifact
notebooks/*.ipynb          # source cells only: outputs stay on the deployment
metrics/*.yaml             # a MetricSet manifest per file
dashboards/*.yaml          # a DashboardSpec manifest per file
features/*.yaml            # entities + feature views
monitors/*.yaml            # anomaly monitors
schedules/*.yaml           # orchestration schedules / sensors
workflows/*.yaml           # procedural workflows (run-if-gated step DAGs)
checks/*.yaml              # data-quality checks (one per <asset>.<name>.yaml)
models/*.yaml              # retrain policies
queries/*.sql              # saved explore queries (a `-- source:` header line)
```

### Which deployment a repo belongs to

Declare the deployments a project may be applied to, and every command is checked against
them rather than trusting whatever `ELBI_URL` happens to hold:

```yaml
targets:
  dev:
    default: true
    host: https://elbi.acme.com
  prod:
    host: https://prod.acme.com
```

`-t prod` picks one; with none named the default is used. A `--url` or `ELBI_URL`
that disagrees with the chosen target is **refused rather than preferred**, because either
could be what was meant and the cost of guessing is somebody's work landing on another
company's deployment. Exactly one target may be `default: true`; with two, what a bare
command means would depend on iteration order.

Declaring no targets leaves a project unconstrained. A laptop wants that, and every
repo predating this looks the same. Databricks puts the host in the committed file for
the same reason: it makes the file portable, while every machine keeps its own way of
authenticating.

### What stays on the deployment

A pulled notebook carries its source and none of its outputs. A cell's output is a
rendering of warehouse rows, and `pull` writes to a laptop, so exporting them would move
data off the deployment for no gain: a notebook is executed on the deployment and never
locally, so an output has no use on the far side.

That is the polarity every comparable tool settled on. `nbstripout` exists to keep outputs
out of version control and names sensitive information among its reasons, and Databricks
Git folders will not commit `.ipynb` output until a workspace administrator enables it.
Here the equivalent switch is `NOTEBOOK_EXPORT_OUTPUTS=1`, which permits a deliberate
download to ask for them; unset, nothing can. Asking is never enough on its own, because
whether data may leave is a question about the deployment rather than about the caller.

Two things do carry values, and both are deliberate:

- **Certificates.** A certificate is a signed, tamper-evident proof of a claim, meant to be
  handed to somebody, so it carries the claim's statistics by design. A repo holding
  `certificates/` therefore holds those aggregates. Omit the folder from `pull` if that
  matters more than the round-trip.
- **`sample_values` in `elbi.yaml`.** Authored by a person, never emitted by the
  app. It is a place where somebody can paste real values into a committed file; treat it
  as documentation rather than as data.

The file's stem is the object's name (`metrics/revenue.yaml` → a metric named
`revenue`). Names must be unique within a folder.

A metrics file is the exception: it may define one metric, or several under a `metrics:`
list that names each, so a ratio can sit beside the two metrics it divides. `pull` writes
one metric per file; grouping is for authoring. Order inside the file does not matter,
because `sync` pushes a metric after whatever it is defined in terms of.

## CLI

```bash
elbi schema                 # print the warehouse schema (tables, columns, types)
elbi schema --json -C .     # write .elbi/schemas/*.json (one per artifact)
elbi plan   -C .            # preview what sync would change
elbi sync   -C .            # apply repo -> app
elbi sync   -C . --prune    # also delete app objects the repo omits
elbi pull   -C .            # write app -> repo (round-trip UI edits into git)
```

`sync` validates every file against these schemas before it touches the app, so a typo
in a metric or monitor is a local error with the exact JSON path, not a failed request.

`--prune` only touches the kinds the repo actually declares: a folder that exists claims
its kind, and an *empty* folder is how a repo asks for none of them. A repo that tracks
only `metrics/` therefore cannot lose its certificates to a prune it did not mean.
The metric, dashboard, and feature-store schemas are the app's own bundled specs (with
the file-supplied `name`/`specVersion` relaxed); the rest describe each file's body.

Pick the deployment with `-t <target>` (see
[which deployment a repo belongs to](#which-deployment-a-repo-belongs-to)), or with `--url`
/ `ELBI_URL` where a project declares no targets. The default is
`http://127.0.0.1:7700`.

An app on this machine needs no credential. Where one is needed, pass `--token` or set
`ELBI_API_KEY`; `--token` takes precedence.

## MCP: let your coding agent see the data

Point Claude Code or Cursor at the app's MCP endpoint (`http://<app>/mcp`) and it gains
warehouse-schema tools:

- `describe_warehouse`: every table with its columns, types, and row counts
- `table_schema(name)`: one table's columns and types
- `sample_rows(name, limit)`: a few example rows

This is the context an agent needs to author a metric, notebook, or derivation without
guessing column names, the prerequisite for building anything against your data.

## Secrets

Secret **values** live in the app (set once in the UI, encrypted at rest, or provided via
the environment); the repo only ever holds a secret **reference by name**, never the
value. A file names the secret it needs; the app resolves the value at run time and
injects only the declared secrets into the code that runs, so the same file works
locally and in production, and no credential is ever committed to git.

## What syncs

Every configurable surface round-trips end to end: create, idempotent re-sync, pull,
and prune all hold:

| Folder        | Object                | Notes                                            |
| ------------- | --------------------- | ------------------------------------------------ |
| `metrics/`    | one metric, or a list | source must be a certified derivation            |
| `dashboards/` | DashboardSpec         | created as a draft; certified bindings on publish|
| `features/`   | entities + views      | one `store.yaml` (entities, then views)          |
| `monitors/`   | anomaly monitor       | target must be a certified metric or derivation  |
| `schedules/`  | materialization sched | cron or data-change sensor                       |
| `workflows/`  | procedural workflow   | a DAG of run-if-gated materialization steps      |
| `checks/`     | data-quality check    | one file per `<asset>.<name>`; boolean SQL expr  |
| `models/`     | retrain policy        | training spec resubmitted on schedule / on change|
| `queries/`    | saved SQL             | a `-- source:` header line                       |
| `notebooks/`  | notebook              | outputs stripped; source cells only              |

Idempotency holds on the **canonical** form the app pulls to: `sync` then `pull` then
`sync` is always a no-op, so a hand-authored file that omits defaults converges to the
canonical file on first pull. Derivations are loaded from `derivations/` by the project
itself (they take effect on app load) rather than pushed over the API.

### An update keeps the object it updates

`sync` changes an existing object **in place**. It never replaces one by deleting it and
creating a substitute, and the difference is not cosmetic. An object's id and everything
hanging off it are part of what it is:

- A **notebook** keeps its id, its folder, and its creation time. A notebook that
  reappeared at the root would lose where it was filed, and nothing would report it.
- A **monitor** keeps its snapshots and its incidents. Those snapshots are the baseline
  anomaly detection compares against, so a monitor re-created to change a threshold would
  start over with nothing to compare to.

Anything referring to an object by id (a dashboard tile, a lineage edge, a link
somebody shared) therefore keeps working across a sync.
