# Notebooks: a reactive authoring surface

Sometimes the fastest way to a derivation is to explore first. A notebook is where you do
that: a cell-based, reactive Python environment for iterating on your bound data, with the
outputs you expect (a printed value, a DataFrame, a chart) rendered inline. It is the
human-facing twin of the loop an agent runs when it answers in chat.

The distinction that matters is what a notebook is *for*. In most platforms the notebook
is the deliverable, and that is exactly the ungoverned, un-versioned artifact elbi
exists to replace. Here the notebook is scratch: durable scratch you can reopen and re-run,
but scratch. The deliverable is what *leaves* it: a cell promoted to a certified
derivation, or a training cell that registers a model. Everything below follows from that.

## Reactive execution

A notebook is a graph, not a script. Each cell defines some variables and reads others,
and running a cell re-runs exactly the cells downstream of it (the ones whose inputs it
just changed) in dependency order, and nothing else. Edit the cell that loads your data
and every cell that used it recomputes; edit a cell nothing depends on and only it runs.

The dependency graph is computed statically, without running anything: elbi reads
each cell's code and works out which global names it binds and which it reads (resolving
names through nested functions the way the Python compiler does). The editor shows this
under each cell (*defines df, revenue · uses orders*), so the dataflow is visible, not
guessed. Cells whose inputs changed since they last ran are flagged **stale**, so you can
see what is out of date before you re-run it.

Two rules keep the model honest, and the UI surfaces both:

- **One producer per variable.** If two cells define the same name, execution order stops
  being well-defined; the notebook flags the conflict rather than picking silently.
- **Mutation is invisible.** Static analysis sees `df = load()` as producing `df`, but it
  cannot see `df["col"] = …` in another cell as changing it; tracking mutation reliably
  is impossible in Python. Mutate a value in the same cell that defines it, or produce a
  new variable, and the graph stays correct. This is the one place to be careful.

Reactivity is a toggle. Turn it off and cells run only when you run them (the classic
model); turn it on and a run cascades to dependents automatically.

## Querying data

Two ways in, and the difference matters once a table is large.

```python
# Runs where the data is. The scan, join and aggregation happen server-side against the
# warehouse; only the result crosses into the kernel.
top = sql("""
    select city, count(*) as orders, avg(total) as avg_total
    from orders group by city order by orders desc
""")
top  # renders as a table
top.to_pandas()  # or .df(), an explicit step, so you can see the cost
```

`sql()` returns the **whole** result; the bound is on what gets *drawn*, not on what
you get. A result renders its first 1,000 rows and says "showing 1,000 of 21,613 rows", the
same division Databricks draws with `display` and pandas with `display.max_rows`: what
falls over at scale is the browser painting a table, not the frame in memory.

Pass `limit=` to fetch less on purpose. `result.truncated` is `True` only if the
deployment's own ceiling (1,000,000 rows) was reached, in which case a mean over these
rows describes a prefix rather than the table. Aggregate in SQL instead.

`.to_pandas()` (or `.df()`), `.to_polars()` and `.to_arrow()` all work; Polars goes via
Arrow, which it is built on.

```python
rows = data["orders"]  # every row, as dicts, in the kernel
```

`data['<name>']` is the convenient path for small tables. It fetches on first access and
caches, but it materialises Python objects (roughly 30–50× the memory of the columnar
form) and is capped at **1,000,000 rows**, past which it warns. If a table is big enough
that this matters, aggregate it in `sql()` and bring back the answer instead of the rows.

That split is deliberately visible rather than automatic: a handle that quietly
materialises just moves the memory failure later.

## Sizing the sandbox

A notebook runs on a *compute profile*: a named shape with its CPU, memory, GPU,
timeouts and egress, defined by whoever runs the deployment and chosen per notebook:

```yaml
# elbi.yaml
sandbox: docker
compute_profiles:
  - name: small
    cpu: "2"
    memory: 4Gi
  - name: large
    cpu: "8"
    memory: 32Gi
default_compute_profile: small
```

Profiles are validated when the project loads, so a typo fails there rather than when
someone opens a notebook. Set none and three sensible sizes are offered.

See [Compute](compute.md) for the full field list, the runners, the governance
(who may use which profile, what a session cost), and why the `subprocess` backend
deliberately enforces no memory limit at all.

## Base environments and their support window

An admin-defined base environment is a package baseline a notebook layers its own
dependencies on, so a common stack resolves and caches once. Each may carry a version and
a dated end of support:

```json
{"name": "DS 2026.1", "version": "2026.1", "deps": ["pandas", "scikit-learn"],
 "end_of_support": "2029-01-31"}
```

Past that date the environment stops being offered for new work and notebooks already on
it keep running unchanged. That asymmetry is the point: a lock records what you got, not
when it stops being maintained, and an upgrade only becomes plannable if the old version
stops accumulating new work rather than being removed. Databricks pairs each serverless
environment version with a dated end of support and behaves the same way at it.

### Limits

Stated rather than left to be discovered, since every one of these is reachable:

| Limit | Value | What happens at it |
| --- | --- | --- |
| Cell wall-clock | 120 s | The cell is interrupted; the kernel and namespace survive. |
| Cell output | 10 MB | Output is dropped from there on, once saying so. The cell keeps running, and its result and traceback still appear. |
| `sql()` rows | 1,000,000 (the host's ceiling) | `truncated` is set and shown. Rendering caps at 1,000 rows and says so. |
| `data['name']` rows | 1,000,000 | `truncated` is set; a warning fires past 100,000. |
| Idle kernel | 30 min | The kernel is reaped and its namespace lost. The notebook is untouched. |

The output limit is enforced by the host, not by the kernel: the kernel is the thing
running the code, so a budget it keeps for itself would be advisory. A cell that prints in
a loop is throttled, not killed; the wall-clock timeout is what stops a runaway.

## The kernel

Every notebook runs against its own kernel: a persistent Python process whose namespace
survives across cells, so a dataframe loaded once is there for the rest of the session.
The bound datasets are available as `data['<name>']` (a list of row dicts) from the first
cell, with no setup, fetched when a cell first names one rather than when the kernel starts, so
opening a notebook costs nothing regardless of how much data is bound. For anything beyond
a small table use **`sql(...)`** instead (below). The kernel supports what a data-science
session needs: the last
expression of a cell shows as its result (`df.head()` on the last line renders the table),
`print()` streams as it happens, `display()` and the `_repr_html_` / `_repr_mimebundle_`
protocol render DataFrames, Vega/Altair charts, and images, and matplotlib figures are
captured inline. A runaway cell can be **interrupted** without losing the namespace; a
**restart** gives a fresh interpreter.

Dependencies are declared per notebook and fixed when the kernel starts, like a Jupyter
kernelspec: to add a package, change the notebook's dependencies and the kernel restarts
with them. Locally the kernel is a host subprocess with a scrubbed environment and network
denied: a bounded sandbox, not a security wall. On the platform it is a hardened
container, the same isolation the certified-derivation sandbox uses. A notebook is
authoring either way, so its state is never the source of a verified result.

## Notebook fluency

The cell surface carries the interactions a notebook user expects, without a Jupyter
server:

- **Markdown and math.** Markdown cells render with GitHub extensions and KaTeX, so
  headings, tables, and `$x^2$` / `$$\int$$` display as formatted output; a cell's
  `text/markdown` and `text/latex` results render the same way.
- **Completion and help.** Tab-completion draws from the live namespace (attributes,
  names, builtins, keywords), and hovering a name shows its signature and docstring,
  the editor's Shift-Tab help, answered by the kernel.
- **Magics and the shell.** Line magics (`%time`, `%timeit`, `%who`, `%pwd`, `%cd`,
  `%env`, `%reset`, `%matplotlib`, `%lsmagic`), cell magics (`%%bash`, `%%capture`,
  `%%writefile`, `%%time`, `%%timeit`, `%%html`, `%%latex`, `%%markdown`,
  `%%javascript`), and `!command` / `!!command` shell escapes all work.
- **Input.** A cell that calls `input()` or `getpass` pauses and prompts in the browser;
  the wall-clock timeout is suspended while the prompt is open.
- **Interactive widgets.** ipywidgets render and stay two-way bound to their Python
  model (a slider drag updates the kernel, a Python assignment updates the view) over a
  per-notebook comm channel. Declare `ipywidgets` in the notebook's dependencies to use
  them.

## From a cell to a derivation

This is the point of the feature. When a cell defines a derivation function, a top-level
`def <name>(ctx): ...` that reads its inputs via `ctx.input(...)` and returns the result,
you can **promote** it. Promoting runs the same authoring loop the chat and the MCP
`propose_derivation` tool use: the code runs once under isolation, the verification oracle
and any golden cases check it, and the certification policy decides whether it is served.
On success the cell becomes a governed, cached, versioned derivation, listed alongside the
rest and served to agents, reached by building it in a notebook, never around one.

```python
def revenue_by_region(ctx):
    "Total revenue grouped by region."
    rows = ctx.input("sales").rows
    by = {}
    for r in rows:
        by[r["region"]] = by.get(r["region"], 0) + float(r["amount"])
    return [{"region": k, "revenue": v} for k, v in by.items()]
```

Put that in a cell, click **Promote to derivation**, and once it verifies it is
`revenue_by_region`, a real derivation. The reverse works too: **Open in notebook** on any
derivation scaffolds a notebook with its source, so editing a certified derivation and
re-promoting it as a new version is a first-class loop, not a copy-paste.

## Training a model here

Two paths, and they land in the same registry.

The notebook's **Train model** action runs the platform's governed AutoML (`train_automl`)
on the server against a dataset, a feature derivation, or a materialized training set. It
passes the prediction gate, so it arrives in the [Models](./models.md) page with its
evidence, lineage, and champion policy intact. The search runs in its own process, so a
model too large for the container fails the run and says so rather than restarting the app.

Or build the model yourself in a cell, with any library and any architecture, and log
it with MLflow:

```python
import mlflow

mlflow.set_experiment("kc-price")
with mlflow.start_run():
    mlflow.log_params(params)
    mlflow.log_metric("rmsle", score)
    mlflow.sklearn.log_model(pipeline, name="model", registered_model_name="kc_price")
```

Use the flavour that matches the library: `mlflow.lightgbm`, `mlflow.pytorch`, or
`mlflow.sklearn` for a genuine sklearn pipeline. `mlflow.sklearn` on a bare booster
refuses it as an untrusted type, and the error names serialisation rather than the flavour.

The kernel is handed this deployment's own tracking URL and no credentials. That is
MLflow's remote-tracking arrangement, with artifacts proxied through the server, so runs
and
versions land in the same store the [Models](./models.md) page and the embedded MLflow UI
read. Iteration is tracked from the first candidate rather than from whichever one you
decided to keep.

What a self-logged version does *not* get is the alias. Serving resolves `champion`, and a
version earns that from the prediction gate, so a model you registered here is listed,
inspectable and loadable by explicit version, and is not served until it is promoted.

If no tracking URL is configured, `register_model` raises and says so: MLflow would
otherwise write to a directory inside the sandbox and report success, losing the run when
the kernel stops. Set `NOTEBOOK_MLFLOW_TRACKING_URI` to point it somewhere durable.

The reverse, **Open in notebook** on a model, scaffolds a notebook from that version's
logged training script, so re-running or adapting a model's training is a live, editable
loop.

## Re-running: parameters and schedules

A notebook you have built is worth re-running, and there are three ways.

- **Run all** re-executes every cell top-to-bottom (optionally on a fresh kernel), which is the
  reproducibility check.
- **Parameters.** Tag one cell `parameters` and give its variables default values; a run
  can then override them. The overrides are spliced in as a cell right after the defaults
  (the papermill model), so the run is reproducible for exactly those inputs. Under
  reactive execution the injected values propagate to every dependent cell.
- **Schedules.** Three modes. `cron` runs at a wall-clock time in a named zone; `interval`
  runs every so many hours; `on_data_change` runs whenever a dataset's contents change,
  keyed on its content hash, the same identity retraining policies use. A scheduled run
  executes the whole notebook with its stored parameters on a fresh kernel.

  Reach for `cron` whenever the time is a calendar one. An interval cannot say "the 1st at
  06:00": 720 hours drifts off it by the length of each run, so a monthly job wanders.
  The zone keeps 06:00 at 06:00 across a daylight-saving boundary.

  Due schedules are noticed on the housekeeping pass, hourly by default, so a run starts
  within that window of its time rather than on the minute. Lower
  `maintenance_interval_seconds` for a tighter one; it is the cadence for every periodic
  task, so a floor of a minute applies.

  ```json
  "schedule": {
    "enabled": true,
    "mode": "cron",
    "cron": "0 6 1 * *",
    "timezone": "America/Phoenix"
  }
  ```

  `on_data_change` is often the better answer than either: it fires when there is new data
  rather than when the clock says there might be. This is the same shape as a model's
  retraining policy: a real data-to-notebook trigger with no external orchestrator.

## The environment

A notebook declares the packages its kernel needs, and the environment is provisioned
before the kernel starts: the declarative-spec model, not a mutable running kernel you
`pip install` into and hope. There are three ways to shape it:

- **The environment panel** (the packages button in the toolbar) edits the notebook's
  dependency list, one spec per line, e.g. `scikit-learn==1.5.2`. Saving resolves the
  spec to a fully-pinned **lock** with `uv` and restarts the kernel, so the environment is
  reproducible and every notebook that shares a dependency set starts from the same cached
  environment.
- **`%pip install <pkg>`** in a cell is the interactive escape hatch. It is not run as
  code: the package is added to the notebook's declared dependencies, the environment
  re-locks, and the kernel restarts, so a package added on the fly is captured in the
  reproducible spec rather than lost on the next session. (Put installs in an early cell,
  since the restart clears the namespace, the same discipline Jupyter and Databricks
  ask for.)
- **Base environments** are admin-curated package baselines (Settings → *Notebook
  environments*) a notebook builds on. A notebook picks a base environment and layers its
  own packages above it: a trusted, cached floor, the way Databricks serverless base
  environments and Hex custom images work.

The pinned lock is what a scheduled or parameterized rerun provisions from, so a run next
month resolves to the same versions it did today. **Re-lock** in the panel re-resolves the
spec when you want to pick up newer versions deliberately.

## The chat agent operates notebooks too

Everything above is drivable by the chat, not just a human in the UI. Ask the assistant to
"build a notebook that profiles this dataset and run it" and it uses the same surface you
do: it creates the notebook, runs it, reads the outputs back to fix any errors, and can
promote a cell to a certified derivation or schedule the notebook to rerun. Whatever it
builds shows up in the Notebooks list for you to open, edit, and take over. The division of
labor mirrors a senior data scientist delegating to an analyst: the agent goes and does the
work on the real platform; the governance (the oracle gate on a promoted cell, the training
gate on a registered model) holds regardless of who, or what, did the building.

## Interchange

Notebooks are the standard `nbformat` v4.5 format under the hood, so one authored here
opens in Jupyter, VS Code, or nbviewer unchanged. **Import** an `.ipynb` to bring existing
work in; **Download** exports the current notebook, outputs and all. Cell ids, the four
output shapes, and MIME bundles round-trip exactly.
