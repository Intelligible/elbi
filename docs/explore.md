# Explore: a SQL workbench and profiler

Most of a data scientist's day is spent before any model exists: pulling data, joining it,
looking at distributions, checking what is missing. The Explore surface is where you do
that. Pick a source, write SQL against it, and read the result as a grid, a chart, or a
column profile. It is the fastest path from a question to a first look at the answer.

Explore is deliberately **ungoverned**. A person querying their own data is trusted, the
same way they would be in any database client or BI tool, so nothing here is routed through
the verification oracle. Exploration is not a derivation and carries no verdict. The single
bridge into the governed world is [**Promote to derivation**](#promote-to-a-derivation),
and it is a step you choose, never a gate you must pass.

## Sources

A query runs against one source at a time:

- **Bound datasets**: the datasets bound in `elbi.yaml`, queried in-process by
  [DuckDB](https://duckdb.org). Each dataset is a table you can name in `FROM` and join
  across; a CSV, Parquet, or JSON file is scanned in place. When every source is in memory,
  external file and URL access is disabled for the query, so it reaches only your data.
- **A registered [data source](data-sources.md)**: a Postgres, MySQL, SQLite, or SQL
  Server database, queried read-only through the same `load_sql` path derivations use. The
  query runs at the source; the schema browser reflects its tables and views.

The schema browser on the left lists the tables and columns available for the chosen
source. Double-click a table, or click a column, to drop its name into the editor. The
editor autocompletes those real table and column names as you type.

## Drafting SQL from a question

If you would rather describe what you want, type it in plain English in the bar above the
editor ("average price by number of bedrooms, most expensive first") and press **Draft
SQL**. The question and the current schema go to the configured model, which returns a
query into the editor for you to read, edit, and run. This is a convenience, not a
governed step: the draft is a starting point, and only a promoted query is ever certified.

## Running a query

Write SQL in the editor and run it with the **Run** button or `⌘↵` / `Ctrl+↵`. Results
come back capped (the default is 1,000 rows) and are shown three ways:

- **Results**: a sortable grid. Click a column header to sort. A banner notes when the
  result was truncated to the cap, so a clipped result is never mistaken for a whole one.
- **Chart**: pick a mark (bar, line, point, area) and the x and y columns to draw the
  result with [Vega-Lite](https://vega.github.io/vega-lite/). Field types are inferred
  from the data.
- **Profile**: per-column statistics computed by the same profiler the
  [data-contract](data-contracts.md) engine uses: completeness, distinct count, inferred
  type, numeric range, and the most common values. This is the one-click version of the
  exploratory summary you would otherwise write by hand.

## Saving a query

**Save** keeps a query under a name so you can reload it later. A saved query is a personal
exploration artifact: a name and its SQL, scoped to you when authentication is enabled. It
is not versioned, cached, or certified, because it is not meant to be a deliverable.

## Promote to a derivation

When an exploratory query turns out to be worth keeping, **Promote** authors it as a
certified [derivation](derivations.md). Promotion generates a derivation that runs the query
over its referenced datasets and puts it through the project's
[authoring loop](authoring.md), meaning a sandboxed run, a reproducibility check, and the oracle
and contract gates when a claim or contract is declared, exactly the path the chat and
[notebooks](notebooks.md) use. The result is a governed, cached, versioned artifact that can
be served to agents, trained on, or built into a [dashboard](dashboards.md), not an
ungoverned snippet.

Because promotion is where a result becomes something others rely on, that is where
verification belongs. Exploration stays free.

A query against an **external data source** is promotable too, but by a different route.
A certified derivation normally runs in the sandbox, which by design has no database
access, so an external-source promotion instead becomes a *trusted, human-origin*
derivation that reads the source in-process through the same read-only `load_sql` path
the runner already uses. It is persisted and rebuilt into the registry on restart, so it
behaves like any other certified derivation. Because it reads external state the runner
cannot version, its result is not cached.

## The SDK primitive

Under the workbench is a small SDK function, usable on its own:

```python
from elbi_core import query_datasets

result = query_datasets(
    {"sales": sales_rows, "regions": region_rows},
    "SELECT r.name, sum(s.amount) AS total "
    "FROM sales s JOIN regions r ON r.id = s.region_id "
    "GROUP BY r.name ORDER BY total DESC",
)
result.columns  # ["name", "total"]
result.rows  # [{"name": ..., "total": ...}, ...]
```

Each source is a `list[dict]`, an Arrow table, or a path to a data file, registered as a
DuckDB view of that name. It needs the `duckdb` extra (`pip install "elbi[duckdb]"`),
which the web app installs by default.
