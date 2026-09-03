# Compute engines

The profiler, the contract checker, and the verification oracle read a dataset's
rows. Held in memory as a list of dicts, that is fine for thousands of rows and
impossible for tens of millions. The **compute seam** lets those trusted, first-party
tools run over a pluggable engine instead: the same profile or contract that runs on
in-memory rows pushes down to DuckDB or Polars, scanning a larger-than-memory file
without loading it. One implementation of each tool scales from a CSV to a warehouse
extract, with no change to the tool.

This is a *trust-and-scale* seam, not the untrusted-code sandbox. The engine runs the
platform's own code (a profile query, a validity check), server-side, where the agent
cannot weaken it. It is distinct from the [execution sandbox](authoring.md#execution-isolation),
which isolates code an agent wrote.

## The idea

A `ComputeBackend` implements a small, fixed catalog of operations: structural queries
(columns, row count), a one-pass per-column profile, the located validity primitives a
data contract is checked in terms of, and a bounded materialization (a deterministic
sample, or a row export). Three backends implement it:

- **`RowsFrame`** over `list[dict]` rows: the dependency-free reference. It defines the
  exact semantics the others must match.
- **`DuckDBFrame`**: pushes every operation to SQL, queries an Arrow table zero-copy,
  and scans a Parquet/CSV/JSON file out of core (spilling to disk under a memory limit).
- **`PolarsFrame`**: the in-process engine, a lazy multi-threaded query graph with a
  streaming collect.

`backend_for(data)` selects one by the data's type. Detection inspects `sys.modules` and
never imports an engine to test for it, so importing the core pulls in no dataframe
engine. The engines are optional extras:

```bash
pip install "elbi[duckdb]"    # or [polars], or [engines] for both
```

## Parity is the contract

The load-bearing guarantee is that every backend returns *exactly* what the reference
returns: the same profile, the same located violations, the same deterministic sample.
A property-based test asserts this across backends over diverse data, so a backend that
diverges is caught. To get there the engines adopt the reference's string-first
semantics: values compare as text (matching `str(value)`), numbers parse with a
non-strict cast, a cell is missing when null or empty, and type conformance calls the
reference's own parser (registered as a UDF in DuckDB, an expression in Polars) so the
richer rules (`"5.0"` is a valid integer, the boolean spellings) match. Profiling is
exact, not sketch-based: on a single node, determinism matters more than the memory an
approximate cardinality would save.

## Engine-backed datasets

Bind a dataset and load it as an engine handle rather than rows:

```python
from elbi import profile_columns, suggest_contract, verify_contract

handle = bindings.load_backend("events", base_dir)  # scans events.parquet out of core
profile = profile_columns(handle)  # the aggregate runs in the engine
contract = suggest_contract(handle)  # thresholds from a pushed-down profile
report = verify_contract(handle, contract)  # checks run in the engine
```

`load_backend` returns a scanning handle when a compute engine is installed and the
binding is a columnar/text file; otherwise it materializes the rows into the reference
backend. It is for the first-party tools, which operate through the seam. A derivation's
`ctx.input(...).rows` still gets materialized rows, because arbitrary Python needs them.

## The oracle: reduce, then compute

A statistical gate (a bootstrap, a variance-inflation factor, a permutation test) cannot
run in SQL, so the oracle does not push its math down. Instead the seam *reduces* the
data in the engine (the scan, the aggregation, a bounded deterministic sample) and hands
the gate a small frame it computes the statistic over in Python. So a claim over a
50-million-row table becomes: aggregate and sample in the engine, then run the gate on a
bounded, reproducible sample. The sample is a pure function of its inputs (a hash of row
position, identical across engines), so a certified result stays reproducible.

## What we do not build

No general dataframe algebra or query IR: the operation catalog is a fixed, small set,
so a thin protocol beats an expression compiler, and the core depends on neither Ibis nor
Narwhals. Arrow is the interchange, but only behind the seam: the public contract stays
`list[dict]`, and crossings into and out of rows are few and explicit, because each is a
copy. No cross-engine plan format (Substrait): the platform targets its own one or two
engines directly.
