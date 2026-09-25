# Data contracts

Most of a data scientist's time goes to cleaning, and a wrong clean looks right:
it runs, it produces rows, and the errors surface later downstream. A **data
contract** makes the quality bar explicit and machine-checkable, so a cleaning
step is certified the same way an analysis is.

A contract is the cleaning counterpart to a [claim](authoring.md#verification).
A claim asserts an inferential conclusion and the oracle checks it for
*soundness*; a contract asserts *data validity* (types, ranges, keys, referential
integrity, shape) and the checker verifies it against the rows. Both gate
certification: a derivation that declares a contract is served only if its output
satisfies it.

## Declaring a contract

Attach a `DataContract` to a derivation. On certification, the checker runs it
over the output; an output that breaks the contract is held, not served.

```python
from elbi_core import (
    Constraints,
    DataContract,
    FieldSpec,
    TableSpec,
    derivation,
    serve,
)

contract = DataContract(
    fields=(
        FieldSpec(
            "customer_id", "integer", Constraints(required=True, unique=True, minimum=1)
        ),
        FieldSpec(
            "email", "string", Constraints(required=True, pattern=r"^[^@]+@[^@]+$")
        ),
        FieldSpec("status", "string", Constraints(enum=("active", "churned", "trial"))),
    ),
    table=TableSpec(primary_key=("customer_id",), row_count_min=1),
)


@derivation(
    inputs={"raw": Dataset("customers")}, serve=serve.table(), contract=contract
)
def clean_customers(ctx):
    "Deduplicated, type-checked customer records."
    ...
```

A field declares a `type` (one of `string`, `integer`, `number`, `boolean`,
`date`, `datetime`) and optional `constraints`. Which constraints are legal
depends on the type: a numeric range only on numbers, length and pattern only on
strings. The contract validates against the Data Contract Spec at definition
time, so an ill-formed one fails at import, not at serve time.

## The check catalog

| Constraint | Applies to | Checks |
| --- | --- | --- |
| `required` | any field | the value is present |
| `unique` | any field | present values do not repeat |
| `minimum` / `maximum` | numeric | the value is within the inclusive range |
| `minLength` / `maxLength` | string | the string length is within bounds |
| `pattern` | string | the string matches a regular expression |
| `enum` | any field | the value is in a closed set |
| `type` | every field | the value parses as the declared type |

Table-level constraints live in `TableSpec`: `primary_key` (present and unique),
`unique_keys` (additional composite uniqueness), `foreign_keys` (referential
integrity into another resource), `row_count` bounds, and `columns_match` (the
columns are exactly the declared fields, a schema-drift guard).

A **foreign key** references another dataset or derivation by name, so a
referential-integrity edge is an edge in the derivation DAG. During certification
the checker resolves the referenced rows through the runner; if the reference
cannot be resolved, that clause is reported inconclusive rather than passed.

## Tolerance

Real data is rarely perfect, and a contract that fails on a single stray value is
often too brittle. Each field takes a `mostly` (a fraction in `[0, 1]`, default
`1.0`): the share of present values that must satisfy the field's constraints.
`mostly=0.99` certifies when at least 99% of present values are valid. Missing
values are judged only by `required`, never by the value constraints, so a null
never counts as an out-of-range number. Structural constraints (keys, referential
integrity, row count, schema shape) are strict and take no tolerance.

## The verdict

Checking a contract returns a three-valued verdict, the same vocabulary the
verification oracle uses:

- **sound**: every clause holds (within tolerance).
- **unsound**: a clause is broken, reported with the located, offending rows
  (`{clause, column, rowNumber, value, note}`), capped so the verdict stays small
  and deterministic on dirty data.
- **inconclusive**: a clause could not be evaluated (a declared column is absent,
  a foreign-key reference is unavailable, the table is empty). Inconclusive is
  never a silent pass; like unsound, it blocks certification.

A contract is a conjunction: all clauses must hold. `verify_contract(rows,
contract, references=...)` runs the checker directly in the SDK.

## Profiling and suggesting a contract

Writing a contract from scratch is tedious, so the profiler proposes one. It reads
a table once and reports, per column, its completeness, distinct count, inferred
type, numeric range, and most common values; the suggester turns that profile into
a contract with a fixed, deterministic rule set (a complete column is required, a
fully distinct one is unique, a non-negative numeric one gets a zero floor, a
low-cardinality string one gets an enum, the first complete-and-unique column
becomes the primary key).

The proposed thresholds are **confidence bounds, not sample rates**. A column seen
92% complete over a few hundred rows yields a looser completeness floor than the
same rate over millions, because the suggester uses the Wilson score lower bound
rather than pinning the observed rate. A contract proposed from a sample therefore
does not overfit it and false-alarm on the next batch. The suggestion is a starting
point to review and tighten, never a certified verdict.

```python
from elbi_core import profile_columns, suggest_contract, verify_contract

profile = profile_columns(rows)  # per-column data-quality summary
contract = suggest_contract(rows)  # a proposed DataContract to review
report = verify_contract(rows, contract)  # check it
```

## Over MCP

`elbi mcp` (and `serve`) expose the cleaning tools alongside the analysis ones:

- **`profile_dataset`**: per-column completeness, distinct count, inferred type,
  range, and top values. Read this before proposing a contract.
- **`suggest_contract`**: a proposed contract for a dataset, with confidence-bound
  tolerances. A starting point to edit, not a verdict.
- **`verify_contract`**: check a dataset against a contract and return the verdict
  with the offending rows. `references` maps a foreign key's target to the dataset
  that holds it.

To *certify* a cleaning step, attach the contract to a derivation and author it
with [`propose_derivation`](authoring.md#over-mcp): the derivation is gated on the
contract and is not served unless its output is valid, so a certified clean table
carries a verified quality bar.

## Engine independence

The check catalog runs over an engine-agnostic backend. Out of the box it operates
on a list of row dicts and needs no third-party dependency. The same contract, and
the same catalog, run against a dataframe or SQL engine by registering a backend
for that engine's type: the contract is data, not code, so it never changes. This
is the seam by which contract checking scales beyond in-memory rows without
rewriting a single constraint.
