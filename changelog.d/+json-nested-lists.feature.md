`WAREHOUSE_JSON_NESTED_LISTS=1` stores any column holding a list of structs as JSON text
rather than as typed columns. Some APIs return collections whose element shape varies
between records — Stripe's invoice line items differ depending on whether a line came
from a subscription or an invoice item — and Delta cannot evolve a list whose element
struct diverges while also gaining a list-typed field, so the sync fails with
"Unsupported CAST from Struct(...) to Struct(...)". Stored as text the sync cannot break
on shape and nothing is discarded, at the cost of column-level access into that field.
Off by default; scalars, plain structs and lists of scalars are never encoded.
