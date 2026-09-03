"""The scoring statements, built as strings and returned with their parameters.

Pure: every function takes values and returns ``(sql, params)``, so the ranking can be
read, diffed and unit-tested without a database. The relation API cannot carry bound
parameters -- ruling out both the query terms and the query embedding -- and has no
``WITH``, so this builds strings like
:mod:`elbi_core.compute.duckdb_backend`. Only values are bound; identifiers
never are, and the constants are interpolated because they are floats from
:mod:`elbi_core.text` rather than anything a caller supplies.

The BM25F expression mirrors :meth:`elbi_core.retrieval.Bm25Retriever.search`
statement for statement, including the parts that look like mistakes and are not: the
saturation divides by ``k1 + wtf`` with no ``k1 + 1`` numerator, and term frequency is
summed across fields *before* saturating. Both are monotone-equivalent variants, and a
generated corpus asserts the two rankers order it identically -- *for an entity that is
one document*, which is the whole claim and is measured rather than assumed:

    unchunked   order differs   0.0%   top-1 differs   0.0%
    chunked     order differs 100.0%   top-1 differs  60.7%

A chunked entity cannot agree and is not meant to: scoring an artifact by its best
chunk, each length-normalized on its own, is a different function from scoring the whole
row as one bag, and that difference is what chunking is *for*. Counting document
frequency per artifact rather than per chunk was tried, and measured no better (61.3%).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from elbi_core.retrieval import DEFAULT_MIN_SIMILARITY
from elbi_core.text import BM25_B, BM25_K1

from .schema import BROWSE_FIELDS, FIELD_WEIGHTS

#: SQL for "no restriction" -- single-user mode, where there is one trusted caller.
UNRESTRICTED = "TRUE"

#: Columns a caller may constrain results by, mapped to how the value is compared. The
#: allow-list is the whole defence: the column name reaches the statement as an
#: identifier, which cannot be bound, so it must never come from a caller unchecked.
#:
#: One entry per bucketable column :class:`~.schema.SearchDoc` actually carries.
#: ``created_at`` is deliberately absent: it is a range rather than a set of buckets,
#: and an ``IN`` list over timestamps is not what anyone means by filtering on a date.
FILTERABLE: dict[str, str] = {
    "entity_type": "VARCHAR",
    "verdict": "VARCHAR",
    "status": "VARCHAR",
    "incident_state": "VARCHAR",
}


def filter_sql(
    filters: Mapping[str, Sequence[Any]] | None,
    *,
    skip: str | None = None,
) -> tuple[str, list[Any]]:
    """A predicate restricting results to the values a caller asked for.

    Applied inside the ranking statement rather than to its output, and that placement
    is the point rather than an optimization. Filtering afterwards means the candidate
    window is spent on documents about to be discarded, so narrowing to a rare type can
    return nothing at all while matches exist. Bound here, the window is spent on
    documents that survive.

    ``skip`` omits one column, for counting a facet's own buckets: the count beside
    "sound" has to say how many there would be if you chose it, so every filter *except*
    that one applies.
    """
    if not filters:
        return UNRESTRICTED, []
    clauses: list[str] = []
    params: list[Any] = []
    for column, values in sorted(filters.items()):
        if column == skip or not values:
            continue
        if column not in FILTERABLE:
            raise ValueError(f"unknown filter {column!r}")
        clauses.append(f"d.{column} IN (SELECT UNNEST(?::{FILTERABLE[column]}[]))")
        params.append(list(values))
    if not clauses:
        return UNRESTRICTED, []
    return "(" + " AND ".join(clauses) + ")", params


def _weight_case() -> str:
    """A CASE mapping each field to its BM25F weight."""
    branches = " ".join(
        f"WHEN '{field}' THEN {float(weight)}"
        for field, weight in sorted(FIELD_WEIGHTS.items())
    )
    return f"CASE t.field {branches} ELSE 1.0 END"


def bm25_sql(
    terms: Sequence[str],
    *,
    n_docs: int,
    limit: int,
    where_sql: str = UNRESTRICTED,
    where_params: Sequence[Any] = (),
) -> tuple[str, list[Any]]:
    """Rank documents by BM25F over the index, best first.

    ``UNRESTRICTED`` restricts which documents may be reached at all; ``UNRESTRICTED``
    restricts which may be matched on their body. A document reachable only at
    ``browse`` matches on its title alone, because a body term that matched would tell
    the caller the word is in a body they may not read.

    ``where_sql`` is the caller's own filtering -- a type, a verdict -- kept separate
    from the two above because they are not the same kind of rule. Authorization decides
    what may be returned at all; this decides what was asked for. They meet here only
    because narrowing before the candidate cut is what makes both correct.

    All default to unrestricted, which is single-user mode with no facets chosen.
    """
    if not terms or n_docs <= 0:
        return "SELECT NULL AS doc_id, 0.0 AS score WHERE FALSE", []

    avg_len = f"(s.total_len / {float(n_docs)})"
    norm = f"(1 - {float(BM25_B)} + {float(BM25_B)} * f.len / {avg_len})"
    idf = f"ln(1 + ({float(n_docs)} - ts.df + 0.5) / (ts.df + 0.5))"

    sql = f"""
        WITH q(term) AS (SELECT UNNEST(?::VARCHAR[])),
        wtf AS (
            SELECT t.doc_id, t.term,
                   SUM(({_weight_case()}) * t.tf / {norm}) AS wtf
            FROM search_term t
            JOIN q ON q.term = t.term
            JOIN search_doc d ON d.doc_id = t.doc_id
            JOIN search_doc_field f
              ON f.doc_id = t.doc_id AND f.field = t.field
            JOIN search_field_stats s ON s.field = t.field
            WHERE s.total_len > 0
              AND ({UNRESTRICTED})
              AND (t.field IN ({_browse_fields_sql()}) OR ({UNRESTRICTED}))
              AND ({where_sql})
            GROUP BY t.doc_id, t.term
        ),
        scored AS (
            SELECT w.doc_id, SUM({idf} * w.wtf / ({float(BM25_K1)} + w.wtf)) AS score
            FROM wtf w
            JOIN search_term_stats ts ON ts.term = w.term
            GROUP BY w.doc_id
        )
        SELECT doc_id, score FROM scored
        WHERE score > 0
        ORDER BY score DESC, doc_id
        LIMIT ?
    """
    params: list[Any] = [
        list(terms),
        *where_params,
        limit,
    ]
    return sql, params


def dense_sql(
    vector: Sequence[float],
    dim: int,
    *,
    limit: int,
    where_sql: str = UNRESTRICTED,
    where_params: Sequence[Any] = (),
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
) -> tuple[str, list[Any]]:
    """Rank documents by cosine similarity to a query vector, best first.

    Written as ``ORDER BY array_cosine_distance(...) LIMIT n`` because that is the shape
    ``vss`` recognises: the HNSW index is only consulted for a distance ordering with a
    limit, and the same query returns the same answer by full scan when the optimizer
    declines. Distance rather than similarity for the same reason: they order
    inversely, and only one of them has an index.

    The predicates stay in the ``WHERE``, so authorization is applied *during*
    retrieval however the plan is chosen. Where they narrow enough that the index
    cannot be used, the exact scan runs instead -- the correct answer either way.

    Restricted to what the caller may read rather than merely browse: a vector is
    derived from the body text, so ranking on it discloses the body.

    The floor is applied outside the ordered subquery rather than beside the predicates,
    which would put a non-indexable expression in the way of the index. It selects the
    same rows: taking the best ``limit`` and then dropping those below a floor gives the
    same set as taking the best ``limit`` of those above it, because the floor and the
    ordering are the same quantity.
    """
    sql = f"""
        SELECT doc_id, sim FROM (
            SELECT d.doc_id,
                   1 - array_cosine_distance(d.vec, ?::FLOAT[{dim}]) AS sim
            FROM search_doc d
            WHERE d.vec IS NOT NULL
              AND ({UNRESTRICTED})
              AND ({UNRESTRICTED})
              AND ({where_sql})
            ORDER BY array_cosine_distance(d.vec, ?::FLOAT[{dim}])
            LIMIT ?
        )
        WHERE sim > {float(min_similarity)}
        ORDER BY sim DESC, doc_id
    """
    # The query vector is bound twice, in the projection and in the ordering: DuckDB's
    # binder will not let the ordering reference the projection's alias here.
    query = list(vector)
    params: list[Any] = [
        query,
        *where_params,
        query,
        limit,
    ]
    return sql, params


def facet_counts_sql(
    column: str,
    *,
    terms: Sequence[str] = (),
    vector: Sequence[float] | None = None,
    dim: int = 0,
    min_similarity: float = DEFAULT_MIN_SIMILARITY,
    where_sql: str = UNRESTRICTED,
    where_params: Sequence[Any] = (),
) -> tuple[str, list[Any]]:
    """Count documents per value of one facet, over the authorized subset only.

    Counted from the same predicate the results come from, so a total can never describe
    rows the caller cannot see. ``column`` is checked against the known facets rather
    than quoted, since an unknown one is a programming error and not a value.

    The caller's other filters narrow these counts, but not this facet's own -- the
    number beside "sound" must say how many results choosing it would give, and applying
    the current verdict to its own counts would collapse every bucket but the chosen one
    to zero. Building that exclusion is the caller's job; see :func:`filter_sql`.

    ``terms`` narrows to the documents a query could match, for the same reason: "how
    many results choosing it would give" is a question about *these* results. Counted
    over the whole corpus instead, a facet advertises buckets that return nothing when
    clicked. Empty ``terms`` is no query, where every reachable document is a candidate.

    Counted in *entities*, not documents, because that is the unit a result is in: a
    chunked derivation is several documents and one hit, so counting rows would put a
    number beside a facet that no page of results could ever reach.
    """
    if column not in FILTERABLE:
        raise ValueError(f"unknown facet {column!r}")
    # Narrowed by the *same* field tier the ranking uses. Counting a browse-only
    # document because a term matched its body tells the caller that word is in a body
    # they may not read -- the inference the two-tier split exists to prevent, arriving
    # through the count instead of the result list.
    #
    # Both halves of the ranking, not the lexical one alone: ``search`` fuses BM25 with
    # dense retrieval, so a document sharing no word with the query still reaches the
    # results. Counted from postings only, such a result appeared in the page and in no
    # facet, and the counts stopped describing the set they sit beside.
    lexical = (
        "d.doc_id IN ("
        "  SELECT t.doc_id FROM search_term t"
        "  WHERE t.term IN (SELECT UNNEST(?::VARCHAR[]))"
        f"   AND (t.field IN ({_browse_fields_sql()}) OR ({UNRESTRICTED}))"
        ")"
        if terms
        else ""
    )
    dense = (
        "d.vec IS NOT NULL"
        f" AND ({UNRESTRICTED})"
        f" AND 1 - array_cosine_distance(d.vec, ?::FLOAT[{dim}])"
        f"   > {float(min_similarity)}"
        if vector is not None and dim
        else ""
    )
    clauses = [half for half in (lexical, dense) if half]
    matching = f"AND ({' OR '.join(clauses)})" if clauses else ""
    sql = f"""
        SELECT d.{column} AS value,
               COUNT(DISTINCT d.entity_type || ':' || d.entity_id) AS count
        FROM search_doc d
        WHERE TRUE
          AND ({UNRESTRICTED}) AND ({where_sql}) AND d.{column} IS NOT NULL
          {matching}
        GROUP BY d.{column}
        ORDER BY count DESC, value
    """
    params: list[Any] = [*where_params]
    if lexical:
        params += [
            sorted(set(terms)),
        ]
    if dense:
        params += [list(vector or ())]
    return sql, params


def _browse_fields_sql() -> str:
    """The fields a browse-level caller may be matched against, as a SQL list."""
    return ", ".join(f"'{field}'" for field in sorted(BROWSE_FIELDS))
