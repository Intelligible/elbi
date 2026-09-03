"""What platform search treats as a document, declared per entity.

This package is the mapping, not the index: :mod:`.specs` names which columns of which
entity become searchable text, how they are weighted, how they are chunked, and which
fields constrain results rather than matching them. The indexer that reads these, and
the query path that scores them, are separate tickets that implement against it.

``docs/search-documents.md`` is generated from these declarations, so the reviewed
document and the code the indexer reads cannot disagree.
"""

from __future__ import annotations

from .schema import (
    BROWSE_FIELDS,
    DESCRIPTION_BOOST,
    FIELD_BODY,
    FIELD_COLUMNS,
    FIELD_TITLE,
    FIELD_WEIGHTS,
    NAME_BOOST,
    SearchDoc,
)

__all__ = [
    "BROWSE_FIELDS",
    "DESCRIPTION_BOOST",
    "FIELD_BODY",
    "FIELD_COLUMNS",
    "FIELD_TITLE",
    "FIELD_WEIGHTS",
    "NAME_BOOST",
    "SearchDoc",
]
