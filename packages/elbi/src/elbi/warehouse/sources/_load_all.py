"""Import every connector module so each registers with ``SourceRegistry``.

Kept out of ``__init__`` so importing the package doesn't drag every vendor SDK; the
registry imports this lazily on first use. Add a connector by importing it here, one
line per source. ``__all__``
re-exports them so the side-effect imports read as intentional (not unused).
"""

from . import (
    csv_file,
    custom,
    dynamodb,
    elasticsearch,
    google_sheets,
    hubspot,
    mongodb,
    object_store,
    posthog,
    saas_rest,
    salesforce,
    shopify,
    sql_database,
    stripe,
    work_rest,
    zendesk,
)

__all__ = [
    "csv_file",
    "custom",
    "dynamodb",
    "elasticsearch",
    "google_sheets",
    "hubspot",
    "mongodb",
    "object_store",
    "posthog",
    "saas_rest",
    "salesforce",
    "shopify",
    "sql_database",
    "stripe",
    "work_rest",
    "zendesk",
]
