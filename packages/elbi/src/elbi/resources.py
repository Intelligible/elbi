"""The kinds of object the app stores, as one enum.

Search, notifications and the trash all need to say *what* an object is without
importing the module that owns it, and a typo in a free-text kind is a row nothing ever
matches. Naming them here keeps that unrepresentable and keeps those three from
depending on each other.
"""

from __future__ import annotations

from enum import Enum


class ResourceType(str, Enum):
    """An object kind.

    Subclasses ``str`` rather than using ``StrEnum``, which needs Python 3.11 while this
    package supports 3.10. One consequence to know: a member compares and stores as its
    value, but ``str(member)`` and ``f"{member}"`` both give
    ``"ResourceType.NOTEBOOK"``. Use ``.value`` where a string is wanted.
    """

    NOTEBOOK = "notebook"
    #: A folder in the notebook tree, which contains notebooks and further folders.
    FOLDER = "folder"
    DASHBOARD = "dashboard"
    DERIVATION = "derivation"
    METRIC = "metric"
    FEATURE_VIEW = "feature_view"
    SAVED_QUERY = "saved_query"
    #: One synced table in the warehouse, keyed by its qualified table name.
    WAREHOUSE_TABLE = "warehouse_table"
    #: A whole connection, and every table a sync has produced from it.
    WAREHOUSE_SOURCE = "warehouse_source"
