A warehouse source's page now shows what its last sync produced, for the source and for
each table, instead of whether a sync job is running. A sync that succeeded but landed no
rows, or one where a table failed, used to read as `idle`; it now shows **Synced, but no
rows** or **Failed** with the error. Each status, column and schedule on the page now
explains itself on hover, with a link to the documentation.
