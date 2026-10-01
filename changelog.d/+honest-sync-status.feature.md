A warehouse source's page now says what its last sync produced instead of showing the
job state. The headline reads `Syncing…`, `Failed` with the error, `Synced` with a row
count, `Never synced`, or a warning when a sync succeeded but a table got no rows, so a
connector that silently lands nothing no longer looks healthy. Each table carries its
own last result, and the page's terms (method, schedule, table names, row counts, last
synced) explain themselves on hover with a link to the new docs section.
