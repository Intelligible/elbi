A warehouse source can be edited in place, from the source page. Previously the only
`PATCH` handled was the sync cadence, so fixing a typo in a manifest meant deleting the
source and rebuilding it — re-entering every secret, and destroying the tables it had
produced on the way out. An **Edit** button now opens the connection form pre-filled with
the stored config. A credential left blank keeps the stored secret — including the
multi-line ones, an SSH private key or a Google Cloud key file — so correcting a query
does not cost credentials that were already working. The new config is tested
before anything is saved, and a resource that disappears from it is disabled rather than
deleted, because its table holds rows the edit did not ask to destroy.
