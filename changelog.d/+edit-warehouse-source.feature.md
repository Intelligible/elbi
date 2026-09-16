A warehouse source can be edited in place. `PATCH /api/warehouse/sources/{id}` already
existed but only changed the sync cadence, so fixing a typo in a manifest meant deleting
the source and rebuilding it — re-entering every secret, and destroying the tables it had
produced on the way out. It now accepts `config`, `name` and `description` too. A
password field left blank or omitted keeps the stored secret, so correcting a query does
not cost the credentials that were already working. An edited config is re-validated
before anything is saved, and a resource that disappears from the new config is disabled
rather than deleted, because its table holds rows the edit did not ask to destroy.
