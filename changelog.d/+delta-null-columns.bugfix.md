A connector sync no longer fails on a page that carries nothing in a field. Each batch's
schema is inferred from the JSON just received, so a field that was empty on that page
arrives as Arrow `null` or as a struct with no fields, and Delta can store neither —
failing the sync with "Invalid data type for Delta Lake: Null". A Stripe account whose
customers have no `description` and no `metadata` could not sync any table at all. Such
fields are now dropped, recursively, so nested cases are handled as well as top-level
ones and a page that does carry a real value still adds the field with its true type,
in either order, with earlier rows reading back as null.
