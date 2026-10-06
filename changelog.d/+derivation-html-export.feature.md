A derivation's output can be downloaded as a page. **Evidence → Export → Output (.html)** on a
derivation, or `GET /api/exports/derivations/{name}/html`, writes the finding and output as
the derivation page shows them — markdown rendered, tables included — as one
self-contained, read-only HTML file anyone can open. The JSON record moves to
**Export → Record (.json)**. The page allows no script and makes no request.
