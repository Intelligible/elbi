Added an HTML snapshot of a derivation page. **Evidence → Export → Page (.html)**, or
`GET /api/exports/derivations/{name}/html`, downloads everything the derivation page
shows (question, verdict, data hash, assumptions, finding, output, verification checks,
result history, claim and source) as one self-contained page that opens offline in any
browser. The JSON record is now under **Export → Record (.json)**.
