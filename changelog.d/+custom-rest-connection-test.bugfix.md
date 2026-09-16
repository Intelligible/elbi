Testing a Custom REST source now fetches a record instead of only parsing the manifest.
A manifest with no `paginator` is structurally valid, so it saved cleanly and dlt then
guessed a paginator — for an endpoint answering in a single response the guess chases a
next page that never arrives, and the sync sat in `pending` with no error and no
timeout. The connection test runs the real resource through `rest_api` behind a
deadline, and a hang now reports which resource stalled and that an explicit paginator
is the likely fix.
