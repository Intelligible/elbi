A running app's derivation cache can be invalidated over HTTP, by tag or by derivation
name, through `POST /api/cache/invalidate`. `elbi cache clear` gains `--derivation`, and
with `--url` or `-t <target>` it invalidates a deployment's cache rather than the local
one, so busting a server's cache no longer needs a shell on the host. Each invalidation
is recorded in the audit log.
