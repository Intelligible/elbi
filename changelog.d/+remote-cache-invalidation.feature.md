A running app's derivation cache can now be invalidated over HTTP, by tag or by
derivation name, through `POST /api/cache/invalidate`; a derivation the caller cannot
reach answers 404. `elbi cache clear` gained `--derivation`, and with `--url` or
`-t <target>` it invalidates a deployment's cache rather than the local one, so busting
a server's cache no longer needs a shell on the host. Each invalidation is recorded in
the audit log.
