# Releasing

`elbi-core`, `elbi-agent`, `elbi-cli` and `elbi` ship together, and
[`release.yml`](.github/workflows/release.yml) handles that part. This page is what happens
before it.

## Why the order matters

`release.yml` fires on `release: published`. By then the tag exists and the distributions are
already building, so anything that edits the repository at that point lands after the release
it was meant to describe. The changelog is built *before* the tag, in an ordinary pull
request.

## Steps

1. Bump `__version__` in `packages/elbi-core/src/elbi_core/__init__.py`. towncrier reads the
   version from there.

2. Build the changelog:

   ```bash
   uv run towncrier build --version <X.Y.Z> --yes
   ```

   This writes a new section into `CHANGELOG.md` from everything in
   [`changelog.d/`](changelog.d), then deletes the fragments. Preview it first with
   `--draft`, which writes nothing and deletes nothing.

3. Open a pull request with the version bump and the changelog, and merge it. Label it
   `skip-news`: cutting a release is not a change a user notices separately from its
   contents.

4. Tag the merge commit and publish a GitHub release. That fires `release.yml`, which builds
   every package, resolves the built wheels the way a user will, and publishes each one to
   PyPI over Trusted Publishing.

## After

The four uploads run as parallel jobs, so a release can land part-way: `elbi` on the index
requiring an `elbi-core` that never uploaded. Re-running the workflow fixes it, because every
upload sets `skip-existing`, but nothing detects it for you.

Confirm from a clean environment:

```bash
uv venv /tmp/elbi-release-check
uv pip install --python /tmp/elbi-release-check elbi==<X.Y.Z>
/tmp/elbi-release-check/bin/elbi --version
```

Installing `elbi` resolves the other three from PyPI, so this fails if any of the four is
missing.
