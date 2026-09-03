# Exports and data portability

Two rights sit behind every serious privacy regime: erasure (delete what is mine) and
**portability** (hand me what is mine, in a form I can use). This page is the second
one. It covers every `/api/exports/*` route, the `elbi export`/`import`
workspace archive, and the format contract a future version has to keep reading.

This is not a substitute for backups (`pg_dump` plus your object-store snapshots
restore a *deployment*; this restores *your work*), and it is not the tabular-egress
answer either: a BI tool reading certified results live, or a dataframe pulling them
into Python, is a warehouse connection, not an export.

## What already exists

Two routes predate this page and are unchanged by it:

| Route | Returns |
| --- | --- |
| `GET /api/conversations/{id}/export` | one chat, as a self-contained JSON document |
| `GET /api/notebooks/{id}/export` | one notebook, as an nbformat `.ipynb` |

## Per-object exports

Three JSON documents, each the same view the app's own detail page reads, plus its
history. Every one opens with `{"schema": "elbi.export/v1", "kind": ..., "created_at": ...}`.

| Route | Definition | Plus |
| --- | --- | --- |
| `GET /api/exports/derivations/{name}` | source, claim, attestation | full result history (see [result history](result-history.md)) and, if certified, the signed [certificate](certificates.md) |
| `GET /api/exports/dashboards/{id}` | the spec | saved versions and every page's current resolved values |
| `GET /api/exports/metrics/{name}` | the manifest | the full definition history |

Each holds the same read gate as its sibling detail route: an owned or project-global
derivation, a dashboard you can see, a metric you own, never a wider one. An
uncertified derivation still exports; its `certificate` is `null` rather than a 404,
because "give me everything you have" is a different question from "give me the
certificate."

## Where these are in the app

Every route above has an affordance, and which one you reach for is the question of who
the file is for:

| You want | Go to |
| --- | --- |
| the proof behind one claim | a derivation's page → **Export record** |
| what a dashboard showed, and how it is built | a dashboard's header → **Export record** |
| a metric's definition and how it changed | the metric's row → **Export record** |
| your own copy of your work | **Settings → Account → Your data** |
| to answer someone's request for their data | **Settings → Users** → *Export data* on their row |
| to move a whole workspace | `elbi export`, a CLI command, not a page |
| the semantic layer as an OSI manifest | Metrics → **Export OSI**, a different thing: the metric *definitions* in the interchange format, not one metric's record |

The Users action is recorded in the audit log against the admin who ran it. The workspace
archive has no button by design: it must run from a checkout (see below).

## The workspace archive

`elbi export` and `elbi import` move a whole project, both halves of
it, as one zip:

```
elbi export --out workspace.zip
elbi import workspace.zip
```

The archive is a [config_sync](analytics-as-code.md) repo with a `records/` tree
added:

```
manifest.json              {"schema": "elbi.export/v1", "surfaces": [...], "records": [...]}
elbi.yaml
derivations/*.py           copied verbatim from the checkout export ran in
metrics/ dashboards/ features/ monitors/ schedules/
workflows/ checks/ models/ queries/ notebooks/     -- written by config_sync.pull
certificates/*.json        -- signed, independently verifiable offline
records/
  derivations/<name>.json       the full per-derivation export document
  derivations/<name>.source.py  the recorded source, header-commented as evidence
  dashboards/<name>.json        definition + versions + current values
  metrics/<name>.json           manifest + version history
```

**`export` must run from the checkout that produced the deployment.** The app stores
only a repo derivation's decorated function, read back with `inspect.getsource`: a
function body, not a module, with no imports. That is enough to *record* as evidence
(`records/derivations/<name>.source.py`), but it will not `import` on its own. Real,
importable `derivations/*.py` files only exist in a real checkout, so this is a CLI
command rather than a server route. A server-only archive could never produce them. Running `export` with no `derivations/` folder present still produces a
complete archive; a warning names what will not be importable.

**`import` applies definitions; it never replays records.** It runs the same `sync`
engine a real checkout uses, for every surface except `certificates`. Certificates are
handled separately: each one is verified against the target instance's own signing
key (`POST /api/certificates/{name}`, the same gate `elbi sync` uses), and the
result is reported rather than written anywhere. Against a *different* instance that
result is always the same: the route verifies the signature against the target's own
public key before it asks whether anything is certified there, and a certificate signed
elsewhere never matches, so every certificate is reported as signed by another
instance. That is expected, not a failure, and not tampering. Only a re-import into the
instance that issued them can report "matches".

This is deliberate, not a shortcut. Writing `records/derivations/<name>.json`'s
result history into the target's database would make it serve a verdict its own
oracle never produced, signed under its own key: a certificate that says "certified"
without anything here having certified it. A record is evidence *that* something was
once verified; it is not, itself, verification.

Derivation source still has to reach the target the way it always does, baked into
an image or pulled by git.
`import` does not restart the app or trigger a reload.

## Round-tripping

`export` then `import` into a clean instance reproduces every definition exactly:
`elbi plan` against the target reports nothing to do, for every surface but
`certificates` (which a clean instance cannot yet match; see above). This is the same
guarantee [analytics-as-code](analytics-as-code.md) already gives `pull`/`sync`; the
archive adds evidence on top without weakening it.

## No secrets, ever

Every export route is built by allow-listing what to include, never by reading a
table and redacting afterward. A test plants a known secret (`Store.save_secret`) and
asserts it in none of the five export documents nor the CLI archive: the same
guarantee the [privacy page](privacy.md) describes for the rest of the
product.

## Format version

`elbi.export/v1` today. A major version bump means a shape change `import`
cannot degrade from gracefully, and it is refused by name (`archive schema
'elbi.export/vN' is not supported by this CLI`) rather than partially applied.
An addition within a major version is read by an older `import` as simply absent;
new fields are always additive, never a renamed or removed one.
