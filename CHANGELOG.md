# Changelog

All notable changes to `elbi` are documented here. The format is driven by
[towncrier](https://towncrier.readthedocs.io/); add a news fragment under
`changelog.d/` with every user-facing change. This project adheres to
[Semantic Versioning](https://semver.org/).

<!-- towncrier release notes start -->

## 0.2.0 (2026-10-14)

### Features

- A dashboard's grid is twice as fine vertically, so a vertical resize moves in 30px
  steps instead of 60px. **The unit of `gridPos.h` changes with it:** the same height is
  twice as many rows, so a tile written as `h: 3` is now `h: 6`. A dashboard authored
  against the old unit renders at about half its height until its `h` and `y` values are
  doubled.
- A running app's derivation cache can now be invalidated over HTTP, by tag or by
  derivation name, through `POST /api/cache/invalidate`; a derivation the caller cannot
  reach answers 404. `elbi cache clear` gained `--derivation`, and with `--url` or
  `-t <target>` it invalidates a deployment's cache rather than the local one, so busting
  a server's cache no longer needs a shell on the host. Each invalidation is recorded in
  the audit log.
- A warehouse source can be edited in place, from the source page. Previously the only
  `PATCH` handled was the sync cadence, so fixing a typo in a manifest meant deleting the
  source and rebuilding it, re-entering every secret and destroying the tables it had
  produced on the way out. An **Edit** button now opens the connection form pre-filled with
  the stored config. A credential left blank keeps the stored secret, including the
  multi-line ones (an SSH private key or a Google Cloud key file), so correcting a query
  does not cost credentials that were already working. The new config is tested
  before anything is saved, and a resource that disappears from it is disabled rather than
  deleted, because its table holds rows the edit did not ask to destroy.
- A warehouse source's page now shows what its last sync produced, for the source and for
  each table, instead of whether a sync job is running. A sync that succeeded but landed no
  rows, or one where a table failed, used to read as `idle`; it now shows **Synced, but no
  rows** or **Failed** with the error. Each status, column and schedule on the page now
  explains itself on hover, with a link to the documentation.
- Added a **PostHog** connector, in beta, for PostHog Cloud (US or EU) and self-hosted
  deployments. It syncs a project's persons, cohorts, feature flags, insights,
  experiments, actions, annotations and surveys, each as a full refresh. Raw events are
  not included: read them through the Custom REST source with a HogQL query, or through
  PostHog's batch export to a bucket.
- Added a `components` serve format. A derivation can now return a list of
  OpenReasoningComponents (ORC) statements, natural-language facts about the data that
  each carry their own evidence, relations and provenance, instead of a table. Every
  served component is stamped with the derivation's name and version, and a new
  `search_components` MCP tool finds components by meaning across every served
  derivation.
- Added a `key` option to monitors over a derivation, naming the rows that breach. Every
  alert from such a monitor now lists those rows, and a row that starts breaching while the
  monitor is already alerting raises `metric.breach_widened`, emailed by default. Without a
  key, a second failing row only moved the total and was never announced.
- Added an HTML snapshot of a derivation page. **Evidence → Export → Page (.html)**, or
  `GET /api/exports/derivations/{name}/html`, downloads everything the derivation page
  shows (question, verdict, data hash, assumptions, finding, output, verification checks,
  result history, claim and source) as one self-contained page that opens offline in any
  browser. The JSON record is now under **Export → Record (.json)**.
- Controls, dropdowns and dialogs across the web app now share one visual language instead
  of each view styling its own. Buttons, form fields, empty states and icon-only actions
  use the same components everywhere, so spacing, color and sizing are consistent from
  page to page. Several controls also carry clearer accessible names and labels for screen
  readers. For contributors: a set of ESLint rules plus a `npm test` conformance check now
  catch raw elements, off-palette colors and arbitrary font sizes, radii and colors before they land.
- Dashboard tiles are editable from the dashboard itself. Each tile has a menu with
  **Edit…** and **Delete**, and the editor changes a tile's title, a text tile's
  markdown, the derivation a bound tile reads, a metric's column, aggregate and number
  format, and the tile's size. The derivation and column fields are pickers listing what
  a tile may bind and the columns that derivation actually returns, and a value no longer
  in the list stays selected and is called out rather than quietly cleared. A save the
  server refuses keeps the editor open with the reason, and deleting asks first.
- Dashboards can be exported as a shareable page. `elbi snapshot <dashboard>`, or
  **Export → Shareable page** on the dashboard, writes one self-contained, read-only
  HTML file: every page of the published dashboard with its current values, sortable
  tables, charts with tooltips, and the derivation or shared metric behind each figure.
  It carries no version history, needs no access to the app, and loads nothing from the
  network when opened. Each data widget can also download its rows as CSV.
- The data warehouse's sources list now shows each source's last sync result, the same
  status as on the source's own page. A source whose sync landed no rows used to show as
  synced in the list; it now shows **Synced, but no rows**, and a source with every table
  turned off shows **No tables enabled**. Each status has a help icon that explains it.
- The tile editor has a **JSON** tab holding the widget exactly as the dashboard spec
  stores it, so a chart's `viz`, a binding's `params` and a tile's `interactions` are
  editable without leaving the tile. It and **Edit spec** are syntax highlighted, and
  check what is typed against the dashboard spec's schema, so an unknown widget type, a
  missing `gridPos` or a tile wider than the grid is underlined where it is, before Save.
  A **Lineage** tab shows where a bound tile's numbers come from: the derivation that
  computed them, and the derivations and warehouse tables that one reads.
- The web app's sidebar has a Docs link, in a new Help section below Operations, that
  opens the documentation at docs.elbi.ai in a new tab.
- `WAREHOUSE_JSON_NESTED_LISTS=1` stores any column holding a list of structs as JSON text
  rather than as typed columns. Some APIs return collections whose element shape varies
  between records (Stripe's invoice line items differ depending on whether a line came
  from a subscription or an invoice item), and Delta cannot evolve a list whose element
  struct diverges while also gaining a list-typed field, so the sync fails with
  "Unsupported CAST from Struct(...) to Struct(...)". Stored as text the sync cannot break
  on shape and nothing is discarded, at the cost of column-level access into that field.
  Off by default; scalars, plain structs and lists of scalars are never encoded. The flag only
  decides new columns: a column the table already has keeps its stored form, so toggling
  it never breaks an incremental sync and takes effect on an existing table at its next
  full refresh.
- ⌘Z undoes a dashboard edit and ⇧⌘Z redoes it, one step per drag, resize, tile save or
  delete. The history lasts for the session only, and the dashboard's saved versions
  remain the durable record. Inside a text field or an open dialog, the keystroke is left
  to the browser.

### Bug Fixes

- **Open in notebook** on a model page works for a model with no champion. It asked the
  server for the training script without naming a version, so the server resolved
  `@champion`, which a model whose versions are all inconclusive does not have. The
  request 404'd, nothing was done with the rejection, and the button appeared dead. It
  now opens the champion's script where there is one and the newest version's otherwise,
  says why when a version has no script, and is not shown at all for a model with no
  versions.
- A claim declared on a derivation in `derivations/*.py` was missing from the derivation's
  page and from `GET /api/derivations/{name}`. It now appears in both, marked "Not checked
  by the oracle." when the derivation has no verdict. The `serve` field of a derivation's
  details and exported record now has one shape for every derivation: its serve settings as
  the spec defines them, plus `deps` when there are any.
- A connector sync no longer fails on a page that carries nothing in a field. Each batch's
  schema is inferred from the JSON just received, so a field that was empty on that page
  arrives as Arrow `null` or as a struct with no fields, and Delta can store neither,
  failing the sync with "Invalid data type for Delta Lake: Null". A Stripe account whose
  customers have no `description` and no `metadata` could not sync any table at all. Such
  fields are now dropped, recursively, so nested cases are handled as well as top-level
  ones and a page that does carry a real value still adds the field with its true type,
  in either order, with earlier rows reading back as null.
- A dashboard's last-edited time now reads in the viewer's own timezone. It was sent
  without its UTC offset, so the dashboard and its exported pages showed the UTC time
  labelled as local, which could put an edit hours after the snapshot that captured it.
- A monitor whose source cannot be read no longer fails silently. A failed check opens an
  incident and raises a `metric.source_failed` alert through the same in-app, email,
  webhook and audit path an anomaly uses, once per run of failures; the next readable
  value closes it as a recovery or as a new anomaly. A metric query that fails with its
  own error type also no longer stops the scheduled pass over later monitors.
- Dashboard tiles resize properly. A tile can be resized from any edge or corner rather
  than only an invisible bottom-right corner, and its handles are visible on hover. A
  chart or table grows with its tile instead of keeping the height it first rendered at,
  which had left the new space empty; a table fills the tile and scrolls.
- Fixed **Open in notebook** giving a notebook that failed on its first line when the
  derivation used imports, constants or helpers from elsewhere in its file or project.
  A cell ending on a derivation now shows the result it returns in production and can be
  run again, and running one cell first runs the cells it depends on. Fixed dollar
  amounts in notebook markdown rendering as math.
- Fixed a dashboard page failing to load when one of its tiles showed a metric that had
  been deleted or never defined. That tile now shows the error and the rest of the page
  loads.
- Fixed a half-written table after a warehouse sync failed part-way through. A full
  refresh now keeps the previous rows until the new ones have all landed, and an
  incremental sync adds its rows all at once or not at all. Fixed a source left at
  `syncing` by a crash or a redeploy never syncing again: it is now marked failed about
  half an hour after the run stopped, and syncs again on its usual schedule. Asking for a
  sync while one is running is now refused with `409 Conflict` instead of starting a
  second run beside it.
- Fixed testing a data source connection reporting the driver's error unmasked. A driver
  that quoted its connection details when failing would have shown the saved password.
  The error still names the host and the cause, now with the password masked.
- Testing a Custom REST source now fetches a record instead of only parsing the manifest.
  A manifest with no `paginator` is structurally valid, so it saved cleanly and dlt then
  guessed a paginator: for an endpoint answering in a single response the guess chases a
  next page that never arrives, and the sync sat in `pending` with no error and no
  timeout. The connection test runs the real resource through `rest_api` behind a
  deadline, and a hang now reports which resource stalled and that an explicit paginator
  is the likely fix. The deadline is on the walk itself, so a manifest that pages forever
  stops paging when the test gives up rather than requesting the next page for as long as
  the server runs.
- The app's own MLflow tracking store now migrates to the installed MLflow's schema on
  start, after the file is copied to a `.bak` beside it, so upgrading to a release with a
  newer MLflow no longer needs a manual `mlflow db upgrade` before models work again. A
  failed migration is logged with the backup's path and the app starts anyway. A tracking
  server's database named by `MLFLOW_TRACKING_URI` is left alone.
- Three faults kept a dashboard from showing what its derivations say. A `text` widget
  could not bind a derivation, so the documented escape hatch (a bespoke visual belongs
  in a derivation returning markdown) did not work, and narrative tiles had to hardcode
  figures that go stale. A text widget did not render markdown at all, emitting its
  source as preformatted text. And a tile's rows crossed the camelCase API boundary as
  field names, so a derivation's `mrr_usd` reached the browser as `mrrUsd` and a widget's
  `viz.field`, written as the derivation writes it, matched nothing: the tile rendered a
  dash and a chart encoding silently found no data.

  A text widget now takes `content` or a `bind`, not both, and renders either as
  markdown. (Markdown there still reads `$…$` as inline TeX, so a dollar amount in a
  bound derivation renders as an integrand; that is fixed separately.) A derivation artifact's `value` is opaque to the casing boundary, like `rows`
  already was.

  Separately, a page that omitted `columns` rendered on a 12-column grid while the spec,
  the schema and `Page.to_manifest` all default to 24, so every page taking the default
  laid out at half width, and each widget spanned twice the intended fraction.

### Documentation

- The README and documentation home page now describe elbi as an empirical memory for the
  agent you already use: what an agent learns from data can be saved as a derivation that
  later sessions and other agents build on. Serving your own computations, the local
  data-science environment, and OSI semantic-model inputs are still covered.
- The documentation sidebar is now grouped: getting started, derivations, the GUI,
  automation and monitoring, and configuration and deployment. Several pages have
  clearer sidebar names, such as "Connect your AI agent (MCP)" and "Writing derivations
  manually".

### Deprecations and Removals

- A dashboard `metric` tile now binds a shared metric, the same definition the Metrics
  page and the agent use, instead of aggregating a derivation's rows in the browser
  through `viz.field` / `viz.agg`. A figure is defined once and reads the same on every
  surface: the tile takes its value and format from the metric, and lineage now links
  each metric to the dashboards that display it. A metric binding's filters may
  reference a dashboard variable as `$name`, so a metric tile still follows the page's
  filters. This is Dashboard Spec 2.0: a Spec 2.0 dashboard whose `metric` tile binds a
  derivation is refused, with a message naming the fix. A dashboard in the earlier
  format, whether saved by an earlier release, pushed with `elbi sync` or brought in by
  `elbi import`, still opens, with each such tile shown as a note saying what it showed
  and which metric to define; see `docs/upgrading.md`.


## 0.1.0 (2026-09-03)

The first public release. Rather than list every change that led here, this is what
Elbi does on the day it arrives.

**Derivations.** Write a Python function with declared inputs and an output format, and
Elbi serves it over the Model Context Protocol (the 2026-07-28 revision and the
handshake-era one) as a versioned, cached tool. Results are
content-addressed on the code, the parameters and the inputs, so a derivation recomputes
exactly when one of those changes and a parent that recomputes to the same output does
not invalidate its children.

**Verified answers.** An agent can propose a derivation, but it cannot report a
directional finding that has not passed the verification oracle: thirty-odd deterministic
gates that select themselves from the shape of the data rather than from a guess at
intent, and grade a certified conclusion on whether it survives perturbation.

**The surfaces around it.** A reactive notebook, a SQL workbench, dashboards bound to
certified derivations, a metrics layer that reads and writes the OSI standard, a feature
store with point-in-time joins, anomaly monitors, lineage and search across every
artifact, asset orchestration, and an ML lifecycle backed by MLflow.

**Finding things.** One search across every artifact, behind ⌘K and behind the agent's
own `search_derivations`. It reads a DuckDB corpus rather than rebuilding one per query,
and a listener on the store keeps it current: save something and it is findable in about
a second, delete it and it leaves results at once. Warehouse column names are indexed
too, so "which datasets have a `customer_id`" is a question with an answer.

**Nothing is deleted by accident.** Notebooks, folders, dashboards, saved queries,
metrics, feature views and derivations are soft-deleted: they leave every list and detail
view, stay recoverable from **Settings → Trash**, and are purged on a retention window
you set. Deleting forever is a separate, explicit act.

**Knowing what happened.** An inbox carries monitor alerts, failed or slow runs,
finished and failed trainings, and drift or expectation checks, with a switch per event
type and per channel. Where SMTP is configured the same events can go to email.

**Your data, portable.** A document per derivation, dashboard or metric, and a
whole-workspace archive pairing each artifact's definition with its evidence: a
derivation's claim, verdict, attestation, certificate and result history; a dashboard's
spec and its current values. Credentials are never read into either.

**Data in.** Forty-five warehouse connectors. Sixteen databases, including PostgreSQL,
MySQL, SQL Server, Oracle, Snowflake, BigQuery, Redshift, ClickHouse, MongoDB, DynamoDB
and Elasticsearch, and the hosted services that speak their protocols (Supabase, Neon,
CockroachDB and PlanetScale) each with the host, port and TLS already right. Object
storage on S3, Google Cloud Storage, Azure Blob and SFTP, matching a glob inside a
bucket and reading CSV, JSON and Parquet by extension, with every row carrying the file
it came from. Google Sheets, and the applications a team runs its work in: Stripe,
Shopify, Salesforce, HubSpot, Zendesk, GitHub, Jira, Notion, Slack, Sentry, Typeform,
Intercom and a dozen more. Anything missing can be described to the custom REST source
as a manifest.

**Databases behind a firewall.** Every connector addressed by a host and port can
connect through an SSH bastion, so a database with no public endpoint needs no
allowlist. Authenticates with the credential you configured and never the host's own
ssh-agent, and will pin the bastion's host key if you give it a fingerprint.

**Upgrading.** `elbi update` reports whether a newer release exists and prints the
command for however Elbi was installed. It reaches the network only when you run it:
nothing checks at startup or on a timer.

**The spec.** The Open Derivation Spec, its JSON Schemas, and a conformance suite, so a
derivation manifest means the same thing to any implementation.
