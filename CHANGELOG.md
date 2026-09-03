# Changelog

All notable changes to `elbi` are documented here. The format is driven by
[towncrier](https://towncrier.readthedocs.io/); add a news fragment under
`changelog.d/` with every user-facing change. This project adheres to
[Semantic Versioning](https://semver.org/).

<!-- towncrier release notes start -->

## 0.1.0 (unreleased)

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
