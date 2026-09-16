# Data sources

A dataset is bound to a concrete source in the local-only `elbi.dev.yaml`
(gitignored; never commit credentials). A binding is either a **file** or a
**SQL source**.

## Files

```yaml
data:
  sales: ./fixtures/sales.csv        # .csv, .json, .jsonl, .parquet
  events: env(EVENTS_PATH)           # env(VAR) → a path
```

Parquet requires the data extra: `pip install "elbi[data]"`.

### Uploading a file

To bring a local CSV or Parquet file into a running deployment, upload it rather than
referencing a path the server cannot see. Use the app's data UI, or a terminal:

```bash
elbi upload ./sales.csv --as-table sales   # staged, registered, and loaded
elbi upload ./sales.csv                    # staged only, for several sources
```

The file streams to the deployment, which stores it in the warehouse (a local
directory, or the object store behind it: S3, MinIO, GCS, Azure) and returns a path.
You send the bytes and the server writes them, so this works whether or not your laptop
can reach the object store. Databricks and Superset take the same server-side approach.

`--as-table NAME` then points a source of that name at the uploaded path and loads it, so
the file arrives as a queryable table in one command and prints the table it made. Without
it the upload is staged and nothing reads it yet: give the path to a CSV/Parquet source,
or pick the file in the app. Staging and loading being separable is what every warehouse
CLI does, and it matters when several sources read one file.

A CSV is read in the encoding and separator it was written in, both detected
automatically, so a spreadsheet export (a Windows-1252 file, or one separated by
semicolons) lands as columns rather than one column of garbage.

Uploads are
capped (`WAREHOUSE_UPLOAD_MAX_BYTES`, default 2 GiB); for larger data, land it in the
bucket yourself and give a source its `s3://`/`gs://` URL, which the connector reads
directly. Behind an ingress, allow a body that large there too; the chart's
`ingress.maxBodySize` does this for the nginx ingress.

## SQL databases and warehouses

```yaml
data:
  orders:
    connection: env(ACME_DB_URL)     # a SQLAlchemy URL, from the environment
    query: "SELECT * FROM dbo.orders WHERE placed_at > '2024-01-01'"
    # or, instead of query:
    # table: dbo.orders
    max_rows: 10000                  # row cap (default 10000)
    timeout: 30                      # statement timeout, seconds (default 30)
```

Install the core plus your engine's driver extra:

| Engine | Extra | Connection URL |
| --- | --- | --- |
| PostgreSQL | `elbi[postgres]` | `postgresql+psycopg://user:pwd@host/db` |
| MySQL / MariaDB | `elbi[mysql]` | `mysql+pymysql://user:pwd@host/db` |
| SQL Server | `elbi[mssql]` | `mssql+pyodbc://user:pwd@host/db?driver=ODBC+Driver+18+for+SQL+Server` |
| Snowflake | `elbi[snowflake]` | `snowflake://user:pwd@account/db/schema?warehouse=WH` |
| BigQuery | `elbi[bigquery]` | `bigquery://project/dataset` |
| SQLite | _(none, stdlib)_ | `sqlite:///path/to.db` |
| Supabase | `elbi[postgres]` | `postgresql+psycopg://postgres.<ref>:pwd@aws-0-<region>.pooler.supabase.com:5432/postgres` |
| Amazon Redshift | `elbi[postgres]` | `postgresql+psycopg://user:pwd@cluster.<id>.<region>.redshift.amazonaws.com:5439/dev` |

> **SQL Server note:** `pyodbc` also needs an OS-level ODBC driver
> (`msodbcsql18` + a driver manager such as unixODBC).

> **Supabase note:** use the **session pooler** host (port 5432) with the
> username `postgres.<project-ref>`. Two other hosts will not work here. The
> direct host `db.<ref>.supabase.co` resolves to IPv6 only unless the project
> has the IPv4 add-on, and the transaction pooler (port 6543) hands each
> statement to whichever backend is free, which breaks the driver's prepared
> statements; a URL cannot turn those off, so use session mode instead.

### How it behaves

The query result flows into the same rows your derivations already read, so no
derivation code changes when you switch a dataset from a file to SQL.

The connector follows read-only analytics best practice:

- **Governance lives at the source.** Connect with a **SELECT-only** database
  role; the framework does not re-implement access control, it relies on your
  database's RBAC and your credentials.
- **Bounded results.** `max_rows` caps what is pulled (protects the warehouse and
  the agent's context window). A statement `timeout` is applied per dialect.
- **Read-only execution** where the dialect supports it (e.g. a read-only
  transaction on PostgreSQL), short-lived connections (`NullPool`), and
  **bound parameters** for any supplied values (identifiers are allow-listed).
- **Credentials never live in git**: always `env(VAR)`.

### Pushing filters down (today vs. later)

Today the query loads a bounded result set and your derivation filters in Python
(e.g. an agent-supplied zipcode). For very large tables, narrow the `query`
itself (add a `WHERE`/`LIMIT`) so the database does the work. Parameter pushdown
(wiring a derivation's `params` straight into the SQL `WHERE`) is a planned
enhancement.

## Managed connectors

The sections above cover data you declare in `elbi.yaml` and the app reads where
it already lives. The other route is a **managed connector**: you set one up once
under **Data warehouse → New source**, and the app pulls the data in on a
schedule and lands it as Delta Lake tables you can query like any other.

Pick that route when the data is behind an API, when it is spread over many
files, or when you want a copy that does not change under you between runs.

There are three shapes.

**Databases** are read over their own protocol and offered table by table:
PostgreSQL, MySQL, SQL Server, SQLite, Oracle, Snowflake, BigQuery, Redshift,
ClickHouse, MongoDB, DynamoDB and Elasticsearch, plus the hosted PostgreSQL and
MySQL services (Supabase, Neon, CockroachDB and PlanetScale), which get their
own entry so the host, port and TLS are right without you working them out.

**File storage** covers Amazon S3 (and any S3-compatible store, via the endpoint
field), Google Cloud Storage, Azure Blob Storage, SFTP and Google Sheets, along
with a single local or remote CSV or Parquet file. The object stores match a glob
inside a bucket, group what they find into a table per folder, and read CSV,
JSON and Parquet according to each file's own extension. Every row carries the
file it came from and when that file last changed, and the second of those is
the incremental cursor: a later sync reads only the files that have changed.

**Applications** are read over their APIs, each with a token you create in that
product: Stripe, HubSpot, Salesforce, Shopify, Zendesk, Chargebee, Mailchimp,
Klaviyo, SendGrid, Braze, Pipedrive, Front, Vercel, Airtable, Mixpanel, PostHog,
GitHub, Jira, Notion, Slack, Sentry, Typeform and Intercom. If yours is not there, the
**Custom REST source** takes a manifest describing any JSON API.

### Incremental sync

A connector offers an incremental cursor only where one would actually help. A
SQL source offers its timestamp and id columns. An object store offers the file's
modification time. MongoDB offers a field only if it *leads* an index, because
filtering on anything else makes the server read the whole collection, which is slower
than the full refresh it was meant to replace. DynamoDB offers nothing, because a
filtered scan reads and costs exactly what an unfiltered one does.

Where no cursor is offered, the sync is a full refresh. That is a property of the
source, not a gap in the connector.

## Reaching a database behind a firewall

A managed connector runs wherever this app runs, which is your own machine or your
own infrastructure. That has one consequence worth stating plainly: **there is no
Elbi IP address to add to an allowlist.** Hosted pipelines publish a list of
theirs because they connect from their servers to your database; here the
connection comes from yours.

So the address to allow is your own, and which one that is depends on where the
app runs:

| Where it runs | What the database sees |
| --- | --- |
| Your laptop | Your current ISP or VPN address, which usually changes |
| A VM with a public address | That machine's address |
| A private subnet | The NAT gateway's address, which is stable |
| Kubernetes | The node addresses, unless a NAT or egress gateway fronts them |

For anything scheduled, put the app somewhere with a stable egress address and
allow that one. A laptop's address changes, and the sync will start failing on a
day nothing was changed.

### SSH tunnel

When the database has no public endpoint at all, connect through a host that can
already see it. Every connector addressed by a host and port has a **Connect
through an SSH tunnel** switch; turn it on, give it the host, and the connection
is made through there instead, so the database only ever has to accept
connections from inside your own network.

That covers PostgreSQL, MySQL, SQL Server, Oracle, ClickHouse, MongoDB,
Elasticsearch, and the hosted variants: Supabase, Neon, CockroachDB, PlanetScale
and Redshift. It does not apply to SQLite (a local file), Snowflake or BigQuery
(APIs addressed by account, not by host), or to the object stores, which are
public endpoints reached over HTTPS.

Authenticate with a password or a private key. Both are stored encrypted, and
neither the SSH agent nor any key in the home directory of the account running the
app is ever consulted; the tunnel uses what you configured, or it does not open.

Set the **host key fingerprint** if you can. Left empty, whichever key the bastion
offers first is trusted, which detects nothing. `ssh-keyscan your-bastion |
ssh-keygen -lf -` prints the value to paste, and a host offering anything else is
then refused before a credential is sent to it.

The bastion needs `AllowTcpForwarding yes` for the account you use. If it is off,
the connection test says so by name rather than reporting a puzzling database
error.

Worth doing on the bastion side, none of which this app can do for you:

- Give the tunnel its own account, and restrict where it may forward with
  `PermitOpen your-database:5432`. `AllowTcpForwarding yes` on its own lets that
  account reach anything the bastion can, which is more than it needs.
- Prefer a key over a password, and prefer Ed25519. Both are accepted here; only
  one of them is a secret that can be guessed.
- Give it a shell of `/usr/sbin/nologin` and `PermitTTY no`. The account exists to
  forward a port, not to log in.

On this side, the connection negotiates AEAD ciphers and SHA-2 encrypt-then-MAC
only. Paramiko's own defaults still include 3DES, the CBC modes and HMAC-MD5, and
rank AES-GCM last; those are all excluded here, so a bastion that can offer
nothing better will fail to connect rather than fall back to it.

Two MongoDB connection strings cannot be tunnelled: a `mongodb+srv://` URI, and
one listing several servers. Both describe a set of servers rather than an
address, and the driver would resolve the set and then connect to the members
directly, going around the tunnel. Name the single server you want to read.

### The other options need nothing from this app

A private link, a VPC peering, or a mesh network such as Tailscale or WireGuard
all work by making the database reachable at an address. Set one of those up and
point the connector at that address; it is an ordinary connection at that point,
with no tunnel configuration involved.
