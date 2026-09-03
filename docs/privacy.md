# Privacy and telemetry

## Does this product phone home?

**No.** There is no usage reporting, no crash reporting, no licence check, and no
automatic version check. Nothing is sent to us, and there is nothing to opt out of
because there is nothing to opt into.

One command reaches the network on purpose: `elbi update` asks PyPI whether a newer
release exists. It runs only when you type it: never at startup, never on a timer,
never inside another command. There is nothing to opt out of. The
request is a GET for a public JSON file, carrying a User-Agent naming the tool and its
version, and nothing else. See [Upgrading](upgrading.md).

Third-party libraries that ship telemetry enabled by default have it disabled here, in code
rather than in a setting an operator has to find. There is a test that fails if a future
change reintroduces one.

## What does leave the network

Exactly three things, all of them yours, all of them to endpoints you configure:

| Destination | Why | Avoidable? |
| --- | --- | --- |
| **Your LLM endpoint** | The agent sends the schema of your data and the question being asked | Point `LLM_BASE_URL` at an internal gateway, or use Bedrock/Vertex in your own account |
| **Your data sources** | Syncing a source means connecting to it | Only the sources you register |
| **Your object storage and database** | Where the warehouse and the app's records live | Both are yours |

Nothing else. An air-gapped install needs no allowlist beyond those, so running with no outbound
network is a supported configuration rather than a best-effort one.

## What the LLM endpoint receives

Worth being specific, because "we send the schema" is easy to under-describe. The agent sends:

- **Table and column names, and column types** for the data it has been asked about.
- **Sample rows**, when a tool asks for them, so it can write code that matches the data's
  actual shape.
- **The question**, and the code it is iterating on, including error output from failed runs.

It does not send your warehouse wholesale, and derivation results stay in your database and
object storage. But sampled rows are real rows: if a column holds regulated data, a sample of
it reaches whatever endpoint you configured.

If that is more than a third-party provider should see, the fix is architectural rather than a
flag: run the model inside your own account (Bedrock, Vertex AI, Azure OpenAI) or behind your
own gateway; set `LLM_BASE_URL` to point at it.

## Are you a data processor under GDPR?

If you self-host, we have no access to your data, your instance, or its logs. There is no
telemetry channel and no support tunnel. On that basis we are not a data processor for your
deployment, and no data processing agreement with us is required for it.

That is a statement about the software, not legal advice, and it stops being true if you send
us something: a support bundle you generate
and choose to share, for instance, contains logs and configuration. Bundles redact secrets as
part of the spec rather than afterwards, but read one before you send it.

## What is recorded inside your deployment

Plenty, and all of it stays with you. The audit log records what ran, which derivations
were certified and by which gates, and every agent tool call with its arguments. That is
the compliance surface, and it is queryable at `GET /api/audit`.

`AUDIT_RETENTION_DAYS` bounds how long it is kept; `0` keeps it forever.

[Exports and data portability](exports.md) covers taking your work with you: a document
per derivation, dashboard or metric, and what those never contain.

## Honoring a right-to-erasure request

Two separate mechanisms exist, and only one of them satisfies a data-subject erasure
request. Deleting a notebook, dashboard, saved query, metric, feature view, or derivation
moves it to trash by default: recoverable, purged automatically after
`TRASH_RETENTION_DAYS`. That default protects against accidents, but relying on it for an
erasure request is a known compliance trap: the data is still in the database for the
whole retention window, and a fixed window is a deferred purge, not an on-request one.

The second mechanism is what an erasure request needs: pass `?permanent=true` on the
delete call, or choose "delete forever" from the trash view, and it is gone immediately
rather than on a timer. Erasure removes the row, its result history and attestations, its search index entry, and
any cached derivation artifacts keyed to it, and the audit log records that the erasure
happened without storing what was erased. What it cannot do is reach backups already
taken. Restoring a backup restores what it held, so a deletion has to be replayed against
any restore, and that boundary is worth stating to whoever is asking.
