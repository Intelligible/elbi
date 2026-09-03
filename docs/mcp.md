# Serving over MCP

MCP is the one interface, in dev and in production. `elbi mcp` (or
`elbi serve`, which serves the same MCP endpoint alongside a chat UI)
is simply that interface served on your machine.

## What gets exposed

For each derivation, the local server registers:

- a **resource** at `elbi://derivation/<name>`, reading current context
  (the `read` capability); and
- a **tool** `run_<name>`, computing and returning the served output (the `run`
  capability).

Each call builds a fresh runner, so the agent always sees a current computation
against your local data bindings.

## Shaping results for the model

LLM accuracy drops on large inline tables, so a `table` tool does not dump every
row into the model's context. It returns:

- a **preview** as text: the rows that fit the `maxCells` budget (rows x columns,
  a portable proxy for a token budget), with a note of the total when trimmed;
- the full rows in MCP **`structuredContent`**, at zero token cost to the model and
  available to a capable host for rendering or processing; and
- a **`resource_link`** to the derivation's resource, for an out-of-band read of
  the full result.

The resource (`read`) still returns the complete rendering. Tune the budget with
`serve.table(max_cells=…)`; other formats return plain text unchanged.

## The audit trail

The serving layer carries an **`AuditSink`** seam, off by default so a local server stays
zero-config. It records one `AuditEvent` per invocation -- the derivation, the
content-addressed version that produced the answer, the params, and the outcome -- with
keys aligned to the OpenTelemetry MCP semantic conventions.

`NullAuditSink` (the default) records nothing. `JsonlAuditSink` appends a durable
one-line-per-event trail. `ChainedJsonlAuditSink` adds tamper-evidence: each line hashes
the one before it, so editing, reordering or dropping a record breaks the chain. Verify
it with `elbi certificate chain`; see
[Verification certificates](./certificates.md#the-audit-log).

A served, certified derivation also carries its signed
[verification certificate](./certificates.md) alongside its result: the tool call's
`structuredContent.certificate` and the resource's `_meta`, so any client can trust and
export what it received without a separate fetch.

## Connecting a client

`elbi mcp` serves streamable HTTP at `http://localhost:7878/mcp` by
default (`elbi serve` serves the same endpoint at
`http://localhost:7700/mcp`, alongside a chat UI). Point any MCP client at it:

- **MCP Inspector**: `npx @modelcontextprotocol/inspector`, then connect to the
  URL.
- **Claude Desktop / Cursor / your own agent**: add the URL as an MCP server.

## Protocol revisions

Both the 2026-07-28 revision and the earlier handshake revision are served, with
nothing to configure on either side. A client announces which it speaks with the
`MCP-Protocol-Version` header; one that sends no header is served the handshake
revision, so an older client keeps working unchanged.

The 2026-07-28 revision has no `initialize` handshake and no session header: each
request carries its own protocol version and capabilities, and a client reads what the
server supports from `server/discover`. Nothing routes on per-connection state, so a
deployment can answer any request from any replica, with neither sticky routing nor a
shared session store. `elbi mcp` owns its socket and still keeps sessions for
handshake-era clients; the endpoint mounted by `elbi serve` does not.

**Tasks** is deliberately absent: a long tool reports through
`run_code(background=true)` with `job_status` and `cancel_job` instead.

**`ttlMs` / `cacheScope`** are left at the conservative defaults the SDK emits on a
resource read, `ttlMs: 0` and `cacheScope: "private"`, which tell a client to cache
nothing. The revision carries them on resource and list results only, and a
derivation's freshness is content-addressed rather than clock-based, so there is no
duration to publish in their place (see
[Caching](./caching.md#how-staleness-is-detected)).

W3C trace context is honoured. A call carries `traceparent` in its own `_meta` rather
than in an HTTP header, so it survives a transport that is not HTTP, and the server
continues that trace instead of starting a new one: a tool call is a span inside the
caller's trace, not an unattributed gap in the middle of it. The spans leave this
process only when the deployment configures an OTLP endpoint.

The client features the revision deprecates are unused here, so a client needs neither
roots nor sampling to connect.

## Auth

Elbi runs as a single-user application: your machine, your data, no account. There is no
token to issue and no identity to check, and `elbi mcp` serves your own configured data
with nothing to manage. Per the MCP authorization spec, local transport should pull
identity from the environment rather than running a browser flow. Doing nothing is
how this obeys that.

Serving the endpoint to anyone but yourself is a different problem, and this runner does
not try to solve it. Put it behind something that authenticates, or reach it over an SSH
tunnel.
