# The CLI

```
elbi init <name>     Scaffold a new project (offline).
elbi mcp             Serve derivations over MCP (the default way to run a project).
elbi serve           Serve the chat UI, and MCP, together on localhost.
elbi validate        Validate the project and derivations against the spec.
elbi cache status    Show the local cache location and entry count.
elbi cache clear     Clear the cache (optionally --tag <tag>).
elbi --version       Print the version.
```

## `init`

```bash
elbi init acme-data-context [--template standard|minimal] [--force]
```

Scaffolds a project and runs offline: nothing is sent anywhere.

## `mcp`

```bash
elbi mcp [--host 127.0.0.1] [--port 7878] [--path /mcp] [--directory .]
```

The default way to run a project: discovers derivations, resolves local data
bindings, and serves them over MCP -- no chat UI, no `elbi` app package needed.
Compute runs on your machine; the serve contract is applied per derivation. Point
any MCP client (Cursor, Claude Code, Claude Desktop, Inspector) at the printed URL;
see [Getting started](getting-started.md). Speaks streamable HTTP only -- a client
cannot spawn it over stdio, so leave the process running.

## `serve`

```bash
elbi serve [--host 127.0.0.1] [--port 7700] [--model <model>] [--directory .]
```

An alternate client: a chat UI plus the same MCP endpoint `mcp` serves, in one
process. Requires the `elbi` app package; if it is not installed, prints how to add
it. Reach for it to inspect the derivation graph or demo without a client of your
own; `mcp` is the default.

## `validate`

```bash
elbi validate [--directory .]
```

Loads the project, imports every derivation (which validates its manifest against
the spec), and warns about dataset inputs not declared in `elbi.yaml`.
Exits non-zero on any failure.

## `certificate`

```bash
elbi certificate issue <name> [--version <prefix>] [--issuer <id>] [-o <file>]
elbi certificate verify <file> [--public-key <key-or-.pub-file>]
elbi certificate chain [--path <audit.jsonl>]
elbi certificate public-key
```

Issues, verifies, and inspects signed [verification certificates](./certificates.md) and
the hash-chained audit log. `verify` and `chain` run fully offline, no project needed.


## `update`

```bash
elbi update
```

```bash
elbi update --apply    # run the upgrade instead of printing it
elbi update --pre      # consider pre-releases
elbi update --quiet    # silent when current, exit 1 when not
```

Asks PyPI whether a newer release exists, and prints the command that upgrades this
particular install: `uv tool upgrade`, `pipx upgrade`, `pip install --upgrade`, or the
compose command inside a container. Where the project publishes release notes, it links
them, so you can see what changed before deciding.

`--apply` runs the command, handing the process to the installer with `exec` so nothing
of Elbi is running while its files are replaced. It refuses a command that needs a shell
and prints it instead.

Other commands reach the network too. `sync`, `pull`, `plan`, `upload` and `run` all
talk to your running app. This is the only one that contacts anything outside your own
infrastructure, and it does so only when you type it: nothing checks at startup or on a
timer. `--quiet` prints nothing when you are current and exits non-zero when you are
not, for use in a script.

See [Upgrading](upgrading.md) for the whole picture.
