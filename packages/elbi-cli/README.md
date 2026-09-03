# elbi-cli

The command-line interface for [elbi](https://github.com/Intelligible/elbi).

```bash
uv tool install elbi-cli
```

```
elbi init <name>     Scaffold a new project (offline).
elbi mcp             Serve your derivations over MCP on localhost.
elbi validate        Validate the project and its derivations against the spec.
```

Installing `elbi-cli` alone gives you `init`, `mcp`, and `validate` with no
heavier dependencies. For the chat UI too (`elbi serve`), install
`elbi` instead: `uv tool install elbi`.

Licensed under [Apache-2.0](https://github.com/Intelligible/elbi/blob/main/LICENSE).
