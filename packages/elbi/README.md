# elbi

The app: a chat surface, a web UI, and an MCP endpoint over your derivations.

Install this if you want the whole thing. `elbi-core` is the SDK on its own and
`elbi-cli` is the command line on its own; this package brings both, plus the server
and the interface.

```bash
uv tool install elbi
elbi init my-project && cd my-project
elbi serve
```

It builds an ASGI app that:

- answers a question by **authoring a derivation** and streaming the run over
  Server-Sent Events, so the browser sees each tool the model calls and a result it
  could not fabricate;
- **mounts the project's MCP server** at `/mcp`, so one process and one port serve both
  the interface and any agent that connects;
- **serves the built single-page app**, which calls those same endpoints, so the same
  code runs as one local process or as an API behind a separately hosted front end.

Models are reached through [LiteLLM](https://docs.litellm.ai), so any provider it
supports works, and usage and cost are recorded the same way whichever you pick.

Built for one person: your machine, your data, no accounts and no sign-in.

Licensed under [Apache-2.0](https://github.com/Intelligible/elbi/blob/main/LICENSE).
