Fixed an agent calling a derivation over MCP, through `elbi mcp` or `elbi serve`,
getting only a generic failure when the derivation refused the call. The refusal's own
message now reaches the agent, and any other failure still carries no detail.
