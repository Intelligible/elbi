The MCP server has a `reconcile_effect` tool: given an agent's estimate of an average
effect (and its interval, if any), it re-estimates the effect independently with
`elbi_core.verification.second_opinion` and returns the answer to report, combining the
two analyses at equal weight when the question asks for the effect itself, and only
checking when it names a method. An estimate outside every elbi model's range is
withheld until the agent confirms it.
