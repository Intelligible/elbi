Added a `components` serve format. A derivation can now return a list of
OpenReasoningComponents (ORC) statements, natural-language facts about the data that
each carry their own evidence, relations and provenance, instead of a table. Every
served component is stamped with the derivation's name and version, and a new
`search_components` MCP tool finds components by meaning across every served
derivation.
