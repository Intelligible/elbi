Added a `verify_citations` MCP tool that checks an answer's numbers against the served
outputs it cites. Each number must round, at the precision it is written with, from a
served value (a percent may match a fraction), and each `name@hash` citation must name
the version the served value was computed under. A derivation the answer ran with params
took them in `params`. Every unsupported number and stale citation is named, so a
misquoted or outdated figure is caught before the answer is sent.
