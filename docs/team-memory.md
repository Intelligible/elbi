# Shared team memory

`elbi-enterprise` is what makes this real, not something aspirational: single
sign-on, users and groups, and per-object grants -- down to one derivation -- over
one shared store, plus row and column policies and directory sync. It's a shipped,
licensed package (free for development, testing and evaluation; a subscription
for production use), not a request that waits on a roadmap. See
[Running Elbi for a team](../README.md#running-elbi-for-a-team).

## What changes with it installed

Without `elbi-enterprise`, Elbi runs for one person: your machine, your data, no
account. Every certified derivation, every signed certificate, every audited call
lives on *your* local `elbi mcp` server. A teammate asking the same question gets
their own server, their own cache, their own copy of whatever they had to certify
from scratch -- even if you certified the identical answer an hour ago.

With it installed and configured, the same store is shared, and every one of these
is scoped by its own grants (`ResourceType.DERIVATION` is a first-class grantable
kind there, so a derivation -- and everything served from it -- can be shared with
one person, one group, or the whole org):

- **Certificates** ([docs/certificates.md](certificates.md)) land where everyone
  with access to that derivation already looks, not a file passed by hand.
- **The audit log** ([docs/mcp.md](mcp.md#the-audit-trail)) becomes a team's record
  of what was asked and answered, not just one person's.
- **Lineage and the catalog** ([docs/lineage.md](lineage.md)) cover everything the
  team has certified, so "has anyone already answered this" is a search, not a
  Slack message.

## The one piece ahead of the access-control layer

`search_components` (this session's own addition to `elbi_core`) finds a relevant,
previously-computed statement by meaning, across every components-format
derivation on one server. `elbi-enterprise`'s grants are precise down to one
derivation -- share `default_risk_components` and everyone with access sees
everything it serves -- but there is not yet a finer-grained kind for one
component-statement *within* that derivation's output. Until there is, share the
derivation, not the individual finding. (Flagging this as the current boundary
rather than verifying it against elbi-enterprise's own test suite directly --
worth confirming with whoever owns that package before this is load-bearing.)

## Ask

Write to solutions@intelligible.ai.
