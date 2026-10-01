A claim declared on a derivation in `derivations/*.py` was missing from the derivation's
page and from `GET /api/derivations/{name}`. It now appears in both, marked "Not checked
by the oracle." when the derivation has no verdict. The `serve` field of a derivation's
details and exported record now has one shape for every derivation: its serve settings as
the spec defines them, plus `deps` when there are any.
