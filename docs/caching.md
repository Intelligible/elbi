# Caching & staleness

Derivations are cached and recomputed incrementally, and served with
stale-while-revalidate semantics (RFC 5861). This is fully local and open source,
with no managed backend required.

## How staleness is detected

Each derivation has a **data version**:

```
data_version = hash(code, parameters, input versions)
```

- **File inputs** are fingerprinted by content hash (cheap; no parse).
- **SQL inputs** are fingerprinted by a `freshness` probe (e.g. `max(updated_at)`)
  when declared, else by hashing the result set.
- **Derivation inputs** contribute their *output version*, composed Merkle-style.

If a cached result exists for the current data version (and is within any TTL),
it is reused: no recompute. Change an input, and the data version changes, so the
derivation recomputes exactly when it needs to.

### Early cutoff

A derivation exposes an **output version** (a hash of its output) to its
consumers. So if a parent recomputes but produces the *same* output, its children
are **not** invalidated; a step that filters out an unchanged column won't
retrain a model downstream of it.

## Per-derivation cache policy

```python
from elbi_core import cache, derivation, serve


@derivation(serve=serve.json(), cache=cache.auto(ttl=3600))
def daily_rollup(ctx): ...


@derivation(serve=serve.text(), cache=cache.never())  # reads wall-clock / an API
def live_status(ctx): ...
```

| Builder | Behavior |
| --- | --- |
| `cache.auto()` (default) | Content-addressed caching. Reuse while code, params, and inputs are unchanged. |
| `cache.auto(ttl=N)` | …plus serve stale + refresh in the background after `N` seconds (bounded staleness for inputs you can't cheaply fingerprint). |
| `cache.auto(expire=M)` | A hard ceiling: never serve a value older than `M` seconds; the next read blocks for a fresh recompute. Bounds stale-if-error. Must be `>= ttl`. |
| `cache.auto(deterministic=False)` | Still cached and reused, but never drives early cutoff (for outputs whose bytes vary across runs, e.g. a trained model). |
| `cache.auto(code_version="v2")` | Pin the logic version explicitly (busts the cache on bump). The escape hatch for changes auto-detection can't see (see below). |
| `cache.never()` | Always recompute. For derivations that read untracked state. |

### How code changes are detected

`code_version` hashes the derivation's own logic (normalized AST, insensitive to
comments and reformatting) **and recurses into the user-defined helper functions
it calls** (and closures/nested functions), so editing a helper busts the cache.
Third-party and stdlib callables are treated as opaque (identified by name).

It cannot see everything, so document the blind spots and use `code_version="…"`
to defeat them:

- calls resolved by attribute (`obj.method()`) or passed as values (callbacks, registries);
- runtime-mutable module globals read inside the function;
- a **library upgrade** changing a dependency's behavior (an environment change, out of scope);
- comparisons are valid only within one Python minor version.

### Safety: signed cache blobs

Cached values are pickled. The store **HMAC-signs every blob and verifies the
signature before unpickling** (constant-time compare), so a tampered or foreign
blob is rejected, never executed. A local cache auto-generates a private key
(`0600`) under the cache dir; a shared or remote store must be given an injected
secret so all readers and writers agree.

Caching is sound for any derivation that is a **pure function of its declared
inputs and parameters**. A derivation that reads untracked state (wall-clock, a
random seed, an external API) must declare `cache.never()`.

## Internal (non-served) derivations

A derivation with **no serve contract** is *internal*: it is a node in the graph
that other derivations consume, but it is not exposed to agents over MCP. Use it
for intermediate steps (a cleaned table, a fitted model, an embedding index)
without exposing them to agents:

```python
@derivation(inputs={"sales": Dataset("sales")})  # no serve → internal
def cleaned_sales(ctx): ...


@derivation(inputs={"clean": cleaned_sales}, serve=serve.table())
def report(ctx): ...  # served; consumes the internal node
```

## Serving: stale-while-revalidate

`elbi mcp` (and `serve`) serve derivations with the ISR contract: an agent gets the
**last-good answer immediately**, and an out-of-date derivation refreshes in the
**background**. Concurrent refreshes are de-duplicated, and a failed refresh keeps
serving the last-good value (`stale-if-error`). So an expensive derivation never
makes an agent wait once it has been computed at least once.

Three lifetime windows: within **`ttl`** the cached value is served as-is; past
`ttl` it is served stale while refreshing in
the background; past **`expire`** it is a hard miss: the next read blocks for a
fresh recompute and never serves something older. `expire` is what stops a
perpetually-failing refresh from serving stale forever.

## The cache CLI

```bash
elbi cache status     # where the cache lives + how many entries
elbi cache clear      # clear everything
elbi cache clear --tag <tag>   # invalidate only tagged entries
elbi cache gc         # reclaim orphaned blobs (mark-and-sweep)
elbi cache gc --max-age-days 30   # also evict entries older than 30 days
```

The cache lives on disk under `.elbi/cache` (an action cache plus a
content-addressable blob store) and is gitignored.

### Garbage collection

Each cached result is tagged with its derivation's name, so removing a derivation
(`elbi cache clear --tag <name>`, or deleting an agent-authored one) purges
its results. `cache gc` is a mark-and-sweep over the two layers: it keeps every
blob an action-cache record references and reclaims the rest, so orphans left by
deletes, evictions, or an interrupted write are collected. A grace window spares
just-written blobs, so a concurrent write (blob first, record next) is never
reclaimed mid-flight. `--max-age-days` also evicts entries past an age, a
backstop against unbounded growth in a long-running autonomous loop.

## Local vs. platform

Everything above runs locally and open-source. The one thing a single local
runner can't do is **coordinate a cache across multiple instances**: that, and
push-based (CDC) SQL invalidation, are deliberately out of scope. The
cache backend is a pluggable interface (`CacheStore`), so the platform can supply a
shared store without changing any derivation code.
