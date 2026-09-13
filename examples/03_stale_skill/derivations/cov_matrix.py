"""Sample covariance of daily returns, recomputed from whatever is in the CSV.

No ML dependency: a plain sample covariance over ``returns`` (long format:
``date, symbol, ret``). It exists to make one point, not to be a real risk
system: this derivation has no serve contract, so it is internal, and its
cache key is a content hash of ``returns.csv`` (see ``cache.auto()`` in
docs/caching.md). Append a row and the hash changes, so the next call
recomputes -- there is no "stale until someone edits a file" state to fall
out of sync with the data, because there is no frozen state at all.
"""

from __future__ import annotations

import statistics
from typing import Any

from elbi_core import Artifact, Context, Dataset, derivation


@derivation(inputs={"returns": Dataset("returns")})  # internal: feeds risk_overlay
def cov_matrix(ctx: Context) -> Artifact:
    """Sample covariance matrix over every symbol's series in ``returns``."""
    rows = ctx.input("returns").rows

    by_symbol: dict[str, dict[str, float]] = {}
    for row in rows:
        by_symbol.setdefault(row["symbol"], {})[row["date"]] = float(row["ret"])

    symbols = sorted(by_symbol)
    dates = sorted({d for series in by_symbol.values() for d in series})
    # Dense panel: every symbol has a value on every date present in the file.
    series = {s: [by_symbol[s][d] for d in dates] for s in symbols}
    means = {s: statistics.fmean(v) for s, v in series.items()}
    n = len(dates)

    cov: dict[str, dict[str, float]] = {a: {} for a in symbols}
    for a in symbols:
        for b in symbols:
            va, vb = series[a], series[b]
            cov[a][b] = sum(
                (va[i] - means[a]) * (vb[i] - means[b]) for i in range(n)
            ) / (n - 1)

    payload: dict[str, Any] = {
        "symbols": symbols,
        "cov": cov,
        "as_of": dates[-1],
        "n_obs": n,
    }
    return Artifact.opaque(payload)
