"""A trim-to-target variance overlay, served fresh on every call.

House rule, stated plainly so it fits in one function: no single name should
carry more than ``TARGET_MAX_VARIANCE_SHARE`` of the book's variance. A name
over that line gets trimmed in proportion to how far over it is; nothing else
in the book moves. This is illustrative, not a production sizing model -- the
point of this example is that the *inputs* are current, not that the rule is
sophisticated.

Because this derivation's only inputs are ``cov_matrix`` (itself built fresh
from ``returns.csv``) and the ``positions`` dataset, every call reflects
whatever is in those files right now. There is no cached "yesterday" answer
sitting in a skill file waiting to go stale.
"""

from __future__ import annotations

import hashlib
import json
import math

from elbi_core import Artifact, Context, Dataset, derivation, serve

from .cov_matrix import cov_matrix

TRADING_DAYS_PER_YEAR = 252
TARGET_MAX_VARIANCE_SHARE = 0.28  # no single name carries more than ~a quarter of book variance


@derivation(
    inputs={"cov": cov_matrix, "positions": Dataset("positions")},
    serve=serve.json(),  # no title: the tool's text content is plain JSON to parse
)
def risk_overlay(ctx: Context) -> Artifact:
    """Recompute gross vol and any variance-share cuts from current data."""
    cov_artifact = ctx.input("cov")
    cov = cov_artifact.value
    symbols: list[str] = cov["symbols"]
    matrix: dict[str, dict[str, float]] = cov["cov"]

    weights = {row["symbol"]: float(row["weight"]) for row in ctx.input("positions").rows}

    # (Sigma w)_a for each name, then the book's variance w^T Sigma w.
    sigma_w = {a: sum(matrix[a][b] * weights.get(b, 0.0) for b in symbols) for a in symbols}
    portfolio_var = sum(weights.get(a, 0.0) * sigma_w[a] for a in symbols)
    gross_vol = math.sqrt(portfolio_var * TRADING_DAYS_PER_YEAR)

    contribution = {a: weights.get(a, 0.0) * sigma_w[a] / portfolio_var for a in symbols}

    cuts: dict[str, float] = {}
    for symbol, share in contribution.items():
        if share <= TARGET_MAX_VARIANCE_SHARE:
            continue
        over_target = (share - TARGET_MAX_VARIANCE_SHARE) / share
        cuts[symbol] = -round(weights[symbol] * over_target, 4)

    # A content tag naming exactly which covariance computation this answer
    # came from -- so "which numbers produced this cut" is never a guess.
    cov_hash = hashlib.sha256(
        json.dumps(matrix, sort_keys=True).encode()
    ).hexdigest()[:8]

    return Artifact.json(
        {
            "as_of": cov["as_of"],
            "gross_vol": round(gross_vol, 4),
            "cuts": cuts,
            "source": f"cov_matrix@{cov_hash}",
        }
    )
