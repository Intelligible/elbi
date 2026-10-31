"""reconcile_effect through the real MCP tool path: elbi's own estimate, combined with
the agent's, is what the tool hands back as the answer to report."""

from __future__ import annotations

import asyncio
import json
import random
from collections.abc import Iterator
from typing import Any

import pytest
from mcp.shared.exceptions import MCPError

from elbi_cli.mcp_server import build_server
from elbi_core import Registry, Runner
from elbi_core.data import Table

so = pytest.importorskip("elbi_core.verification.second_opinion")


@pytest.fixture(autouse=True, scope="module")
def _few_threads() -> Iterator[None]:
    """Cap native threads so parallel test workers do not oversubscribe the cores."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=2):
        yield


def _text(result: Any) -> str:
    blocks = getattr(result, "content", result)
    return "".join(getattr(b, "text", "") for b in blocks)


def _table(n: int = 160, seed: int = 0) -> Table:
    """Binary treatment confounded by z; true ATE 2 (estimates here run near 1-1.7)."""
    rng = random.Random(seed)
    rows = []
    for _ in range(n):
        z = rng.gauss(0, 1)
        t = 1.0 if rng.random() < (0.75 if z > 0 else 0.25) else 0.0
        rows.append({"t": t, "y": 2 * t + 3 * z + rng.gauss(0, 1), "z": z})
    return Table(rows=rows)


def _dose_table(n: int = 160) -> Table:
    """A continuous treatment: no doubly robust estimate exists."""
    rng = random.Random(1)
    rows = []
    for _ in range(n):
        z = rng.gauss(0, 1)
        x = z + rng.gauss(0, 1)
        rows.append({"x": x, "y": 1.5 * x + 2 * z + rng.gauss(0, 1), "z": z})
    return Table(rows=rows)


@pytest.fixture(scope="module")
def server() -> Any:
    tables = {"d": _table(), "tiny": _table(8), "dose": _dose_table()}
    registry = Registry()
    return build_server(
        registry, lambda: Runner(registry), load_dataset=tables.__getitem__
    )


def _call(server: Any, **args: Any) -> dict[str, Any]:
    out = asyncio.run(
        server.call_tool(
            "reconcile_effect",
            {"dataset": "d", "x": "t", "y": "y", "adjust": ["z"], **args},
        )
    )
    return json.loads(_text(out))


@pytest.fixture(scope="module")
def opinion() -> Any:
    return so.estimate_effect(_table().rows, "t", "y", ["z"])


def test_point_answer_is_the_mean_with_elbis_doubly_robust_estimate(
    server: Any, opinion: Any
) -> None:
    out = _call(server, estimate=1.8)
    assert out["combined"].startswith("your analysis and elbi's")
    assert out["final"]["estimate"] == pytest.approx((1.8 + opinion.doubly_robust) / 2)
    assert out["final"]["ci_lower"] is None
    assert out["elbi_estimates"] == pytest.approx(opinion.estimates)


def test_interval_answer_is_the_mixture_and_holds_the_point(
    server: Any, opinion: Any
) -> None:
    out = _call(server, estimate=1.8, ci_lower=1.6, ci_upper=2.0)
    lo, hi = so.combine_interval(1.6, 2.0, opinion)
    assert out["final"]["ci_lower"] == pytest.approx(lo)
    assert out["final"]["ci_upper"] == pytest.approx(hi)
    assert lo <= out["final"]["estimate"] <= hi


def test_named_procedure_is_checked_not_combined(server: Any) -> None:
    out = _call(
        server, estimate=1.8, ci_lower=1.6, ci_upper=2.0, procedure_specified=True
    )
    assert out["final"] == {"estimate": 1.8, "ci_lower": 1.6, "ci_upper": 2.0}
    assert out["combined"] == "your answer, checked"


def test_estimate_outside_every_model_is_withheld_until_confirmed(server: Any) -> None:
    out = _call(server, estimate=-5.0)
    assert out["final"] is None
    assert "outside every elbi model" in out["problem"]
    kept = _call(server, estimate=-5.0, confirm=True)
    assert kept["final"]["estimate"] == -5.0
    assert kept["combined"] == "your answer, confirmed"


def test_too_little_data_leaves_the_answer_unchecked(server: Any) -> None:
    out = json.loads(
        _text(
            asyncio.run(
                server.call_tool(
                    "reconcile_effect",
                    {
                        "dataset": "tiny",
                        "x": "t",
                        "y": "y",
                        "adjust": ["z"],
                        "estimate": 1.0,
                    },
                )
            )
        )
    )
    assert out["final"]["estimate"] == 1.0
    assert "unchecked" in out["note"]


@pytest.mark.parametrize(
    ("args", "message"),
    [
        ({"estimand": "late"}, "estimand"),
        ({"ci_lower": 1.0}, "both"),
        ({"ci_lower": 2.0, "ci_upper": 1.0}, "below"),
    ],
)
def test_malformed_requests_are_refused(
    server: Any, args: dict[str, Any], message: str
) -> None:
    with pytest.raises(MCPError, match=message):
        _call(server, estimate=2.0, **args)


def test_without_a_doubly_robust_estimate_the_point_is_reported_as_checked(
    server: Any,
) -> None:
    args = {
        "dataset": "dose",
        "x": "x",
        "y": "y",
        "adjust": ["z"],
        "x0": 0.0,
        "x1": 1.0,
    }
    out = json.loads(
        _text(
            asyncio.run(server.call_tool("reconcile_effect", {**args, "estimate": 1.5}))
        )
    )
    assert out["final"]["estimate"] == 1.5
    assert out["combined"].startswith("your answer, checked")
    both = json.loads(
        _text(
            asyncio.run(
                server.call_tool(
                    "reconcile_effect",
                    {**args, "estimate": 1.5, "ci_lower": 1.3, "ci_upper": 1.7},
                )
            )
        )
    )
    assert both["combined"].startswith("interval combined")
