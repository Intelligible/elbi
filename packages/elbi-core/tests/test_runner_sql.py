"""End-to-end: a derivation backed by a SQL dataset, run and served."""

from __future__ import annotations

from pathlib import Path

import pytest

from elbi_core import (
    Artifact,
    Context,
    Dataset,
    Registry,
    Runner,
    derivation,
    serve,
)
from elbi_core.config import DataBindings

sa = pytest.importorskip("sqlalchemy")


def test_derivation_over_sql_dataset(tmp_path: Path) -> None:
    db = f"sqlite:///{tmp_path / 'orders.db'}"
    engine = sa.create_engine(db)
    with engine.begin() as conn:
        conn.execute(sa.text("CREATE TABLE orders (region TEXT, amount REAL)"))
        conn.execute(
            sa.text(
                "INSERT INTO orders VALUES ('west', 100), ('west', 40), ('east', 5)"
            )
        )
    engine.dispose()

    registry = Registry()

    @derivation(
        inputs={"orders": Dataset("orders")},
        serve=serve.table(title="Revenue by region", columns=["region", "revenue"]),
        registry=registry,
    )
    def revenue_by_region(ctx: Context) -> Artifact:
        totals: dict[str, float] = {}
        for row in ctx.input("orders").rows:
            totals[row["region"]] = totals.get(row["region"], 0.0) + float(
                row["amount"]
            )
        rows = [{"region": k, "revenue": v} for k, v in sorted(totals.items())]
        return Artifact.table(rows)

    bindings = DataBindings(bindings={"orders": {"connection": db, "table": "orders"}})
    runner = Runner(registry, bindings=bindings, base_dir=tmp_path)

    artifact = runner.run("revenue_by_region")
    assert artifact.value == [
        {"region": "east", "revenue": 5.0},
        {"region": "west", "revenue": 140.0},
    ]
    rendered = runner.serve("revenue_by_region")
    assert "Revenue by region" in rendered
    assert "west" in rendered
