"""The SQL connectors, and the hosted services that reuse one of their wires.

Supabase and Redshift are PostgreSQL underneath, so what is worth testing is not that
they read rows -- the Postgres path already does that -- but the three things they
change: which driver and port the URL gets, whether the driver is told to skip prepared
statements, and whether a host typed the way the vendor's own dashboard shows it is
understood.
"""

from __future__ import annotations

import pytest

from elbi.warehouse.sources.registry import SourceRegistry

_SUPABASE = {
    "host": "aws-0-us-east-1.pooler.supabase.com",
    "database": "postgres",
    "user": "postgres.abcdefghijklm",
    "password": "secret",
}


def test_hosted_postgres_sources_are_in_the_catalog() -> None:
    catalogued = (("supabase", "Supabase"), ("redshift", "Amazon Redshift"))
    for source_type, label in catalogued:
        source = SourceRegistry.get(source_type)
        assert source.config.name == source_type
        assert source.config.label == label
        assert source.config.category == "Databases"


def test_supabase_speaks_the_postgres_driver_on_the_session_pooler_port() -> None:
    url = SourceRegistry.get("supabase")._url(_SUPABASE)  # type: ignore[attr-defined]
    assert url.startswith("postgresql+psycopg://")
    assert "aws-0-us-east-1.pooler.supabase.com:5432/postgres" in url


def test_redshift_keeps_the_postgres_driver_but_its_own_port() -> None:
    config = {"host": "c.abc.us-east-1.redshift.amazonaws.com", "database": "dev"}
    url = SourceRegistry.get("redshift")._url(config)  # type: ignore[attr-defined]
    assert url.startswith("postgresql+psycopg://")
    # 5439, not Postgres' 5432: the port follows the connector, the driver follows
    # the wire, and this is the assertion that keeps those two lookups apart.
    assert url.endswith("c.abc.us-east-1.redshift.amazonaws.com:5439/dev")


def test_a_host_pasted_with_its_scheme_still_connects() -> None:
    config = {**_SUPABASE, "host": "  https://aws-0-us-east-1.pooler.supabase.com/  "}
    url = SourceRegistry.get("supabase")._url(config)  # type: ignore[attr-defined]
    assert url.endswith("@aws-0-us-east-1.pooler.supabase.com:5432/postgres")
    assert "https" not in url


@pytest.mark.parametrize(
    ("host", "prepared"),
    [
        ("aws-0-us-east-1.pooler.supabase.com", False),
        ("db.abcdefghijklm.supabase.co", True),
    ],
)
def test_prepared_statements_are_disabled_only_on_the_pooler(
    host: str, prepared: bool
) -> None:
    source = SourceRegistry.get("supabase")
    args = source._connect_args({**_SUPABASE, "host": host})  # type: ignore[attr-defined]
    assert ("prepare_threshold" not in args) is prepared


def test_redshift_never_prepares() -> None:
    source = SourceRegistry.get("redshift")
    args = source._connect_args({"host": "c.abc.us-east-1.redshift.amazonaws.com"})  # type: ignore[attr-defined]
    assert args == {"prepare_threshold": None}


def test_the_project_url_is_refused_before_anything_is_dialled() -> None:
    """The REST endpoint is not a database, and the DNS error alone would not say so."""
    ok, errors = SourceRegistry.get("supabase").validate(
        {**_SUPABASE, "host": "https://AbcDefGhIjklm.supabase.co"}
    )
    assert not ok
    message = " ".join(errors)
    assert "project URL" in message
    # Lowercased, because the pooler username is case-sensitive and the whole point of
    # the message is that it can be copied into the form as it stands.
    assert "postgres.abcdefghijklm" in message
    assert "aws-0-<region>.pooler.supabase.com" in message


def test_a_missing_field_is_not_answered_with_an_explanation_about_ipv6() -> None:
    ok, errors = SourceRegistry.get("supabase").validate(
        {"host": "db.abcdefghijklm.supabase.co"}
    )
    assert not ok
    assert any("database" in e for e in errors)
    assert not any("IPv4" in e for e in errors)


def test_a_failed_direct_host_explains_the_ipv6_default() -> None:
    ok, errors = SourceRegistry.get("supabase").validate(
        {
            "host": "db.abcdefghijklm.supabase.co",
            "database": "postgres",
            "user": "postgres",
            "password": "secret",
        }
    )
    assert not ok
    assert any("IPv4 add-on" in e for e in errors)


def test_the_host_field_points_at_the_pooler() -> None:
    fields = {f.name: f for f in SourceRegistry.get("supabase").config.fields}
    assert fields["host"].placeholder == "aws-0-us-east-1.pooler.supabase.com"
    assert "Session pooler" in fields["host"].caption
    # Inherited from Postgres rather than restated, which is the point of the subclass:
    # the same fields, differing only where this connector deliberately changed one.
    assert set(fields) == {f.name for f in SourceRegistry.get("postgres").config.fields}


# --- the branded wire connectors --------------------------------------------------


@pytest.mark.parametrize(
    ("source_type", "label", "port", "driver"),
    [
        ("neon", "Neon", 5432, "postgresql+psycopg"),
        ("cockroachdb", "CockroachDB", 26257, "postgresql+psycopg"),
        ("planetscale", "PlanetScale", 3306, "mysql+pymysql"),
        ("clickhouse", "ClickHouse", 8123, "clickhouse+http"),
    ],
)
def test_each_hosted_database_keeps_its_own_port_and_its_engines_driver(
    source_type: str, label: str, port: int, driver: str
) -> None:
    source = SourceRegistry.get(source_type)
    assert source.config.label == label
    url = source._url(  # type: ignore[attr-defined]
        {"host": "h", "database": "d", "user": "u", "password": "p"}
    )
    assert url.startswith(f"{driver}://")
    assert f":{port}/" in url


@pytest.mark.parametrize("source_type", ["neon", "cockroachdb"])
def test_a_hosted_postgres_refuses_to_fall_back_to_plaintext(source_type: str) -> None:
    """libpq's default is `prefer`, which continues unencrypted when TLS fails.

    For a database that exists only on the public internet that is the wrong outcome,
    and there is no local deployment whose convenience it would be protecting.
    """
    args = SourceRegistry.get(source_type)._connect_args({})  # type: ignore[attr-defined]
    assert args["sslmode"] == "require"


def test_planetscale_hands_pymysql_a_certificate_authority() -> None:
    """Unlike libpq, pymysql finds no CA on its own, so TLS would fail without one."""
    import certifi

    args = SourceRegistry.get("planetscale")._connect_args({})  # type: ignore[attr-defined]
    assert args["ssl"]["ca"] == certifi.where()


def test_clickhouse_asks_for_tls_by_naming_the_protocol_not_the_scheme() -> None:
    """The scheme picks the dialect; only `protocol` switches the transport to https."""
    source = SourceRegistry.get("clickhouse")
    base = {"host": "h", "database": "d", "user": "u", "password": "p"}
    assert "protocol=https" not in source._url(base)  # type: ignore[attr-defined]
    secure = source._url({**base, "secure": True})  # type: ignore[attr-defined]
    assert secure.endswith("?protocol=https")
    # Cloud listens for TLS on another port, so the default follows the switch.
    assert ":8443/" in secure


# --- the shared engine, against real servers --------------------------------------


def _live(source_type: str, config: dict[str, object], ddl: list[str]) -> object:
    """Skip unless the server is up, then apply the DDL and hand back the connector."""
    import sqlalchemy as sa

    source = SourceRegistry.get(source_type)
    ok, errors = source.validate(config)
    if not ok:
        pytest.skip(f"no live {source_type}: {errors}")
    with source._engine(config).begin() as connection:  # type: ignore[attr-defined]
        for statement in ddl:
            connection.execute(sa.text(statement))
    return source


def _extract(
    source: object, config: dict[str, object], table: str, **kw: object
) -> list:
    from elbi.warehouse.sources.base import SourceInputs

    batches = source.extract(SourceInputs(config=config, schema=table, **kw))  # type: ignore[attr-defined]
    return [row for batch in batches for row in batch.to_pylist()]


_PG = {
    "host": "127.0.0.1",
    "port": 15432,
    "database": "lake",
    "user": "postgres",
    "password": "elbi",
    "schema": "public",
}
_CLICKHOUSE = {
    "host": "127.0.0.1",
    "port": 18123,
    "database": "default",
    "user": "default",
    "password": "elbi",
}


def test_live_postgres_reads_a_table_and_honours_a_cursor() -> None:
    source = _live(
        "postgres",
        _PG,
        [
            "DROP TABLE IF EXISTS live_orders",
            "CREATE TABLE live_orders (id int primary key, sku text)",
            "INSERT INTO live_orders VALUES (1,'a'),(2,'b'),(3,'c')",
        ],
    )
    assert "live_orders" in [s.name for s in source.schemas(_PG)]  # type: ignore[attr-defined]
    assert len(_extract(source, _PG, "live_orders")) == 3
    later = _extract(
        source, _PG, "live_orders", incremental_field="id", incremental_since=1
    )
    assert [r["id"] for r in later] == [2, 3]


def test_live_postgres_reads_a_table_outside_the_default_schema() -> None:
    """The reflected table is keyed `schema.table`, which the cursor path needs."""
    config = {**_PG, "schema": "sales"}
    source = _live(
        "postgres",
        config,
        [
            "CREATE SCHEMA IF NOT EXISTS sales",
            "DROP TABLE IF EXISTS sales.live_leads",
            "CREATE TABLE sales.live_leads (id int primary key, name text)",
            "INSERT INTO sales.live_leads VALUES (1,'x'),(2,'y'),(3,'z')",
        ],
    )
    later = _extract(
        source, config, "live_leads", incremental_field="id", incremental_since=1
    )
    assert [r["id"] for r in later] == [2, 3]


def test_live_clickhouse_does_not_duplicate_rows_on_an_incremental_read() -> None:
    """The regression test for a cross join against the same table.

    Reflection may replace the Table object it was handed, and clickhouse-sqlalchemy
    does. Building the WHERE from the stale object's columns named a second table, so
    the query compiled to `FROM hits, hits` and returned every matching row once per
    row in the table -- silently, with no error and a plausible-looking result.
    """
    source = _live(
        "clickhouse",
        _CLICKHOUSE,
        [
            "DROP TABLE IF EXISTS live_hits",
            "CREATE TABLE live_hits (id UInt32, page String)"
            " ENGINE=MergeTree ORDER BY id",
            "INSERT INTO live_hits VALUES (1,'/a'),(2,'/b'),(3,'/c')",
        ],
    )
    assert len(_extract(source, _CLICKHOUSE, "live_hits")) == 3
    later = _extract(
        source, _CLICKHOUSE, "live_hits", incremental_field="id", incremental_since=1
    )
    assert [r["id"] for r in later] == [2, 3]
