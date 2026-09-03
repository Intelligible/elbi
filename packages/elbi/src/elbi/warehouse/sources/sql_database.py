"""SQL database connectors: Postgres, MySQL, MS SQL, SQLite, and hosted variants.

One shared engine (introspect tables, stream them into the warehouse full-refresh or
incremental on a cursor column) behind a connector per dialect, so each is its own
catalog tile with the right label, icon, and defaults. Reads read-only through
SQLAlchemy, batched to bound memory on large tables.

A hosted service that speaks one of these protocols (Supabase and Redshift both speak
Postgres) sets ``wire`` to the engine underneath and inherits the whole path. What it
adds is the part that is genuinely its own: the host its dashboard shows, and the
connection mistakes that host invites.
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
from typing import TYPE_CHECKING, Any

import pyarrow as pa

if TYPE_CHECKING:
    # sqlalchemy is imported lazily below; this is for the annotation only.
    from sqlalchemy import Engine

from ..config import SourceConfig, SourceField, SourceSchema
from .base import SimpleSource, Source, SourceInputs
from .registry import SourceRegistry
from .tunnel import routed, tunnel_fields

_DRIVERS = {
    "postgres": "postgresql+psycopg",
    "mysql": "mysql+pymysql",
    "mssql": "mssql+pymssql",
    "sqlite": "sqlite",
    # HTTP rather than ClickHouse's native protocol: it is the port Cloud exposes, the
    # one that survives a corporate proxy, and the only one reachable over TLS without
    # further configuration. The native protocol is faster and is not worth the reach.
    "clickhouse": "clickhouse+http",
    # python-oracledb in its thin mode, which speaks the wire protocol directly and
    # needs no Oracle Instant Client installed beside it.
    "oracle": "oracle+oracledb",
}
# Keyed by dialect, not by wire: Redshift speaks Postgres but listens on 5439, and
# Supabase's pooler on 5432 (session mode) or 6543 (transaction mode).
_DEFAULT_PORTS = {
    "postgres": 5432,
    "mysql": 3306,
    "mssql": 1433,
    "redshift": 5439,
    "supabase": 5432,
    "neon": 5432,
    "cockroachdb": 26257,
    "planetscale": 3306,
    "clickhouse": 8123,
    "oracle": 1521,
}

#: ClickHouse answers HTTP on 8123 and HTTPS on 8443, and Cloud offers only the second.
_CLICKHOUSE_SECURE_PORT = 8443
_BATCH = 50_000


class _SqlSource(SimpleSource):
    """Shared SQL engine; a dialect subclass fixes ``dialect``/``label``/``icon``."""

    dialect: str = ""
    #: The dialect whose *driver* this connector speaks, when that is not its own name.
    #: A hosted service earns its own catalog tile and its own default port while
    #: reusing an engine above, and this is the one line that says which.
    wire: str = ""
    label: str = ""
    icon: str = ""
    supports_column_selection = True
    #: Whether this connector reaches its database over a host and port, and so can be
    #: tunnelled. False for the ones addressed another way: a file path, an account
    #: name, a service-account key.
    tunnellable = True

    def _remote_port(self, config: dict[str, Any]) -> int:
        """The port the database itself listens on, before any tunnel is involved."""
        return int(config.get("port") or _DEFAULT_PORTS.get(self.dialect, 0))

    @contextmanager
    def _routed(self, config: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """The config, pointed at a tunnel when one is configured.

        The port is resolved first: left at its default the config carries no port at
        all, and a forward to port zero goes nowhere. The dialect's default is what the
        user meant, so it is written in before the forward opens.
        """
        if not self.tunnellable:
            yield config
            return
        with routed({**config, "port": self._remote_port(config)}) as reachable:
            yield reachable

    @property
    def _driver_dialect(self) -> str:
        return self.wire or self.dialect

    @property
    def source_type(self) -> str:
        """The registry key: the dialect id (``postgres``, ``mysql``, …)."""
        return self.dialect

    @property
    def config(self) -> SourceConfig:
        """A networked-database connection form (host, port, database, credentials)."""
        return SourceConfig(
            name=self.dialect,
            label=self.label,
            category="Databases",
            icon=self.icon,
            caption=f"Sync tables from a {self.label} database.",
            fields=[
                SourceField(name="host", label="Host", placeholder="db.example.com"),
                SourceField(name="port", label="Port", type="number", required=False),
                SourceField(name="database", label="Database", placeholder="analytics"),
                SourceField(name="user", label="User", required=False),
                SourceField(
                    name="password",
                    label="Password",
                    type="password",
                    required=False,
                ),
                SourceField(
                    name="schema", label="Schema", required=False, placeholder="public"
                ),
                *tunnel_fields(),
            ],
        )

    def _url(self, config: dict[str, Any]) -> str:
        driver = _DRIVERS[self._driver_dialect]
        if self._driver_dialect == "sqlite":
            return f"sqlite:///{config['database']}"
        user = config.get("user", "")
        pw = config.get("password", "")
        auth = f"{user}:{pw}@" if user else ""
        port = config.get("port") or _DEFAULT_PORTS.get(self.dialect, "")
        host = config.get("host", "")
        hostport = f"{host}:{port}" if port else host
        return f"{driver}://{auth}{hostport}/{config['database']}"

    def _connect_args(self, config: dict[str, Any]) -> dict[str, Any]:
        """Driver-level connect arguments, for a host that needs one."""
        del config
        return {}

    def _engine(self, config: dict[str, Any]) -> Engine:
        from sqlalchemy import create_engine, pool

        # A fresh engine per call, so pooling it buys nothing and costs a connection
        # that nothing disposes: the pool outlives its last use and its connection is
        # left to the garbage collector, which CPython 3.13 reports as an unclosed
        # database. NullPool closes on return instead, as the datasource probes do.
        return create_engine(
            self._url(config),
            poolclass=pool.NullPool,
            connect_args=self._connect_args(config),
        )

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Check required fields, then probe with ``SELECT 1``."""
        ok, errors = super().validate(config)
        if not ok:
            return ok, errors
        try:
            from sqlalchemy import text

            with (
                self._routed(config) as reachable,
                self._engine(reachable).connect() as conn,
            ):
                conn.execute(text("SELECT 1"))
            return True, []
        except Exception as e:  # surface the connection error to the user, any type
            return False, [f"Could not connect: {e}"]

    def schemas(self, config: dict[str, Any]) -> list[SourceSchema]:
        """Introspect the database for its tables and their cursor candidates."""
        from sqlalchemy import inspect

        schema = config.get("schema") or None
        out: list[SourceSchema] = []
        with self._routed(config) as reachable:
            inspector = inspect(self._engine(reachable))
            for table in inspector.get_table_names(schema=schema):
                cols = inspector.get_columns(table, schema=schema)
                candidates = ("updated_at", "modified_at", "created_at", "id")
                incremental = [
                    c["name"] for c in cols if c["name"].lower() in candidates
                ]
                out.append(SourceSchema(name=table, incremental_fields=incremental))
        return out

    def extract(self, inputs: SourceInputs) -> Iterator[pa.Table]:
        """Stream a table in batches, filtered past the cursor when incremental.

        Built with SQLAlchemy Core (``select(table)``) rather than string SQL, so table
        and column identifiers are quoted by the engine and the cursor value is bound.
        """
        with self._routed(inputs.config) as reachable:
            yield from self._stream(inputs, reachable)

    def _stream(
        self, inputs: SourceInputs, config: dict[str, Any]
    ) -> Iterator[pa.Table]:
        """Read one table over a connection that is already reachable."""
        from sqlalchemy import MetaData, Table, select

        engine = self._engine(config)
        db_schema = config.get("schema") or None
        metadata = MetaData()
        Table(inputs.schema, metadata, autoload_with=engine, schema=db_schema)
        # Taken from the metadata rather than from what Table() returned. Reflection is
        # allowed to replace the object it was handed, and clickhouse-sqlalchemy does:
        # the returned one is then stale, its columns belong to the replacement, and a
        # WHERE built from them names a second table. The query compiles to a cross join
        # against itself and every row comes back duplicated, with no error to notice.
        key = f"{db_schema}.{inputs.schema}" if db_schema else inputs.schema
        table = metadata.tables[key]
        stmt = select(table)
        if inputs.incremental_field and inputs.incremental_since is not None:
            stmt = stmt.where(
                table.c[inputs.incremental_field] > inputs.incremental_since
            )
        with engine.connect().execution_options(stream_results=True) as conn:
            result = conn.execute(stmt)
            columns = list(result.keys())
            while True:
                rows = result.fetchmany(_BATCH)
                if not rows:
                    break
                yield pa.Table.from_pylist(
                    [dict(zip(columns, r, strict=True)) for r in rows]
                )


@SourceRegistry.register
class PostgresSource(_SqlSource):
    """PostgreSQL."""

    dialect = "postgres"
    label = "PostgreSQL"
    icon = "🐘"


# Supabase shows three different hostnames in its dashboard and only two of them are a
# database. Each pattern below is one of the ways that goes wrong.
_SUPABASE_POOLER_HOST = re.compile(r"\.pooler\.supabase\.com$", re.IGNORECASE)
_SUPABASE_DIRECT_HOST = re.compile(r"^db\.[a-z0-9]+\.supabase\.co$", re.IGNORECASE)
_SUPABASE_PROJECT_HOST = re.compile(
    r"^(?P<ref>[a-z0-9]+)\.supabase\.co$", re.IGNORECASE
)

_SUPABASE_HOST_CAPTION = (
    "In the Supabase dashboard, click **Connect** and open the **Direct** tab. Use the "
    "**Session pooler** host, `aws-0-<region>.pooler.supabase.com`, with the "
    "username `postgres.<project-ref>`. The direct host `db.<ref>.supabase.co` "
    "resolves to IPv6 "
    "only unless the project has the IPv4 add-on."
)

_SUPABASE_IPV4_HINT = (
    "The direct host db.<ref>.supabase.co resolves to IPv6 only unless the project has "
    "Supabase's IPv4 add-on (Project settings -> Add-ons). Use the session pooler "
    "host, aws-0-<region>.pooler.supabase.com, with the username "
    "postgres.<project-ref>."
)


def _bare_host(value: str) -> str:
    """A pasted value reduced to a bare host: no scheme, no path, no stray space."""
    without_scheme = re.sub(
        r"^[a-z][a-z0-9+.-]*://", "", (value or "").strip(), flags=re.IGNORECASE
    )
    return without_scheme.split("/", 1)[0]


@SourceRegistry.register
class SupabaseSource(PostgresSource):
    """Supabase: hosted PostgreSQL, over the ordinary Postgres wire.

    Everything that reads rows is inherited unchanged, because a Supabase database is a
    PostgreSQL database. What is added is the part that is Supabase's own: its dashboard
    offers three hostnames, two of which are databases, and picking the wrong one fails
    in a way the Postgres error does not explain.

    The project URL (``<ref>.supabase.co``) is the REST endpoint and is refused before a
    connection is attempted, because nothing is listening for Postgres there and the
    resulting DNS error says only that the name did not resolve. The direct host
    (``db.<ref>.supabase.co``) is a real database, so it is tried; only when that
    attempt fails is the IPv6 explanation added, since the host does work for a project
    holding the IPv4 add-on and refusing it outright would be wrong.
    """

    dialect = "supabase"
    wire = "postgres"
    label = "Supabase"
    icon = "⚡"

    @property
    def config(self) -> SourceConfig:
        """The Postgres form, with the host field pointed at the pooler."""
        base = super().config
        fields = [
            replace(
                f,
                placeholder="aws-0-us-east-1.pooler.supabase.com",
                caption=_SUPABASE_HOST_CAPTION,
            )
            if f.name == "host"
            else f
            for f in base.fields
        ]
        return replace(
            base,
            name=self.dialect,
            label=self.label,
            icon=self.icon,
            caption="Sync tables from a Supabase database.",
            docs_url="https://supabase.com/docs/guides/database/connecting-to-postgres",
            fields=fields,
        )

    def _url(self, config: dict[str, Any]) -> str:
        # Normalised here as well as in `validate`, so a host pasted with its scheme
        # still connects rather than being accepted and then failing at sync time.
        return super()._url({**config, "host": _bare_host(config.get("host", ""))})

    def _connect_args(self, config: dict[str, Any]) -> dict[str, Any]:
        if not _SUPABASE_POOLER_HOST.search(_bare_host(config.get("host", ""))):
            return {}
        # The transaction-mode pooler (port 6543) gives each statement whichever backend
        # is free, so a statement prepared on one is absent on the next and psycopg's
        # reuse fails. Session mode is unaffected, but a bulk read runs each statement
        # once and gains nothing from preparing, so both get the safe setting rather
        # than making it depend on a port the user can change without telling us.
        return {"prepare_threshold": None}

    def validate(self, config: dict[str, Any]) -> tuple[bool, list[str]]:
        """Reject the project URL outright; explain a failed direct-host attempt."""
        host = _bare_host(config.get("host", ""))
        project = _SUPABASE_PROJECT_HOST.match(host)
        if project:
            # Project refs are lowercase and the pooler username is case-sensitive, so
            # the suggestion below is canonicalised to be copy-pasteable.
            ref = project.group("ref").lower()
            return False, [
                f"{host!r} is the Supabase project URL, not a database host. Use the "
                f"session pooler host aws-0-<region>.pooler.supabase.com with the "
                f"username postgres.{ref}, or the direct host db.{ref}.supabase.co."
            ]
        # The field check runs first and on its own, so a missing database name is not
        # answered with an explanation about IPv6.
        fields_ok, field_errors = Source.validate(self, config)
        if not fields_ok:
            return False, field_errors
        ok, errors = super().validate(config)
        if not ok and _SUPABASE_DIRECT_HOST.match(host):
            return False, [*errors, _SUPABASE_IPV4_HINT]
        return ok, errors


@SourceRegistry.register
class RedshiftSource(PostgresSource):
    """Amazon Redshift: the Postgres wire against a Redshift cluster.

    Redshift answers the Postgres protocol, so the shared engine introspects and streams
    it unchanged. Two settings differ: the cluster listens on 5439, and psycopg's
    automatic statement preparation is turned off. A sync runs each SELECT once, so
    preparing buys nothing here, and leaving it on is a known source of failures against
    a server this far diverged from the one psycopg targets.

    Marked beta: the code path is the tested Postgres one, but the introspection queries
    have not been run against a live cluster, and Redshift's catalog is where a Postgres
    client most often finds the fork showing through.
    """

    dialect = "redshift"
    wire = "postgres"
    label = "Amazon Redshift"
    icon = "🟥"

    @property
    def config(self) -> SourceConfig:
        """The Postgres form, with the cluster endpoint as the host."""
        base = super().config
        fields = [
            replace(
                f,
                placeholder="cluster.abc123.us-east-1.redshift.amazonaws.com",
                caption="The cluster endpoint, without the trailing port and database.",
            )
            if f.name == "host"
            else f
            for f in base.fields
        ]
        return replace(
            base,
            name=self.dialect,
            label=self.label,
            icon=self.icon,
            caption="Sync tables from an Amazon Redshift cluster.",
            docs_url="https://docs.aws.amazon.com/redshift/latest/mgmt/configuring-connections.html",
            release_status="beta",
            fields=fields,
        )

    def _connect_args(self, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {"prepare_threshold": None}


@SourceRegistry.register
class MySqlSource(_SqlSource):
    """MySQL."""

    dialect = "mysql"
    label = "MySQL"
    icon = "🐬"


#: The three connectors below reach a service that exists only on the public internet,
#: so each demands TLS rather than accepting the driver's default. Plain PostgreSQL and
#: MySQL keep that default, because a database on localhost or a private subnet is a
#: legitimate thing to connect to without it.
_REQUIRE_TLS = "the connection is refused unless it can be encrypted"


@SourceRegistry.register
class NeonSource(PostgresSource):
    """Neon: serverless PostgreSQL, over the ordinary Postgres wire.

    Neon exists only as a hosted service, so there is no unencrypted case to preserve
    and ``sslmode`` is set to require rather than left at libpq's ``prefer``. Under
    ``prefer`` a failed TLS negotiation silently continues in plaintext, which is the
    wrong outcome for a database reached across the internet.

    Nothing else differs. Compute endpoints are addressed by hostname and separated by
    SNI, which psycopg sends as a matter of course, so the branch older drivers needed
    is not required here.
    """

    dialect = "neon"
    wire = "postgres"
    label = "Neon"
    icon = "🌱"

    @property
    def config(self) -> SourceConfig:
        """The Postgres form, with a Neon compute endpoint as the host."""
        base = super().config
        fields = [
            replace(
                f,
                placeholder="ep-cool-darkness-123456.us-east-2.aws.neon.tech",
                caption="The endpoint host from the Neon dashboard's connection "
                f"string. TLS is always used: {_REQUIRE_TLS}.",
            )
            if f.name == "host"
            else f
            for f in base.fields
        ]
        return replace(
            base,
            name=self.dialect,
            label=self.label,
            icon=self.icon,
            caption="Sync tables from a Neon serverless PostgreSQL database.",
            docs_url="https://neon.tech/docs/connect/connect-from-any-app",
            fields=fields,
        )

    def _connect_args(self, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {"sslmode": "require"}


@SourceRegistry.register
class CockroachDbSource(PostgresSource):
    """CockroachDB: the Postgres wire against a CockroachDB cluster.

    Two settings differ from plain Postgres: the cluster listens on 26257, and TLS is
    required. CockroachDB Cloud refuses anything else, and a self-hosted secure cluster
    accepts it, so requiring it costs nothing that is worth having. An insecure local
    cluster is a development mode and is reachable through the PostgreSQL connector.
    """

    dialect = "cockroachdb"
    wire = "postgres"
    label = "CockroachDB"
    icon = "🪳"

    @property
    def config(self) -> SourceConfig:
        """The Postgres form, with a CockroachDB cluster host."""
        base = super().config
        fields = [
            replace(
                f,
                placeholder="my-cluster-1234.j77.aws-eu-central-1.cockroachlabs.cloud",
                caption=f"The cluster host. TLS is always used: {_REQUIRE_TLS}.",
            )
            if f.name == "host"
            else f
            for f in base.fields
        ]
        return replace(
            base,
            name=self.dialect,
            label=self.label,
            icon=self.icon,
            caption="Sync tables from a CockroachDB cluster.",
            docs_url="https://www.cockroachlabs.com/docs/stable/connect-to-the-database",
            fields=fields,
        )

    def _connect_args(self, config: dict[str, Any]) -> dict[str, Any]:
        del config
        return {"sslmode": "require"}


@SourceRegistry.register
class PlanetScaleSource(MySqlSource):
    """PlanetScale: the MySQL wire against a PlanetScale database.

    PlanetScale is TLS-only and hosted-only, and unlike libpq, pymysql will not find a
    certificate authority on its own. The bundle certifi already ships is handed to it,
    which is the same set of roots the rest of this application trusts and works
    identically on a laptop and in a container that has no system CA store.
    """

    dialect = "planetscale"
    wire = "mysql"
    label = "PlanetScale"
    icon = "🪐"

    @property
    def config(self) -> SourceConfig:
        """The MySQL form, with a PlanetScale host and its branch password."""
        base = super().config
        fields = [
            replace(
                f,
                placeholder="aws.connect.psdb.cloud",
                caption="The host from the branch's connection details. TLS is always "
                f"used: {_REQUIRE_TLS}.",
            )
            if f.name == "host"
            else f
            for f in base.fields
        ]
        return replace(
            base,
            name=self.dialect,
            label=self.label,
            icon=self.icon,
            caption="Sync tables from a PlanetScale database.",
            docs_url="https://planetscale.com/docs/concepts/connection-strings",
            fields=fields,
        )

    def _connect_args(self, config: dict[str, Any]) -> dict[str, Any]:
        import certifi

        del config
        return {"ssl": {"ca": certifi.where()}}


@SourceRegistry.register
class MsSqlSource(_SqlSource):
    """Microsoft SQL Server."""

    dialect = "mssql"
    label = "MS SQL Server"
    icon = "🧱"


@SourceRegistry.register
class OracleSource(_SqlSource):
    """Oracle Database: tables and views, over the thin driver.

    Addressed by service name rather than SID. Both still work on the server, but a SID
    names one instance where a service name names the database a client should be routed
    to, which is what every Oracle release this century has told clients to use and what
    a RAC or Autonomous connection string will give you.

    The schema field matters more here than elsewhere: in Oracle a schema is a user, and
    left empty the listing would return every table the account can see across the whole
    database, including the dictionary. It is required for that reason.
    """

    dialect = "oracle"
    label = "Oracle"
    icon = "🔺"

    @property
    def config(self) -> SourceConfig:
        """Host, port, service name, credentials, and the schema to read."""
        return SourceConfig(
            name=self.dialect,
            label=self.label,
            category="Databases",
            icon=self.icon,
            caption="Sync tables and views from an Oracle database.",
            docs_url="https://python-oracledb.readthedocs.io/en/latest/user_guide/connection_handling.html",
            fields=[
                SourceField(name="host", label="Host", placeholder="db.example.com"),
                SourceField(name="port", label="Port", type="number", required=False),
                SourceField(
                    name="service_name",
                    label="Service name",
                    placeholder="ORCLPDB1",
                    caption="Not the SID. The connection string your DBA gives you "
                    "names it after a slash.",
                ),
                SourceField(name="user", label="User"),
                SourceField(name="password", label="Password", type="password"),
                SourceField(
                    name="schema",
                    label="Schema",
                    placeholder="HR",
                    caption="A schema is a user in Oracle, and it is usually the "
                    "owner's name in capitals.",
                ),
                *tunnel_fields(),
            ],
        )

    def _url(self, config: dict[str, Any]) -> str:
        from urllib.parse import quote_plus

        user = quote_plus(str(config.get("user") or ""))
        password = quote_plus(str(config.get("password") or ""))
        host = config.get("host", "")
        port = config.get("port") or _DEFAULT_PORTS["oracle"]
        service = config.get("service_name", "")
        driver = _DRIVERS[self.dialect]
        return f"{driver}://{user}:{password}@{host}:{port}/?service_name={service}"


@SourceRegistry.register
class ClickHouseSource(_SqlSource):
    """ClickHouse: the columnar analytics database, over its HTTP interface.

    Reached through SQLAlchemy like the others, so introspection, batching and the
    incremental cursor are all inherited. What it adds is the TLS switch, because
    ClickHouse listens for plaintext and TLS on different ports and a self-hosted
    server and ClickHouse Cloud disagree about which one is normal.
    """

    dialect = "clickhouse"
    label = "ClickHouse"
    icon = "🏠"

    @property
    def config(self) -> SourceConfig:
        """Host, port, database, credentials, and whether to speak TLS."""
        return SourceConfig(
            name=self.dialect,
            label=self.label,
            category="Databases",
            icon=self.icon,
            caption="Sync tables from a ClickHouse database.",
            docs_url="https://clickhouse.com/docs/en/interfaces/http",
            fields=[
                SourceField(
                    name="host",
                    label="Host",
                    placeholder="abc123.us-east-1.aws.clickhouse.cloud",
                ),
                SourceField(
                    name="port",
                    label="Port",
                    type="number",
                    required=False,
                    caption="Defaults to 8123, or 8443 when TLS is on.",
                ),
                SourceField(name="database", label="Database", placeholder="default"),
                SourceField(
                    name="user", label="User", required=False, default="default"
                ),
                SourceField(
                    name="password", label="Password", type="password", required=False
                ),
                SourceField(
                    name="secure",
                    label="Use TLS",
                    type="switch",
                    required=False,
                    default=False,
                    caption="Required by ClickHouse Cloud. Leave off for a plaintext "
                    "server on your own network.",
                ),
                *tunnel_fields(),
            ],
        )

    def _remote_port(self, config: dict[str, Any]) -> int:
        """8443 when TLS is on, because that is the other port ClickHouse listens on."""
        if port := config.get("port"):
            return int(port)
        secure = bool(config.get("secure"))
        return _CLICKHOUSE_SECURE_PORT if secure else _DEFAULT_PORTS["clickhouse"]

    def _url(self, config: dict[str, Any]) -> str:
        from urllib.parse import quote_plus

        secure = bool(config.get("secure"))
        user = quote_plus(str(config.get("user") or "default"))
        password = quote_plus(str(config.get("password") or ""))
        port = config.get("port") or (
            _CLICKHOUSE_SECURE_PORT if secure else _DEFAULT_PORTS["clickhouse"]
        )
        host = config.get("host", "")
        database = config.get("database", "")
        # The scheme names the dialect, not the transport; `protocol` is what tells the
        # HTTP driver to use TLS, and without it a secure port answers nothing readable.
        suffix = "?protocol=https" if secure else ""
        driver = _DRIVERS[self.dialect]
        return f"{driver}://{user}:{password}@{host}:{port}/{database}{suffix}"


@SourceRegistry.register
class SqliteSource(_SqlSource):
    """SQLite: a single database file, so it needs only a path."""

    dialect = "sqlite"
    label = "SQLite"
    icon = "📁"
    # A path on this machine's own disk; there is nothing to tunnel to.
    tunnellable = False

    @property
    def config(self) -> SourceConfig:
        """A file-path connection form (SQLite has no host or credentials)."""
        return SourceConfig(
            name=self.dialect,
            label=self.label,
            category="Databases",
            icon=self.icon,
            caption="Sync tables from a local SQLite database file.",
            fields=[
                SourceField(
                    name="database",
                    label="Database file",
                    placeholder="/data/app.db",
                ),
                SourceField(
                    name="schema", label="Schema", required=False, placeholder="main"
                ),
            ],
        )


@SourceRegistry.register
class SnowflakeSource(_SqlSource):
    """Snowflake: introspect and sync tables from a Snowflake warehouse.

    Needs the ``snowflake-sqlalchemy`` driver (the ``snowflake`` extra); the shared
    ``_SqlSource`` engine handles introspection and streaming once the URL is built.
    """

    dialect = "snowflake"
    label = "Snowflake"
    icon = "❄️"
    # Addressed by account identifier over Snowflake's own endpoint, not host and port.
    tunnellable = False

    @property
    def config(self) -> SourceConfig:
        """A Snowflake connection form (account, warehouse, database, schema, role)."""
        return SourceConfig(
            name=self.dialect,
            label=self.label,
            category="Databases",
            icon=self.icon,
            caption="Sync tables from a Snowflake warehouse.",
            docs_url="https://docs.snowflake.com/en/developer-guide/python-connector/sqlalchemy",
            fields=[
                SourceField(
                    name="account_id",
                    label="Account id",
                    placeholder="xy12345.us-east-1",
                ),
                SourceField(name="user", label="Username", placeholder="User1"),
                SourceField(name="password", label="Password", type="password"),
                SourceField(name="database", label="Database", placeholder="analytics"),
                SourceField(name="schema", label="Schema", placeholder="public"),
                SourceField(
                    name="warehouse", label="Warehouse", placeholder="COMPUTE_WAREHOUSE"
                ),
                SourceField(name="role", label="Role", required=False),
            ],
        )

    def _url(self, config: dict[str, Any]) -> str:
        from urllib.parse import quote_plus

        user = quote_plus(config.get("user", ""))
        pw = quote_plus(config.get("password", ""))
        account = config["account_id"]
        database = config.get("database", "")
        schema = config.get("schema", "")
        params = {k: config[k] for k in ("warehouse", "role") if config.get(k)}
        qs = "&".join(f"{k}={v}" for k, v in params.items())
        query = f"?{qs}" if qs else ""
        return f"snowflake://{user}:{pw}@{account}/{database}/{schema}{query}"


@SourceRegistry.register
class BigQuerySource(_SqlSource):
    """BigQuery: introspect and sync tables/views from a Google BigQuery dataset.

    Needs the ``sqlalchemy-bigquery`` driver (the ``bigquery`` extra). Auth is a service
    account key JSON, passed to the engine as ``credentials_info``; the dataset is the
    SQLAlchemy schema.
    """

    dialect = "bigquery"
    label = "BigQuery"
    icon = "🔎"
    # A Google API reached by service account; there is no host to forward to.
    tunnellable = False

    @property
    def config(self) -> SourceConfig:
        """A BigQuery connection form (project, dataset, service-account key)."""
        return SourceConfig(
            name=self.dialect,
            label=self.label,
            category="Databases",
            icon=self.icon,
            caption="Sync tables and views from a Google BigQuery dataset.",
            docs_url="https://cloud.google.com/bigquery/docs",
            fields=[
                SourceField(
                    name="key_file",
                    label="Google Cloud JSON key file",
                    type="textarea",
                    placeholder='{"type": "service_account", ...}',
                    caption="A service-account key with BigQuery read access.",
                ),
                # The SQLAlchemy schema is the BigQuery dataset (labelled "Dataset ID").
                SourceField(name="schema", label="Dataset ID", placeholder="analytics"),
            ],
        )

    def _credentials(self, config: dict[str, Any]) -> dict[str, Any]:
        import json

        return json.loads(config["key_file"]) if config.get("key_file") else {}

    def _url(self, config: dict[str, Any]) -> str:
        return f"bigquery://{self._credentials(config).get('project_id', '')}"

    def _engine(self, config: dict[str, Any]):  # type: ignore[no-untyped-def]
        from sqlalchemy import create_engine

        info = self._credentials(config) or None
        return create_engine(self._url(config), credentials_info=info)
