"""SSH tunnelling: reaching a database that is not reachable directly.

The live half is the point. A tunnel that is only unit-tested proves the config was
assembled, not that bytes moved, so these tests run against a real OpenSSH bastion with
the databases on a Docker network that has no published ports at all. Every one of them
first asserts that the direct connection *fails* -- if it did not, the tunnelled
connection would prove nothing.

To bring that environment up::

    docker network create elbi-tunnel-net
    docker run -d --name elbi-tun-pg --network elbi-tunnel-net \\
        -e POSTGRES_PASSWORD=elbi -e POSTGRES_DB=lake postgres:17-alpine
    docker run -d --name elbi-tun-mongo --network elbi-tunnel-net mongo:8
    # plus an sshd with AllowTcpForwarding yes, published on 12200, user jump/jumppass

Without it, the live tests skip and the rest still run.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

pytest.importorskip("pyarrow")
pytest.importorskip("paramiko")

from elbi.warehouse.sources.base import SourceInputs
from elbi.warehouse.sources.registry import SourceRegistry
from elbi.warehouse.sources.tunnel import (
    SshTunnel,
    fingerprint,
    private_key,
    routed,
    routed_url,
    tunnel_fields,
)

BASTION_PORT = 12200
TUNNEL: dict[str, Any] = {
    "ssh_host": "127.0.0.1",
    "ssh_port": BASTION_PORT,
    "ssh_user": "jump",
    "ssh_password": "jumppass",
}

# Docker-network names: resolvable from the bastion, meaningless from this host, which
# is exactly the property the tests rely on.
PG = {
    "host": "elbi-tun-pg",
    "port": 5432,
    "database": "lake",
    "user": "postgres",
    "password": "elbi",
    "schema": "public",
}
MONGO = {"connection_string": "mongodb://elbi-tun-mongo:27017", "database": "lake"}


def _bastion() -> None:
    """Skip unless an SSH bastion is listening."""
    probe = socket.socket()
    probe.settimeout(3)
    try:
        probe.connect(("127.0.0.1", BASTION_PORT))
        banner = probe.recv(64)
    except Exception:
        pytest.skip(
            f"no SSH bastion on port {BASTION_PORT}; see this module's docstring"
        )
    finally:
        probe.close()
    if not banner.startswith(b"SSH-"):
        pytest.skip(f"whatever is on port {BASTION_PORT} is not SSH")


# --- the form ---------------------------------------------------------------------


def test_the_tunnel_fields_are_all_optional() -> None:
    """A connector that gains these must not change for anyone not using them."""
    assert all(not f.required for f in tunnel_fields())


def test_the_secrets_among_them_are_secrets() -> None:
    kinds = {f.name: f.type for f in tunnel_fields()}
    assert kinds["ssh_password"] == "password"
    assert kinds["ssh_key_passphrase"] == "password"
    assert kinds["ssh_private_key"] == "textarea"


@pytest.mark.parametrize(
    "source_type",
    [
        "postgres",
        "mysql",
        "mssql",
        "oracle",
        "clickhouse",
        "supabase",
        "redshift",
        "neon",
        "cockroachdb",
        "planetscale",
        "mongodb",
        "elasticsearch",
    ],
)
def test_every_host_addressed_source_offers_a_tunnel(source_type: str) -> None:
    names = {f.name for f in SourceRegistry.get(source_type).config.fields}
    assert "ssh_host" in names, f"{source_type} cannot be tunnelled"


@pytest.mark.parametrize("source_type", ["sqlite", "snowflake", "bigquery"])
def test_a_source_with_no_host_does_not_offer_a_tunnel(source_type: str) -> None:
    """SQLite is a local file; the other two are APIs addressed by account, not host.

    Offering the fields anyway would invite someone to fill them in and wonder why
    nothing changed.
    """
    names = {f.name for f in SourceRegistry.get(source_type).config.fields}
    assert "ssh_host" not in names


# --- configuration, before anything is dialled ------------------------------------


def test_no_ssh_host_means_no_tunnel() -> None:
    assert SshTunnel.from_config({"host": "db"}) is None


def test_the_host_is_what_turns_it_on() -> None:
    tunnel = SshTunnel.from_config({"ssh_host": "bastion", "ssh_user": "u"})
    assert tunnel is not None
    assert tunnel.host == "bastion"
    assert tunnel.port == 22


@pytest.mark.parametrize(
    ("config", "expected"),
    [
        ({"ssh_host": "b", "ssh_password": "p"}, "username is required"),
        ({"ssh_host": "b", "ssh_user": "u"}, "password or private key is required"),
        (
            {"ssh_host": "b", "ssh_user": "u", "ssh_password": "p", "ssh_port": 99999},
            "not a valid SSH port",
        ),
        (
            {
                "ssh_host": "b",
                "ssh_user": "u",
                "ssh_password": "p",
                "ssh_host_key": "deadbeef",
            },
            "SHA256:",
        ),
    ],
)
def test_a_half_configured_tunnel_says_what_is_missing(
    config: dict[str, Any], expected: str
) -> None:
    tunnel = SshTunnel.from_config(config)
    assert tunnel is not None
    assert any(expected in problem for problem in tunnel.check())


def test_an_unreadable_private_key_is_caught_before_connecting() -> None:
    tunnel = SshTunnel.from_config(
        {"ssh_host": "b", "ssh_user": "u", "ssh_private_key": "not a key"}
    )
    assert tunnel is not None
    assert any("Ed25519" in problem for problem in tunnel.check())


def test_an_unreadable_private_key_reports_every_loaders_reason() -> None:
    """One of them is the real cause -- a wrong passphrase, a truncated paste."""
    with pytest.raises(ValueError, match="Ed25519Key:") as raised:
        private_key("-----BEGIN OPENSSH PRIVATE KEY-----\nnope\n")
    assert "RSAKey:" in str(raised.value)


# --- rewriting the address ---------------------------------------------------------


def test_without_a_tunnel_the_config_is_handed_back_untouched() -> None:
    config = {"host": "db.example.com", "port": 5432}
    with routed(config) as reachable:
        assert reachable is config


def test_without_a_tunnel_a_url_is_handed_back_untouched() -> None:
    config = {"url": "https://es.example.com:9243"}
    with routed_url(config, "url", 9200) as reachable:
        assert reachable is config


def test_a_url_rewrite_keeps_everything_but_the_address(monkeypatch: Any) -> None:
    """The driver has to receive the same URL, pointed somewhere else."""
    from contextlib import contextmanager

    from elbi.warehouse.sources import tunnel as module

    @contextmanager
    def fake_forward(self: Any, host: str, port: int) -> Any:
        assert (host, port) == ("es.internal", 9243)
        yield "127.0.0.1", 51234

    monkeypatch.setattr(module.SshTunnel, "forward", fake_forward)
    config = {
        "url": "https://user:secret@es.internal:9243/some/path?a=1",
        **TUNNEL,
    }
    with routed_url(config, "url", 9200) as reachable:
        assert reachable["url"] == "https://user:secret@127.0.0.1:51234/some/path?a=1"


def test_a_url_with_no_port_falls_back_to_the_services_default(
    monkeypatch: Any,
) -> None:
    from contextlib import contextmanager

    from elbi.warehouse.sources import tunnel as module

    seen: list[tuple[str, int]] = []

    @contextmanager
    def fake_forward(self: Any, host: str, port: int) -> Any:
        seen.append((host, port))
        yield "127.0.0.1", 1

    monkeypatch.setattr(module.SshTunnel, "forward", fake_forward)
    with routed_url({"url": "http://es.internal", **TUNNEL}, "url", 9200):
        pass
    assert seen == [("es.internal", 9200)]


def test_a_fingerprint_is_printed_the_way_ssh_keygen_prints_it() -> None:
    """Base64, no padding -- because that is the form a user will have copied."""
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric import ed25519

    pem = (
        ed25519.Ed25519PrivateKey.generate()
        .private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.OpenSSH,
            encryption_algorithm=serialization.NoEncryption(),
        )
        .decode()
    )
    printed = fingerprint(private_key(pem))
    assert printed.startswith("SHA256:")
    # ssh-keygen prints it unpadded, so a trailing "=" would not match what a user
    # copied out of it.
    assert not printed.endswith("=")
    assert len(printed) == len("SHA256:") + 43


# --- MongoDB's two refusals --------------------------------------------------------


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("mongodb+srv://cluster.mongodb.net", "mongodb+srv://"),
        ("mongodb://a:27017,b:27017/lake", "several servers"),
        ("mongodb://user:pw@a:27017,b:27017/lake", "several servers"),
    ],
)
def test_mongodb_refuses_a_uri_that_names_a_set_of_servers(
    uri: str, expected: str
) -> None:
    """A forward reaches one address; the driver would leave the tunnel for the rest.

    Refused with the reason rather than allowed to start and hang, which is what the
    driver does when it connects to hostnames only the far side can resolve.
    """
    ok, errors = SourceRegistry.get("mongodb").validate(
        {"connection_string": uri, "database": "lake", **TUNNEL}
    )
    assert not ok
    assert expected in " ".join(errors)


def test_mongodb_allows_those_uris_when_no_tunnel_is_configured() -> None:
    """The restriction belongs to the tunnel, not to the connector."""
    from elbi.warehouse.sources.mongodb import _tunnel_problem

    assert _tunnel_problem({"connection_string": "mongodb+srv://c.net"}) is None


# --- against a real bastion --------------------------------------------------------


def test_live_postgres_is_not_reachable_without_the_tunnel() -> None:
    """The premise every test below rests on."""
    _bastion()
    ok, _ = SourceRegistry.get("postgres").validate(PG)
    assert not ok, "the database is reachable directly, so the tunnel proves nothing"


def test_live_postgres_reads_through_the_tunnel() -> None:
    _bastion()
    source = SourceRegistry.get("postgres")
    config = {**PG, **TUNNEL}
    ok, errors = source.validate(config)
    if not ok:
        pytest.skip(f"bastion present but the database is not set up: {errors}")
    import sqlalchemy as sa

    with (
        source._routed(config) as reachable,  # type: ignore[attr-defined]
        source._engine(reachable).begin() as connection,  # type: ignore[attr-defined]
    ):
        connection.execute(sa.text("DROP TABLE IF EXISTS tunnelled"))
        connection.execute(
            sa.text("CREATE TABLE tunnelled (id int primary key, note text)")
        )
        connection.execute(
            sa.text("INSERT INTO tunnelled VALUES (1,'a'),(2,'b'),(3,'c')")
        )

    assert "tunnelled" in [s.name for s in source.schemas(config)]
    rows = [
        row
        for batch in source.extract(SourceInputs(config=config, schema="tunnelled"))
        for row in batch.to_pylist()
    ]
    assert [r["note"] for r in rows] == ["a", "b", "c"]


def test_live_an_incremental_read_works_through_the_tunnel() -> None:
    """The forward has to survive being opened once per call, not just the first."""
    _bastion()
    source = SourceRegistry.get("postgres")
    config = {**PG, **TUNNEL}
    if not source.validate(config)[0]:
        pytest.skip("bastion present but the database is not set up")
    rows = [
        row
        for batch in source.extract(
            SourceInputs(
                config=config,
                schema="tunnelled",
                incremental_field="id",
                incremental_since=1,
            )
        )
        for row in batch.to_pylist()
    ]
    assert [r["id"] for r in rows] == [2, 3]


def test_live_a_wrong_ssh_password_is_refused() -> None:
    _bastion()
    ok, errors = SourceRegistry.get("postgres").validate(
        {**PG, **TUNNEL, "ssh_password": "wrong"}
    )
    assert not ok
    assert "Authentication failed" in " ".join(errors)


def test_live_a_database_the_bastion_cannot_reach_says_which_half_failed() -> None:
    """Two problems that look identical without this: the tunnel, or the database."""
    _bastion()
    ok, errors = SourceRegistry.get("postgres").validate(
        {**PG, **TUNNEL, "host": "no-such-database"}
    )
    assert not ok
    message = " ".join(errors)
    assert "could not open a connection to no-such-database" in message
    assert "the database is what is unreachable" in message


def test_live_a_pinned_host_key_that_does_not_match_is_refused() -> None:
    """The only setting that detects an interception rather than recording it."""
    _bastion()
    ok, errors = SourceRegistry.get("postgres").validate(
        {
            **PG,
            **TUNNEL,
            "ssh_host_key": "SHA256:" + "A" * 43,
        }
    )
    assert not ok
    message = " ".join(errors)
    assert "the SSH host offered" in message
    assert "nothing was sent to it" in message


def test_live_mongodb_reads_through_the_tunnel() -> None:
    _bastion()
    source = SourceRegistry.get("mongodb")
    config = {**MONGO, **TUNNEL}
    ok, errors = source.validate(config)
    if not ok:
        pytest.skip(f"bastion present but MongoDB is not set up: {errors}")
    assert source.schemas(config)


# --- host key pinning, across the several keys a host publishes -------------------


def _tunnel(host_key: str) -> SshTunnel:
    made = SshTunnel.from_config({**TUNNEL, "ssh_host_key": host_key})
    assert made is not None
    return made


def test_a_single_fingerprint_is_accepted() -> None:
    assert _tunnel("SHA256:" + "A" * 43).pinned() == {"SHA256:" + "A" * 43}


@pytest.mark.parametrize("separator", [", ", ",", "\n", " ", ",\n  "])
def test_several_fingerprints_can_be_pasted_however_they_were_copied(
    separator: str,
) -> None:
    """`ssh-keyscan | ssh-keygen -lf -` prints a line per key; people paste lines."""
    one, two = "SHA256:" + "A" * 43, "SHA256:" + "B" * 43
    assert _tunnel(separator.join([one, two])).pinned() == {one, two}


def test_anything_that_is_not_a_fingerprint_is_ignored() -> None:
    """A pasted ssh-keyscan line carries the bit length and the type around the hash."""
    pasted = "256 SHA256:" + "A" * 43 + " user@host (ED25519)"
    assert _tunnel(pasted).pinned() == {"SHA256:" + "A" * 43}


def test_a_pin_with_no_fingerprint_in_it_is_a_configuration_error() -> None:
    assert any("SHA256:" in problem for problem in _tunnel("deadbeef").check())


def test_live_pinning_any_one_of_the_hosts_keys_is_enough() -> None:
    """The trap this exists for.

    A host publishes a key per algorithm and `ssh-keyscan` prints all of them, but the
    client negotiates exactly one. Matching only the negotiated key would reject a user
    who pasted a different line -- correct input, refused connection.
    """
    _bastion()
    import subprocess

    printed = subprocess.run(
        "ssh-keyscan -p 12200 127.0.0.1 2>/dev/null | ssh-keygen -lf - 2>/dev/null",
        shell=True,
        capture_output=True,
        text=True,
    ).stdout
    fingerprints = [line.split()[1] for line in printed.splitlines() if line.strip()]
    if len(fingerprints) < 2:
        pytest.skip("the bastion publishes only one host key, so there is no trap here")

    source = SourceRegistry.get("postgres")
    config = {**PG, **TUNNEL}
    if not source.validate(config)[0]:
        pytest.skip("bastion present but the database is not set up")

    # All of them together: what the caption tells the user to paste.
    assert source.validate({**config, "ssh_host_key": ", ".join(fingerprints)})[0]
    # And any single one of them, whichever the client goes on to negotiate.
    assert any(
        source.validate({**config, "ssh_host_key": one})[0] for one in fingerprints
    )


def test_live_a_refused_host_key_names_the_algorithm_it_saw() -> None:
    """Without the algorithm, a mismatch reads as a break-in, not a wrong line."""
    _bastion()
    ok, errors = SourceRegistry.get("postgres").validate(
        {**PG, **TUNNEL, "ssh_host_key": "SHA256:" + "Z" * 43}
    )
    assert not ok
    assert "ssh-ed25519" in " ".join(errors) or "ssh-rsa" in " ".join(errors)


# --- the byte pump -----------------------------------------------------------------


def test_each_direction_of_a_forward_closes_on_its_own() -> None:
    """A driver that has finished sending is still waiting for its answer.

    Tearing both halves down on the first end-of-file cuts off the reply, which is the
    shape of a result that arrives truncated for no visible reason. Here the driver
    sends a request, half-closes, and must still receive what comes back.
    """
    from elbi.warehouse.sources.tunnel import _pump

    class _Channel:
        """A paramiko channel's surface, over a socket so `select` behaves normally."""

        def __init__(self, sock: socket.socket) -> None:
            self._sock = sock
            self.shut = False

        def fileno(self) -> int:
            return self._sock.fileno()

        def sendall(self, data: bytes) -> None:
            self._sock.sendall(data)

        def recv(self, size: int) -> bytes:
            return self._sock.recv(size)

        def shutdown_write(self) -> None:
            self.shut = True
            self._sock.shutdown(socket.SHUT_WR)

    driver, pump_side = socket.socketpair()
    chan_side, server = socket.socketpair()
    channel = _Channel(chan_side)

    def answer() -> None:
        assert server.recv(64) == b"query"
        server.sendall(b"answer")
        server.shutdown(socket.SHUT_WR)

    import threading

    driver.sendall(b"query")
    driver.shutdown(socket.SHUT_WR)
    responder = threading.Thread(target=answer, daemon=True)
    responder.start()
    _pump(pump_side, channel)
    responder.join(timeout=5)

    assert channel.shut, "the far side was never told the request had ended"
    assert driver.recv(64) == b"answer", "the reply was cut off"
    for sock in (driver, pump_side, chan_side, server):
        sock.close()


# --- the algorithms this client will agree to --------------------------------------


def test_no_broken_cipher_is_offered() -> None:
    """paramiko's own list still ends with 3des-cbc, and ranks AES-GCM last of all.

    3DES has a 64-bit block and has been out of OpenSSH's defaults for years; the CBC
    modes are the construction the Terrapin work made everyone move off. Neither belongs
    in front of a database.
    """
    from elbi.warehouse.sources.tunnel import _CIPHERS

    assert not any("3des" in c for c in _CIPHERS)
    assert not any(c.endswith("-cbc") for c in _CIPHERS)
    # AEAD first, so it is what gets negotiated rather than what is left over.
    assert _CIPHERS[0].endswith("gcm@openssh.com")


def test_no_deprecated_mac_is_offered_and_etm_comes_first() -> None:
    """MD5 and SHA-1 are still in paramiko's list, and it ranks plain MACs above ETM."""
    from elbi.warehouse.sources.tunnel import _MACS

    assert not any("md5" in m for m in _MACS)
    assert not any(m.startswith("hmac-sha1") for m in _MACS)
    assert _MACS[0].endswith("-etm@openssh.com")


def test_the_hardened_lists_are_actually_applied(monkeypatch: Any) -> None:
    """Declaring them and not setting them would be worse than not declaring them."""
    import paramiko

    from elbi.warehouse.sources.tunnel import _CIPHERS, _MACS

    applied: dict[str, Any] = {}

    class _Options:
        ciphers: Any = None
        digests: Any = None

    class _FakeTransport:
        def __init__(self, address: Any) -> None:
            self._options = _Options()

        def get_security_options(self) -> Any:
            return self._options

        def start_client(self, timeout: int = 0) -> None:
            applied["ciphers"] = self._options.ciphers
            applied["digests"] = self._options.digests
            raise RuntimeError("stop here; the negotiation is all this test wanted")

        def close(self) -> None:
            return None

    monkeypatch.setattr(paramiko, "Transport", _FakeTransport)
    tunnel = SshTunnel.from_config(TUNNEL)
    assert tunnel is not None
    with pytest.raises(RuntimeError), tunnel.forward("db", 5432):
        pass
    assert applied["ciphers"] == _CIPHERS
    assert applied["digests"] == _MACS


def test_the_tunnel_fields_are_grouped_under_one_heading() -> None:
    """Seven optional inputs in a flat list read as seven more things to fill in."""
    from elbi.warehouse.sources.tunnel import SECTION

    assert {f.section for f in tunnel_fields()} == {SECTION}
    assert "optional" in SECTION.lower()


def test_a_connectors_own_fields_stay_out_of_that_group() -> None:
    """The heading stops meaning anything if every field is filed under it."""
    fields = SourceRegistry.get("postgres").config.fields
    own = [f for f in fields if not f.name.startswith("ssh_")]
    assert own, "postgres declares no fields of its own"
    assert all(f.section == "" for f in own)


def test_the_grouped_fields_come_last_so_the_form_reads_in_order() -> None:
    """A heading in the middle of a connector's own fields would split them in two."""
    names = [f.name for f in SourceRegistry.get("postgres").config.fields]
    first_ssh = next(i for i, n in enumerate(names) if n.startswith("ssh_"))
    assert all(n.startswith("ssh_") for n in names[first_ssh:])


# --- the switch, and the choice between the two credentials -----------------------


def test_the_switch_turns_the_tunnel_on_without_a_host_being_typed_yet() -> None:
    """Turning it on but leaving the host blank must not connect straight to the DB.

    Returning no tunnel there would do exactly that -- silently, to a database the user
    had just said was only reachable through a bastion.
    """
    tunnel = SshTunnel.from_config({"ssh_enabled": True})
    assert tunnel is not None
    assert any("SSH host is required" in problem for problem in tunnel.check())


def test_a_host_alone_still_works_for_a_config_written_to_the_api() -> None:
    """The switch is the form's doing; a config posted directly need not carry it."""
    assert SshTunnel.from_config({"ssh_host": "bastion"}) is not None


def test_neither_the_switch_nor_a_host_means_no_tunnel() -> None:
    assert SshTunnel.from_config({"ssh_enabled": False, "host": "db"}) is None


def test_choosing_password_ignores_a_key_left_behind_by_the_other_choice() -> None:
    """Both values persist in the saved config; only the chosen one may be used."""
    tunnel = SshTunnel.from_config(
        {
            "ssh_host": "b",
            "ssh_auth": "password",
            "ssh_password": "secret",
            "ssh_private_key": "-----BEGIN OPENSSH PRIVATE KEY-----",
        }
    )
    assert tunnel is not None
    assert tunnel.password == "secret"
    assert tunnel.key == ""


def test_choosing_a_key_ignores_a_password_left_behind() -> None:
    tunnel = SshTunnel.from_config(
        {
            "ssh_host": "b",
            "ssh_auth": "key",
            "ssh_password": "secret",
            "ssh_private_key": "KEY",
        }
    )
    assert tunnel is not None
    assert tunnel.key == "KEY"
    assert tunnel.password == ""


def test_with_no_choice_recorded_a_key_is_taken_to_mean_a_key() -> None:
    tunnel = SshTunnel.from_config(
        {"ssh_host": "b", "ssh_password": "secret", "ssh_private_key": "KEY"}
    )
    assert tunnel is not None
    assert tunnel.key == "KEY"


# --- what the form is told to show -------------------------------------------------


def test_only_the_switch_shows_until_it_is_turned_on() -> None:
    """Eight fields on a form nobody is going to fill in is eight too many."""
    unconditional = [f for f in tunnel_fields() if not f.depends_on]
    assert [f.name for f in unconditional] == ["ssh_enabled"]


def test_everything_else_hangs_off_the_switch_or_the_auth_choice() -> None:
    parents = {f.depends_on for f in tunnel_fields() if f.depends_on}
    assert parents == {"ssh_enabled", "ssh_auth"}


def test_the_two_credentials_are_shown_one_at_a_time() -> None:
    """A form offering both invites filling in both, and only one is ever used."""
    conditions = {
        f.name: (f.depends_on, f.depends_value)
        for f in tunnel_fields()
        if f.depends_on == "ssh_auth"
    }
    assert conditions["ssh_password"] == ("ssh_auth", "password")
    assert conditions["ssh_private_key"] == ("ssh_auth", "key")
    assert conditions["ssh_key_passphrase"] == ("ssh_auth", "key")


def test_the_switch_is_a_switch_and_the_choice_is_a_choice() -> None:
    kinds = {f.name: f.type for f in tunnel_fields()}
    assert kinds["ssh_enabled"] == "switch"
    assert kinds["ssh_auth"] == "select"
    options = next(f for f in tunnel_fields() if f.name == "ssh_auth").options
    assert [o["value"] for o in options] == ["password", "key"]
