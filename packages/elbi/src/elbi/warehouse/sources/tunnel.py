"""SSH tunnelling, for a database that is not reachable directly.

A great many databases are not on the public internet, and the ones that are usually sit
behind an allowlist. Because this app runs on the user's own infrastructure rather than
a vendor's, there is no fixed address to hand an administrator: the egress address is
the laptop's, or the NAT gateway's, and on a laptop it changes. A bastion is the answer
that does not depend on any of that, and it is what every comparable tool offers.

The other answers are network-layer and need no code here at all. A private link, a VPC
peering, a Tailscale or WireGuard network -- each of those makes the database reachable
at an address, and a connector pointed at that address is an ordinary connector. This
module exists for the case where none of that has been set up and there is an SSH host
that can already see the database.

How it works is the classic local forward. A socket is opened on loopback, and each
connection accepted there is carried over the SSH transport to the real host and port.
The connector is then pointed at the loopback address and knows nothing about any of it,
which is what lets one implementation serve every driver rather than each of them
growing its own.

The forward is written here rather than taken from a library. The obvious dependency,
``sshtunnel``, last released in January 2021 and declares support up to Python 3.8; the
work it does is one socket server and a byte pump, and paramiko -- which it wraps, and
which this app already installs -- does the difficult half.

One tunnel is opened per call, and closed when that call returns. A sync of twenty
tables therefore performs twenty-one SSH handshakes rather than one, which is the price
of the connectors staying stateless. It is not usually noticeable next to the data
transfer between them, but a bastion on OpenSSH 9.8 or later with
``PerSourcePenalties`` at its defaults can refuse a burst of them -- the symptom is
"Not allowed at this time" where an authentication error would be expected.

Two deliberate departures from what the comparable tools do.

Nothing falls back to the ambient SSH agent or to the private keys in the home directory
of whoever started the server. A tunnel authenticates as the credential the user
configured or it does not open, so a source cannot quietly connect as the host account.

The bastion's host key can be pinned. Left empty the first key offered is accepted,
which is what the other tools do and is trust-on-first-use; given a fingerprint, a key
that does not match is refused, which is the only configuration that detects an
interception rather than recording it.
"""

from __future__ import annotations

import base64
import contextlib
import hashlib
import io
import logging
import re
import select
import socket
import socketserver
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from paramiko import PKey, Transport

from ..config import SourceField

logger = logging.getLogger("elbi")

#: Bytes moved per read while pumping a forwarded connection.
_CHUNK = 32_768

#: How long a pump waits on a quiet socket before checking whether it should stop.
_POLL = 0.5

#: Where the local end of a forward is bound. Loopback only: a forward bound to any
#: other interface would offer the database to everything that can reach this host.
_BIND = "127.0.0.1"

#: Seconds between keepalives on the SSH connection. A long extract can leave the
#: control channel quiet for minutes at a time, and an idle NAT or firewall in between
#: will drop a connection it believes is finished. This keeps it visibly alive.
_KEEPALIVE = 30

#: The ciphers this client will negotiate, best first. paramiko's own list still carries
#: 3des-cbc and the CBC modes, and ranks AES-GCM last of all; OpenSSH dropped 3DES from
#: its defaults years ago and prefers AEAD. Narrowing to AEAD and CTR costs nothing a
#: bastion built this decade can offer, and a host that can offer only 3DES is not one
#: to route a database through.
_CIPHERS = (
    "aes256-gcm@openssh.com",
    "aes128-gcm@openssh.com",
    "aes256-ctr",
    "aes192-ctr",
    "aes128-ctr",
)

#: Message authentication, best first. paramiko's list ends with hmac-md5 and hmac-sha1;
#: both are long deprecated, and the encrypt-then-MAC variants it ranks *below* the
#: plain ones are the stronger construction, so the order here is reversed from its own.
_MACS = (
    "hmac-sha2-512-etm@openssh.com",
    "hmac-sha2-256-etm@openssh.com",
    "hmac-sha2-512",
    "hmac-sha2-256",
)

#: How long to wait for the server's banner, and for authentication to conclude.
#: paramiko's own defaults, restated because a tunnel that hangs on setup is worse than
#: one that fails, and because leaving them implicit invites them to change underneath.
_BANNER_TIMEOUT = 15
_AUTH_TIMEOUT = 30


#: The heading these fields are grouped under in the connection form, so seven optional
#: inputs read as one thing a user can skip rather than seven more to fill in.
SECTION = "SSH tunnel (optional)"


def tunnel_fields() -> list[SourceField]:
    """The connection-form fields a tunnellable connector appends to its own.

    A switch first, and everything else conditional on it, so a connector that gains
    these shows one more control rather than eight. The two credentials are conditional
    on a choice between them, because a form offering both invites filling in both and
    only one is ever used.
    """
    return [
        SourceField(
            section=SECTION,
            name="ssh_enabled",
            label="Connect through an SSH tunnel",
            type="switch",
            required=False,
            default=False,
            caption="For a database with no public endpoint. The connection is made "
            "through a host that can already reach it, so the database only has to "
            "accept connections from inside your own network.",
        ),
        SourceField(
            section=SECTION,
            name="ssh_host",
            label="SSH host",
            required=False,
            placeholder="bastion.example.com",
            depends_on="ssh_enabled",
        ),
        SourceField(
            section=SECTION,
            name="ssh_port",
            label="SSH port",
            type="number",
            required=False,
            default=22,
            depends_on="ssh_enabled",
        ),
        SourceField(
            section=SECTION,
            name="ssh_user",
            label="SSH username",
            required=False,
            depends_on="ssh_enabled",
        ),
        SourceField(
            section=SECTION,
            name="ssh_auth",
            label="Authenticate with",
            type="select",
            required=False,
            default="password",
            options=[
                {"value": "password", "label": "Password"},
                {"value": "key", "label": "Private key"},
            ],
            depends_on="ssh_enabled",
        ),
        SourceField(
            section=SECTION,
            name="ssh_password",
            label="SSH password",
            type="password",
            required=False,
            depends_on="ssh_auth",
            depends_value="password",
        ),
        SourceField(
            section=SECTION,
            name="ssh_private_key",
            label="SSH private key",
            type="textarea",
            secret=True,
            required=False,
            placeholder="-----BEGIN OPENSSH PRIVATE KEY-----",
            caption="The whole file, including its header and footer.",
            depends_on="ssh_auth",
            depends_value="key",
        ),
        SourceField(
            section=SECTION,
            name="ssh_key_passphrase",
            label="Private key passphrase",
            type="password",
            required=False,
            caption="Only if the key above is encrypted.",
            depends_on="ssh_auth",
            depends_value="key",
        ),
        SourceField(
            section=SECTION,
            name="ssh_host_key",
            label="Host key fingerprint",
            required=False,
            placeholder="SHA256:47DEQpj8HBSa+/TImW+5JCeuQeRkm5NMpJWZG3hSuFU",
            caption="Optional, and worth setting. Left empty, whichever key the host "
            "offers is trusted. `ssh-keyscan host | ssh-keygen -lf -` prints one line "
            "per key the host publishes; paste any or all of them, separated by "
            "commas, and a host offering none of them is refused.",
            depends_on="ssh_enabled",
        ),
    ]


def fingerprint(key: PKey) -> str:
    """A host key's SHA-256 fingerprint, in the form OpenSSH prints.

    Base64 with the padding removed, which is what ``ssh-keygen -lf`` shows and so
    what a user will have copied.
    """
    digest = hashlib.sha256(key.asbytes()).digest()
    return "SHA256:" + base64.b64encode(digest).decode().rstrip("=")


def private_key(raw: str, passphrase: str = "") -> PKey:
    """A pasted private key as a paramiko key object.

    The key type is not asked for: the header already says which it is, and a mismatched
    answer would be one more thing to get wrong. Each loader is tried and the first that
    accepts the material wins. DSA is absent because paramiko dropped it.

    When none accepts it, every loader's own complaint goes into the error. One of them
    is the real reason -- a wrong passphrase, a truncated paste -- and discarding them
    all would leave the user with "could not be read" and nothing to act on.
    """
    import paramiko

    secret = passphrase or None
    refusals: list[str] = []
    for loader in (paramiko.Ed25519Key, paramiko.ECDSAKey, paramiko.RSAKey):
        try:
            return loader.from_private_key(io.StringIO(raw), password=secret)
        except Exception as e:
            refusals.append(f"{loader.__name__}: {e}")
    raise ValueError(
        "the private key could not be read as Ed25519, ECDSA or RSA; check that the "
        "whole file was pasted, including its header and footer, and that the "
        "passphrase is right (" + "; ".join(refusals) + ")"
    )


@dataclass(frozen=True)
class SshTunnel:
    """A bastion to reach a database through."""

    host: str
    port: int
    user: str
    password: str = ""
    key: str = ""
    key_passphrase: str = ""
    host_key: str = ""

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> SshTunnel | None:
        """A tunnel from a source's saved fields, or ``None`` if none was configured.

        Either the switch or a host turns it on. The switch is what the form sets; the
        host alone is honoured too, so a config written straight to the API without the
        switch still tunnels rather than silently connecting direct.

        A switch turned on with no host still produces a tunnel, which then fails its
        own check with the missing field named. Returning ``None`` there would connect
        to the database directly instead, which is the one outcome someone who asked
        for a tunnel must not silently get.
        """
        host = str(config.get("ssh_host") or "").strip()
        if not (host or config.get("ssh_enabled")):
            return None
        # The form's chosen method decides which credential is used, so a value left
        # behind by the other one is ignored rather than quietly preferred. With no
        # choice recorded -- a config written straight to the API -- a key present is
        # taken to mean a key was intended.
        chosen = str(config.get("ssh_auth") or "").strip().lower()
        password = str(config.get("ssh_password") or "")
        key = str(config.get("ssh_private_key") or "").strip()
        if chosen == "password":
            key = ""
        elif chosen == "key":
            password = ""
        return cls(
            host=host,
            port=int(config.get("ssh_port") or 22),
            user=str(config.get("ssh_user") or "").strip(),
            password=password,
            key=key,
            key_passphrase=str(config.get("ssh_key_passphrase") or ""),
            host_key=str(config.get("ssh_host_key") or "").strip(),
        )

    def check(self) -> list[str]:
        """What is missing or malformed, before anything is dialled."""
        problems: list[str] = []
        if not self.host:
            problems.append("An SSH host is required to use a tunnel")
        if not self.user:
            problems.append("An SSH username is required to use a tunnel")
        if not (self.password or self.key):
            problems.append(
                "An SSH password or private key is required to use a tunnel"
            )
        if not 0 < self.port <= 65535:
            problems.append(f"{self.port} is not a valid SSH port")
        if self.key:
            try:
                private_key(self.key, self.key_passphrase)
            except ValueError as e:
                problems.append(str(e))
        if self.host_key and not self.pinned():
            problems.append(
                "The host key fingerprint should look like "
                "'SHA256:...'; `ssh-keyscan host | ssh-keygen -lf -` prints it"
            )
        return problems

    def pinned(self) -> frozenset[str]:
        """The fingerprints this tunnel will accept, from what the user pasted.

        A host publishes a key per algorithm, so ``ssh-keyscan`` prints several lines
        and a user may reasonably paste more than one -- or paste the RSA one while the
        client goes on to negotiate Ed25519. Any of them matching is a match, which is
        what ``known_hosts`` does with the same situation.
        """
        return frozenset(
            token
            for token in re.split(r"[\s,]+", self.host_key)
            if token.startswith("SHA256:")
        )

    @contextmanager
    def forward(self, remote_host: str, remote_port: int) -> Iterator[tuple[str, int]]:
        """Open a forward to ``remote_host:remote_port``; yield the local end.

        The port is chosen by the operating system rather than picked, so two syncs
        running at once cannot collide on it.
        """
        import paramiko

        transport = paramiko.Transport((self.host, self.port))
        transport.banner_timeout = _BANNER_TIMEOUT
        transport.auth_timeout = _AUTH_TIMEOUT
        options = transport.get_security_options()
        options.ciphers = _CIPHERS
        options.digests = _MACS
        try:
            transport.start_client(timeout=30)
            self._verify(transport)
            self._authenticate(transport)
            transport.set_keepalive(_KEEPALIVE)
            self._probe(transport, remote_host, remote_port)
            server = _Forwarder(
                (_BIND, 0),
                _Handler,
                transport=transport,
                target=(remote_host, remote_port),
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                yield _BIND, server.server_address[1]
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=10)
        finally:
            transport.close()

    def _probe(self, transport: Transport, host: str, port: int) -> None:
        """Open one channel and close it, so a refusal is reported as itself.

        Without this the first channel is opened inside the driver's connect, where a
        refusal reaches the user as whatever the driver made of a socket that closed on
        it -- for Postgres, "server closed the connection unexpectedly", which names
        neither the bastion nor the reason. Opening one here costs a TCP connect that is
        immediately dropped, and turns two common misconfigurations into their own
        messages.
        """
        try:
            channel = transport.open_channel("direct-tcpip", (host, port), (_BIND, 0))
        except Exception as e:
            detail = str(e)
            if "administratively prohibited" in detail.lower():
                raise ValueError(
                    f"the SSH host {self.host} refuses port forwarding. Its "
                    "sshd_config needs `AllowTcpForwarding yes` for this account "
                    "before it can be used as a tunnel."
                ) from e
            raise ValueError(
                f"the SSH host {self.host} could not open a connection to "
                f"{host}:{port}: {detail}. The tunnel itself worked, so the database "
                "is what is unreachable from that host."
            ) from e
        channel.close()

    def _verify(self, transport: Transport) -> None:
        """Refuse a bastion whose key does not match the pinned fingerprint."""
        offered = transport.get_remote_server_key()
        actual = fingerprint(offered)
        accepted = self.pinned()
        if not accepted:
            # Nothing to check against, so this is trust-on-first-use -- the thing every
            # static analyser flags about paramiko's AutoAddPolicy, and it is flagged
            # for good reason. The fingerprint is logged so it can be copied into the
            # form, which is the only way the next connection becomes verifiable.
            logger.info(
                "ssh tunnel to %s: host key %s (%s). Set it as the host key "
                "fingerprint on this source to verify it from now on.",
                self.host,
                actual,
                offered.get_name(),
            )
            return
        if actual not in accepted:
            expected = ", ".join(sorted(accepted))
            raise ValueError(
                f"the SSH host offered {actual} ({offered.get_name()}), which is not "
                f"among the {expected} this source pins. Either the host key changed, "
                "or this is not the host you think it is; nothing was sent to it."
            )

    def _authenticate(self, transport: Transport) -> None:
        """Authenticate as the configured credential, and only as that.

        The agent and the home-directory keys are never consulted. Left to its defaults
        paramiko would try both, so a source with a wrong password could still open a
        tunnel as whoever runs the server.
        """
        if self.key:
            transport.auth_publickey(
                self.user, private_key(self.key, self.key_passphrase)
            )
        else:
            transport.auth_password(self.user, self.password)


class _Forwarder(socketserver.ThreadingTCPServer):
    """A loopback listener whose connections are carried over one SSH transport."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(
        self,
        address: tuple[str, int],
        handler: type[socketserver.BaseRequestHandler],
        *,
        transport: Transport,
        target: tuple[str, int],
    ) -> None:
        self.transport = transport
        self.target = target
        super().__init__(address, handler)


class _Handler(socketserver.BaseRequestHandler):
    """One forwarded connection: a channel to the target, and bytes pumped both ways."""

    server: _Forwarder

    def handle(self) -> None:
        try:
            channel = self.server.transport.open_channel(
                "direct-tcpip", self.server.target, self.request.getpeername()
            )
        except Exception:
            # The bastion refused to open the channel -- most often because it can
            # reach the SSH host but not the database behind it. Closing the local end
            # lets the driver report a connection failure, which is what happened.
            return
        if channel is None:
            return
        try:
            _pump(self.request, channel)
        finally:
            channel.close()


def _pump(local: socket.socket, channel: Any) -> None:
    """Move bytes between the local socket and the channel until both directions end.

    Each direction is closed on its own. A driver that has finished sending and is
    waiting for the rest of a result would otherwise have its answer cut off the moment
    it stopped talking, because tearing both directions down on the first end-of-file
    ends the half that still had data to deliver.
    """
    sending, receiving = True, True
    while sending or receiving:
        watch = [s for s, live in ((local, sending), (channel, receiving)) if live]
        readable, _, _ = select.select(watch, [], [], _POLL)
        if sending and local in readable:
            data = local.recv(_CHUNK)
            if data:
                channel.sendall(data)
            else:
                sending = False
                channel.shutdown_write()
        if receiving and channel in readable:
            data = channel.recv(_CHUNK)
            if data:
                local.sendall(data)
            else:
                receiving = False
                with contextlib.suppress(OSError):
                    # Already gone if the peer closed first, which is not an error.
                    local.shutdown(socket.SHUT_WR)


@contextmanager
def routed(
    config: dict[str, Any], host_field: str = "host", port_field: str = "port"
) -> Iterator[dict[str, Any]]:
    """The config, with its host and port pointed at a tunnel when one is configured.

    A connector wraps its work in this and is otherwise unchanged: with no tunnel the
    config is handed back as it stands, and with one the host and port name the local
    end of the forward. That keeps the tunnel out of every individual driver.
    """
    tunnel = SshTunnel.from_config(config)
    if tunnel is None:
        yield config
        return
    problems = tunnel.check()
    if problems:
        raise ValueError("; ".join(problems))
    remote_host = str(config.get(host_field) or "")
    remote_port = int(config.get(port_field) or 0)
    with tunnel.forward(remote_host, remote_port) as (local_host, local_port):
        yield {**config, host_field: local_host, port_field: local_port}


@contextmanager
def routed_url(
    config: dict[str, Any], url_field: str, default_port: int
) -> Iterator[dict[str, Any]]:
    """As :func:`routed`, for a connector addressed by a URL rather than by fields.

    The URL's host and port are what gets forwarded; everything else about it -- the
    scheme, any credentials, the path -- is put back unchanged, so the driver receives
    the same URL pointed somewhere else.

    Note what this cannot do. A URL naming several hosts, or one whose scheme makes the
    driver look more up in DNS, describes a set of servers rather than an address; a
    forward to one of them is not a forward to the set, and the driver would leave the
    tunnel as soon as it learned the others. Callers that accept such a URL check for it
    themselves, because only they know which form theirs takes.
    """
    from urllib.parse import urlsplit, urlunsplit

    tunnel = SshTunnel.from_config(config)
    if tunnel is None:
        yield config
        return
    problems = tunnel.check()
    if problems:
        raise ValueError("; ".join(problems))

    parts = urlsplit(str(config.get(url_field) or ""))
    if not parts.hostname:
        raise ValueError(f"{url_field} does not name a host to tunnel to")
    remote_port = parts.port or default_port
    with tunnel.forward(parts.hostname, remote_port) as (local_host, local_port):
        credentials = ""
        if parts.username:
            credentials = parts.username
            if parts.password:
                credentials += f":{parts.password}"
            credentials += "@"
        rewritten = parts._replace(netloc=f"{credentials}{local_host}:{local_port}")
        yield {**config, url_field: urlunsplit(rewritten)}
