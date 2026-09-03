"""The egress allowlist proxy, at its behavior boundary: which hosts it permits, and how
it answers a CONNECT. The allow/refuse decision is the security-critical part, tested
directly and over a real socket pair for the two refusal paths."""

from __future__ import annotations

import socket

from elbi_core import _egress_proxy


def test_allowed_matches_exact_and_dot_suffix() -> None:
    assert _egress_proxy._allowed("pypi.org", ["pypi.org"])  # exact
    assert _egress_proxy._allowed("a.pythonhosted.org", ["pythonhosted.org"])  # suffix
    assert _egress_proxy._allowed("PyPI.org", ["pypi.org"])  # case-insensitive
    assert not _egress_proxy._allowed("evil.com", ["pypi.org"])
    # a suffix match must be dot-bounded, so a shared tail is not a match
    assert not _egress_proxy._allowed("notpypi.org", ["pypi.org"])
    assert not _egress_proxy._allowed("pypi.org", [])  # empty allowlist permits nothing


def test_handle_refuses_a_non_connect_request() -> None:
    client, proxy = socket.socketpair()
    try:
        client.sendall(b"GET / HTTP/1.1\r\n\r\n")
        _egress_proxy._handle(proxy, ["pypi.org"])  # closes proxy on the refusal path
        assert b"405" in client.recv(200)  # only CONNECT tunnels are supported
    finally:
        client.close()


def test_handle_refuses_a_disallowed_host() -> None:
    client, proxy = socket.socketpair()
    try:
        client.sendall(b"CONNECT evil.com:443 HTTP/1.1\r\n\r\n")
        _egress_proxy._handle(proxy, ["pypi.org"])  # closes proxy on the refusal path
        assert b"403" in client.recv(200)  # not on the allowlist
    finally:
        client.close()
