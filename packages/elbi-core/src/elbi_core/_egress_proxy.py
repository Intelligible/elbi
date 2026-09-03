"""A minimal filtering HTTPS-CONNECT proxy that enforces an egress allowlist.

Not a public API. Run inside a small sidecar container next to an exploration sandbox
that has no direct internet (an ``--internal`` Docker network): the sandbox's
``HTTPS_PROXY``/``HTTP_PROXY`` point here, so every outbound connection is a ``CONNECT``
this proxy sees. It permits a connection only when the target host matches the allowlist
(exact host or a ``.suffix`` domain match) and refuses everything else, which is how an
egress allowlist is enforced when the platform cannot set per-host firewall rules.

Stdlib only, so it runs by file path in a bare container. The allowlist is the argv.
"""

from __future__ import annotations

import contextlib
import socket
import sys
import threading

#: Where the proxy listens inside the sidecar; the sandbox points HTTPS_PROXY here.
_PORT = 8888
_BUFFER = 65536


def _allowed(host: str, allowlist: list[str]) -> bool:
    """Whether ``host`` is on the allowlist (exact, or a dot-suffix domain match)."""
    host = host.lower().strip(".")
    for entry in allowlist:
        entry = entry.lower().strip(".")
        if host == entry or host.endswith("." + entry):
            return True
    return False


def _pipe(src: socket.socket, dst: socket.socket) -> None:
    """Copy bytes one way until the source closes; best-effort, errors end the copy."""
    try:
        while True:
            chunk = src.recv(_BUFFER)
            if not chunk:
                break
            dst.sendall(chunk)
    except OSError:
        pass
    finally:
        for sock in (src, dst):
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)


def _handle(client: socket.socket, allowlist: list[str]) -> None:
    """Serve one client: parse its CONNECT, allow or refuse, then tunnel."""
    try:
        request = b""
        while b"\r\n\r\n" not in request:
            chunk = client.recv(_BUFFER)
            if not chunk:
                client.close()
                return
            request += chunk
        line = request.split(b"\r\n", 1)[0].decode("latin-1")
        parts = line.split()
        # Only CONNECT (HTTPS tunnels) is supported; pip and apt use it for TLS.
        if len(parts) < 2 or parts[0].upper() != "CONNECT":
            client.sendall(b"HTTP/1.1 405 Method Not Allowed\r\n\r\n")
            client.close()
            return
        host, _, port_s = parts[1].partition(":")
        if not _allowed(host, allowlist):
            client.sendall(b"HTTP/1.1 403 Forbidden\r\n\r\n")
            client.close()
            return
        upstream = socket.create_connection((host, int(port_s or 443)), timeout=30)
        client.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        threading.Thread(target=_pipe, args=(client, upstream), daemon=True).start()
        _pipe(upstream, client)
    except OSError:
        with contextlib.suppress(OSError):
            client.close()


def main(allowlist: list[str]) -> int:
    """Serve the proxy until killed, permitting CONNECT only to ``allowlist`` hosts."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("0.0.0.0", _PORT))  # noqa: S104 - the sidecar is isolated by design
    listener.listen(64)
    while True:
        client, _ = listener.accept()
        threading.Thread(target=_handle, args=(client, allowlist), daemon=True).start()


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess/container
    raise SystemExit(main(sys.argv[1:]))
