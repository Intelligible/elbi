"""One HTTP client for every command that talks to a running app.

Shared rather than per-command so authentication cannot be a property of which verb you
happened to type: a command group with its own client supports only whatever credential
its author remembered, and against a real deployment that reads as an unexplained 401.

An app on this machine needs no credential at all, which is the usual case. Where one
is needed, an explicit ``--token`` or ``ELBI_API_KEY`` covers it. A plugin that knows
how to obtain credentials some other way supplies them through
:func:`set_auth_provider`.
"""

from __future__ import annotations

import os
from collections.abc import Callable

import httpx

#: Where a command looks when no target and no ``--url`` say otherwise.
DEFAULT_URL = "http://127.0.0.1:7700"

#: A run can take as long as the work does; a configuration call should not hang.
RUN_TIMEOUT = 600.0
CONFIG_TIMEOUT = 60.0

#: Returns the authentication to use for a base URL, or ``None`` to send none.
AuthProvider = Callable[[str], "httpx.Auth | None"]

_provider: AuthProvider | None = None


class LoginRequired(Exception):
    """The app refused the credential and the CLI cannot obtain a new one itself.

    Raised by an installed auth provider whose stored credential has expired and could
    not be renewed. Commands catch it to print the message rather than a stack trace, so
    the text a provider raises with is what the person reads.
    """


def set_auth_provider(provider: AuthProvider | None) -> None:
    """Install ``provider`` as the source of credentials this CLI cannot obtain itself.

    Called by a plugin during startup. Passing ``None`` restores the default, which is
    to send an explicit token if one was given and otherwise nothing.
    """
    global _provider
    _provider = provider


def resolve_url(url: str | None) -> str:
    """The app a command should talk to: the argument, the environment, or localhost."""
    return (url or os.environ.get("ELBI_URL") or DEFAULT_URL).rstrip("/")


def client_for(
    url: str | None, token: str | None, *, timeout: float = CONFIG_TIMEOUT
) -> httpx.Client:
    """An HTTP client for the target app, carrying whatever credential applies.

    Precedence: an explicit ``--token``, then the environment, then whatever an
    installed provider holds for that host. The provider comes last so neither of the
    first two stops working, and it is asked *by host*, so a credential cannot be sent
    anywhere but the app it was issued for.
    """
    base = resolve_url(url)
    given = token or os.environ.get("ELBI_API_KEY")
    if given:
        return httpx.Client(
            base_url=base,
            headers={"Authorization": f"Bearer {given}"},
            timeout=timeout,
        )
    auth = _provider(base) if _provider is not None else None
    return httpx.Client(base_url=base, auth=auth, timeout=timeout)
