"""camelCase the JSON API boundary, so the wire has one field-name convention.

The app's REST API emits and accepts camelCase field names, matching the camelCase spec
manifests it already serves (Open Derivation / Metric Spec) and the TypeScript client,
instead of the historical mix of snake_case and camelCase.

Three kinds of payload are governed by another standard or are opaque user data, and are
left byte-for-byte: OSI documents (snake_case by the OSI spec), notebook documents
(nbformat: ``cell_type``, ``output_type``, …), and the values of tabular-data keys
(``rows``, ``variables``, ``claims``, ``value``) whose keys are user column names or an
identity provider's claim names, not API fields.
Route-level exemptions handle the first two; the key set handles the third. Streaming
responses (the chat SSE) and non-JSON bodies are passed through untouched by the
middleware that uses these helpers.

The transform is conservative and idempotent: only a clearly snake_case token (lowercase
with at least one underscore) is rewritten, so already-camel keys, MIME types
(``text/plain``), and dotted keys are never touched; and ``camelize``/``snakeify`` are
inverses on the field names they apply to.
"""

from __future__ import annotations

import json
import re
from typing import Any

from starlette.types import ASGIApp, Message, Receive, Scope, Send

#: A key that is unambiguously snake_case (lowercase, digits, at least one underscore).
_SNAKE = re.compile(r"^[a-z][a-z0-9]*(?:_[a-z0-9]+)+$")
#: A key that is unambiguously camelCase (a lowercase head then an uppercase hump).
_CAMEL = re.compile(r"^[a-z][a-z0-9]*(?:[A-Z][a-z0-9]*)+$")

#: Keys whose values are opaque data (row dicts keyed by user column names, live kernel
#: variables, run-history cells keyed by asset name, a derivation artifact's ``value``):
#: the value is copied verbatim rather than recursed into, so its data keys survive the
#: API's casing boundary intact. ``value`` is the artifact a dashboard tile renders; the
#: API's other ``value`` fields hold scalars, which the transform never rewrites anyway.
OPAQUE_KEYS = frozenset({"rows", "variables", "cells", "claims", "value"})


def _to_camel(key: str) -> str:
    if not _SNAKE.match(key):
        return key
    head, *rest = key.split("_")
    return head + "".join(word[:1].upper() + word[1:] for word in rest)


def _to_snake(key: str) -> str:
    if not _CAMEL.match(key):
        return key
    return re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()


def _transform(value: Any, rename: Any) -> Any:
    if isinstance(value, dict):
        out: dict[Any, Any] = {}
        for key, val in value.items():
            if isinstance(key, str):
                out[rename(key)] = (
                    val if key in OPAQUE_KEYS else _transform(val, rename)
                )
            else:  # non-string keys are never field names; leave key and value as-is
                out[key] = val
        return out
    if isinstance(value, list):
        return [_transform(item, rename) for item in value]
    return value


def camelize(value: Any) -> Any:
    """Rewrite snake_case object keys to camelCase (opaque-key values left verbatim)."""
    return _transform(value, _to_camel)


def snakeify(value: Any) -> Any:
    """Rewrite camelCase object keys to snake_case (the inverse of :func:`camelize`)."""
    return _transform(value, _to_snake)


#: Routes whose JSON follows an external wire standard and must pass byte-for-byte, so
#: they keep that standard's field convention rather than the app's camelCase: the OSI
#: export (snake_case by OSI), notebook documents (nbformat: ``cell_type``), and the
#: chat/conversation surface (the Vercel AI SDK UI Message Stream, whose SSE parts and
#: persisted ``result`` payloads must match, the SSE itself is never buffered anyway).
_STANDARD_ROUTES = (
    "/api/metrics/osi",
    "/api/notebooks",
    "/api/chat",
    "/api/conversations",
    # Shares the conversations envelope, so it must share its spelling: two endpoints
    # naming the same cursor field differently is a trap for one client.
    "/api/search",
    "/api/messages",
    "/api/certificates",
    # Export documents embed a certificate's signed bytes verbatim (IP-31); a
    # camelCase rewrite here would silently invalidate every signature it touches.
    "/api/exports",
)

#: Request bodies left verbatim on the way in: spec manifests (already camelCase, parsed
#: by ``from_manifest``) and provider-specific config passed through to a connector. The
#: app-shaped bodies on these routes (e.g. a metric query) snake-case themselves.
_VERBATIM_REQUEST_ROUTES = (
    *_STANDARD_ROUTES,
    "/api/metrics",
    "/api/dashboards",
    "/api/features",
    "/api/data-sources",
    "/api/warehouse",
    "/api/secrets",
)


def _content_type(headers: list[tuple[bytes, bytes]]) -> str:
    for key, value in headers:
        if key.lower() == b"content-type":
            return value.decode("latin-1").lower()
    return ""


class CamelCaseResponses:
    """Rewrite JSON API response bodies to camelCase field names.

    JSON responses under ``/api`` are camelCased; streaming responses (the chat SSE),
    non-JSON bodies, the SPA, mounts, and the standard-governed routes pass through
    untouched. Buffering happens only for a JSON body, so SSE is never held up.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Rewrite an eligible ``/api`` JSON response body to camelCase keys."""
        path = scope.get("path", "")
        if (
            scope["type"] != "http"
            or not path.startswith("/api")
            or path.startswith(_STANDARD_ROUTES)
        ):
            await self._app(scope, receive, send)
            return

        state: dict[str, Any] = {"json": False, "start": None, "body": bytearray()}

        async def send_wrapper(message: Message) -> None:
            if message["type"] == "http.response.start":
                if _content_type(message.get("headers", [])).startswith(
                    "application/json"
                ):
                    state["json"] = True
                    state["start"] = message  # deferred until the body is assembled
                    return
                await send(message)
            elif message["type"] == "http.response.body" and state["json"]:
                state["body"].extend(message.get("body", b""))
                if message.get("more_body"):
                    return
                await _flush_camelized(send, state["start"], bytes(state["body"]))
            else:
                await send(message)

        await self._app(scope, receive, send_wrapper)


async def _flush_camelized(send: Send, start: Message, raw: bytes) -> None:
    try:
        body = json.dumps(camelize(json.loads(raw)), default=str).encode()
    except (ValueError, TypeError):  # not JSON after all: send verbatim
        body = raw
    headers = [
        (key, value)
        for key, value in start.get("headers", [])
        if key.lower() != b"content-length"
    ]
    headers.append((b"content-length", str(len(body)).encode()))
    start["headers"] = headers
    await send(start)
    await send({"type": "http.response.body", "body": body, "more_body": False})


class SnakeCaseRequests:
    """Snake-case JSON request bodies so handlers keep their snake_case field access.

    The frontend sends camelCase; this converts it back on the way in for app-shaped
    bodies. Spec-manifest and provider-config routes are left verbatim (they are parsed
    as camelCase manifests or passed through to a connector), as are non-JSON bodies.
    """

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Rewrite an eligible ``/api`` JSON request body to snake_case keys."""
        path = scope.get("path", "")
        if (
            scope["type"] != "http"
            or scope.get("method") not in ("POST", "PUT", "PATCH")
            or not path.startswith("/api")
            or path.startswith(_VERBATIM_REQUEST_ROUTES)
        ):
            await self._app(scope, receive, send)
            return

        body = bytearray()
        more = True
        while more:
            message = await receive()
            if message["type"] != "http.request":
                # a disconnect before the body finished; let the app see it
                await self._app(scope, receive, send)
                return
            body.extend(message.get("body", b""))
            more = message.get("more_body", False)

        payload = bytes(body)
        if body and _content_type(scope.get("headers", [])).startswith(
            "application/json"
        ):
            try:
                payload = json.dumps(snakeify(json.loads(body))).encode()
            except (ValueError, TypeError):
                payload = bytes(body)

        served = False

        async def receive_wrapper() -> Message:
            nonlocal served
            if not served:
                served = True
                return {"type": "http.request", "body": payload, "more_body": False}
            return {"type": "http.request", "body": b"", "more_body": False}

        await self._app(scope, receive_wrapper, send)
