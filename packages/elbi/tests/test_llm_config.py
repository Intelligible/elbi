"""The LLM client is resolved from explicit configuration only: no silent fallbacks.

There is no built-in default model and no fail-over list: with nothing configured the
app still boots (so a profile can be added in the UI), and a surface that needs the LLM
answers with a clear, actionable error rather than silently picking a model or key.
"""

from __future__ import annotations

import inspect

from fastapi.testclient import TestClient

from elbi import create_app
from elbi.serve import _build_client
from elbi_core.errors import ElbiError


def _unconfigured_make_client(_profile: str | None = None):
    # Stands in for serve.resolve_client when no model is configured: it raises rather
    # than returning a default client.
    raise ElbiError("No LLM model is configured. Add a model profile.")


def test_app_boots_unconfigured_and_chat_errors_clearly() -> None:
    # With no client and a factory that raises, the server must still start (a plain GET
    # works), and a chat turn must fail with a clean 400 naming the fix, not a 500, and
    # not a silent default model.
    app = create_app(
        load_datasets=lambda: {},
        client=None,
        make_client=_unconfigured_make_client,
    )
    with TestClient(app) as http:
        assert http.get("/api/models").status_code == 200  # booted fine
        turn = http.post(
            "/api/chat",
            json={
                "messages": [
                    {"role": "user", "parts": [{"type": "text", "text": "hi"}]}
                ]
            },
        )
        assert turn.status_code == 400
        assert "no llm model is configured" in turn.json()["detail"].lower()


def test_build_client_has_no_fallback_parameter() -> None:
    # The fail-over ("fallbacks") mechanism was removed; the client builder must no
    # longer accept it, so a stray LLM_FALLBACKS can never reintroduce silent fail-over.
    # Asserted as an absence rather than an exact signature, so adding a legitimate
    # parameter does not fail a test that is about fail-over.
    params = set(inspect.signature(_build_client).parameters)
    assert {"model", "api_key", "base_url"} <= params
    assert not [p for p in params if "fallback" in p.lower()]
