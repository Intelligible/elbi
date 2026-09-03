"""The /api/chat endpoint is tested with a scripted LLM, so no network or key is needed.

It asserts the streamed contract in the AI SDK UI Message Stream format the `useChat`
hook consumes: the agent's steps arrive as custom data parts, the answer as a text part,
and the verification payload as a data-result part carrying a verified, sound conclusion
(the same enforcement the runtime guarantees) and the run persists as a conversation
with its authored derivation in the native store.
"""

from __future__ import annotations

import json
import random
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from fastapi.testclient import TestClient
from mcp.server.mcpserver import Context as McpContext

from elbi import create_app
from elbi.db import Derivation, Store, open_store
from elbi_agent import (
    DeriveFn,
    DeriveOutcome,
    Step,
    ToolCall,
    ToolSpec,
    Transcript,
)
from elbi_core import verify_all
from elbi_core.versioning import hash_json

N = 400


def _derive_factory(
    store: Store, rows: list[dict[str, str]]
) -> Callable[[str, str], DeriveFn]:
    """Derive factory that runs the real oracle and persists (stands in for author)."""

    def factory(conversation_id: str, question: str) -> DeriveFn:
        def derive(
            name: str,
            source: str,
            claim: Mapping[str, Any] | None,
            contract: Mapping[str, Any] | None,
            fmt: str,
            assumptions: Sequence[str],
            deps: Sequence[str],
        ) -> DeriveOutcome:
            report = verify_all(rows, **dict(claim or {}))
            checks = tuple((g.name, g.verdict, g.detail) for g in report.ran)
            if report.verdict != "sound":
                return DeriveOutcome(
                    False, report.verdict, "output", checks, detail=report.render()
                )
            store.save_derivation(
                Derivation(
                    name=name,
                    conversation_id=conversation_id,
                    question=question,
                    source=source,
                    verdict="sound",
                    rendered="| effect |\n| 0.6 |",
                    data_hash=hash_json(rows),
                )
            )
            return DeriveOutcome(
                True,
                "sound",
                "output",
                checks,
                hash_json(rows),
                # forward the oracle's own estimate, as the real derive bridge does
                estimate=report.estimate,
                estimate_label=report.estimate_label,
                adjusted_for=report.adjusted_for,
            )

        return derive

    return factory


def _robust() -> list[dict[str, str]]:
    # A real effect (x raises y by ~0.6) that survives its adjustment set: w refines
    # the estimate without reversing it, so the controlled report is verified sound.
    rng = random.Random(0)
    w = [rng.gauss(0, 1) for _ in range(N)]
    x = [0.5 * wi + rng.gauss(0, 1) for wi in w]
    y = [0.6 * xi + 0.4 * wi + rng.gauss(0, 0.5) for xi, wi in zip(x, w, strict=True)]
    return [{"x": str(x[i]), "y": str(y[i]), "w": str(w[i])} for i in range(N)]


class _Scripted:
    """Replays steps in order, or by what the turn was asked.

    ``when`` maps a substring of a turn's prompt to the step that turn gets. It exists
    because a background job's completion runs a *second* agent loop concurrently with
    the request that started it: with only an ordered script the two loops race for the
    same index, and which turn gets which answer depends on how fast the job finished.
    Dispatching on the prompt gives each turn its own script whatever the timing.

    A ``when`` value may also be an exception, which that turn raises instead of
    stepping -- how a test makes one specific turn fail without the failure landing on
    whichever turn happens to run first.
    """

    def __init__(
        self, *steps: Step, when: dict[str, Step | Exception] | None = None
    ) -> None:
        self._steps = list(steps)
        self._when = dict(when or {})
        self._i = 0
        self._lock = threading.Lock()

    def step(self, transcript: Transcript, tools: Sequence[ToolSpec]) -> Step:
        asked = " ".join(
            block.get("text", "")
            for message in transcript.messages
            if message.get("role") == "user"
            for block in message.get("content", [])
            if isinstance(block, dict)
        )
        for marker, step in self._when.items():
            if marker in asked:
                if isinstance(step, Exception):
                    raise step
                return step
        with self._lock:
            step = self._steps[self._i]
            self._i += 1
            return step


def _call(name: str, **arguments: object) -> ToolCall:
    return ToolCall(id=f"c-{name}", name=name, arguments=dict(arguments))


def _chunks(body: str) -> list[dict[str, Any]]:
    """Parse an AI SDK UI Message Stream body into its JSON chunks (skips [DONE])."""
    out: list[dict[str, Any]] = []
    for line in body.splitlines():
        if line.startswith("data:"):
            payload = line.split(":", 1)[1].strip()
            if payload and payload != "[DONE]":
                out.append(json.loads(payload))
    return out


def _derive_call(name: str, **claim: object) -> ToolCall:
    return ToolCall(
        id="c-derive",
        name="derive",
        arguments={"name": name, "source": "def d(ctx): ...", "claim": dict(claim)},
    )


def test_chat_streams_the_ai_sdk_protocol(tmp_path: object) -> None:
    from pathlib import Path

    rows = _robust()
    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'app.db'}")
    client = _Scripted(
        Step(tool_calls=(_call("structure_map", dataset="d"),)),
        Step(tool_calls=(_derive_call("x_on_y", x="x", y="y", controls=["w"]),)),
        Step(tool_calls=(_call("answer", summary="x raises y, holding w fixed."),)),
    )
    app = create_app(
        load_datasets=lambda: {"d": rows},
        client=client,
        store=store,
        derive_factory=_derive_factory(store, rows),
    )
    with TestClient(app) as http:
        # the request is an AI SDK `useChat` payload (messages with text parts)
        response = http.post(
            "/api/chat",
            json={
                "messages": [
                    {"role": "user", "parts": [{"type": "text", "text": "effect?"}]}
                ]
            },
        )
        assert response.status_code == 200
        assert response.headers["x-vercel-ai-ui-message-stream"] == "v1"
        assert response.text.rstrip().endswith("[DONE]")

        chunks = _chunks(response.text)
        types = [c["type"] for c in chunks]
        assert types[0] == "start" and "finish" in types
        # the agent's steps arrive as custom data parts
        steps = [c["data"]["name"] for c in chunks if c["type"] == "data-tool"]
        assert "derive" in steps
        # the answer is a text part: the model's own words, then a system-rendered
        # certified-findings block carrying the oracle's OWN estimate (so the reported
        # magnitude is the certified one, not a figure the model wrote while exploring)
        text = "".join(c["delta"] for c in chunks if c["type"] == "text-delta")
        assert text.startswith("x raises y, holding w fixed.")
        assert "Certified findings" in text and "per +1 unit of x" in text
        # the verification payload rides in a custom data-result part
        result = next(c["data"] for c in chunks if c["type"] == "data-result")
        assert result["verified"] is True and result["verdict"] == "sound"
        assert result["spec"]["derivation"] == "x_on_y"
        assert any(c["name"] == "effect" for c in result["checks"])

        # the run persisted: a conversation, its two turns, and the authored derivation
        derivations = http.get("/api/derivations").json()
        assert [d["name"] for d in derivations] == ["x_on_y"]
        convos = http.get("/api/conversations").json()["items"]
        messages = http.get(f"/api/conversations/{convos[0]['id']}/messages").json()
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[1]["result"]["verified"] is True


def test_the_openapi_schema_generates() -> None:
    """FastAPI builds this from every response model, so one bad model breaks it all.

    Nothing asked for it, and the failure is a 500 on a route no test drove: the schema
    is what an SDK generator and the docs page read.
    """

    class _Idle:
        def step(self, transcript: object, tools: object) -> object:
            raise NotImplementedError

    app = create_app(load_datasets=lambda: {"d": []}, client=_Idle())
    with TestClient(app) as http:
        schema = http.get("/openapi.json")
        assert schema.status_code == 200, schema.text
        paths = schema.json()["paths"]
        assert "/api/derivations" in paths
        assert "/api/secrets" in paths


def test_chat_turn_failure_finishes_with_a_visible_error(tmp_path: object) -> None:
    # An LLM or tool failure mid-turn must not abort the stream truncated; the stream
    # finishes with an unverified error result the client can show, not a silent hang.
    class _Boom:
        def step(self, transcript: Transcript, tools: Sequence[ToolSpec]) -> Step:
            raise RuntimeError("the model backend is unavailable")

    app = create_app(load_datasets=lambda: {"d": []}, client=_Boom())
    with TestClient(app) as http:
        response = http.post("/api/chat", json=_chat_body("c1", "effect?"))
        assert response.status_code == 200
        assert response.text.rstrip().endswith("[DONE]")  # stream terminated cleanly
        chunks = _chunks(response.text)
        assert any(c["type"] == "finish" for c in chunks)
        result = next(c["data"] for c in chunks if c["type"] == "data-result")
        assert result["verified"] is False
        assert "server error" in result["narrative"]


def test_failed_turn_is_persisted_not_orphaned(tmp_path: object) -> None:
    # A turn that errors must still record an assistant reply, so reopening the
    # conversation shows the question AND that it failed, not the question alone.
    from pathlib import Path

    class _Boom:
        def step(self, transcript: Transcript, tools: Sequence[ToolSpec]) -> Step:
            raise RuntimeError("the model backend is unavailable")

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'orphan.db'}")
    app = create_app(load_datasets=lambda: {"d": []}, client=_Boom(), store=store)
    with TestClient(app) as http:
        assert (
            http.post("/api/chat", json=_chat_body("c1", "effect?")).status_code == 200
        )
        msgs = http.get("/api/conversations/c1/messages").json()
        assert [m["role"] for m in msgs] == ["user", "assistant"]  # not orphaned
        assert "server error" in msgs[1]["content"]
        assert msgs[1]["result"]["verdict"] == "unverified"


def test_repo_derivation_detail_shows_source_and_renders_output(
    tmp_path: object,
) -> None:
    # A repo (human-authored) derivation is mirrored into the store with its source but
    # no stored rendering; the detail endpoint must show the source and render the
    # output on demand, rather than a blank page (regression: repo rows showed blank).
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'repo.db'}")
    store.save_derivation(
        Derivation(
            name="regional_orders",
            question="Project the orders table to the fields metrics aggregate.",
            source="def regional_orders(ctx):\n    return ctx.input('orders')\n",
            origin="repo",
            rendered="",  # repo rows carry no stored rendering
        )
    )
    app = create_app(
        load_datasets=dict,
        store=store,
        render_derivation=lambda name: "| region | orders |\n| west | 3 |",
    )
    with TestClient(app) as http:
        detail = http.get("/api/derivations/regional_orders").json()
    assert "def regional_orders" in detail["source"]  # source is no longer blank
    assert "west" in detail["rendered"]  # output rendered on demand
    assert detail["claim"] is None  # a human derivation has no oracle claim


def test_repo_derivation_detail_omits_output_when_unrenderable(
    tmp_path: object,
) -> None:
    # When the output cannot be produced (unbound data, internal derivation), the detail
    # leaves the rendering empty rather than failing: the view omits it.
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'repo2.db'}")
    store.save_derivation(
        Derivation(
            name="unbound", question="x", source="def unbound(ctx): ...", origin="repo"
        )
    )
    app = create_app(
        load_datasets=dict, store=store, render_derivation=lambda name: None
    )
    with TestClient(app) as http:
        detail = http.get("/api/derivations/unbound").json()
    assert detail["rendered"] == ""


def _tool_outputs(chunks: list[dict[str, Any]]) -> list[str]:
    """The run_code/tool outputs streamed to the client as data-tool-output parts."""
    return [c["data"]["output"] for c in chunks if c["type"] == "data-tool-output"]


def test_chat_run_code_workspace_persists_within_a_conversation(
    tmp_path: object,
) -> None:
    # End to end through /api/chat: a chat writes a file in one run_code call and reads
    # it back in the next, so exploration output carries across calls (a fitted model,
    # an intermediate table). The persisted value reaches the second tool's output.
    from pathlib import Path

    scratch_root = Path(str(tmp_path)) / "workspaces"
    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'app.db'}")
    client = _Scripted(
        Step(tool_calls=(_call("run_code", code="open('m.txt','w').write('V7')"),)),
        Step(tool_calls=(_call("run_code", code="result = open('m.txt').read()"),)),
        Step(tool_calls=(_call("answer", summary="reused the saved artifact."),)),
    )
    app = create_app(
        load_datasets=dict, client=client, store=store, scratch_root=scratch_root
    )
    with TestClient(app) as http:
        response = http.post(
            "/api/chat",
            json={
                "messages": [
                    {"role": "user", "parts": [{"type": "text", "text": "reuse it"}]}
                ]
            },
        )
        assert response.status_code == 200
        outputs = _tool_outputs(_chunks(response.text))
        # the second run_code read back what the first wrote
        assert any("V7" in out for out in outputs)

        # the workspace lives under scratch_root, keyed by the conversation id
        convo_id = http.get("/api/conversations").json()["items"][0]["id"]
        assert (scratch_root / convo_id).is_dir()


def test_chat_run_code_keeps_variables_across_calls(tmp_path: object) -> None:
    # Through /api/chat: the chat's run_code is stateful, so a variable from one call
    # is still defined in the next (a model stays in memory), not just files on disk.
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'app.db'}")
    client = _Scripted(
        Step(tool_calls=(_call("run_code", code="trained = 123"),)),
        Step(tool_calls=(_call("run_code", code="result = trained + 1"),)),
        Step(tool_calls=(_call("answer", summary="reused the in-memory value."),)),
    )
    app = create_app(load_datasets=dict, client=client, store=store)
    with TestClient(app) as http:
        response = http.post(
            "/api/chat",
            json={
                "messages": [
                    {"role": "user", "parts": [{"type": "text", "text": "reuse"}]}
                ]
            },
        )
        outputs = _tool_outputs(_chunks(response.text))
        assert any("124" in out for out in outputs)  # trained persisted across calls


def test_chat_workspaces_are_isolated_per_conversation(tmp_path: object) -> None:
    # Each conversation gets its own workspace, so one chat's scratch never leaks into
    # another: the second conversation cannot read a file the first wrote.
    from pathlib import Path

    scratch_root = Path(str(tmp_path)) / "workspaces"
    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'app.db'}")
    client = _Scripted(
        # conversation A: write a file, then answer
        Step(tool_calls=(_call("run_code", code="open('shared.txt','w').write('A')"),)),
        Step(tool_calls=(_call("answer", summary="wrote it."),)),
        # conversation B: try to read the same name, then answer
        Step(tool_calls=(_call("run_code", code="open('shared.txt').read()"),)),
        Step(tool_calls=(_call("answer", summary="tried to read it."),)),
    )
    app = create_app(
        load_datasets=dict, client=client, store=store, scratch_root=scratch_root
    )
    with TestClient(app) as http:
        msg = {"messages": [{"role": "user", "parts": [{"type": "text", "text": "x"}]}]}
        http.post("/api/chat", json=msg)  # conversation A writes shared.txt
        second = http.post("/api/chat", json=msg)  # conversation B, a fresh workspace
        outputs = _tool_outputs(_chunks(second.text))
        # B's read fails: the file is in A's workspace, not B's
        assert any("FileNotFoundError" in out for out in outputs)


def test_chat_forwards_declared_deps_to_the_derive_factory(tmp_path: object) -> None:
    # Through the HTTP API: the packages the model declares for its derivation must
    # reach the authoring layer, or a certified model that imports numpy/sklearn is
    # impossible (the derivation sandbox is stdlib-only).
    from pathlib import Path

    seen: dict[str, Sequence[str]] = {}

    def factory(conversation_id: str, question: str) -> DeriveFn:
        def derive(
            name: str,
            source: str,
            claim: Mapping[str, Any] | None,
            contract: Mapping[str, Any] | None,
            fmt: str,
            assumptions: Sequence[str],
            deps: Sequence[str],
        ) -> DeriveOutcome:
            seen["deps"] = tuple(deps)
            return DeriveOutcome(True, "sound", "ok", (), hash_json([{"v": 1}]))

        return derive

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'app.db'}")
    client = _Scripted(
        Step(
            tool_calls=(
                ToolCall(
                    id="c-derive",
                    name="derive",
                    arguments={
                        "name": "m",
                        "source": "def m(ctx): ...",
                        "claim": {"x": "x", "y": "y"},
                        "deps": ["scikit-learn"],
                    },
                ),
            )
        ),
        Step(tool_calls=(_call("answer", summary="done"),)),
    )
    app = create_app(
        load_datasets=lambda: {"d": [{"x": "1", "y": "2"}]},
        client=client,
        store=store,
        derive_factory=factory,
    )
    with TestClient(app) as http:
        http.post(
            "/api/chat",
            json={
                "messages": [
                    {"role": "user", "parts": [{"type": "text", "text": "e?"}]}
                ]
            },
        )
    assert seen["deps"] == ("scikit-learn",)


def _chat_body(conversation_id: str, text: str) -> dict[str, Any]:
    return {
        "conversationId": conversation_id,
        "messages": [{"role": "user", "parts": [{"type": "text", "text": text}]}],
    }


def test_chat_continues_the_conversation_across_turns(tmp_path: object) -> None:
    # A second turn with the same conversation id reuses the conversation and replays
    # the prior turn (question + answer) into the model's context, so a follow-up works.
    from pathlib import Path

    class _Recording(_Scripted):
        def __init__(self, *steps: Step) -> None:
            super().__init__(*steps)
            self.seen: Transcript | None = None

        def step(self, transcript: Transcript, tools: Sequence[ToolSpec]) -> Step:
            self.seen = transcript
            return super().step(transcript, tools)

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'app.db'}")
    client = _Recording(
        Step(tool_calls=(_call("answer", summary="trained xgboost, R2 0.9"),)),
        Step(tool_calls=(_call("answer", summary="tuned it"),)),
    )
    app = create_app(load_datasets=dict, client=client, store=store)
    with TestClient(app) as http:
        http.post("/api/chat", json=_chat_body("c1", "train a model"))
        http.post("/api/chat", json=_chat_body("c1", "now tune it"))
        assert client.seen is not None
        blob = "\n".join(str(m["content"]) for m in client.seen.messages)
        assert "train a model" in blob  # prior question replayed
        assert "trained xgboost, R2 0.9" in blob  # prior answer replayed
        assert "now tune it" in blob  # the new question
        # the conversation was reused, not recreated
        convos = http.get("/api/conversations").json()["items"]
        assert [c["id"] for c in convos] == ["c1"]


def test_chat_launches_a_background_training_job(tmp_path: object) -> None:
    # `derive` with background=true submits a durable job and the turn continues; the
    # job authors on a worker and is observable through /api/jobs until it certifies.
    import time
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'app.db'}")

    def derive_factory(conversation_id: str, question: str) -> DeriveFn:
        def derive(name, source, claim, contract, fmt, assumptions, deps):  # type: ignore[no-untyped-def]
            return DeriveOutcome(
                certified=True, verdict="sound", rendered="R2 0.9", data_hash="h"
            )

        return derive

    client = _Scripted(
        Step(
            tool_calls=(
                ToolCall(
                    id="c-derive",
                    name="derive",
                    arguments={
                        "name": "price_model",
                        "source": "def price_model(ctx): return []",
                        "background": True,
                    },
                ),
            )
        ),
        Step(tool_calls=(_call("answer", summary="training launched"),)),
    )
    app = create_app(
        load_datasets=lambda: {"houses": []},
        client=client,
        store=store,
        derive_factory=derive_factory,
    )
    with TestClient(app) as http:
        http.post("/api/chat", json=_chat_body("c1", "train a SOTA model"))
        listed = http.get("/api/jobs").json()
        assert len(listed) == 1
        job_id = listed[0]["id"]
        job = listed[0]
        for _ in range(100):
            job = http.get(f"/api/jobs/{job_id}").json()
            if job["state"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.02)
        assert job["state"] == "succeeded"
        assert job["result"]["certified"] is True
        assert job["result"]["verdict"] == "sound"


def test_claimless_background_job_followup_is_not_marked_verified(
    tmp_path: object,
) -> None:
    # AutoCertifyOnVerify certifies a background job on a clean run alone, even with
    # no claim declared (verdict stays None). The chat follow-up must not fabricate
    # "sound" or verified=True for a result nobody's claim was ever checked against.
    import time
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'app.db'}")

    def derive_factory(conversation_id: str, question: str) -> DeriveFn:
        def derive(name, source, claim, contract, fmt, assumptions, deps):  # type: ignore[no-untyped-def]
            return DeriveOutcome(certified=True, rendered="42", data_hash="h")

        return derive

    client = _Scripted(
        Step(
            tool_calls=(
                ToolCall(
                    id="c-derive",
                    name="derive",
                    arguments={
                        "name": "revenue",
                        "source": "def revenue(ctx): return []",
                        "background": True,
                    },
                ),
            )
        ),
        Step(tool_calls=(_call("answer", summary="training launched"),)),
    )
    app = create_app(
        load_datasets=lambda: {"houses": []},
        client=client,
        store=store,
        derive_factory=derive_factory,
    )
    with TestClient(app) as http:
        http.post("/api/chat", json=_chat_body("c1", "train a model"))
        listed = http.get("/api/jobs").json()
        job_id = listed[0]["id"]
        job = listed[0]
        for _ in range(100):
            job = http.get(f"/api/jobs/{job_id}").json()
            if job["state"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.02)
        assert job["state"] == "succeeded" and job["result"]["certified"] is True
        assert job["result"].get("verdict") is None  # nothing was claimed or checked

        followup = None
        for _ in range(100):
            msgs = http.get("/api/conversations/c1/messages").json()
            candidates = [
                m
                for m in msgs
                if m["role"] == "assistant" and m.get("result") is not None
            ]
            if candidates:
                followup = candidates[-1]
                break
            time.sleep(0.02)
        assert followup is not None
        assert followup["result"]["verified"] is False
        assert followup["result"]["verdict"] == "unverified"


def test_chat_launches_a_background_run_code_job(tmp_path: object) -> None:
    # run_code with background=true submits a job that runs in its own isolated session;
    # the turn continues and the job's stdout is observable through /api/jobs.
    import time
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'app.db'}")
    client = _Scripted(
        Step(
            tool_calls=(
                ToolCall(
                    id="c-rc",
                    name="run_code",
                    arguments={"code": "print('bg ran')", "background": True},
                ),
            )
        ),
        Step(tool_calls=(_call("answer", summary="launched exploration"),)),
    )
    app = create_app(load_datasets=dict, client=client, store=store)
    with TestClient(app) as http:
        http.post("/api/chat", json=_chat_body("c1", "run a long computation"))
        listed = http.get("/api/jobs").json()
        assert len(listed) == 1
        job_id = listed[0]["id"]
        job = listed[0]
        for _ in range(1200):
            job = http.get(f"/api/jobs/{job_id}").json()
            if job["state"] in ("succeeded", "failed", "cancelled"):
                break
            time.sleep(0.02)
        assert job["state"] == "succeeded"
        assert "bg ran" in (job["result"]["stdout"] or "")


def test_background_completion_posts_a_verified_followup(tmp_path: object) -> None:
    # When a background derivation job certifies, the agent is re-pinged: an off-request
    # turn narrates the certified result and a verified follow-up answer is posted back
    # to the conversation (no re-derive, and it carries the oracle's verdict).
    import time
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'app.db'}")

    def derive_factory(conversation_id: str, question: str) -> DeriveFn:
        def derive(name, source, claim, contract, fmt, assumptions, deps):  # type: ignore[no-untyped-def]
            return DeriveOutcome(
                certified=True,
                verdict="sound",
                rendered="held-out R2 0.90",
                data_hash="h",
                checks=(("prediction", "sound", "held-out R2 0.90"),),
            )

        return derive

    client = _Scripted(
        Step(
            tool_calls=(
                ToolCall(
                    id="c-d",
                    name="derive",
                    arguments={
                        "name": "price_model",
                        "source": "def price_model(ctx): return []",
                        "background": True,
                    },
                ),
            )
        ),
        Step(tool_calls=(_call("answer", summary="training launched"),)),
        # The off-request completion turn narrates the certified result. Keyed on its
        # prompt rather than on position: it runs concurrently with the request that
        # started it, so position is a race.
        when={
            "has finished and CERTIFIED": Step(
                tool_calls=(_call("answer", summary="The model predicts price well."),)
            )
        },
    )
    app = create_app(
        load_datasets=lambda: {"houses": []},
        client=client,
        store=store,
        derive_factory=derive_factory,
    )
    with TestClient(app) as http:
        http.post("/api/chat", json=_chat_body("c1", "train a model"))
        followups: list[dict[str, Any]] = []
        # Waits for the message this test is about, not for a count of them. Counting
        # can be satisfied by the launch note plus anything else that happens to land
        # first, which then fails on the `next` below -- a slow runner reports it as a
        # StopIteration in a test that looks like it is about narration.
        for _ in range(1200):
            msgs = http.get("/api/conversations/c1/messages").json()
            followups = [m for m in msgs if m["role"] == "assistant"]
            if any("predicts price" in f["content"] for f in followups):
                break
            time.sleep(0.05)
        assert len(followups) >= 2
        # Delivery order is not part of the contract: a fast job can post its
        # follow-up before the turn's own message persists. What matters is
        # that the certified follow-up exists and is verified.
        finding = next(f for f in followups if "predicts price" in f["content"])
        assert finding["result"]["verified"] is True
        assert finding["result"]["verdict"] == "sound"


def test_background_completion_falls_back_when_narration_fails(
    tmp_path: object,
) -> None:
    # If the off-request narration turn errors (an LLM or API failure), the promised
    # follow-up must not vanish silently: the certified result is posted verbatim.
    # Reverting to a blanket suppress leaves only the launch note, so this asserts the
    # follow-up survives a narration failure.
    import time
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'app.db'}")

    def derive_factory(conversation_id: str, question: str) -> DeriveFn:
        def derive(name, source, claim, contract, fmt, assumptions, deps):  # type: ignore[no-untyped-def]
            return DeriveOutcome(
                certified=True,
                verdict="sound",
                rendered="held-out R2 0.90",
                data_hash="h",
                checks=(("prediction", "sound", "held-out R2 0.90"),),
            )

        return derive

    client = _Scripted(
        Step(
            tool_calls=(
                ToolCall(
                    id="c-d",
                    name="derive",
                    arguments={
                        "name": "price_model",
                        "source": "def price_model(ctx): return []",
                        "background": True,
                    },
                ),
            )
        ),
        Step(tool_calls=(_call("answer", summary="training launched"),)),
        # The off-request completion turn raises, standing in for an LLM failure. Keyed
        # on its prompt rather than left off the end of the script: it runs concurrently
        # with the request that started it, so the turn that runs past the last step is
        # whichever one arrives second, which need not be this one.
        when={"has finished and CERTIFIED": RuntimeError("narration failed")},
    )
    app = create_app(
        load_datasets=lambda: {"houses": []},
        client=client,
        store=store,
        derive_factory=derive_factory,
    )
    with TestClient(app) as http:
        http.post("/api/chat", json=_chat_body("c1", "train a model"))
        followups: list[dict[str, Any]] = []
        # The verbatim fallback specifically, for the reason given in the sibling test
        # above: a count is satisfied by whichever two messages arrive first.
        for _ in range(1200):
            msgs = http.get("/api/conversations/c1/messages").json()
            followups = [m for m in msgs if m["role"] == "assistant"]
            if any("held-out R2 0.90" in f["content"] for f in followups):
                break
            time.sleep(0.05)
        assert len(followups) >= 2
        # Order-independent for the same reason as the narrated follow-up above.
        finding = next(
            f for f in followups if "held-out R2 0.90" in f["content"]
        )  # certified result, verbatim
        assert finding["result"]["verified"] is True
        assert finding["result"]["verdict"] == "sound"


class _AutoDerive:
    """A stateless client: launch one background derive per turn, then answer.

    It decides from the transcript, not a fixed script, so it is robust to how the
    off-request completion turns interleave with request turns across threads. A turn
    seeded with the certified-completion prompt always answers (never re-derives).
    """

    _SOURCE = "def m(ctx): return []"

    @staticmethod
    def _text(message: dict[str, Any]) -> str:
        return " ".join(
            b.get("text", "") for b in message["content"] if isinstance(b, dict)
        )

    def step(self, transcript: Transcript, tools: Sequence[ToolSpec]) -> Step:
        msgs = transcript.messages
        last_user = next((m for m in reversed(msgs) if m["role"] == "user"), None)
        if last_user is not None and "CERTIFIED" in self._text(last_user):
            return Step(tool_calls=(_call("answer", summary="the certified finding"),))
        derived = any(
            b.get("name") == "derive"
            for m in msgs
            if m["role"] == "assistant"
            for b in m["content"]
            if isinstance(b, dict)
        )
        if not derived:
            return Step(
                tool_calls=(
                    ToolCall(
                        id="c-d",
                        name="derive",
                        arguments={
                            "name": "m",
                            "source": self._SOURCE,
                            "background": True,
                        },
                    ),
                )
            )
        return Step(tool_calls=(_call("answer", summary="training launched"),))


def test_deduped_job_delivers_a_followup_to_each_conversation(tmp_path: object) -> None:
    # Content-addressed dedupe runs the derivation once, but every conversation that
    # asked for it must still get its own certified follow-up. c2 submits identical
    # content after c1's job finished, so it dedupes onto the completed job and is
    # delivered from cache; dropping that path (or the last-writer-wins map) loses c2's.
    import time
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'app.db'}")

    def derive_factory(conversation_id: str, question: str) -> DeriveFn:
        def derive(name, source, claim, contract, fmt, assumptions, deps):  # type: ignore[no-untyped-def]
            return DeriveOutcome(
                certified=True,
                verdict="sound",
                rendered="held-out R2 0.90",
                data_hash="h",
                checks=(("prediction", "sound", "held-out R2 0.90"),),
            )

        return derive

    app = create_app(
        load_datasets=lambda: {"houses": []},
        client=_AutoDerive(),
        store=store,
        derive_factory=derive_factory,
    )

    def wait_certified(http: TestClient, conversation_id: str) -> list[dict[str, Any]]:
        for _ in range(1200):
            msgs = http.get(f"/api/conversations/{conversation_id}/messages").json()
            certified = [
                m
                for m in msgs
                if m["role"] == "assistant" and (m.get("result") or {}).get("verified")
            ]
            if certified:
                return certified
            time.sleep(0.05)
        return []

    with TestClient(app) as http:
        http.post("/api/chat", json=_chat_body("c1", "train a model"))
        assert wait_certified(http, "c1")  # c1's follow-up arrives
        http.post("/api/chat", json=_chat_body("c2", "train a model"))
        assert wait_certified(http, "c2")  # c2 dedupes onto it and still gets one
        # The identical content ran as a single job (deduped), not two.
        assert len(http.get("/api/jobs").json()) == 1


def test_job_endpoints_404_when_absent(tmp_path: object) -> None:
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'app.db'}")
    app = create_app(load_datasets=dict, client=_Scripted(), store=store)
    with TestClient(app) as http:
        assert http.get("/api/jobs").json() == []
        assert http.get("/api/jobs/nope").status_code == 404
        assert http.post("/api/jobs/nope/cancel").status_code == 404


def test_models_endpoint_lists_the_catalogue() -> None:
    models = [
        {"id": "claude-x", "name": "Claude X", "provider": "anthropic", "default": True}
    ]
    app = create_app(load_datasets=dict, client=_Scripted(), models=models)
    with TestClient(app) as http:
        assert http.get("/api/models").json() == models


def test_health() -> None:
    app = create_app(load_datasets=dict, client=_Scripted())
    with TestClient(app) as http:
        assert http.get("/health").json() == {"status": "ok"}


def test_viz_carries_the_models_vega_lite_spec_with_rows() -> None:
    # The model's own Vega-Lite spec wins verbatim (any mark/transform), streamed with
    # the certified rows injected as the data downstream.
    from elbi.app import _viz_for
    from elbi_agent import AnswerResult

    spec = {
        "mark": "boxplot",
        "encoding": {"x": {"field": "grade"}, "y": {"field": "price"}},
    }
    rows = tuple({"grade": str(g), "price": str(g * 1000)} for g in range(20))
    viz = _viz_for(AnswerResult(True, "sound", "", rows=rows, viz=spec))
    assert viz is not None and viz["spec"] == spec
    assert len(viz["rows"]) == 20  # the certified rows ride along


def test_viz_falls_back_to_a_spec_from_claim_roles() -> None:
    # With no model chart, a minimal Vega-Lite spec is built from the claim's roles.
    from elbi.app import _viz_for
    from elbi_agent import AnswerResult

    claim = {"x": "lat_partial", "y": "price_partial_log", "controls": ["longitude"]}
    rows = tuple(
        {
            "lat_partial": str(i * 0.001),
            "price_partial_log": str(i * 0.002),
            "longitude": "-122.2",
        }
        for i in range(20)
    )
    viz = _viz_for(AnswerResult(True, "sound", "", spec={"claim": claim}, rows=rows))
    assert viz is not None
    assert viz["spec"]["encoding"]["x"]["field"] == "lat_partial"


def test_chart_validation_rejects_a_phantom_column() -> None:
    # Any Vega-Lite chart is allowed, but every field must be a certified column or a
    # transform output, so a spec cannot invent data. The error names the columns.
    from elbi_agent.runtime import _validate_chart

    rows = [{"grade": "7", "price": "500000"}]
    ok = _validate_chart(
        {"mark": "bar", "encoding": {"x": {"field": "grade"}, "y": {"field": "price"}}},
        rows,
    )
    assert ok is None
    # a derived field (transform `as`) is allowed; a phantom one is rejected
    derived = _validate_chart(
        {
            "transform": [{"calculate": "datum.price/1000", "as": "k"}],
            "mark": "bar",
            "encoding": {"x": {"field": "grade"}, "y": {"field": "k"}},
        },
        rows,
    )
    assert derived is None
    bad = _validate_chart({"mark": "bar", "encoding": {"y": {"field": "sqft"}}}, rows)
    assert bad is not None and "sqft" in bad


def test_serves_spa_when_present(tmp_path: object) -> None:
    # A built SPA directory is served at "/" (html=True), without shadowing the API.
    from pathlib import Path

    static = Path(str(tmp_path))
    (static / "index.html").write_text("<div id='root'>app</div>", encoding="utf-8")
    app = create_app(load_datasets=dict, client=_Scripted(), static_dir=static)
    with TestClient(app) as http:
        assert http.get("/health").json() == {"status": "ok"}  # API still wins
        root = http.get("/")
        assert root.status_code == 200 and "id='root'" in root.text
        # a client-side route with no file on disk falls back to index.html
        spa_route = http.get("/derivations")
        assert spa_route.status_code == 200 and "id='root'" in spa_route.text


def test_mcp_server_is_mounted_and_its_lifespan_runs(tmp_path: object) -> None:
    # What create_app owns is the wiring: it must call streamable_http_app(), mount
    # it at /mcp, and run its lifespan (the streamable-HTTP session manager is only
    # started inside it). A fake stands in for the MCP so the wiring is asserted
    # precisely, without the real server's stream machinery (whose in-process GC
    # teardown is unrelated third-party noise). A real server over the real mount is
    # exercised by test_mounted_mcp_answers_a_deployed_host.
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from starlette.routing import Mount

    class _FakeMCP:
        def __init__(self) -> None:
            self.entered: list[str] = []
            self.router = SimpleNamespace(lifespan_context=self._lifespan)
            # create_app decides the sub-app's transport settings, so the fake records
            # them rather than guessing.
            self.app_kwargs: dict[str, Any] = {}
            # create_app installs the tool-scope gate here.
            self.middleware: list[Any] = []

        @asynccontextmanager
        async def _lifespan(self, _app: Any) -> Any:
            self.entered.append("start")
            yield
            self.entered.append("stop")

        def streamable_http_app(self, **kwargs: Any) -> _FakeMCP:
            self.app_kwargs = kwargs
            return self

        async def __call__(self, scope: Any, receive: Any, send: Any) -> None:
            await send({"type": "http.response.start", "status": 204, "headers": []})
            await send({"type": "http.response.body", "body": b""})

    fake = _FakeMCP()
    # A static dir, because the single-page app's catch-all mount is what makes the bare
    # `/mcp` path interesting: without it Starlette's own slash-redirect handles the
    # case, and the test would pass while the shipped app did not.
    from pathlib import Path

    static = Path(str(tmp_path)) / "web"
    static.mkdir()
    (static / "index.html").write_text("<!doctype html><title>Elbi</title>")
    app = create_app(
        load_datasets=dict, client=_Scripted(), mcp_server=fake, static_dir=static
    )

    assert any(isinstance(r, Mount) and r.path == "/mcp" for r in app.routes)
    with TestClient(app) as http:
        assert http.get("/health").json() == {"status": "ok"}
        # The advertised endpoint is `.../mcp`, and a client posts to exactly that.
        # The mount serves only paths *under* it, and the single-page app's catch-all
        # answers everything else -- with 405 for a POST, since it serves files. So the
        # bare path needs its own route, or the documented URL is simply broken.
        # 307 specifically: the only redirect that obliges a client to repeat the POST
        # with its body intact.
        redirect = http.post("/mcp", json={"jsonrpc": "2.0"}, follow_redirects=False)
        assert redirect.status_code == 307
        assert redirect.headers["location"].endswith("/mcp/")
        # And following it reaches the mounted transport rather than the catch-all.
        assert http.post("/mcp", json={"jsonrpc": "2.0"}).status_code == 204
    assert fake.entered == ["start", "stop"]  # the mounted app's lifespan ran
    # Mounted at /mcp, so the sub-app answers at its own root, and it does not police
    # the Host header: a localhost-only allowlist would answer 421 to every deployed
    # hostname.
    assert fake.app_kwargs["streamable_http_path"] == "/"
    security = fake.app_kwargs["transport_security"]
    assert security.enable_dns_rebinding_protection is False


def test_mounted_mcp_answers_a_deployed_host() -> None:
    # The MCP server is built for host 127.0.0.1, which makes it auto-enable DNS
    # rebinding protection with a localhost-only Host allowlist. Mounted behind a real
    # domain that allowlist answers 421 to everything, so the deployed /mcp is
    # unreachable unless create_app clears it. A fake cannot catch this: the check
    # lives in the real transport.
    #
    # The invariant is that the Host header does not change the answer, which holds
    # however the transport evolves. The request deliberately opens no session: the
    # Host check runs ahead of session handling, and a live session owns streams that
    # would outlive the test.
    from elbi_cli.mcp_server import build_server
    from elbi_core.registry import Registry

    def answer(base_url: str) -> tuple[int, str]:
        app = create_app(
            load_datasets=dict,
            client=_Scripted(),
            mcp_server=build_server(Registry(), lambda: None),
        )
        with TestClient(app, base_url=base_url) as http:
            reply = http.post(
                "/mcp/",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers={
                    "Accept": "application/json, text/event-stream",
                    "Content-Type": "application/json",
                },
            )
        return reply.status_code, reply.text

    deployed = answer("https://elbi.example.com")
    assert deployed == answer("http://127.0.0.1:7700")
    assert deployed[0] != 421 and "Invalid Host header" not in deployed[1]


def test_mounted_mcp_serves_a_handshake_client_without_a_session() -> None:
    # Deployments scale this app with no sticky routing, so a handshake-era client must
    # not depend on which replica answered its initialize. Keeping per-process session
    # state would mint an Mcp-Session-Id and answer the next POST "session not found",
    # so the property under test is that the second request carries nothing from the
    # first and still works.
    from elbi_cli.mcp_server import build_server
    from elbi_core import Registry, Runner

    registry = Registry()
    app = create_app(
        load_datasets=dict,
        client=_Scripted(),
        mcp_server=build_server(registry, lambda: Runner(registry)),
    )
    sent = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    with TestClient(app, base_url="https://elbi.example.com") as http:
        opened = http.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "probe", "version": "0"},
                },
            },
            headers=sent,
        )
        # Deliberately carries nothing forward: no session header, as a second replica
        # would have none to send.
        listed = http.post(
            "/mcp/",
            json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
            headers=sent,
        )

    assert opened.status_code == 200, opened.text
    assert "mcp-session-id" not in {k.lower() for k in opened.headers}
    assert listed.status_code == 200, listed.text
    assert "search_derivations" in listed.text  # a real tool list came back


def test_mounted_mcp_notifies_a_handshake_client_of_a_tool_list_change() -> None:
    # Serving without sessions costs the standalone notification stream, so a tool-list
    # change has to ride the request that caused it. Agents author and delete
    # derivations at runtime, and a client that never hears about it cannot call the new
    # `run_<name>`. The probe tool stands in for propose/delete, whose sandbox is beside
    # the point here: what is under test is delivery, not authoring.
    from mcp.server import MCPServer

    from elbi_cli.mcp_server import _notify_tools_changed

    server = MCPServer("probe")

    async def touch(ctx: McpContext[Any, Any]) -> str:
        """Announce a tool-list change the way the authoring tools do."""
        await _notify_tools_changed(ctx)
        return "done"

    server.tool(name="touch", description="probe")(touch)

    app = create_app(load_datasets=dict, client=_Scripted(), mcp_server=server)
    sent = {
        "Accept": "application/json, text/event-stream",
        "Content-Type": "application/json",
    }
    with TestClient(app, base_url="https://elbi.example.com") as http:
        http.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "initialize",
                "params": {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {},
                    "clientInfo": {"name": "probe", "version": "0"},
                },
            },
            headers=sent,
        )
        called = http.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "touch", "arguments": {}},
            },
            headers=sent,
        )

    assert called.status_code == 200, called.text
    assert "notifications/tools/list_changed" in called.text


def test_mounted_mcp_serves_the_2026_07_28_revision_statelessly() -> None:
    # The revision drops the initialize handshake and the session header: a request
    # carries its own protocol version and client capabilities in `_meta`, and a client
    # reads capabilities from `server/discover`. That is what lets a replica behind a
    # plain load balancer answer any request, so it is asserted over real HTTP rather
    # than through an in-memory client (which skips JSON-RPC framing entirely).
    from elbi_cli.mcp_server import build_server
    from elbi_core import Artifact, Context, Registry, Runner, derivation, serve

    registry = Registry()

    @derivation(name="hello", serve=serve.text(), registry=registry)
    def hello(ctx: Context) -> Artifact:
        """A friendly greeting."""
        return Artifact.text("hi")

    revision = "2026-07-28"
    envelope = {
        "_meta": {
            "io.modelcontextprotocol/protocolVersion": revision,
            "io.modelcontextprotocol/clientCapabilities": {},
        }
    }

    def headers(method: str, name: str | None = None) -> dict[str, str]:
        sent = {
            "Accept": "application/json, text/event-stream",
            "Content-Type": "application/json",
            "mcp-protocol-version": revision,
            "mcp-method": method,
        }
        if name is not None:
            sent["mcp-name"] = name
        return sent

    app = create_app(
        load_datasets=dict,
        client=_Scripted(),
        mcp_server=build_server(registry, lambda: Runner(registry)),
    )
    with TestClient(app, base_url="https://elbi.example.com") as http:
        found = http.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "server/discover",
                "params": envelope,
            },
            headers=headers("server/discover"),
        )
        called = http.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {**envelope, "name": "run_hello", "arguments": {}},
            },
            headers=headers("tools/call", "run_hello"),
        )
        # Infrastructure routes on the headers, so the server refuses one that disagrees
        # with the body rather than trusting either.
        mismatched = http.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/list",
                "params": envelope,
            },
            headers=headers("resources/list"),
        )

    assert found.status_code == 200, found.text
    assert revision in found.text
    assert '"listChanged":true' in found.text  # the dynamic tool list is advertised
    assert called.status_code == 200, called.text
    assert "hi" in called.text
    # No handshake ran and no session was minted, so any replica can answer the next.
    assert "mcp-session-id" not in {k.lower() for k in called.headers}
    assert mismatched.status_code == 400


def test_a_list_result_is_cacheable_only_by_the_caller_that_asked() -> None:
    """`ttlMs` and `cacheScope` on a list result, and why the scope has to be private.

    The revision requires both fields on the list and read results so a client can cache
    instead of polling, and `cacheScope` decides whether an intermediary between the two
    may keep a copy. Private, because the warehouse tools answer from the catalog the
    server resolves per call, so a response a shared cache could hand on is the wrong
    shape here whether or not the surfaces differ today.

    Both values come from the SDK rather than from anything this repo sets, which is
    the reason to pin them: they are part of the contract the server publishes, and a
    changed default would otherwise move them quietly.
    """
    import json

    from elbi_cli.mcp_server import build_server
    from elbi_core import Artifact, Context, Registry, Runner, derivation, serve

    registry = Registry()

    @derivation(name="hello", serve=serve.text(), registry=registry)
    def hello(ctx: Context) -> Artifact:
        """A friendly greeting."""
        return Artifact.text("hi")

    revision = "2026-07-28"
    app = create_app(
        load_datasets=dict,
        client=_Scripted(),
        mcp_server=build_server(registry, lambda: Runner(registry)),
    )
    with TestClient(app, base_url="https://elbi.example.com") as http:
        listed = http.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list",
                "params": {
                    "_meta": {
                        "io.modelcontextprotocol/protocolVersion": revision,
                        "io.modelcontextprotocol/clientCapabilities": {},
                    }
                },
            },
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "mcp-protocol-version": revision,
                "mcp-method": "tools/list",
            },
        )

    assert listed.status_code == 200, listed.text
    # The response may arrive as an SSE frame, so read the payload out of it either way.
    body = listed.text
    if "data: " in body:
        body = next(
            line[len("data: ") :]
            for line in body.splitlines()
            if line.startswith("data: ")
        )
    result = json.loads(body)["result"]

    assert result["resultType"] == "complete"
    assert isinstance(result["ttlMs"], int)
    assert result["cacheScope"] == "private"


def test_mcp_continues_the_caller_s_trace() -> None:
    """W3C trace context crosses the MCP boundary, per the 2026-07-28 revision.

    A caller's trace must continue into the work the server does on its behalf,
    otherwise a tool call is an unattributed gap in the middle of the caller's trace.
    The carrier is the request's own `_meta`, not an HTTP header, so it survives a
    transport that is not HTTP.

    Asserted over real HTTP against the mounted endpoint, because that is the seam that
    matters: the server extracts the context and starts its span, and only dispatch
    exercises both.
    """
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
        InMemorySpanExporter,
    )

    from elbi_cli.mcp_server import build_server
    from elbi_core import Artifact, Context, Registry, Runner, derivation, serve

    registry = Registry()

    @derivation(name="hello", serve=serve.text(), registry=registry)
    def hello(ctx: Context) -> Artifact:
        """A friendly greeting."""
        return Artifact.text("hi")

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    # Synchronous export, so the spans are readable as soon as the request returns.
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # The SDK binds its tracer at import, but the handle is a proxy that resolves the
    # provider per span, so setting one here is picked up whatever ran first.
    trace.set_tracer_provider(provider)

    trace_id = "4bf92f3577b34da6a3ce929d0e0e4736"
    revision = "2026-07-28"
    app = create_app(
        load_datasets=dict,
        client=_Scripted(),
        mcp_server=build_server(registry, lambda: Runner(registry)),
    )
    with TestClient(app, base_url="https://elbi.example.com") as http:
        called = http.post(
            "/mcp/",
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "_meta": {
                        "io.modelcontextprotocol/protocolVersion": revision,
                        "io.modelcontextprotocol/clientCapabilities": {},
                        "traceparent": f"00-{trace_id}-00f067aa0ba902b7-01",
                    },
                    "name": "run_hello",
                    "arguments": {},
                },
            },
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
                "mcp-protocol-version": revision,
                "mcp-method": "tools/call",
                "mcp-name": "run_hello",
            },
        )

    assert called.status_code == 200, called.text
    # Only the MCP layer's own spans, so a deployment that also traces HTTP (when
    # OTEL_EXPORTER_OTLP_ENDPOINT is set) does not change what this reads.
    spans = [
        s
        for s in exporter.get_finished_spans()
        if "mcp.method.name" in dict(s.attributes or {})
    ]
    assert spans, "the MCP server emitted no span for the call"
    served = spans[0]
    assert served.context is not None
    assert format(served.context.trace_id, "032x") == trace_id, (
        "the server started a new trace instead of continuing the caller's"
    )
    assert dict(served.attributes or {})["gen_ai.tool.name"] == "run_hello"


def test_conversation_timestamps_carry_a_utc_offset(tmp_path: object) -> None:
    # SQLite hands stored UTC datetimes back naive, so a bare isoformat() drops the zone
    # and the browser parses the value as LOCAL time: shifting every chat into the
    # future, which the sidebar's relative clock clamps to "just now". The API must emit
    # an explicit UTC offset so the client reads the instant the chat was recorded.
    from datetime import datetime
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'ts.db'}")
    store.create_conversation("hello", conversation_id="c-ts")
    app = create_app(load_datasets=lambda: {"d": []}, client=_Scripted(), store=store)
    with TestClient(app) as http:
        item = http.get("/api/conversations").json()["items"][0]
    for key in ("created_at", "updated_at"):
        parsed = datetime.fromisoformat(item[key])
        offset = parsed.utcoffset()
        assert offset is not None, f"{key} has no timezone marker: {item[key]!r}"
        assert offset.total_seconds() == 0  # UTC, so the client reads the right instant


def test_delete_conversation_endpoint(tmp_path: object) -> None:
    # Removes the conversation from the history list; deleting an unknown id is a 404.
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'del.db'}")
    store.create_conversation("first", conversation_id="c-1")
    store.create_conversation("second", conversation_id="c-2")
    app = create_app(load_datasets=lambda: {"d": []}, client=_Scripted(), store=store)
    with TestClient(app) as http:
        ids = {c["id"] for c in http.get("/api/conversations").json()["items"]}
        assert ids == {"c-1", "c-2"}

        resp = http.delete("/api/conversations/c-1")
        assert resp.status_code == 200 and resp.json() == {"ok": True}

        remaining = {c["id"] for c in http.get("/api/conversations").json()["items"]}
        assert remaining == {"c-2"}  # gone from the list

        assert http.delete("/api/conversations/c-1").status_code == 404  # already gone
        assert http.delete("/api/conversations/nope").status_code == 404


def test_rename_conversation_endpoint(tmp_path: object) -> None:
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'rename.db'}")
    store.create_conversation("old", conversation_id="c1")
    app = create_app(load_datasets=lambda: {"d": []}, client=_Scripted(), store=store)
    with TestClient(app) as http:
        r = http.patch("/api/conversations/c1", json={"title": "  New Name  "})
        assert r.status_code == 200 and r.json()["title"] == "New Name"  # trimmed
        items = http.get("/api/conversations").json()["items"]
        assert items[0]["title"] == "New Name"  # reflected in the list
        assert (
            http.patch("/api/conversations/c1", json={"title": " "}).status_code == 400
        )
        assert (
            http.patch("/api/conversations/x", json={"title": "y"}).status_code == 404
        )


def test_search_conversations_endpoint(tmp_path: object) -> None:
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'search.db'}")
    store.create_conversation("housing prices", conversation_id="c1")
    store.create_conversation("customer churn", conversation_id="c2")
    app = create_app(load_datasets=lambda: {"d": []}, client=_Scripted(), store=store)
    with TestClient(app) as http:
        items = http.get("/api/conversations", params={"q": "hous"}).json()["items"]
        assert [c["id"] for c in items] == ["c1"]  # title filter, case-insensitive
        assert len(http.get("/api/conversations").json()["items"]) == 2  # no filter


def test_messages_expose_ids(tmp_path: object) -> None:
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'ids.db'}")
    store.create_conversation("t", conversation_id="c1")
    store.add_message("c1", "user", content="hi")
    app = create_app(load_datasets=lambda: {"d": []}, client=_Scripted(), store=store)
    with TestClient(app) as http:
        msgs = http.get("/api/conversations/c1/messages").json()
        assert msgs[0]["id"] and isinstance(msgs[0]["id"], str)  # stable message id


def test_new_conversation_gets_an_llm_title(tmp_path: object) -> None:
    # A new chat starts with the raw question as its title, then a background LLM title
    # replaces it. Titling runs off the request path, so we poll briefly for the swap.
    import time
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'title.db'}")
    client = _Scripted(Step(tool_calls=(_call("answer", summary="ok"),)))
    app = create_app(
        load_datasets=lambda: {"d": []},
        client=client,
        store=store,
        generate_title=lambda q: "Housing price drivers",
    )
    with TestClient(app) as http:
        r = http.post("/api/chat", json=_chat_body("c1", "how does price vary"))
        assert r.status_code == 200
        title = ""
        for _ in range(100):  # up to ~2s for the background thread to set it
            items = http.get("/api/conversations").json()["items"]
            title = items[0]["title"] if items else ""
            if title == "Housing price drivers":
                break
            time.sleep(0.02)
        assert title == "Housing price drivers"


def test_title_generation_failure_keeps_the_raw_title(tmp_path: object) -> None:
    # If titling returns nothing (or errs), the plain truncated-question title stays,
    # so a title is always present and titling can never break the conversation.
    import time
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'title2.db'}")
    client = _Scripted(Step(tool_calls=(_call("answer", summary="ok"),)))
    app = create_app(
        load_datasets=lambda: {"d": []},
        client=client,
        store=store,
        generate_title=lambda q: None,  # no title produced
    )
    with TestClient(app) as http:
        http.post("/api/chat", json=_chat_body("c1", "how does price vary"))
        time.sleep(0.2)  # give any background work time to (not) change the title
        items = http.get("/api/conversations").json()["items"]
        assert items[0]["title"] == "how does price vary"  # raw title preserved


def test_llm_profile_crud_and_default(tmp_path: object) -> None:
    # Register multiple named profiles (e.g. OpenAI and Claude), each with its own key;
    # the first becomes the default and keys are masked. Deleting the default promotes
    # another. This is what lets a user "set up multiple models" and pick between them.
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'profiles.db'}")
    app = create_app(load_datasets=lambda: {"d": []}, client=_Scripted(), store=store)
    with TestClient(app) as http:
        assert http.get("/api/settings/llm/profiles").json() == {
            "profiles": [],
            "default": "",
            "titleProfile": "",
            "configured": True,
        }
        a = http.put(
            "/api/settings/llm/profiles/OpenAI",
            json={"model": "openai/gpt-5", "api_key": "sk-a"},
        ).json()
        assert a == {
            "name": "OpenAI",
            "model": "openai/gpt-5",
            "baseUrl": "",
            # Per-profile, since a reasoning model and a cheap title model want
            # different budgets; empty leaves it to the model's own default.
            "reasoningEffort": "",
            "apiKeySet": True,
            # Resolved server-side for the UI: which provider this routes to, and the
            # budgets this model accepts. The ladder is per-model -- gpt-5 takes
            # neither `none` nor `xhigh` where a newer sibling takes both -- which is
            # the whole reason it is read from the registry and not listed here.
            "provider": "openai",
            # A display name for the providers whose key is not presentable ("azure_ai",
            # "vertex_ai"), and the label the picker shows beside a model whose provider
            # we have no logo for.
            "providerLabel": "OpenAI",
            "reasoningEfforts": ["minimal", "low", "medium", "high"],
        }
        assert "apiKey" not in a  # the secret is never returned
        http.put(
            "/api/settings/llm/profiles/Claude",
            json={"model": "anthropic/claude-sonnet-5", "api_key": "sk-c"},
        )
        listing = http.get("/api/settings/llm/profiles").json()
        assert {p["name"] for p in listing["profiles"]} == {"OpenAI", "Claude"}
        assert listing["default"] == "OpenAI"  # first saved is the default

        # editing without an api_key keeps the stored key
        http.put(
            "/api/settings/llm/profiles/OpenAI", json={"model": "openai/gpt-5-mini"}
        )
        assert store.get_profile("OpenAI").api_key == "sk-a"  # type: ignore[union-attr]

        http.put("/api/settings/llm/default", json={"name": "Claude"})
        assert http.get("/api/settings/llm/profiles").json()["default"] == "Claude"

        # deleting the default promotes the remaining profile
        assert http.delete("/api/settings/llm/profiles/Claude").status_code == 200
        assert http.get("/api/settings/llm/profiles").json()["default"] == "OpenAI"
        # names are display labels: spaces are allowed ("Sonnet 5")...
        spaced = http.put(
            "/api/settings/llm/profiles/Sonnet 5",
            json={"model": "anthropic/claude-sonnet-5"},
        )
        assert spaced.status_code == 200, spaced.text
        assert spaced.json()["name"] == "Sonnet 5"
        # ...but a name with a disallowed character is a clean 400 that says why, not a
        # silent failure (the frontend surfaces this detail).
        bad = http.put("/api/settings/llm/profiles/bad@name!", json={})
        assert bad.status_code == 400
        assert "letters, digits" in bad.json()["detail"]
        assert http.delete("/api/settings/llm/profiles/nope").status_code == 404


def test_chat_resolves_the_profile_per_conversation(tmp_path: object) -> None:
    # A conversation runs on the profile the switcher chose; a different chat can run on
    # a different one. make_client records which profile each turn resolved.
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'perconv.db'}")
    seen: list[str | None] = []

    def make_client(profile: str | None) -> object:
        seen.append(profile)
        return _Scripted(Step(tool_calls=(_call("answer", summary="ok"),)))

    app = create_app(
        load_datasets=lambda: {"d": []},
        client=_Scripted(),
        make_client=make_client,  # type: ignore[arg-type]
        store=store,
    )
    with TestClient(app) as http:
        http.put("/api/settings/llm/profiles/OpenAI", json={"model": "openai/gpt-5"})
        http.put("/api/settings/llm/profiles/Claude", json={"model": "anthropic/x"})
        # one chat picks Claude explicitly
        body = {**_chat_body("c1", "hi"), "profile": "Claude"}
        http.post("/api/chat", json=body)
        assert seen[-1] == "Claude"
        assert store.get_conversation("c1").profile == "Claude"  # type: ignore[union-attr]
        # another chat with no pick falls to the default profile (OpenAI, first saved)
        http.post("/api/chat", json=_chat_body("c2", "hi"))
        assert seen[-1] == "OpenAI"


def test_usage_is_tracked_per_conversation(tmp_path: object) -> None:
    # Each turn's token counts and cost accumulate on the conversation and ride in the
    # result payload, so the UI can show what a chat spent.
    from pathlib import Path

    from elbi_agent import Usage

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'usage.db'}")
    client = _Scripted(
        Step(
            tool_calls=(_call("answer", summary="ok"),),
            usage=Usage(prompt_tokens=100, completion_tokens=40, cost=0.02),
        )
    )
    app = create_app(load_datasets=lambda: {"d": []}, client=client, store=store)
    with TestClient(app) as http:
        http.post("/api/chat", json=_chat_body("c1", "hi"))
        u = http.get("/api/conversations/c1/usage").json()
        assert u["prompt_tokens"] == 100 and u["completion_tokens"] == 40
        assert abs(u["cost"] - 0.02) < 1e-9
        # the per-turn result payload also carries the usage
        msgs = http.get("/api/conversations/c1/messages").json()
        assert msgs[1]["result"]["usage"]["prompt_tokens"] == 100


def test_export_downloads_the_conversation(tmp_path: object) -> None:
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'export.db'}")
    client = _Scripted(Step(tool_calls=(_call("answer", summary="ok"),)))
    app = create_app(load_datasets=lambda: {"d": []}, client=client, store=store)
    with TestClient(app) as http:
        http.post("/api/chat", json=_chat_body("c1", "hello there"))
        r = http.get("/api/conversations/c1/export")
        assert r.status_code == 200
        assert "attachment" in r.headers["content-disposition"]
        doc = r.json()
        assert doc["id"] == "c1"
        assert [m["role"] for m in doc["messages"]] == ["user", "assistant"]
        assert doc["messages"][0]["content"] == "hello there"
        # an unknown conversation is a 404, not an empty file
        assert http.get("/api/conversations/nope/export").status_code == 404


def test_title_profile_setting(tmp_path: object) -> None:
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'titleprof.db'}")
    app = create_app(load_datasets=lambda: {"d": []}, client=_Scripted(), store=store)
    with TestClient(app) as http:
        http.put(
            "/api/settings/llm/profiles/Cheap", json={"model": "openai/gpt-5-mini"}
        )
        assert (
            http.put(
                "/api/settings/llm/title-profile", json={"name": "Cheap"}
            ).status_code
            == 200
        )
        assert http.get("/api/settings/llm/profiles").json()["titleProfile"] == "Cheap"
        # unknown profile rejected
        assert (
            http.put("/api/settings/llm/title-profile", json={"name": "x"}).status_code
            == 404
        )
        # clearing falls back to the default
        http.put("/api/settings/llm/title-profile", json={"name": ""})
        assert http.get("/api/settings/llm/profiles").json()["titleProfile"] == ""


def test_message_feedback(tmp_path: object) -> None:
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'feedback.db'}")
    client = _Scripted(Step(tool_calls=(_call("answer", summary="ok"),)))
    app = create_app(load_datasets=lambda: {"d": []}, client=client, store=store)
    with TestClient(app) as http:
        http.post("/api/chat", json=_chat_body("c1", "hi"))
        msgs = http.get("/api/conversations/c1/messages").json()
        assistant_id = msgs[1]["id"]
        assert msgs[1]["feedback"] is None
        assert (
            http.put(
                f"/api/messages/{assistant_id}/feedback", json={"feedback": "up"}
            ).status_code
            == 200
        )
        reloaded = http.get("/api/conversations/c1/messages").json()
        assert reloaded[1]["feedback"] == "up"
        # clearing sets it back to null; a bad value is rejected; unknown id is 404
        http.put(f"/api/messages/{assistant_id}/feedback", json={"feedback": None})
        assert http.get("/api/conversations/c1/messages").json()[1]["feedback"] is None
        assert (
            http.put(
                f"/api/messages/{assistant_id}/feedback", json={"feedback": "meh"}
            ).status_code
            == 400
        )
        assert (
            http.put("/api/messages/nope/feedback", json={"feedback": "up"}).status_code
            == 404
        )


def test_long_conversation_condenses_and_caches(tmp_path: object) -> None:
    # A conversation past the replay window summarizes its older turns and caches the
    # summary, so a normal follow-up carries a short note instead of every message.
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'condense.db'}")
    store.create_conversation("t", conversation_id="c1")
    for i in range(14):  # 14 prior turns; the window keeps the last 12, so 2 are older
        store.add_message(
            "c1", "user" if i % 2 == 0 else "assistant", content=f"turn {i}"
        )
    client = _Scripted(
        Step(text="USER_CONTEXT: earlier goal"),  # the condense call
        Step(tool_calls=(_call("answer", summary="ok"),)),  # the chat turn
    )
    app = create_app(load_datasets=lambda: {"d": []}, client=client, store=store)
    with TestClient(app) as http:
        http.post("/api/chat", json=_chat_body("c1", "follow up"))
        conv = store.get_conversation("c1")
        assert conv is not None
        assert conv.summary == "USER_CONTEXT: earlier goal"
        assert conv.summary_turns == 2  # 14 prior turns minus the 12-turn window


def test_edit_and_resend_truncates_then_reruns(tmp_path: object) -> None:
    # Editing a past question drops it and the turns after it, then re-runs from there,
    # so the conversation replaces that point instead of appending a branch.
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'edit.db'}")
    client = _Scripted(
        Step(tool_calls=(_call("answer", summary="first"),)),
        Step(tool_calls=(_call("answer", summary="second"),)),
    )
    app = create_app(load_datasets=lambda: {"d": []}, client=client, store=store)
    with TestClient(app) as http:
        http.post("/api/chat", json=_chat_body("c1", "original question"))
        first = http.get("/api/conversations/c1/messages").json()
        user_id = first[0]["id"]
        assert [m["content"] for m in first] == ["original question", "first"]
        # resend an edited version of that first user message
        body = {**_chat_body("c1", "edited question"), "editMessageId": user_id}
        http.post("/api/chat", json=body)
        after = http.get("/api/conversations/c1/messages").json()
        # the old turn is gone; the conversation now holds only the edited exchange
        assert [m["content"] for m in after] == ["edited question", "second"]


def test_budget_gate_blocks_chat(tmp_path: object) -> None:
    # Once the instance is over its budget window, a new turn is refused before spending
    # (HTTP 402), refusing the turn before any spend.
    from pathlib import Path

    from elbi_agent import Usage

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'budget.db'}")
    client = _Scripted(
        Step(tool_calls=(_call("answer", summary="ok"),), usage=Usage(cost=0.6)),
        Step(tool_calls=(_call("answer", summary="ok"),), usage=Usage(cost=0.6)),
    )
    app = create_app(load_datasets=lambda: {"d": []}, client=client, store=store)
    with TestClient(app) as http:
        http.put("/api/settings/budget", json={"max_budget": 1.0, "window": "30d"})
        assert http.post("/api/chat", json=_chat_body("c1", "hi")).status_code == 200
        assert http.get("/api/settings/budget").json()["spend"] == 0.6
        # second turn pushes spend to 1.2 (>= 1.0); the third is blocked pre-call
        http.post("/api/chat", json=_chat_body("c2", "hi"))
        assert http.post("/api/chat", json=_chat_body("c3", "hi")).status_code == 402


def test_audit_trail_records_answers(tmp_path: object) -> None:
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'audit.db'}")
    client = _Scripted(Step(tool_calls=(_call("answer", summary="ok"),)))
    app = create_app(load_datasets=lambda: {"d": []}, client=client, store=store)
    with TestClient(app) as http:
        http.post("/api/chat", json=_chat_body("c1", "hi"))
        audit = http.get("/api/audit").json()
        assert any(e["action"] == "answer" and e["targetId"] == "c1" for e in audit)


def test_reasoning_effort_can_be_turned_off_not_only_dialled_down(tmp_path) -> None:
    """`none` has to be expressible, or some models cannot answer at all.

    OpenAI's newer reasoning models apply an effort of their own and then refuse
    function tools alongside it on /v1/chat/completions. Every turn this app takes
    carries tools, so such a model fails on every question until the effort is
    explicitly off, and with only low/medium/high accepted there was no way to say so.
    """
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'effort.db'}")
    app = create_app(load_datasets=lambda: {"d": []}, client=_Scripted(), store=store)
    with TestClient(app) as http:
        for effort in ("none", "minimal", "low", "medium", "high"):
            saved = http.put(
                f"/api/settings/llm/profiles/p-{effort}",
                json={
                    "model": "openai/gpt-5",
                    "api_key": "sk",
                    "reasoning_effort": effort,
                },
            )
            assert saved.status_code == 200, (effort, saved.text)
            assert saved.json()["reasoningEffort"] == effort

        # Still empty-or-known: a typo must not reach the provider as a 400 mid-chat.
        bad = http.put(
            "/api/settings/llm/profiles/typo",
            json={"model": "openai/gpt-5", "api_key": "sk", "reasoning_effort": "max"},
        )
        assert bad.status_code == 400
        assert "reasoning_effort must be one of" in bad.json()["detail"]


def test_a_provider_rejection_is_not_reported_as_try_again() -> None:
    """A 4xx from the provider recurs, so saying otherwise sends you in circles.

    This is the message that cost a log dive: the provider had explained exactly what to
    change and the app replaced it with "server error, please try again", which is both
    wrong about the cause and wrong about the remedy.
    """
    from elbi.app import _turn_failure_message

    class Rejected(Exception):
        status_code = 400

    detail = "Function tools with reasoning_effort are not supported for gpt-5."
    message = _turn_failure_message(Rejected(detail))
    assert detail in message
    assert "configuration problem" in message
    assert "try again" not in message.lower()

    # A timeout or a dropped connection carries no status; retrying is the right advice.
    for transient in (TimeoutError("read timeout"), RuntimeError("connection reset")):
        fallback = _turn_failure_message(transient)
        assert "Please try again." in fallback

    # A 5xx is the provider's problem and may well clear on its own.
    class Upstream(Exception):
        status_code = 503

    assert "Please try again." in _turn_failure_message(Upstream("bad gateway"))


def test_the_chat_stream_carries_the_provider_reason_through(tmp_path) -> None:
    """And it has to reach the user, not just exist as a helper.

    Asserted through the endpoint because the failure was a wiring one: the message the
    provider sent was replaced on its way out, and a unit test of the formatter would
    have passed the whole time that was happening.
    """
    from pathlib import Path

    detail = "Function tools with reasoning_effort are not supported for gpt-5."

    class Rejecting:
        """A client whose provider refuses the request, as a 4xx, every time."""

        def step(self, transcript, tools):  # type: ignore[no-untyped-def]
            raise self._rejection()

        def stream(self, transcript, tools):  # type: ignore[no-untyped-def]
            raise self._rejection()

        @staticmethod
        def _rejection() -> Exception:
            exc = RuntimeError(detail)
            exc.status_code = 400  # type: ignore[attr-defined]
            return exc

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'reject.db'}")
    app = create_app(load_datasets=lambda: {"d": []}, client=Rejecting(), store=store)
    with TestClient(app) as http:
        response = http.post("/api/chat", json=_chat_body("c1", "hi"))

    # The stream still completes -- a truncated stream leaves the client hanging.
    assert response.status_code == 200
    assert response.text.rstrip().endswith("[DONE]")
    assert detail in response.text
    assert "server error" not in response.text


def test_a_profile_tells_the_ui_its_provider_and_which_efforts_fit(tmp_path) -> None:
    """Both are resolved server-side, because the browser cannot work them out.

    Splitting the model string on "/" is what the UI used to do, and it is wrong in both
    directions: a bare `gpt-5.6-terra` has no prefix and is still OpenAI, while
    `bedrock/us.anthropic.…` has one that is not the model's family. The efforts come
    from the model registry, which the browser has no copy of.
    """
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'provider.db'}")
    app = create_app(load_datasets=lambda: {"d": []}, client=_Scripted(), store=store)
    with TestClient(app) as http:
        prefixless = http.put(
            "/api/settings/llm/profiles/bare",
            json={"model": "gpt-5.6-terra", "api_key": "sk"},
        ).json()
        assert prefixless["provider"] == "openai", (
            "an unprefixed model still has a home"
        )

        hosted = http.put(
            "/api/settings/llm/profiles/hosted",
            json={"model": "bedrock/us.anthropic.claude-sonnet-5", "api_key": "sk"},
        ).json()
        assert hosted["provider"] == "bedrock", "the prefix is where it routes"

        # A reasoning model offers a ladder, ordered by how much thinking it asks for
        # rather than alphabetically, since this populates a control read left to right.
        efforts = prefixless["reasoningEffort"], prefixless["reasoningEfforts"]
        assert efforts[0] == ""  # unset: the model's own default
        ladder = ["none", "minimal", "low", "medium", "high", "xhigh", "max"]
        assert efforts[1] == [e for e in ladder if e in efforts[1]]
        assert "none" in efforts[1]

        # A model with no reasoning to budget offers no control at all.
        plain = http.put(
            "/api/settings/llm/profiles/plain",
            json={"model": "openai/gpt-4o", "api_key": "sk"},
        ).json()
        assert plain["reasoningEfforts"] == []


def test_every_provider_label_is_one_litellm_actually_has() -> None:
    """The label map is keyed on LiteLLM's provider names, so a rename must not pass.

    A stale key fails silently: the provider stops matching and its display name reverts
    to the raw key. Checking against the library's own list turns that into a failing
    test on the upgrade that renamed it.
    """
    import litellm

    from elbi.app import _PROVIDER_LABELS

    known = {str(p.value if hasattr(p, "value") else p) for p in litellm.provider_list}
    unknown = sorted(set(_PROVIDER_LABELS) - known)
    assert not unknown, f"labelled providers LiteLLM does not have: {unknown}"


def test_every_logo_alias_names_a_provider_litellm_can_route() -> None:
    """The picker's logo aliases are keyed on provider names LiteLLM emits.

    The web app maps a few provider names onto another brand's logo, because the name is
    not the brand: `watsonx` is IBM's, `triton` is NVIDIA's, `dashscope` is Alibaba
    Cloud's, `fireworks_ai` carries a routing suffix. Those keys only ever match what
    `_model_provider` returns, so a provider LiteLLM renames leaves a dead entry
    matching nothing. Since the rule is a real logo or none at all, the symptom is a
    model quietly losing its logo. Checked from Python because this is the side that
    knows the provider names.
    """
    import re
    from pathlib import Path

    import litellm

    web = Path(__file__).resolve().parents[1] / "web"
    source = (web / "src" / "components" / "ProviderIcon.tsx").read_text()
    body = re.search(
        r"ICON_ALIASES: Record<string, string> = \{(.*?)\n\}", source, re.S
    )
    assert body, "could not find ICON_ALIASES in ProviderIcon.tsx"
    aliases = re.findall(r'^\s*"?([\w.-]+)"?:\s*"([\w.-]+)",', body.group(1), re.M)
    assert aliases, "ICON_ALIASES parsed as empty; the guard would pass vacuously"

    known = {str(getattr(p, "value", p)) for p in litellm.provider_list}
    unknown = sorted(name for name, _ in aliases if name not in known)
    assert not unknown, f"logo aliases for providers LiteLLM cannot route: {unknown}"


def test_the_app_reports_its_version_and_how_to_upgrade_it(tmp_path) -> None:
    """The interface has no other way to know either.

    A browser cannot see whether the process behind it came from a container, a uv tool
    install or a virtualenv, and the upgrade command differs for each, so the server
    works it out and says.
    """
    from pathlib import Path

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'version.db'}")
    app = create_app(load_datasets=lambda: {"d": []}, client=_Scripted(), store=store)
    with TestClient(app) as http:
        body = http.get("/api/version").json()

    assert body["version"]
    assert body["package"] in {"elbi", "elbi-cli"}
    assert body["install"]["kind"]


def test_the_version_endpoint_does_not_ask_pypi_anything(tmp_path, monkeypatch) -> None:
    """The whole design in one assertion.

    Reporting the running version is local. Asking whether a newer one exists is an
    outbound request, and the server never makes one: `elbi update` is where a person
    asks that, deliberately. A future change that folded a check in here would turn
    every page load into a connection nobody requested.
    """
    from pathlib import Path

    import httpx

    def refuse(*args, **kwargs):
        raise AssertionError("the version endpoint made a network request")

    monkeypatch.setattr(httpx, "get", refuse)
    monkeypatch.setattr(httpx, "post", refuse)

    store = open_store(f"sqlite:{Path(str(tmp_path)) / 'version2.db'}")
    app = create_app(load_datasets=lambda: {"d": []}, client=_Scripted(), store=store)
    with TestClient(app) as http:
        assert http.get("/api/version").status_code == 200
