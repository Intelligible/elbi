"""The LiteLLM adapter's translation, tested without a network call.

The risk in a model-agnostic client is the mapping between the runtime's Anthropic
content-block transcript and LiteLLM's OpenAI shape (and back). These tests pin that
mapping; the live provider call is exercised separately.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

from elbi_agent.litellm_client import (
    _to_step,
    to_openai_messages,
    to_openai_tools,
)
from elbi_agent.llm import Step, ToolCall, ToolSpec, Transcript


def test_transcript_maps_to_openai_messages() -> None:
    transcript = Transcript(system="You are helpful.")
    transcript.add_user_text("effect of x on y?")
    transcript.add_assistant(
        Step(
            text="checking",
            tool_calls=(ToolCall("call-1", "run_code", {"code": "1+1"}),),
        )
    )
    transcript.add_tool_results([("call-1", "2")])

    messages = to_openai_messages(transcript)
    assert messages == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "effect of x on y?"},
        {
            "role": "assistant",
            "content": "checking",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "run_code",
                        "arguments": json.dumps({"code": "1+1"}),
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": "2"},
    ]


def test_assistant_text_only_has_no_tool_calls() -> None:
    transcript = Transcript(system="S")
    transcript.add_assistant_text("a prior answer")
    messages = to_openai_messages(transcript)
    assert messages[-1] == {"role": "assistant", "content": "a prior answer"}
    assert "tool_calls" not in messages[-1]


def test_tools_map_to_openai_functions() -> None:
    schema = {"type": "object", "properties": {"code": {"type": "string"}}}
    tools = to_openai_tools([ToolSpec("run_code", "Run code.", schema)])
    assert tools == [
        {
            "type": "function",
            "function": {
                "name": "run_code",
                "description": "Run code.",
                "parameters": schema,
            },
        }
    ]
    assert to_openai_tools([]) == []


def _response(content: str | None, tool_calls: list, finish_reason: str):
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason=finish_reason)]
    )


def _tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id, function=SimpleNamespace(name=name, arguments=arguments)
    )


def test_text_response_maps_to_a_finished_step() -> None:
    step = _to_step(_response("here is the answer", [], "stop"))
    assert step.text == "here is the answer"
    assert step.tool_calls == () and step.end is True


def test_tool_call_response_maps_to_an_unfinished_step() -> None:
    resp = _response(
        None,
        [_tool_call("c1", "run_code", json.dumps({"code": "x"}))],
        "tool_calls",
    )
    step = _to_step(resp)
    assert step.end is False
    assert step.tool_calls == (ToolCall("c1", "run_code", {"code": "x"}),)


def test_malformed_tool_arguments_default_to_empty() -> None:
    resp = _response(None, [_tool_call("c1", "answer", "not json")], "tool_calls")
    step = _to_step(resp)
    assert step.tool_calls[0].arguments == {}


class _FakeLiteLLM:
    """A stand-in for the litellm module: records kwargs, returns fixed text."""

    def __init__(self, content: str) -> None:
        self._content = content
        self.drop_params = True
        self.calls: list[dict] = []

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self._content, tool_calls=None)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=message, finish_reason="stop")]
        )

    def supports_function_calling(self, model: str) -> bool:  # pragma: no cover
        return True


def test_non_native_model_drives_tools_through_the_prompt() -> None:
    # With native tool calling off, the client folds tools into the prompt, sets a
    # stop word, passes no `tools` param, and parses the <function=...> reply back into
    # a tool call, so a model without native FC still drives the loop.
    from elbi_agent.litellm_client import LiteLLMClient

    # the stop word trims the trailing </function>, which the client re-adds
    reply = "Running.\n<function=run_code>\n<parameter=code>1+1</parameter>"
    client = LiteLLMClient(model="ollama/llama3", native_tool_calling=False)
    fake = _FakeLiteLLM(reply)
    client._litellm = fake  # type: ignore[attr-defined]

    tool = ToolSpec("run_code", "Run code.", {"type": "object"})
    transcript = Transcript(system="S")
    transcript.add_user_text("run 1+1")
    step = client.step(transcript, [tool])

    sent = fake.calls[-1]
    assert "tools" not in sent  # no native tools param
    assert sent["stop"] == ["</function>"]
    assert "## run_code" in sent["messages"][0]["content"]  # tools in the system prompt
    assert step.end is False
    assert step.tool_calls == (ToolCall("call_0", "run_code", {"code": "1+1"}),)


def test_fallbacks_are_passed_through_to_completion() -> None:
    from elbi_agent.litellm_client import LiteLLMClient

    client = LiteLLMClient(model="openai/gpt-5", fallbacks=["openai/gpt-5-mini"])
    fake = _FakeLiteLLM("hello")
    client._litellm = fake  # type: ignore[attr-defined]
    transcript = Transcript(system="S")
    transcript.add_user_text("q")
    client.step(transcript, [])
    assert fake.calls[-1]["fallbacks"] == ["openai/gpt-5-mini"]


def test_prompt_caching_marks_the_system_message() -> None:
    # The large, stable system prompt is tagged as an ephemeral cache breakpoint so
    # repeated turns reuse it cheaply; providers without caching drop the marker.
    from elbi_agent.litellm_client import LiteLLMClient

    client = LiteLLMClient(model="anthropic/claude-sonnet-5")  # caching on by default
    fake = _FakeLiteLLM("hi")
    client._litellm = fake  # type: ignore[attr-defined]
    transcript = Transcript(system="S")
    transcript.add_user_text("q")
    client.step(transcript, [])
    system = fake.calls[-1]["messages"][0]
    assert system["role"] == "system"
    assert system["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_reasoning_effort_passed_through() -> None:
    from elbi_agent.litellm_client import LiteLLMClient

    client = LiteLLMClient(
        model="openai/o3", reasoning_effort="high", caching_prompt=False
    )
    fake = _FakeLiteLLM("hi")
    client._litellm = fake  # type: ignore[attr-defined]
    transcript = Transcript(system="S")
    transcript.add_user_text("q")
    client.step(transcript, [])
    assert fake.calls[-1]["reasoning_effort"] == "high"


def test_usage_is_read_from_the_response() -> None:
    from elbi_agent.litellm_client import LiteLLMClient

    class _WithUsage(_FakeLiteLLM):
        def completion(self, **kwargs):
            self.calls.append(kwargs)
            message = SimpleNamespace(content="hi", tool_calls=None)
            usage = SimpleNamespace(
                prompt_tokens=30, completion_tokens=12, prompt_tokens_details=None
            )
            return SimpleNamespace(
                choices=[SimpleNamespace(message=message, finish_reason="stop")],
                usage=usage,
            )

        def completion_cost(self, completion_response=None):
            return 0.005

    client = LiteLLMClient(model="openai/gpt-5", caching_prompt=False)
    client._litellm = _WithUsage("hi")  # type: ignore[attr-defined]
    transcript = Transcript(system="S")
    transcript.add_user_text("q")
    step = client.step(transcript, [])
    assert step.usage is not None
    assert step.usage.prompt_tokens == 30 and step.usage.completion_tokens == 12
    assert abs(step.usage.cost - 0.005) < 1e-9


# -- learning a provider's tools-and-reasoning constraint --------------------------


class _Rejects:
    """A fake LiteLLM that refuses tools alongside a reasoning effort, as OpenAI does.

    Records every call so a test can assert what the retry actually sent, which is the
    part that matters: retrying with the effort merely absent would fail again, because
    these models apply an effort of their own when the field is omitted.
    """

    def __init__(self, detail: str, status: int = 400) -> None:
        self.calls: list[dict] = []
        self._detail, self._status = detail, status
        self.drop_params = self.modify_params = False

    def completion(self, **params):  # type: ignore[no-untyped-def]
        self.calls.append(params)
        # Absent counts as a refusal, which is what the real provider does: with no
        # field the model applies an effort of its own, and only an explicit "none"
        # satisfies it. Modelling this the other way round is what let a broken guard
        # pass its tests.
        if params.get("tools") and params.get("reasoning_effort") != "none":
            exc = RuntimeError(self._detail)
            exc.status_code = self._status  # type: ignore[attr-defined]
            raise exc
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )

    def stream_chunk_builder(self, chunks, messages=None):  # type: ignore[no-untyped-def]
        return self.completion(model="x")


def _client(fake, effort="high", forget=True):  # type: ignore[no-untyped-def]
    """A client wired to a fake LiteLLM.

    ``forget`` resets what has been learned about models. The set is class-level on
    purpose -- one model's lesson should outlive the client that learned it -- so a test
    of that persistence opts out of the reset rather than fighting it.
    """
    from elbi_agent.litellm_client import LiteLLMClient

    if forget:
        LiteLLMClient._tools_need_no_reasoning.clear()
    client = LiteLLMClient.__new__(LiteLLMClient)
    client._litellm = fake
    client._model = "openai/gpt-5.6-terra"
    client._max_tokens, client._num_retries = 64, 0
    client._api_key = client._base_url = client._api_version = None
    client._temperature = client._fallbacks = None
    client._fallbacks = []
    client._reasoning_effort = effort
    client._caching_prompt = False
    client._native_tool_calling = True
    client._extra = {}
    return client


_REFUSAL = (
    "Function tools with reasoning_effort are not supported for gpt-5.6-terra in "
    "/v1/chat/completions."
)


def test_a_refused_pairing_is_retried_with_the_effort_explicitly_off() -> None:
    """The provider is the authority, so the constraint is learned from its rejection.

    LiteLLM's registry reports this model as supporting reasoning *and* function
    calling, which is true separately and false together: no capability field says so,
    so a table cannot know it and only the rejection can teach it.
    """
    fake = _Rejects(_REFUSAL)
    client = _client(fake)
    tools = (ToolSpec("ping", "p", {"type": "object", "properties": {}}),)

    step = client.step(Transcript(system="s"), tools)

    assert step.text == "ok"
    assert len(fake.calls) == 2, "expected one rejection and one retry"
    assert fake.calls[0]["reasoning_effort"] == "high"
    # Explicitly off, not merely absent: omitting it lets the model pick its own.
    assert fake.calls[1]["reasoning_effort"] == "none"


def test_the_lesson_is_remembered_so_only_the_first_turn_pays_for_it() -> None:
    fake = _Rejects(_REFUSAL)
    tools = (ToolSpec("ping", "p", {"type": "object", "properties": {}}),)

    _client(fake).step(Transcript(system="s"), tools)  # learns the constraint
    fake.calls.clear()
    # A second client for the same model, as a fresh request would build.
    _client(fake, effort="high", forget=False).step(Transcript(system="s"), tools)

    assert len(fake.calls) == 1, (
        "the second turn should not repeat the rejected request"
    )
    assert fake.calls[0]["reasoning_effort"] == "none"


def test_an_unrelated_rejection_is_not_quietly_stripped_of_reasoning() -> None:
    """Matching loosely here would hide real errors behind a silent downgrade."""
    fake = _Rejects("Incorrect API key provided.", status=401)
    client = _client(fake)
    tools = (ToolSpec("ping", "p", {"type": "object", "properties": {}}),)

    try:
        client.step(Transcript(system="s"), tools)
    except RuntimeError as exc:
        assert "Incorrect API key" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("an auth failure must not be swallowed")

    assert len(fake.calls) == 1, "no retry belongs on an unrelated failure"


def test_the_refusal_is_caught_when_no_effort_was_sent_at_all() -> None:
    """The production case, and the one the first version of this guard missed.

    A profile that leaves the effort unset sends no `reasoning_effort` field, the model
    applies one of its own, and the request is refused. Requiring a value we had sent
    meant the retry never fired precisely when it was needed: the deployment surfaced
    the provider's complaint on every message instead of recovering from it.
    """
    fake = _Rejects(_REFUSAL)
    client = _client(fake, effort="")  # unset, as a fresh profile is
    tools = (ToolSpec("ping", "p", {"type": "object", "properties": {}}),)

    step = client.step(Transcript(system="s"), tools)

    assert step.text == "ok"
    assert len(fake.calls) == 2
    assert "reasoning_effort" not in fake.calls[0], "nothing was sent the first time"
    assert fake.calls[1]["reasoning_effort"] == "none", "and it is off explicitly after"


def test_a_refusal_while_already_off_is_surfaced_rather_than_looped() -> None:
    """If `none` was already sent, the complaint is about something else."""

    class AlwaysRefuses(_Rejects):
        def completion(self, **params):  # type: ignore[no-untyped-def]
            self.calls.append(params)
            exc = RuntimeError(_REFUSAL)
            exc.status_code = 400  # type: ignore[attr-defined]
            raise exc

    fake = AlwaysRefuses(_REFUSAL)
    client = _client(fake, effort="none")
    tools = (ToolSpec("ping", "p", {"type": "object", "properties": {}}),)

    try:
        client.step(Transcript(system="s"), tools)
    except RuntimeError:
        pass
    else:  # pragma: no cover
        raise AssertionError("the refusal should have propagated")

    assert len(fake.calls) == 1, "retrying the same request would be a loop"
