"""A model-agnostic :class:`~elbi_agent.llm.LLMClient` backed by LiteLLM.

Available under the ``litellm`` extra. LiteLLM presents one OpenAI-shaped interface over
~100 providers (OpenAI, Anthropic, Google, Bedrock, Azure, OpenRouter, Ollama, local,
...), selected by a ``provider/model`` string, so the runtime's loop can run on any of
them without change. This adapter translates the runtime's Anthropic content-block
transcript into LiteLLM's OpenAI messages, offers the tools as OpenAI functions, and
maps the response back to a :class:`~elbi_agent.llm.Step`.

The default install stays Anthropic-only via
:class:`~elbi_agent.anthropic_client.AnthropicClient`; this client is the opt-in
path to every other provider.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from typing import Any, ClassVar

from .llm import Step, StepDelta, ToolCall, ToolSpec, Transcript, Usage

logger = logging.getLogger("elbi")


class LiteLLMClient:
    """Drive the runtime's loop with any LiteLLM-supported model.

    ``model`` is a LiteLLM model string that names the provider (``openai/gpt-5``,
    ``anthropic/claude-sonnet-5``, ``gemini/gemini-2.5-pro``, ``bedrock/...``,
    ``ollama/llama3``, ``openrouter/...``). ``api_key``/``base_url``/``api_version`` are
    passed through for hosted or self-hosted endpoints; unset, LiteLLM reads the
    provider's conventional environment variable. ``num_retries`` retries transient
    failures with LiteLLM's own backoff, so a flaky provider does not drop a turn.
    """

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: str | None = None,
        temperature: float | None = None,
        max_tokens: int = 4096,
        num_retries: int = 5,
        native_tool_calling: bool | None = None,
        fallbacks: Sequence[str] | None = None,
        reasoning_effort: str | None = None,
        caching_prompt: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> None:
        import litellm

        # Drop parameters a given provider does not support rather than erroring, so one
        # request shape works across providers (LiteLLM's cross-provider normalisation).
        litellm.drop_params = True
        # Let LiteLLM repair message shapes a provider needs but an OpenAI-shaped caller
        # does not send. The case that matters here is Anthropic: extended thinking with
        # tool calling requires the previous assistant turn's thinking blocks to be
        # replayed, which an OpenAI-shaped transcript has nowhere to carry. Without
        # this,
        # turning thinking on for a Claude model breaks every tool-using turn -- and
        # every
        # turn this app takes carries tools.
        litellm.modify_params = True
        self._litellm = litellm
        self._model = model
        self._api_key = api_key
        self._base_url = base_url
        self._api_version = api_version
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._num_retries = num_retries
        # Whether the model has native function-calling. Left None, it is read from
        # LiteLLM's capability table; without it, tools are driven through the prompt.
        self._native_tool_calling = native_tool_calling
        # Models to fail over to, in order, when the primary errors (LiteLLM does so).
        self._fallbacks = list(fallbacks or ())
        # A reasoning model's effort ("low"/"medium"/"high"); None lets the model pick.
        self._reasoning_effort = reasoning_effort
        # Mark the (large) system prompt as a prompt-cache breakpoint on providers that
        # support it (Anthropic), cutting cost and latency on repeated turns.
        self._caching_prompt = caching_prompt
        self._extra = dict(extra or {})

    #: Models found to refuse function tools alongside a reasoning effort, discovered
    #: from
    #: the provider's own rejection rather than from a table. Class-level so one model's
    #: lesson is not relearned by every client instance built for it.
    _tools_need_no_reasoning: ClassVar[set[str]] = set()

    def _tool_reasoning_conflict(self, exc: Exception, params: dict[str, Any]) -> bool:
        """Whether the provider just refused tools *because* of the reasoning effort.

        The authority on this is the provider, not a capability table. LiteLLM's
        registry
        reports gpt-5.6-terra as supporting both reasoning and function calling, which
        is
        true separately and false together on /v1/chat/completions -- there is no field
        that says so, and the only way to find out is to be told.

        So the rejection is read, narrowly: a 4xx naming both concerns. Matching loosely
        here would silently strip reasoning from a model that failed for some other
        reason, which is a worse outcome than surfacing the error.

        Note what is deliberately *not* required: that we sent an effort. The refusal
        happens when the field is absent, because the model then applies one of its
        own -- which is the shape of the whole problem, and why the remedy is to send
        `none` rather than to stop sending something. An earlier version of this guard
        required a value here, and so never fired on the case that actually occurs.
        """
        if "tools" not in params:
            return False
        if params.get("reasoning_effort") == "none":
            return False  # already off; the refusal is about something else
        status = getattr(exc, "status_code", None)
        if not (isinstance(status, int) and 400 <= status < 500):
            return False
        detail = str(exc).lower()
        return "reasoning_effort" in detail and "tool" in detail

    def _without_reasoning(self, params: dict[str, Any]) -> dict[str, Any]:
        """The same request with the effort explicitly off, and the lesson recorded.

        Explicitly ``none`` rather than absent: these models apply an effort of their
        own
        when the field is omitted, which is what the rejection was about.
        """
        self._tools_need_no_reasoning.add(self._model)
        logger.info(
            "%s refuses function tools with a reasoning effort; retrying with "
            "reasoning_effort='none' and remembering it for this model",
            self._model,
        )
        return {**params, "reasoning_effort": "none"}

    def _params(self) -> dict[str, Any]:
        """The shared completion parameters (model, limits, credentials, options)."""
        params: dict[str, Any] = {
            "model": self._model,
            "max_tokens": self._max_tokens,
            "num_retries": self._num_retries,
            **self._extra,
        }
        if self._fallbacks:
            params["fallbacks"] = self._fallbacks
        if self._reasoning_effort:
            # A model already known to refuse the pairing gets `none` from the start, so
            # only the turn that discovered it pays for a rejected request.
            params["reasoning_effort"] = (
                "none"
                if self._model in self._tools_need_no_reasoning
                else self._reasoning_effort
            )
        if self._temperature is not None:
            params["temperature"] = self._temperature
        if self._api_key is not None:
            params["api_key"] = self._api_key
        if self._base_url is not None:
            params["base_url"] = self._base_url
        if self._api_version is not None:
            params["api_version"] = self._api_version
        return params

    def step(self, transcript: Transcript, tools: Sequence[ToolSpec]) -> Step:
        """Send the transcript to the model via LiteLLM and parse its next turn."""
        messages = to_openai_messages(transcript)
        params = self._params()
        if tools and not self._uses_native_tools():
            return self._prompt_tool_step(messages, tools, params)
        # Tag the system prompt as a cache breakpoint (native path only; the non-native
        # path folds the system prompt into text and those models lack prompt caching).
        if self._caching_prompt:
            messages = _mark_cacheable(messages)
        params["messages"] = messages
        tool_params = to_openai_tools(tools)
        if tool_params:
            params["tools"] = tool_params
        try:
            response = self._litellm.completion(**params)
        except Exception as exc:
            if not self._tool_reasoning_conflict(exc, params):
                raise
            response = self._litellm.completion(**self._without_reasoning(params))
        return _to_step(response, self._usage(response))

    def stream(
        self, transcript: Transcript, tools: Sequence[ToolSpec]
    ) -> Iterator[StepDelta | Step]:
        """Stream a turn: yield text as the model produces it, then the final Step.

        Only the native tool path streams; a model driven through the prompt (non-native
        tools) returns its Step whole, since its ``<function=...>`` blocks are not worth
        surfacing token by token. Deltas carry the model's prose (its reasoning); the
        assembled response parses to the same tool calls and usage as ``step``.
        """
        messages = to_openai_messages(transcript)
        params = self._params()
        if tools and not self._uses_native_tools():
            yield self._prompt_tool_step(messages, tools, params)
            return
        if self._caching_prompt:
            messages = _mark_cacheable(messages)
        params["messages"] = messages
        tool_params = to_openai_tools(tools)
        if tool_params:
            params["tools"] = tool_params
        params["stream"] = True
        params["stream_options"] = {"include_usage": True}
        chunks: list[Any] = []
        # The first chunk is what raises, so the retry has to wrap opening the stream
        # rather than the loop: nothing has been yielded to the caller yet, so swapping
        # the request and starting again is invisible downstream.
        try:
            stream = iter(self._litellm.completion(**params))
            first = next(stream, None)
        except Exception as exc:
            if not self._tool_reasoning_conflict(exc, params):
                raise
            stream = iter(self._litellm.completion(**self._without_reasoning(params)))
            first = next(stream, None)
        for chunk in [] if first is None else [first, *stream]:
            chunks.append(chunk)
            choices = getattr(chunk, "choices", None)
            delta = choices[0].delta if choices else None
            text = getattr(delta, "content", None) if delta is not None else None
            if text:
                yield StepDelta(text=text)
        # Reassemble the streamed chunks into one response, then parse it exactly as the
        # non-streaming path so tool calls and usage are identical.
        response = self._litellm.stream_chunk_builder(chunks, messages=messages)
        yield _to_step(response, self._usage(response))

    def _usage(self, response: Any) -> Usage:
        """Token counts and cost for a completion, from the response + cost table."""
        raw = getattr(response, "usage", None)
        details = getattr(raw, "prompt_tokens_details", None)
        try:
            cost = float(
                self._litellm.completion_cost(completion_response=response) or 0.0
            )
        except Exception:
            cost = 0.0
        return Usage(
            prompt_tokens=int(getattr(raw, "prompt_tokens", 0) or 0),
            completion_tokens=int(getattr(raw, "completion_tokens", 0) or 0),
            cache_read_tokens=int(getattr(details, "cached_tokens", 0) or 0),
            cache_write_tokens=int(getattr(raw, "cache_creation_input_tokens", 0) or 0),
            cost=cost,
        )

    def _uses_native_tools(self) -> bool:
        """Use the native tool API? From config, else LiteLLM's capability table."""
        if self._native_tool_calling is not None:
            return self._native_tool_calling
        try:
            return bool(self._litellm.supports_function_calling(model=self._model))
        except Exception:
            return True  # assume native for an unknown model; opt out via config

    def _prompt_tool_step(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[ToolSpec],
        params: dict[str, Any],
    ) -> Step:
        """Drive tools through the prompt for a model without native tool calling."""
        from .non_native_tools import STOP_WORD, parse_calls, to_prompt_messages

        params["messages"] = to_prompt_messages(messages, tools)
        params["stop"] = [STOP_WORD]
        response = self._litellm.completion(**params)
        text = response.choices[0].message.content
        # LiteLLM strips the stop word; re-add it so the final call block closes.
        if text and "<function=" in text and STOP_WORD not in text:
            text = text + STOP_WORD
        prose, calls = parse_calls(text)
        return Step(
            text=prose,
            tool_calls=tuple(calls),
            end=not calls,
            usage=self._usage(response),
        )


def to_openai_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    """The tools as OpenAI function definitions (empty list when there are none)."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema,
            },
        }
        for tool in tools
    ]


def to_openai_messages(transcript: Transcript) -> list[dict[str, Any]]:
    """Translate the Anthropic content-block transcript into OpenAI chat messages.

    The runtime keeps history in Anthropic's block shape (text / tool_use /
    tool_result); OpenAI's shape carries assistant tool calls on the message and tool
    results as their own ``tool`` messages. This maps between the two for any provider.
    """
    out: list[dict[str, Any]] = []
    if transcript.system:
        out.append({"role": "system", "content": transcript.system})
    for message in transcript.messages:
        blocks = message.get("content") or []
        if message.get("role") == "assistant":
            out.append(_assistant_message(blocks))
        else:
            out.extend(_user_messages(blocks))
    return out


def _assistant_message(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    text_parts: list[str] = []
    tool_calls: list[dict[str, Any]] = []
    for block in blocks:
        if block.get("type") == "text":
            text_parts.append(block["text"])
        elif block.get("type") == "tool_use":
            tool_calls.append(
                {
                    "id": block["id"],
                    "type": "function",
                    "function": {
                        "name": block["name"],
                        "arguments": json.dumps(block.get("input") or {}),
                    },
                }
            )
    message: dict[str, Any] = {
        "role": "assistant",
        "content": "\n".join(text_parts) or None,
    }
    if tool_calls:
        message["tool_calls"] = tool_calls
    return message


def _user_messages(blocks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    # A user turn is either plain text (the question) or the results of the model's tool
    # calls; the latter become one OpenAI ``tool`` message each, keyed by call id.
    tool_results = [b for b in blocks if b.get("type") == "tool_result"]
    if tool_results:
        return [
            {
                "role": "tool",
                "tool_call_id": block["tool_use_id"],
                "content": _as_text(block.get("content")),
            }
            for block in tool_results
        ]
    text = "\n".join(b["text"] for b in blocks if b.get("type") == "text")
    images = [b for b in blocks if b.get("type") == "image"]
    if images:
        # Rebuild each Anthropic image block into the OpenAI image_url data URL; a mixed
        # text-and-image turn becomes one message with a content array.
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        for image in images:
            source = image.get("source", {})
            media = source.get("media_type", "image/png")
            url = f"data:{media};base64,{source.get('data', '')}"
            content.append({"type": "image_url", "image_url": {"url": url}})
        return [{"role": "user", "content": content}]
    return [{"role": "user", "content": text}]


def _as_text(content: Any) -> str:
    """A tool result as text (it is a string in this runtime; be defensive anyway)."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(b.get("text", "") for b in content if isinstance(b, dict))
    return str(content)


def _mark_cacheable(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark the system prompt as an ephemeral prompt-cache breakpoint.

    Anthropic caches everything up to a ``cache_control`` marker, so tagging the large,
    stable system prompt lets repeated turns reuse it at a fraction of the token cost.
    Providers without prompt caching ignore the marker (LiteLLM drops it), so it is safe
    across models. Only the system message is tagged; the per-turn messages are not.
    """
    out: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if message.get("role") == "system" and isinstance(content, str):
            out.append(
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            )
        else:
            out.append(message)
    return out


def _to_step(response: Any, usage: Usage | None = None) -> Step:
    """Map a LiteLLM completion response to the runtime's :class:`Step`."""
    choice = response.choices[0]
    message = choice.message
    calls: list[ToolCall] = []
    for tool_call in getattr(message, "tool_calls", None) or []:
        raw_args = tool_call.function.arguments or "{}"
        try:
            arguments = json.loads(raw_args)
        except (json.JSONDecodeError, TypeError):
            arguments = {}
        calls.append(
            ToolCall(id=tool_call.id, name=tool_call.function.name, arguments=arguments)
        )
    return Step(
        text=message.content,
        tool_calls=tuple(calls),
        end=choice.finish_reason != "tool_calls",
        usage=usage,
    )
