"""An :class:`~elbi_agent.llm.LLMClient` backed by the Anthropic Messages API.

Available under the ``anthropic`` extra. The API key is read from the environment
(``ANTHROPIC_API_KEY``) unless passed, so bring-your-own-key is the default. The
transcript already uses Anthropic's content-block shape, so the mapping is a thin
translation of the request and the response.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, cast

import anthropic
from anthropic.types import TextBlock, ToolUseBlock

from .llm import Step, StepDelta, ToolCall, ToolSpec, Transcript, Usage

#: A balanced default for a multi-step loop; override with ``model=`` for a
#: stronger or cheaper tier.
DEFAULT_MODEL = "claude-sonnet-5"


@dataclass(frozen=True)
class ModelInfo:
    """A selectable model: its API id, a display name, provider, and default flag."""

    id: str
    name: str
    provider: str = "anthropic"
    default: bool = False


#: The models offered in the picker: the current-generation Claude family, with Sonnet 5
#: the default. Only these ids are honoured per request.
MODELS: tuple[ModelInfo, ...] = (
    ModelInfo("claude-opus-4-8", "Claude Opus 4.8"),
    ModelInfo("claude-sonnet-5", "Claude Sonnet 5", default=True),
    ModelInfo("claude-haiku-4-5", "Claude Haiku 4.5"),
    ModelInfo("claude-fable-5", "Claude Fable 5"),
)


class AnthropicClient:
    """Drive the runtime's loop with a Claude model."""

    def __init__(
        self,
        *,
        model: str = DEFAULT_MODEL,
        max_tokens: int = 4096,
        api_key: str | None = None,
    ) -> None:
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    def _request(self, tools: Sequence[ToolSpec]) -> dict[str, Any]:
        """The tool spec passed to every request/stream call."""
        return {
            "tools": cast(
                "Any",
                [
                    {
                        "name": tool.name,
                        "description": tool.description,
                        "input_schema": tool.input_schema,
                    }
                    for tool in tools
                ],
            )
        }

    def step(self, transcript: Transcript, tools: Sequence[ToolSpec]) -> Step:
        """Send the transcript to the model and parse its next turn."""
        response = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=transcript.system,
            messages=cast("Any", transcript.messages),
            **self._request(tools),
        )
        return _to_step(response)

    def stream(
        self, transcript: Transcript, tools: Sequence[ToolSpec]
    ) -> Iterator[StepDelta | Step]:
        """Stream a turn: yield text as the model produces it, then the final Step."""
        with self._client.messages.stream(
            model=self._model,
            max_tokens=self._max_tokens,
            system=transcript.system,
            messages=cast("Any", transcript.messages),
            **self._request(tools),
        ) as streamed:
            for text in streamed.text_stream:
                if text:
                    yield StepDelta(text=text)
            final = streamed.get_final_message()
        yield _to_step(final)


def _to_step(response: Any) -> Step:
    """Parse an Anthropic message (from create or a stream) into a :class:`Step`."""
    text: str | None = None
    calls: list[ToolCall] = []
    for block in response.content:
        if isinstance(block, TextBlock):
            text = (text or "") + block.text
        elif isinstance(block, ToolUseBlock):
            calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=cast("dict[str, Any]", block.input),
                )
            )
    raw = response.usage
    usage = Usage(
        prompt_tokens=getattr(raw, "input_tokens", 0) or 0,
        completion_tokens=getattr(raw, "output_tokens", 0) or 0,
        cache_read_tokens=getattr(raw, "cache_read_input_tokens", 0) or 0,
        cache_write_tokens=getattr(raw, "cache_creation_input_tokens", 0) or 0,
    )
    return Step(
        text=text,
        tool_calls=tuple(calls),
        end=response.stop_reason != "tool_use",
        usage=usage,
    )
