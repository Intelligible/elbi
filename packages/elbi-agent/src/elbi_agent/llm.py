"""The LLM seam the runtime drives its loop against.

The runtime is provider-agnostic: it speaks to the model through :class:`LLMClient`,
one method that takes the running transcript plus the available tools and returns
the model's next move (free text and/or tool calls, and whether the model is done).
A fake client makes the loop fully testable without a network or an API key; the
Anthropic client (under the ``anthropic`` extra) maps the same shape onto the
Messages API.

Transcript messages use the Anthropic content-block shape (``role`` plus a list of
``text`` / ``tool_use`` / ``tool_result`` blocks); it is the most direct mapping for
the default client, and other providers translate from it.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class ToolSpec:
    """A tool offered to the model: its name, what it does, and its argument schema.

    Anthropic's guidance is that clear, example-rich tool descriptions outperform
    terse ones, so ``description`` carries the usage notes the model needs.
    """

    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    """One tool invocation the model requested, with the id needed to reply to it."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class Usage:
    """Token counts and cost for one model call, summable across a run.

    ``cache_read``/``cache_write`` track prompt-cache hits and writes (0 when the
    provider does not report them); ``cost`` is in US dollars (0 when it cannot be
    computed). Adding two usages sums the counts, so a loop totals its calls.
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    cost: float = 0.0

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cache_read_tokens=self.cache_read_tokens + other.cache_read_tokens,
            cache_write_tokens=self.cache_write_tokens + other.cache_write_tokens,
            cost=self.cost + other.cost,
        )


@dataclass(frozen=True)
class StepDelta:
    """An incremental slice of a streaming model turn: text as it is generated.

    A client that supports streaming yields these as the model produces tokens, then a
    final :class:`Step` with the assembled turn. A client without streaming just returns
    the ``Step``.
    """

    text: str


@dataclass(frozen=True)
class Step:
    """The model's output for one turn of the loop.

    ``end`` is true when the model stopped for any reason other than calling a tool
    (so the loop should not feed results back and continue). ``usage`` is the call's
    token counts and cost when the client reports them, so the loop can total them.
    """

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    end: bool = False
    usage: Usage | None = None


def _image_block(data_url: str) -> dict[str, Any] | None:
    """Parse a ``data:<mime>;base64,<data>`` URL into an Anthropic image block.

    Returns ``None`` for anything that is not a base64 data URL, so a malformed
    attachment is dropped rather than corrupting the request.
    """
    if not data_url.startswith("data:") or ";base64," not in data_url:
        return None
    header, data = data_url.split(",", 1)
    media_type = header[len("data:") :].split(";", 1)[0] or "image/png"
    return {
        "type": "image",
        "source": {"type": "base64", "media_type": media_type, "data": data},
    }


@dataclass
class Transcript:
    """The running conversation, in the content-block shape clients consume."""

    system: str
    messages: list[dict[str, Any]] = field(default_factory=list)

    def add_user_text(self, text: str) -> None:
        """Append a plain user message (the opening question)."""
        self.messages.append(
            {"role": "user", "content": [{"type": "text", "text": text}]}
        )

    def add_user_message(self, text: str, images: Sequence[str] = ()) -> None:
        """Append a user message with text and any attached images.

        ``images`` are ``data:<mime>;base64,<data>`` URLs (as a browser produces from an
        upload). Each becomes an Anthropic image block; the LiteLLM adapter rewrites
        them to the OpenAI ``image_url`` shape, so one transcript drives any provider.
        """
        content: list[dict[str, Any]] = []
        if text:
            content.append({"type": "text", "text": text})
        for data_url in images:
            block = _image_block(data_url)
            if block is not None:
                content.append(block)
        self.messages.append({"role": "user", "content": content})

    def add_assistant_text(self, text: str) -> None:
        """Append a prior assistant answer when replaying history (no tool calls)."""
        self.messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": text}]}
        )

    def add_assistant(self, step: Step) -> None:
        """Record the model's turn so its tool calls can be answered next."""
        content: list[dict[str, Any]] = []
        if step.text:
            content.append({"type": "text", "text": step.text})
        for call in step.tool_calls:
            content.append(
                {
                    "type": "tool_use",
                    "id": call.id,
                    "name": call.name,
                    "input": call.arguments,
                }
            )
        self.messages.append({"role": "assistant", "content": content})

    def add_tool_results(self, results: Sequence[tuple[str, str]]) -> None:
        """Append the outputs of the model's tool calls, keyed by call id."""
        self.messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": call_id, "content": output}
                    for call_id, output in results
                ],
            }
        )


class LLMClient(Protocol):
    """A single conversational step: given the transcript and tools, decide the move."""

    def step(self, transcript: Transcript, tools: Sequence[ToolSpec]) -> Step:
        """Return the model's next turn (text and/or tool calls, and whether done)."""
        ...
