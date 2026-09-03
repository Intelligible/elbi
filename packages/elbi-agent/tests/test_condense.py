"""The rolling condenser summarizes older turns and the runtime carries the note."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from elbi_agent import Step, ToolSpec, Transcript, condense_turns, stream
from elbi_agent.runtime import Workspace


class _Recorder:
    """A client that records each transcript it is asked to step and replays steps."""

    def __init__(self, *steps: Step) -> None:
        self._steps = list(steps)
        self._i = 0
        self.seen: list[Transcript] = []

    def step(self, transcript: Transcript, tools: Sequence[ToolSpec]) -> Step:
        self.seen.append(transcript)
        step = self._steps[self._i]
        self._i += 1
        return step


def _text_of(message: dict[str, Any]) -> str:
    """The message text, whether stored as a string or Anthropic content blocks."""
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    return " ".join(b.get("text", "") for b in content if isinstance(b, dict))


def test_condense_turns_summarizes_via_the_client() -> None:
    client = _Recorder(Step(text="USER_CONTEXT: wants the effect of x on y"))
    turns = [("user", "what is the effect of x on y?"), ("assistant", "about 0.6")]
    summary = condense_turns(client, turns)
    assert summary == "USER_CONTEXT: wants the effect of x on y"
    # the turns were rendered into the summarization prompt
    assert "effect of x on y" in _text_of(client.seen[-1].messages[-1])


def test_condense_turns_empty_is_blank() -> None:
    client = _Recorder(Step(text="unused"))
    assert condense_turns(client, []) == ""
    assert condense_turns(client, [("user", "   ")]) == ""


def test_stream_carries_the_summary_into_context() -> None:
    # The summary is replayed as a background note before the recent window, so the
    # model sees the condensed earlier turns without replaying them verbatim.
    client = _Recorder(Step(text="done", tool_calls=(), end=True))
    events = list(
        stream(
            "follow up",
            Workspace(datasets={}),
            client,
            summary="USER_CONTEXT: earlier goal",
        )
    )
    assert any(e.kind == "result" for e in events)
    replayed = " ".join(_text_of(m) for m in client.seen[0].messages)
    assert "USER_CONTEXT: earlier goal" in replayed
    assert "condensed" in replayed  # framed as background, not verified fact
