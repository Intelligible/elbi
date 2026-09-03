"""A streaming client surfaces reasoning deltas; a non-streaming one still works."""

from __future__ import annotations

from collections.abc import Sequence

from elbi_agent import Step, StepDelta, ToolCall, ToolSpec, Transcript, stream
from elbi_agent.runtime import Workspace


class _Streamer:
    """A client that streams two text slices, then a final answer step."""

    def step(
        self, transcript: Transcript, tools: Sequence[ToolSpec]
    ) -> Step:  # pragma: no cover
        return Step(tool_calls=(ToolCall("a", "answer", {"summary": "ok"}),), end=True)

    def stream(self, transcript: Transcript, tools: Sequence[ToolSpec]):
        yield StepDelta(text="thinking ")
        yield StepDelta(text="about it")
        yield Step(tool_calls=(ToolCall("a", "answer", {"summary": "ok"}),), end=True)


def test_streaming_client_yields_reasoning_deltas() -> None:
    events = list(stream("q", Workspace(datasets={}), _Streamer()))
    deltas = [e.text for e in events if e.kind == "reasoning_delta"]
    assert deltas == ["thinking ", "about it"]
    # the assembled step still drives the loop to a final result
    assert any(e.kind == "result" for e in events)
    # a streaming turn does not also emit the whole-text reasoning event (no dupes)
    assert not any(e.kind == "reasoning" for e in events)
