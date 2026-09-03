"""An attached image reaches the model as an Anthropic block / an OpenAI image_url."""

from __future__ import annotations

from collections.abc import Sequence

from elbi_agent import Step, ToolSpec, Transcript, stream
from elbi_agent.litellm_client import to_openai_messages
from elbi_agent.runtime import Workspace

_PNG = "data:image/png;base64,iVBORw0KGgoAAAANS"


def test_transcript_carries_text_and_image_blocks() -> None:
    t = Transcript(system="S")
    t.add_user_message("what is this chart?", [_PNG])
    content = t.messages[-1]["content"]
    assert content[0] == {"type": "text", "text": "what is this chart?"}
    assert content[1]["type"] == "image"
    assert content[1]["source"]["media_type"] == "image/png"
    assert content[1]["source"]["data"] == "iVBORw0KGgoAAAANS"


def test_a_malformed_data_url_is_dropped() -> None:
    t = Transcript(system="S")
    t.add_user_message("q", ["not-a-data-url"])
    assert t.messages[-1]["content"] == [{"type": "text", "text": "q"}]


def test_image_maps_to_openai_image_url() -> None:
    t = Transcript(system="S")
    t.add_user_message("what is this chart?", [_PNG])
    messages = to_openai_messages(t)
    user = messages[-1]
    parts = user["content"]
    assert parts[0] == {"type": "text", "text": "what is this chart?"}
    assert parts[1]["type"] == "image_url"
    assert parts[1]["image_url"]["url"] == _PNG


class _Recorder:
    def __init__(self, step: Step) -> None:
        self._step = step
        self.seen: list[Transcript] = []

    def step(self, transcript: Transcript, tools: Sequence[ToolSpec]) -> Step:
        self.seen.append(transcript)
        return self._step


def test_stream_attaches_images_to_the_question() -> None:
    client = _Recorder(Step(text="ok", end=True))
    list(stream("describe it", Workspace(datasets={}), client, images=[_PNG]))
    # The transcript is mutated after the step, so scan all messages for the image.
    blocks = [
        b
        for message in client.seen[0].messages
        if isinstance(message["content"], list)
        for b in message["content"]
    ]
    assert any(b.get("type") == "image" for b in blocks)
