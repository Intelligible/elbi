"""Title generation: an LLM title when the model answers, truncation as the fallback.

Model-agnostic: it drives whatever ``LLMClient`` is injected, so these tests use a tiny
scripted client rather than any real provider.
"""

from __future__ import annotations

from collections.abc import Sequence

from elbi.titles import fallback_title, llm_title, make_title
from elbi_agent import Step, ToolSpec, Transcript


class _TitleClient:
    """An LLMClient that returns fixed text (or raises), for titling only."""

    def __init__(self, text: str | None = None, *, boom: bool = False) -> None:
        self._text = text
        self._boom = boom

    def step(self, transcript: Transcript, tools: Sequence[ToolSpec]) -> Step:
        if self._boom:
            raise RuntimeError("model unavailable")
        return Step(text=self._text, tool_calls=())


def test_llm_title_cleans_the_model_reply() -> None:
    # Quotes and a trailing period are stripped; only the bare title survives.
    client = _TitleClient('"Housing price drivers."')
    assert llm_title("how does each variable affect price?", client) == (
        "Housing price drivers"
    )


def test_llm_title_takes_the_first_line_only() -> None:
    client = _TitleClient("Churn analysis\n(here is why)")
    assert llm_title("why do customers churn", client) == "Churn analysis"


def test_llm_title_truncates_to_max_length() -> None:
    client = _TitleClient("A very long descriptive title that runs on and on and on")
    out = llm_title("q", client, max_length=20)
    assert out is not None and len(out) == 20 and out.endswith("…")


def test_llm_title_none_on_error_or_empty() -> None:
    assert llm_title("q", _TitleClient(boom=True)) is None  # error -> None
    assert llm_title("q", _TitleClient("   ")) is None  # empty -> None
    assert llm_title("   ", _TitleClient("x")) is None  # no message -> None


def test_fallback_title_truncates_the_first_line() -> None:
    assert fallback_title("effect of x on y") == "effect of x on y"
    assert fallback_title("") == "analysis"
    long = "a" * 80
    assert fallback_title(long).endswith("…") and len(fallback_title(long)) == 50


def test_make_title_falls_back_without_a_client_or_on_failure() -> None:
    # No client, or an LLM failure, yields the truncated first line, never nothing.
    assert make_title("effect of x on y", None) == "effect of x on y"
    assert make_title("effect of x on y", _TitleClient(boom=True)) == "effect of x on y"
    assert make_title("effect of x on y", _TitleClient("Nice Title")) == "Nice Title"
