"""Prompt-based tool calling: render tools into the prompt, parse calls back out."""

from __future__ import annotations

from elbi_agent.llm import ToolCall, ToolSpec
from elbi_agent.non_native_tools import (
    describe_tools,
    parse_calls,
    to_prompt_messages,
)

_TOOL = ToolSpec(
    "run_code",
    "Run Python.",
    {
        "type": "object",
        "properties": {"code": {"type": "string", "description": "the code"}},
        "required": ["code"],
    },
)


def test_describe_tools_lists_name_and_parameters() -> None:
    text = describe_tools([_TOOL])
    assert "## run_code" in text and "Run Python." in text
    assert "code (string, required): the code" in text
    assert "<function=" in text  # the invocation instructions are included


def test_parse_calls_extracts_a_call_and_strips_it() -> None:
    text = (
        "I'll run it.\n"
        "<function=run_code>\n<parameter=code>print(1 + 1)</parameter>\n</function>"
    )
    prose, calls = parse_calls(text)
    assert prose == "I'll run it."
    assert calls == [ToolCall("call_0", "run_code", {"code": "print(1 + 1)"})]


def test_parse_calls_coerces_json_parameter_values() -> None:
    text = (
        "<function=derive>\n"
        "<parameter=n>42</parameter>\n"
        '<parameter=claim>{"x": "a", "y": "b"}</parameter>\n'
        "</function>"
    )
    _, calls = parse_calls(text)
    assert calls[0].arguments == {"n": 42, "claim": {"x": "a", "y": "b"}}


def test_parse_calls_without_a_call_returns_the_text() -> None:
    assert parse_calls("just an answer") == ("just an answer", [])


def test_to_prompt_messages_rewrites_the_exchange() -> None:
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "run it"},
        {
            "role": "assistant",
            "content": "ok",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "run_code", "arguments": '{"code": "1"}'},
                }
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": "1"},
    ]
    out = to_prompt_messages(messages, [_TOOL])
    # tools are folded into the system message
    assert out[0]["role"] == "system" and "## run_code" in out[0]["content"]
    # the assistant's native call becomes the prompt syntax, with no tool_calls field
    assert "tool_calls" not in out[2]
    assert "<function=run_code>" in out[2]["content"]
    # the tool result becomes a user message naming the tool
    assert out[3]["role"] == "user"
    assert out[3]["content"] == "EXECUTION RESULT of [run_code]:\n1"
