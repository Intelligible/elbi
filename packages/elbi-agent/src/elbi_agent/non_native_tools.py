"""Prompt-based tool calling for models without native function-calling.

Many capable models (a lot of open and local ones) do not support the providers' native
tool-call API. Rather than exclude them, this renders the tools into the system
prompt and asks the model to emit calls in a simple, parseable syntax, then parses that
syntax back into tool calls. The runtime loop is unchanged: it still receives tool calls
and feeds results back; only the transport differs.

The invocation format mirrors a widely-used convention::

    <function=run_code>
    <parameter=code>print(1 + 1)</parameter>
    </function>

so a model that has seen it (or is given the instructions here) can drive the same loop.
The parser accepts one or more such blocks per turn.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any

from .llm import ToolCall, ToolSpec

# Emitted after each call so the model stops; passed to the provider as a stop word.
STOP_WORD = "</function>"

_FN_RE = re.compile(r"<function=([^>]+)>\n?(.*?)</function>", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>]+)>(.*?)</parameter>", re.DOTALL)

_INSTRUCTIONS = """
You have access to the following tools. To call one, write EXACTLY:

<function=TOOL_NAME>
<parameter=PARAM_NAME>PARAM_VALUE</parameter>
</function>

Rules:
- Start a call with <function= and end it with </function>.
- Give every required parameter. A value that is an object or a list must be written as
  compact JSON.
- You may call a tool or answer in plain text; a call must come last, not narrated.

Available tools:
"""


def describe_tools(tools: Sequence[ToolSpec]) -> str:
    """A plain-text description of the tools for the system prompt."""
    lines = [_INSTRUCTIONS.rstrip()]
    for tool in tools:
        lines.append(f"\n## {tool.name}\n{tool.description}")
        properties = (tool.input_schema or {}).get("properties") or {}
        required = set((tool.input_schema or {}).get("required") or [])
        if properties:
            lines.append("Parameters:")
            for name, schema in properties.items():
                kind = schema.get("type", "any") if isinstance(schema, dict) else "any"
                flag = "required" if name in required else "optional"
                desc = schema.get("description", "") if isinstance(schema, dict) else ""
                lines.append(f"- {name} ({kind}, {flag}){f': {desc}' if desc else ''}")
    return "\n".join(lines)


def to_prompt_messages(
    messages: list[dict[str, Any]], tools: Sequence[ToolSpec]
) -> list[dict[str, Any]]:
    """Rewrite OpenAI tool-call messages into the prompt-based form for such models.

    Tools are folded into the system message; an assistant's native ``tool_calls``
    become ``<function=...>`` text; and ``tool`` results become user messages, so the
    whole exchange is plain chat the provider accepts without a tools parameter.
    """
    names_by_id: dict[str, str] = {}
    out: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            suffix = "\n\n" + describe_tools(tools)
            out.append(
                {"role": "system", "content": (message["content"] or "") + suffix}
            )
        elif role == "assistant" and message.get("tool_calls"):
            rendered = []
            for call in message["tool_calls"]:
                names_by_id[call["id"]] = call["function"]["name"]
                rendered.append(_render_call(call["function"]))
            text = message.get("content") or ""
            out.append(
                {
                    "role": "assistant",
                    "content": (text + "\n" + "\n".join(rendered)).strip(),
                }
            )
        elif role == "tool":
            name = names_by_id.get(message.get("tool_call_id", ""), "tool")
            out.append(
                {
                    "role": "user",
                    "content": f"EXECUTION RESULT of [{name}]:\n{message['content']}",
                }
            )
        else:
            out.append(message)
    return out


def parse_calls(text: str | None) -> tuple[str | None, list[ToolCall]]:
    """Split a model reply into its leading prose and the tool calls it emitted.

    Returns the text with the call blocks removed (None if nothing remains) and the
    parsed calls. A parameter value is decoded as JSON when it can be (numbers, bools,
    objects, lists arrive as themselves) and kept as a string otherwise.
    """
    if not text:
        return text, []
    calls: list[ToolCall] = []
    for index, match in enumerate(_FN_RE.finditer(text)):
        name = match.group(1).strip()
        arguments = {
            param.strip(): _coerce(value.strip())
            for param, value in _PARAM_RE.findall(match.group(2))
        }
        calls.append(ToolCall(id=f"call_{index}", name=name, arguments=arguments))
    prose = _FN_RE.sub("", text).strip()
    return (prose or None), calls


def _render_call(function: dict[str, Any]) -> str:
    try:
        arguments = json.loads(function.get("arguments") or "{}")
    except (json.JSONDecodeError, TypeError):
        arguments = {}
    parts = [f"<function={function['name']}>"]
    for name, value in arguments.items():
        rendered = value if isinstance(value, str) else json.dumps(value)
        parts.append(f"<parameter={name}>{rendered}</parameter>")
    parts.append("</function>")
    return "\n".join(parts)


def _coerce(value: str) -> Any:
    try:
        return json.loads(value)
    except (json.JSONDecodeError, ValueError):
        return value
