"""A ~15-line MCP client: call run_risk_overlay and print only the served JSON.

This is what "the same question as a tool" means -- no chat framework, no
prompt. One tool call against the server ``elbi mcp`` is already running.
"""

import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client


async def main() -> None:
    async with (
        streamable_http_client("http://127.0.0.1:7878/mcp") as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await session.call_tool("run_risk_overlay", {})
        data = json.loads(result.content[0].text)

    print("=== run_risk_overlay ===")
    print(f"{'as_of':<10} {data['as_of']}")
    print(f"{'gross_vol':<10} {data['gross_vol']}")
    print(f"{'cuts':<10} " + ", ".join(f"{k} {v}" for k, v in data["cuts"].items()))
    print(f"{'source':<10} {data['source']}")


if __name__ == "__main__":
    asyncio.run(main())
