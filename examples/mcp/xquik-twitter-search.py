"""
Use Xquik's remote MCP server with Langroid for X post search.

Before running:

    export XQUIK_API_KEY=<your-xquik-api-key>

Run like this (-m model optional; defaults to gpt-4.1-mini):

    uv run examples/mcp/xquik-twitter-search.py -m ollama/qwen2.5-coder:32b

The default query uses X advanced search syntax.
"""

import asyncio
import os
from textwrap import dedent

from fastmcp.client.transports import StreamableHttpTransport
from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp.fastmcp_client import get_tools_async
from langroid.mytypes import NonToolAction

XQUIK_MCP_URL = "https://xquik.com/mcp"


def xquik_transport() -> StreamableHttpTransport:
    api_key = os.getenv("XQUIK_API_KEY")
    if not api_key:
        raise ValueError("Set XQUIK_API_KEY before running this example.")
    return StreamableHttpTransport(
        url=XQUIK_MCP_URL,
        headers={"x-api-key": api_key},
    )


async def main(
    model: str = "",
    query: str = 'from:xquikcom "API"',
    limit: int = 5,
) -> None:
    tools = await get_tools_async(xquik_transport())

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            handle_llm_no_tool=NonToolAction.FORWARD_USER,
            llm=lm.OpenAIGPTConfig(
                chat_model=model or "gpt-4.1-mini",
                max_output_tokens=2000,
                async_stream_quiet=False,
            ),
            system_message=dedent(
                """
                Use the available Xquik MCP tools for X post search and lookup
                requests. Return concise results with author, text, and post URL
                when the MCP response includes them.
                """
            ),
        )
    )
    agent.enable_message(tools)

    task = lr.Task(agent, interactive=False)
    result = await task.run_async(
        dedent(
            f"""
            Search X for `{query}` with Xquik. Return up to {limit} relevant
            posts. Include the author, text, and URL when available.
            """
        ).strip()
    )
    print(result.content)


def run_main(**kwargs) -> None:
    asyncio.run(main(**kwargs))


if __name__ == "__main__":
    Fire(run_main)
