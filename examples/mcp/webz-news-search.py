"""Search current news through Webz.io's remote MCP server.

Set ``WEBZ_API_TOKEN`` before running. The MCP server exposes the
``news_search_by_webz`` tool and accepts natural-language queries plus its
documented news filters.

Run like this (``--model`` is optional):

    uv run examples/mcp/webz-news-search.py --query="latest AI policy news"

The Webz MCP endpoint and bearer-token transport are kept in this example so
the provider-specific integration does not become a Langroid dependency.
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

WEBZ_MCP_URL = "https://news-search-mcp.webz.io/mcp"


def webz_transport() -> StreamableHttpTransport:
    """Build an authenticated transport for the Webz news MCP server."""
    token = os.getenv("WEBZ_API_TOKEN", "").strip()
    if not token:
        raise ValueError("Set WEBZ_API_TOKEN before running this example.")
    return StreamableHttpTransport(
        url=WEBZ_MCP_URL,
        headers={"Authorization": f"Bearer {token}"},
    )


async def main(model: str = "", query: str = "latest AI policy news") -> None:
    """Search Webz news and print the agent's answer.

    Args:
        model: Optional model identifier for the Langroid agent.
        query: Natural-language news query passed to the MCP tool.
    """
    tools = await get_tools_async(webz_transport())

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            handle_llm_no_tool=NonToolAction.FORWARD_USER,
            llm=lm.OpenAIGPTConfig(
                chat_model=model or "gpt-4.1-mini",
                max_output_tokens=2000,
                async_stream_quiet=False,
            ),
            system_message=dedent("""
                Use the `news_search_by_webz` MCP tool for news searches.
                Respect the available tool schema and filters, and return a
                concise answer with article titles, dates, sources, and URLs
                when they are present in the tool response.
                """),
        )
    )
    agent.enable_message(tools)

    task = lr.Task(agent, interactive=False)
    result = await task.run_async(query)
    print(result.content if result is not None else "No answer returned.")


def run_main(model: str = "", query: str = "latest AI policy news") -> None:
    """Run the asynchronous example through Python Fire."""
    asyncio.run(main(model=model, query=query))


if __name__ == "__main__":
    Fire(run_main)
