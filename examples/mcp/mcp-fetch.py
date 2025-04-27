"""
Simple example of using the Anthropic Fetch MCP Server to get web-site content.

Fetch MCP Server: https://github.com/modelcontextprotocol/servers/tree/main/src/fetch

Run like this:

    uv run examples/mcp/mcp-fetch.py --model gpt-4.1-mini

Ask questions like:

Summarize the content of this page:
https://www.anthropic.com/news/model-context-protocol
"""

import langroid as lr
import langroid.language_models as lm
from langroid.mytypes import NonToolAction
from langroid.agent.tools.mcp.fastmcp_client import FastMCPClient
from fastmcp.client.transports import UvxStdioTransport
from fire import Fire


async def main(model: str = ""):
    transport = UvxStdioTransport(
        tool_name="mcp-server-fetch",
    )
    async with FastMCPClient(transport) as client:
        # get the `fetch` MCP tool as a Langroid ToolMessage sub-class
        fetch_tool = await client.make_tool("fetch")

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            # forward to user when LLM doesn't use a tool
            handle_llm_no_tool=NonToolAction.FORWARD_USER,
            llm=lm.OpenAIGPTConfig(
                chat_model=model or "gpt-4.1-mini",
                max_output_tokens=1000,
                async_stream_quiet=False,
            ),
        )
    )

    # enable the agent to use the fetch tool
    agent.enable_message(fetch_tool)
    # make task with interactive=False =>
    # waits for user only when LLM doesn't use a tool
    task = lr.Task(agent, interactive=False)
    await task.run_async()


if __name__ == "__main__":
    Fire(main)
