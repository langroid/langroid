"""
Generic script to connect to any MCP Server.

Steps:
- from the MCP server page, determine what type of transport is need to connect.
- import the appropriate transport
- set up the `transport` variable in the first line

Run like this (omitting the `--model` argument will use the default GPT-4.1-Mini):

    uv run examples/mcp/any-mcp.py --model ollama/qwen2.5-coder:32b

See docs on various types of transports that are available:
https://langroid.github.io/langroid/notes/mcp-tools/
"""

import langroid as lr
import langroid.language_models as lm
from langroid.mytypes import NonToolAction
from langroid.agent.tools.mcp.fastmcp_client import get_tools_async
from fastmcp.client.transports import StdioTransport
from fire import Fire


async def main(model: str = ""):
    transport = StdioTransport(  # or any other transport
        command="...",
        args=[],
        env=dict(MY_VAR="blah"),
    )
    all_tools = await get_tools_async(transport)

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

    # enable the agent to use all tools
    agent.enable_message(all_tools)
    # make task with interactive=False =>
    # waits for user only when LLM doesn't use a tool
    task = lr.Task(agent, interactive=False)
    await task.run_async(
        "Based on the TOOLs available to you, greet the user and"
        "tell them what kinds of help you can provide."
    )


if __name__ == "__main__":
    Fire(main)
