"""
Simple example of using the BioMCP server.

https://github.com/genomoncology/biomcp

The server offers several tools, and we can enable ALL of them to be used
by a Langroid agent.

Run like this:

    uv run examples/mcp/biomcp.py --model gpt-4.1-mini

"""

from fastmcp.client.transports import StdioTransport
from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp.fastmcp_client import get_tools_async
from langroid.mytypes import NonToolAction


async def main(model: str = ""):
    transport = StdioTransport(
        command="uv", args=["run", "--with", "biomcp-python", "biomcp", "run"]
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
