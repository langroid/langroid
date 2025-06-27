"""
Simple example of using the Memory MCP server:
https://github.com/modelcontextprotocol/servers/tree/main/src/memory
This server gives your agent persistent memory using a local Knowledge Graph,
so when you re-start the chat it will remember what you talked about last time.


The server offers several tools, and we can enable ALL of them to be used
by a Langroid agent.

Run like this (-m model optional; defaults to gpt-4.1-mini):

    uv run examples/mcp/memory.py --m ollama/qwen2.5-coder:32b

"""

from fastmcp.client.transports import NpxStdioTransport
from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp.fastmcp_client import get_tools_async
from langroid.mytypes import NonToolAction


async def main(model: str = ""):
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            # forward to user when LLM doesn't use a tool
            handle_llm_no_tool=NonToolAction.FORWARD_USER,
            llm=lm.OpenAIGPTConfig(
                chat_model=model or "gpt-4.1-mini",
                max_output_tokens=1000,
                async_stream_quiet=False,
            ),
            system_message="""
            To be helpful to the user, think about which of your several TOOLs
            you can use, possibly one after the other, to answer the user's question.
            """,
        )
    )

    transport = NpxStdioTransport(
        package="@modelcontextprotocol/server-memory",
        args=["-y"],
    )
    tools = await get_tools_async(transport)

    # enable the agent to use all tools
    agent.enable_message(tools)
    # make task with interactive=False =>
    # waits for user only when LLM doesn't use a tool
    task = lr.Task(agent, interactive=False)
    await task.run_async(
        "Based on the TOOLs available to you, greet the user and"
        "tell them what kinds of help you can provide."
    )


if __name__ == "__main__":
    Fire(main)
