"""
Interact with Claude Code's MCP server.


Run like this (omitting the `--model` argument will use the default GPT-4.1-Mini):

    uv run examples/mcp/claude-code-mcp.py --model gpt-4.1-mini


"""

from fastmcp.client.transports import (
    StdioTransport,
)
from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp.fastmcp_client import get_tools_async
from langroid.mytypes import NonToolAction


async def main(model: str = ""):
    transport = StdioTransport(
        command="claude",
        args=["mcp", "serve"],
        env={},
    )
    all_tools = await get_tools_async(transport)

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            # forward to user when LLM doesn't use a tool
            handle_llm_no_tool=NonToolAction.FORWARD_USER,
            system_message="""
            You are a coding assistant who has access to 
            various tools from Claude Code. You can use these tools to
            to help the user with their coding-related tasks
            or code-related questions.
            """,
            llm=lm.OpenAIGPTConfig(
                chat_model=model or "gpt-4.1",
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
