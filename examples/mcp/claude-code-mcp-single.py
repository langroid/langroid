"""
Enable a Langroid agent to use a SINGLE MCP Tool from 
Claude Code's MCP server.

Similar to claude-code-mcp.py but showing how to use a single tool, i.e.,
Claude-Code's special Grep tool that is built on ripgrep.

Run like this (omitting the `--model` argument will use the default gpt-5-mini):

    uv run examples/mcp/claude-code-mcp-single.py --model gpt-5-mini


"""

from fastmcp.client.transports import (
    StdioTransport,
)
from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp import mcp_tool
from langroid.agent.tools.mcp.fastmcp_client import get_tools_async
from langroid.mytypes import NonToolAction


transport = StdioTransport(
    command="claude",
    args=["mcp", "serve"],
    env={},
)


# Illustrating how we can:
# - use the MCP tool decorator to create a Langroid ToolMessage subclass
# - override the handle_async() method to customize the output, sent to the LLM


@mcp_tool(transport, "Grep")
class GrepTool(lr.ToolMessage):
    async def handle_async(self):
        # call the actual tool
        result: str = await self.call_tool_async()

        # modify the result as desired
        return f"""
        Below are the results of the grep:
        
        <GrepResult>
        {result}
        </GrepResult>
        
        Use these results to answer the user's original question.
        """


async def main(model: str = ""):
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            # forward to user when LLM doesn't use a tool
            handle_llm_no_tool=NonToolAction.FORWARD_USER,
            llm=lm.OpenAIGPTConfig(
                chat_model=model or "gpt-5-mini",
                max_output_tokens=1000,
                # this defaults to True, but we set it to False so we can see output
                async_stream_quiet=False,
            ),
        )
    )

    # enable the agent to use the grep tool
    agent.enable_message(GrepTool)
    # make task with interactive=False =>
    # waits for user only when LLM doesn't use a tool
    task = lr.Task(agent, interactive=False)
    await task.run_async()


if __name__ == "__main__":
    import asyncio

    def run_main(**kwargs) -> None:
        """Run the async main function with a proper event loop.

        Args:
            **kwargs: Keyword arguments to pass to the main function.
        """
        asyncio.run(main(**kwargs))

    Fire(run_main)
