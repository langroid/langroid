"""
Simple example of using the Exa Web Search MCP Server to
answer questions using web-search.

Exa MCP Server: https://docs.exa.ai/examples/exa-mcp

Run like this (omitting the `--model` argument will use the default GPT-4.1-Mini):

    uv run examples/mcp/exa-web-search --model ollama/qwen2.5

"""

import os

from fastmcp.client.transports import NpxStdioTransport
from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp import mcp_tool
from langroid.mytypes import NonToolAction

transport = NpxStdioTransport(
    package="exa-mcp-server",
    env_vars=dict(EXA_API_KEY=os.getenv("EXA_API_KEY")),
)


# Illustrating how we can:
# - use the MCP tool decorator to create a Langroid ToolMessage subclass
# - override the handle_async() method to customize the output, sent to the LLM


@mcp_tool(transport, "web_search_exa")
class ExaSearchTool(lr.ToolMessage):
    async def handle_async(self):
        result: str = await self.call_tool_async()
        return f"""
        Below are the results of the web search:
        
        <WebSearchResult>
        {result}
        </WebSearchResult>
        
        Use these results to answer the user's original question.
        """


async def main(model: str = ""):
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            # forward to user when LLM doesn't use a tool
            handle_llm_no_tool=NonToolAction.FORWARD_USER,
            llm=lm.OpenAIGPTConfig(
                chat_model=model or "gpt-4.1-mini",
                max_output_tokens=1000,
                # this defaults to True, but we set it to False so we can see output
                async_stream_quiet=False,
            ),
        )
    )

    # enable the agent to use the web-search tool
    agent.enable_message(ExaSearchTool)
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
