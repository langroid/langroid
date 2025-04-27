import os
import langroid as lr
import langroid.language_models as lm
from langroid.mytypes import NonToolAction
from langroid.agent.tools.mcp.fastmcp_client import FastMCPClient
from fastmcp.client.transports import NpxStdioTransport
from fire import Fire


async def main(model: str = ""):
    transport = NpxStdioTransport(
        package="exa-mcp-server",
        env_vars=dict(EXA_API_KEY=os.getenv("EXA_API_KEY")),
    )
    async with FastMCPClient(transport) as client:
        # get the `web_search_exa` MCP tool as a Langroid ToolMessage sub-class
        web_search_tool = await client.make_tool("web_search_exa")

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

    # enable the agent to use the fetch tool
    agent.enable_message(web_search_tool)
    # make task with interactive=False =>
    # waits for user only when LLM doesn't use a tool
    task = lr.Task(agent, interactive=False)
    await task.run_async()


if __name__ == "__main__":
    Fire(main)
