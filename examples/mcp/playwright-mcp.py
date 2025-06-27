"""
Playwright MCP Example - Langroid integration with Playwright MCP server

This example demonstrates how to use Langroid with the Playwright MCP server
to create an agent that can automate web interactions, take screenshots,
and perform web browsing tasks.

What this example shows:
- Integration with Playwright MCP server for web automation
- How to connect to and use Playwright's web interaction tools within a Langroid agent
- Creation of a web automation agent that can navigate, click, type, and capture web content

What is Playwright MCP?
- Playwright MCP is a Model Context Protocol server that provides web automation capabilities
- It allows LLMs to interact with web pages through browser automation
- The MCP server provides tools for navigation, interaction, and content capture
- This example demonstrates using these web automation capabilities within a Langroid agent

References:
https://github.com/microsoft/playwright-mcp

Steps to run:
1. Ensure Node.js 18+ is installed
2. The script will automatically start the Playwright MCP server via npx

Run like this (-m model optional; defaults to gpt-4.1-mini):
    uv run examples/mcp/playwright/playwright-mcp.py -m ollama/qwen2.5-coder:32b

NOTE: This simple example is hardcoded to answer a single question,
but you can easily extend this with a loop to enable a
continuous chat with the user.

"""

from fastmcp.client.transports import NpxStdioTransport
from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp.fastmcp_client import FastMCPClient
from langroid.agent.tools.orchestration import DoneTool


async def main(model: str = ""):
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            # forward to user when LLM doesn't use a tool
            handle_llm_no_tool="You FORGOT to use one of your TOOLs!",
            llm=lm.OpenAIGPTConfig(
                chat_model=model or "gpt-4.1",
                max_output_tokens=1000,
                async_stream_quiet=False,
            ),
            system_message=f"""
           Your goal is to answer the user's question by
            using browsing tools to navigate Wikipedia.

            Access the web through the provided browsing
            tool. Begin by using the `browser_navigate`
            tool/message to navigate to wikipedia.org.

            Unless you are done, be SURE that you use a
            browsing tool in each step. Think carefully
            about the next step you want to take, and then
            call the appropriate tool. NEVER attempt to
            use more than one tool at a time.

            If you are done, submit the answer with the TOOL
            `{DoneTool.name()}`; give me a succinct
            answer from the results of your browsing.            
            """,
        )
    )

    transport = NpxStdioTransport(
        package="@playwright/mcp@latest",
        args=[],  # "--isolated", "--storage-path={./playwright-storage.json}"],
    )
    async with FastMCPClient(transport, persist_connection=True) as client:
        tools = await client.get_tools_async()
        for t in tools:
            # limit the max tokens for each tool-result to 1000
            t._max_result_tokens = 5000

        # enable the agent to use all tools
        agent.enable_message(tools)
        # make task with interactive=False =>
        task = lr.Task(agent, interactive=False, recognize_string_signals=False)
        await task.run_async(
            """
            What was the first award won by the person who had the featured
            article on English Wikipedia on June 12, 2025? You may need to 
            check the "archive" to find older featured pages. Give me the 
            award which is shown first when sorted by year.
            """,
        )


if __name__ == "__main__":
    Fire(main)
