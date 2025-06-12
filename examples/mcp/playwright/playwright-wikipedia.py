"""
Simple example of Playwright MCP browser which answers questions by
browsing Wikipedia.

Playwright requires specific versions of browsers to be installed
 hence this example requires Docker.
"""

import langroid as lr
import langroid.language_models as lm
import docker
from fastmcp.client.transports import StreamableHttpTransport
from langroid.agent.tools.mcp import FastMCPClient
from typing import Optional
import asyncio
from langroid.agent.tools.orchestration import DoneTool
from rich import print
from fire import Fire
from httpx import ReadError
import docker.errors
import os


class BrowserAgent(lr.ChatAgent):
    def __init__(self, config: lr.ChatAgentConfig):
        super().__init__(config)
        self.enable_message(DoneTool)

    def handle_message_fallback(self, msg: str | lr.ChatDocument) -> Optional[str]:
        if isinstance(msg, str) or msg.metadata.sender != lr.Entity.LLM:
            return

        return """
        Unless you are done, be sure to use a browser tool/function at each step!
        If you need to plan your actions, think step by step and then call the appropriate
        tool/function.

        If you are done, submit your answer with the `done_tool` tool/function.
        """

    async def agent_response_async(
        self, msg: Optional[str | lr.ChatDocument] = None
    ) -> Optional[lr.ChatDocument]:
        result = await super().agent_response_async(msg)

        # Playwright responds with a large snapshot at each step As
        # this can exceed maximum context length we truncate all but
        # the most recent response
        for message in self.message_history:
            if message.role == lm.Role.USER:
                message.content = (
                    message.content
                    if len(message.content) < 500
                    else message.content[:500] + "\n...[Truncated]"
                )

        return result

    def agent_response(
        self, msg: Optional[str | lr.ChatDocument] = None
    ) -> Optional[lr.ChatDocument]:
        result = super().agent_response(msg)

        # Playwright responds with a large snapshot at each step As
        # this can exceed maximum context length we truncate all but
        # the most recent response
        for message in self.message_history:
            if message.role == lm.Role.USER:
                message.content = (
                    message.content
                    if len(message.content) < 500
                    else message.content[:500] + "\n...[Truncated]"
                )

        return result


async def main(question: str = "", model: str = ""):
    question = (
        question
        or """
    What was the first award won by the person who had the featured
    article on English Wikipedia on June 12, 2025? Give me the award
    which is shown first when sorted by year.
    """
    )
    client = docker.from_env()
    tag = "playwright-mcp"

    try:
        client.images.get(tag)
    except docker.errors.APIError:
        path = os.path.dirname(os.path.abspath(__file__))
        client.images.build(path=path, tag=tag)

    container = client.containers.run(
        tag,
        ports={"8931/tcp": 8931},
        remove=True,
        detach=True,
    )

    # Wait for the server to start
    connected = False
    while not connected:
        try:
            transport = StreamableHttpTransport(
                "http://localhost:8931/mcp",
            )

            async with FastMCPClient(transport, persist_connection=True) as client:
                connected = True
                browser_tools = await client.get_tools_async()
                agent = BrowserAgent(
                    lr.ChatAgentConfig(
                        llm=lm.AzureConfig(
                            chat_model=model or lm.openai_gpt.OpenAIChatModel.GPT4_1
                        ),
                        system_message="""
                        Your goal is to answer the user's question by using
                        browsing tools to navigate Wikipedia.

                        Access the web through the provided browsing tool. Begin
                        by using the `browser_navigate` tool/message to navigate
                        to wikipedia.org.

                        Unless you are done, be SURE that you use a tool in each
                        step. Think carefully about the next step you want to
                        take, and then call the appropriate tool. NEVER attempt to
                        use more than one tool at a time.

                        If you are done, submit the answer with the `done_tool` tool/function;
                        give me a succinct answer from the results of your browsing. 
                        """,
                    )
                )

                for tool in browser_tools:
                    agent.enable_message(tool)

                answer = await lr.Task(
                    agent, interactive=False, recognize_string_signals=False
                ).run_async(question)

                if answer:
                    print(f"[green]Answer: {answer.content}")
        except ReadError:
            await asyncio.sleep(0.1)

    container.stop()


if __name__ == "__main__":

    def run_main(**kwargs) -> None:
        """Run the async main function with a proper event loop.

        Args:
            **kwargs: Keyword arguments to pass to the main function.
        """
        asyncio.run(main(**kwargs))

    Fire(run_main)
