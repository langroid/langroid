"""
Variant of gitmcp.py that works via the Chainlit UI library,
hardcoded to work for a specific github repo.


Simple example of using the GitMCP server to "chat" about a GitHub repository.

https://github.com/idosal/git-mcp

The server offers several tools, and we can enable ALL of them to be used
by a Langroid agent.

Run like this (-m model optional; defaults to gpt-4.1-mini):

    uv run chainlit run examples/mcp/chainlit-mcp.py

"""

import chainlit as cl
from textwrap import dedent
import langroid as lr
import langroid.language_models as lm
from langroid.pydantic_v1 import Field
from langroid.agent.tools.mcp.fastmcp_client import get_tools_async
from langroid.agent.tools.orchestration import SendTool
from langroid.mytypes import NonToolAction
from fastmcp.client.transports import SSETransport


class SendUserTool(SendTool):
    request: str = "send_user"
    purpose: str = "Send <content> to user"
    to: str = "user"
    content: str = Field(
        ...,
        description="""
        Message to send to user, typically answer to user's request,
        or a clarification question to the user, if user's task/question
        is not completely clear.
        """,
    )


@cl.on_chat_start
async def start():

    lm_config = lm.OpenAIGPTConfig(
        chat_model="gpt-4.1-mini",
    )
    transport = SSETransport(url="https://gitmcp.io/langroid/langroid-examples")
    tools: list[type] = await get_tools_async(transport)
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            handle_llm_no_tool=NonToolAction.FORWARD_USER,
            llm=lm_config,
            system_message=dedent(
                """
                  Make best use of any of the TOOLs available to you,
                  to answer the user's questions.
                  You are a DevOps assistant"
                  """
            ),
        )
    )  # Pass config as a dictionary
    agent.enable_message(tools)
    task_cfg = lr.TaskConfig(recognize_string_signals=False)
    task = lr.Task(agent, config=task_cfg, interactive=False)
    lr.ChainlitTaskCallbacks(task)
    await task.run_async(
        "Based on the TOOLs available to you, greet the user and"
        "tell them what kinds of help you can provide."
    )
