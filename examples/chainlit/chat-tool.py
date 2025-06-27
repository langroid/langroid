"""
Basic single-agent chat example, using task.run(), with a tool, with streaming,
using ChainlitTaskCallbacks.

After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/chat-tool.py
"""

from textwrap import dedent

import chainlit as cl

import langroid as lr
from langroid.agent.callbacks.chainlit import add_instructions


class CapitalTool(lr.ToolMessage):
    request = "capital"
    purpose = "To present the capital of given <country>."
    country: str
    capital: str

    def handle(self) -> str:
        return f"""
        Success! LLM responded with a tool/function-call, with result:
        
        Capital of {self.country} is {self.capital}.
        """


@cl.on_chat_start
async def on_chat_start():
    config = lr.ChatAgentConfig(
        name="CapitalExpert",
        system_message="""
        When asked for the <capital> of a <country>, present
        your response using the `capital` tool/function-call.
        """,
    )
    agent = lr.ChatAgent(config)
    agent.enable_message(CapitalTool)

    await add_instructions(
        title="Instructions",
        content=dedent(
            """
        Interact with a **Langroid Task**, whose ChatAgent has access 
        to a `capital` tool. You can ask about anything, but whenever you ask 
        about a country's capital, the agent will use the `capital` tool to present 
        the capital of that country. This "tool-message" is handled by the Agent's 
        handler method, and the result is presented as plain text.
        """
        ),
    )
    # inject callbacks into the agent
    task = lr.Task(
        agent,
        interactive=True,
    )
    cl.user_session.set("task", task)


@cl.on_message
async def on_message(message: cl.Message):
    task = cl.user_session.get("task")
    lr.ChainlitTaskCallbacks(task)
    await task.run_async(message.content)
