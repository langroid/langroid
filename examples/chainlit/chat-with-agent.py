"""
Basic single-agent chat example, to directly use an Agent (i.e. without Task)
using callbacks, which also enables streaming.

After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/chat-with-agent.py

"""

import logging

import chainlit as cl

import langroid as lr
from langroid.agent.callbacks.chainlit import add_instructions

# set info logger
logging.basicConfig(level=logging.INFO)


@cl.on_chat_start
async def on_chat_start():
    config = lr.ChatAgentConfig(
        name="Demo",
        system_message="You are a helpful assistant. Be concise in your answers.",
    )
    agent = lr.ChatAgent(config)

    cl.user_session.set("agent", agent)

    await add_instructions(
        title="Instructions",
        content="Interact with a **Langroid ChatAgent**",
    )


@cl.on_message
async def on_message(message: cl.Message):
    agent: lr.ChatAgent = cl.user_session.get("agent")
    # important: only apply callbacks after getting first msg.
    lr.ChainlitAgentCallbacks(agent)
    await agent.llm_response_async(message.content)
