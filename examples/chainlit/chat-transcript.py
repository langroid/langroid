"""
Variant of chat-agent.py, that waits for user to type "/s" (meaning submit)
to store chat transcript in a file.

Directly uses an Agent (i.e. without Task) 
using callbacks, which also enables streaming.

After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/chat-transcript.py

or:
    
uv run chainlit run examples/chainlit/chat-transcript.py

"""

import logging

import chainlit as cl

import langroid as lr
from langroid.agent.callbacks.chainlit import add_instructions

# set info logger
logging.basicConfig(level=logging.INFO)

FILE = "examples/chainlit/chat-transcript.txt"


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
    if message.content.startswith("/s"):
        content = message.content
        # get transcript of entire conv history as a string
        history = (
            "\n\n".join(
                [
                    f"{msg.role.value.upper()}: {msg.content}"
                    for msg in agent.message_history
                ]
            )
            + "\n\n"
            + "FINAL User Answer: "
            + content[2:]
        )

        # save chat transcript to file
        with open(FILE, "w") as f:
            f.write(f"Chat transcript:\n\n{history}\n")
            await cl.Message(
                content=f"Chat transcript saved to {FILE}.",
                author="System",
            ).send()
        return

    await agent.llm_response_async(message.content)
