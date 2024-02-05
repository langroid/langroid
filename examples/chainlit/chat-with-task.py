"""
Basic single-agent chat example using Task along with ChainlitTaskCallbacks.

After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/chat-with-task.py
"""
import langroid as lr
import chainlit as cl
from langroid.agent.callbacks.chainlit import ChainlitTaskCallbacks
from langroid.agent.callbacks.chainlit import add_instructions


@cl.on_chat_start
async def on_chat_start():
    config = lr.ChatAgentConfig(
        name="Assistant",
        system_message="You are a helpful assistant. Be concise in your answers.",
    )
    agent = lr.ChatAgent(config)
    task = lr.Task(
        agent,
        interactive=True,
    )
    await add_instructions(
        title="Instructions",
        content="Interact with a **Langroid Task**",
    )

    # inject callbacks into the task's agent
    ChainlitTaskCallbacks(task)
    cl.user_session.set("task", task)


@cl.on_message
async def on_message(message: cl.Message):
    task = cl.user_session.get("task")
    await task.run_async(message.content)
