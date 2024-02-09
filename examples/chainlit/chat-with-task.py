"""
Basic single-agent chat example using Task along with ChainlitTaskCallbacks.

After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/chat-with-task.py
"""
import langroid as lr
import chainlit as cl
from langroid.agent.callbacks.chainlit import (
    add_instructions,
    make_llm_settings_widgets,
    update_llm,
    setup_llm,
)


@cl.on_settings_update
async def on_settings_update(settings: cl.ChatSettings):
    await update_llm(settings, "agent")
    setup_agent_task()


async def setup_agent_task():
    await setup_llm()
    llm_config = cl.user_session.get("llm_config")
    config = lr.ChatAgentConfig(
        llm=llm_config,
        name="Demo",
        system_message="You are a helpful assistant. Be concise in your answers.",
    )
    agent = lr.ChatAgent(config)
    task = lr.Task(
        agent,
        interactive=True,
    )
    cl.user_session.set("task", task)


@cl.on_chat_start
async def on_chat_start():
    await add_instructions(
        title="Basic Langroid Chatbot",
        content="Uses Langroid's `Task.run()`",
    )
    await make_llm_settings_widgets()
    await setup_agent_task()


@cl.on_message
async def on_message(message: cl.Message):
    task = cl.user_session.get("task")
    # sometimes we may want the User to NOT have agent name in front,
    # and just show them as YOU.
    callback_config = lr.ChainlitCallbackConfig(user_has_agent_name=False)
    lr.ChainlitTaskCallbacks(task, message, config=callback_config)
    await task.run_async(message.content)
