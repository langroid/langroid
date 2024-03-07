"""
Basic single-agent chat example, without streaming.

DEPCRECATED: Script kept only for reference.
The better way is shown in chat-agent.py or chat-task.py, which uses callbacks.

After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/chat-no-callback.py
"""

import chainlit as cl

import langroid as lr


@cl.on_chat_start
async def on_chat_start():
    sys_msg = "You are a helpful assistant. Be concise in your answers."
    config = lr.ChatAgentConfig(
        system_message=sys_msg,
    )
    agent = lr.ChatAgent(config)
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message):
    agent: lr.ChatAgent = cl.user_session.get("agent")
    response = await agent.llm_response_async(message.content)
    msg = cl.Message(content=response.content)
    await msg.send()
