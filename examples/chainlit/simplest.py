"""
Absolute bare-bones way to set up a simple chatbot using all default settings,
using a Langroid Task + callbacks.

After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/simplest.py
"""

import langroid as lr
import langroid.language_models as lm
import chainlit as cl


@cl.on_chat_start
async def on_chat_start():
    lm_config = lm.OpenAIGPTConfig(chat_model="ollama/mistral")
    agent = lr.ChatAgent(lr.ChatAgentConfig(llm=lm_config))
    task = lr.Task(agent, interactive=True)

    msg = "Help me with some questions"
    lr.ChainlitTaskCallbacks(task)
    await task.run_async(msg)
