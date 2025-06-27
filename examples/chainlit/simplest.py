"""
Absolute bare-bones way to set up a simple chatbot using all default settings,
using a Langroid Task + callbacks.

After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/simplest.py
"""

import chainlit as cl

import langroid as lr
import langroid.language_models as lm


@cl.on_message
async def on_message(message: cl.Message):
    lm_config = lm.OpenAIGPTConfig()
    agent = lr.ChatAgent(lr.ChatAgentConfig(llm=lm_config))
    task = lr.Task(agent, interactive=True)

    lr.ChainlitTaskCallbacks(task)
    await task.run_async(message.content)
