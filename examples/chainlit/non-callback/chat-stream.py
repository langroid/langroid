"""
DEPRECATED, not guaranteed to work: We are keeping this example for reference,
but do not use this as way to chat with streaming.
See chat-callback.py for the best way to do this
(i.e. use ChainlitAgentCallbacks when interacting directly an Agent,
or use ChainlitTaskCallbacks when interacting with a Task).

Basic single-agent chat example, with streaming,
using an older method, rather than the best way,
which is via callbacks, as in chat-callback.py.


After setting up the virtual env as in README,
and you have your OpenAI API Key in the .env file, run like this:

chainlit run examples/chainlit/chat-stream.py
"""

import asyncio
import re
import sys

import chainlit as cl

from langroid import ChatAgent, ChatAgentConfig
from langroid.utils.configuration import settings

settings.stream = True  # works if False as well


class ContinuousCaptureStream:
    """
    Capture stdout in a stream.
    This allows capturing of streaming output that would normally be printed to stdout,
    e.g. streaming tokens coming from OpenAI's API.
    """

    def __init__(self):
        self.content = ""
        self.new_content_event = asyncio.Event()
        self.is_finished = False  # Flag to indicate completion

    def write(self, data):
        self.content += data
        self.new_content_event.set()

    def flush(self):
        pass

    async def get_new_content(self):
        await self.new_content_event.wait()
        self.new_content_event.clear()
        new_content, self.content = self.content, ""
        return new_content

    def set_finished(self):
        self.is_finished = True
        self.new_content_event.set()  # T


def strip_ansi_codes(text):
    ansi_escape = re.compile(
        r"(?:\x1B[@-_]|[\x80-\x9A\x9C-\x9F]|[\x1A-\x1C\x1E-\x1F])+\[[0-?]*[ -/]*[@-~]"
    )
    return ansi_escape.sub("", text)


@cl.on_chat_start
async def on_chat_start():
    sys_msg = "You are a helpful assistant. Be concise in your answers."
    config = ChatAgentConfig(
        system_message=sys_msg,
        show_stats=False,
    )
    agent = ChatAgent(config)
    cl.user_session.set("agent", agent)


@cl.on_message
async def on_message(message: cl.Message):
    agent: ChatAgent = cl.user_session.get("agent")
    msg = cl.Message(content="")
    await msg.send()

    capture_stream = ContinuousCaptureStream()
    original_stdout = sys.stdout
    sys.stdout = capture_stream

    # Run response() in a separate thread or as a non-blocking call
    asyncio.create_task(run_response(agent, message, capture_stream))

    while not capture_stream.is_finished:
        new_output = await capture_stream.get_new_content()
        new_output = strip_ansi_codes(new_output)
        if new_output:
            await msg.stream_token(new_output)

    # Restore original stdout when done
    sys.stdout = original_stdout

    await msg.update()


async def run_response(agent: ChatAgent, message: cl.Message, stream):
    await agent.llm_response_async(message.content)
    stream.set_finished()
