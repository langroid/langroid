"""
Refer
https://mem0.ai/blog/how-to-make-your-clients-more-context-aware-with-openmemory-mcp/
https://docs.mem0.ai/openmemory/quickstart
https://github.com/mem0ai/mem0/tree/main/openmemory

OR


Steps to create and connect to openmemory mcp server

- git clone <https://github.com/mem0ai/mem0.git>
- cd mem0/openmemory
- cp api/.env.example api/.env
- add your OPENAI_API_KEY
- make build # builds the mcp server and ui
- make up  # runs openmemory mcp server and ui
"""

import os

from fastmcp.client.transports import SSETransport
from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp.fastmcp_client import get_tools_async
from langroid.mytypes import NonToolAction

# trying to connect to openmemory
URL = "http://localhost:8765/mcp/openmemory/sse/"
# set userid to my own, got from os: $USER
userid = os.getenv("USER")


async def main(model: str = ""):
    transport = SSETransport(
        url=URL + userid,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    all_tools = await get_tools_async(transport)

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            # forward to user when LLM doesn't use a tool
            handle_llm_no_tool=NonToolAction.FORWARD_USER,
            llm=lm.OpenAIGPTConfig(
                # chat_model=model or "gpt-4.1-mini",
                chat_model="gemini/gemini-2.0-flash",
                max_output_tokens=1000,
                async_stream_quiet=False,
            ),
        )
    )

    # enable the agent to use all tools
    agent.enable_message(all_tools)
    # make task with interactive=False =>
    # waits for user only when LLM doesn't use a tool
    task = lr.Task(agent, interactive=False)
    await task.run_async(
        "Based on the TOOLs available to you, greet the user and"
        "tell them what kinds of help you can provide."
    )


if __name__ == "__main__":
    Fire(main)
