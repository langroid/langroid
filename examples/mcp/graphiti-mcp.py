"""
Refer
https://github.com/getzep/graphiti/tree/main/mcp_server

OR


Steps to create and connect to openmemory mcp server

- git clone https://github.com/getzep/graphiti.git
- cd graphiti/mcp_server
- cp .env.example .env
- add your OPENAI_API_KEY
- docker compose up -d

"""

from fastmcp.client.transports import SSETransport
from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp.fastmcp_client import get_tools_async
from langroid.mytypes import NonToolAction

# trying to connect to openmemory
URL = "http://0.0.0.0:8000/sse"


async def main(model: str = ""):
    transport = SSETransport(
        url=URL,
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
