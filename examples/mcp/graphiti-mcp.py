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

# trying to connect to graphiti-mcp
URL = "http://0.0.0.0:8000/sse"


async def main(model: str = ""):
    SYSTEM_MESSAGE = """Add an episode with proper formatting.

    - `episode_body` must be an **escaped JSON string** if `source='json'` (not a Python dict).
    - `source` can be:
    - 'text' – plain text (default)
    - 'json' – structured data
    - `source_description` optional field
    - 'message' – conversation-style
    - `uuid` is not needed dont add `uuid` at any cost

    Examples:
    - Text: "Acme launched a new product."
    - JSON: "{\\\"company\\\": {\\\"name\\\": \\\"Acme\\\"}, \\\"products\\\": [{\\\"id\\\": \\\"P001\\\"}]}",
    - Message: "user: Hello\nassistant: Hi there"

    Entities and relationships are auto-extracted from JSON.
    Use `group_id` to group episodes (optional).
    """

    transport = SSETransport(
        url=URL,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    all_tools = await get_tools_async(transport)

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            system_message=SYSTEM_MESSAGE,
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
