"""
Refer
https://github.com/getzep/graphiti/tree/main/mcp_server

OR


Steps to create and connect to graphiti mcp server

If you want to use docker 
- git clone https://github.com/getzep/graphiti.git
- cd graphiti/mcp_server
- cp .env.example .env
- add your OPENAI_API_KEY
- docker compose up -d


Without using docker ,

#For data persistence
mkdir -p $HOME/neo4j/data
sudo chown -R 7474:7474 $HOME/neo4j/data


#Run neo4j container
docker run \
  --name=my-neo4j \
  --publish=7474:7474 \
  --publish=7687:7687 \
  --env NEO4J_AUTH=neo4j/demodemo \
  --volume=$HOME/neo4j/data:/data \
  neo4j:latest


git clone https://github.com/getzep/graphiti.git
cd mcp_server
cp .env.example .env

Add you OPENAI_API_KEY in .env

uv sync --upgrade
uv run graphiti_mcp_server.py
"""

from fastmcp.client.transports import SSETransport
from fire import Fire

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp.fastmcp_client import get_tools_async
from langroid.mytypes import NonToolAction

import json

from typing import List, Dict, Any


from langroid.pydantic_v1 import Field
from langroid.agent.tool_message import ToolMessage

# trying to connect to graphiti-mcp
URL = "http://0.0.0.0:8000/sse"


class MemoryFormatterTool(ToolMessage):
    """
    A tool to format a dictionary into a specially-escaped JSON string
    required by the `add_memory` tool's `episode_body`.
    """

    request: str = "memory_formatter"
    purpose: str = """
        Use this tool to correctly format a dictionary into the required JSON string format
        before calling the `add_memory` tool. This is for when the source is 'json'.
        """
    data_to_format: Dict[str, Any] = Field(
        ..., description="The key-value data to be formatted into a JSON string."
    )

    def handle(self) -> str:
        """
        Takes a dictionary, converts it to a standard JSON string,
        and then replaces all double-quotes with the required triple-backslash-escaped
        version (\\").
        """
        standard_json_string = json.dumps(self.data_to_format)
        # In a Python regular string, to get 3 literal backslashes, you need 6.
        # The replacement string is '\\\\\\"'.
        triple_escaped_string = standard_json_string.replace('"', '\\\\\\"')
        return triple_escaped_string


async def main(model: str = ""):
    SYSTEM_MESSAGE = """
        You are Graphiti, an intelligent memory agent. If a user asks a question, always
        use `search_memory_facts` or `search_memory_nodes`.

        **IMPORTANT: To add structured JSON data to memory, you MUST follow this
        two-step process:**

        1.  **FIRST, use the `memory_formatter` tool.** Pass the structured data as a
            simple dictionary to the `data_to_format` argument.

        2.  **SECOND, use the `add_memory` tool.** Take the string result from the
            `memory_formatter` and use it as the `episode_body` for the `add_memory` call.
            You should also intelligently fill in the `name` and `source_description` fields.
            
        **ADDITIONALLY**
            **`delete_entity_edge`, `delete_episode`, `get_entity_edge`:** These tools require a precise `uuid`.
          
            **`clear_graph`:** This action is irreversible and deletes ALL data. Only use if explicitly commanded and confirmed by the user.

        Your goal is to be helpful and accurate by leveraging your memory capabilities.
    """
    transport = SSETransport(
        url=URL,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    all_tools = await get_tools_async(transport)

    # Injecting (Monkey Patching) the AddMemoryTool examples because
    # langroid pydantic_v1 and graphiti mcp pydantic_v2 version mismatch
    # Hence the tool calling format is not recognized by langroid agents
    # for same reason subclassing tool would not work as well

    AddMemoryTool = next((t for t in all_tools if t.name() == "add_memory"), None)

    if AddMemoryTool:

        @classmethod
        def new_examples(cls) -> List[ToolMessage]:
            # These examples show the FINAL format the LLM should generate
            # FOR THE `add_memory` tool, which is the output of our formatter.
            return [
                cls(
                    request="add_memory",
                    name__="Apollo 11 Moon Landing",
                    episode_body='{\\\\\\"mission\\\\\\": \\\\\\"Apollo 11\\\\\\", \\\\\\"event_date\\\\\\": \\\\\\"1969-07-20\\\\\\", \\\\\\"first_step_by\\\\\\": \\\\\\"Neil Armstrong\\\\\\"}',
                    source="json",
                    source_description="Structured data of the first human moon landing",
                ),
                cls(
                    request="add_memory",
                    name__="Customer Profile",
                    episode_body='{\\\\\\"company\\\\\\": {\\\\\\"name\\\\\\": \\\\\\"Acme Technologies\\\\\\"}, \\\\\\"products\\\\\\": [{\\\\\\"id\\\\\\": \\\\\\"P001\\\\\\", \\\\\\"name\\\\\\": \\\\\\"CloudSync\\\\\\"}, {\\\\\\"id\\\\\\": \\\\\\"P002\\\\\\", \\\\\\"name\\\\\\": \\\\\\"DataMiner\\\\\\"}]}',
                    source="json",
                    source_description="CRM data",
                ),
                cls(
                    request="add_memory",
                    name__="Fall of the Berlin Wall",
                    episode_body="The Berlin Wall fell on November 9, 1989, marking a symbolic end to the Cold War and leading to German reunification.",
                    source="text",
                    source_description="Summary of the fall of the Berlin Wall",
                ),
                cls(
                    request="add_memory",
                    name__="User Booked Doctor Appointment",
                    episode_body="user: Can you schedule a doctor appointment for Tuesday?\\nassistant: Sure, I’ve booked a doctor appointment for Tuesday at 3 PM.",
                    source="message",
                    source_description="Conversation about scheduling an appointment",
                ),
            ]

        # Replace the original examples method with our new one
        AddMemoryTool.examples = new_examples

    graphiti_agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            system_message=SYSTEM_MESSAGE,
            handle_llm_no_tool=NonToolAction.FORWARD_USER,
            llm=lm.OpenAIGPTConfig(
                chat_model="gemini/gemini-2.0-flash",
                max_output_tokens=2000,
                async_stream_quiet=False,
            ),
        )
    )

    # Enable all MCP tools PLUS our custom formatter tool.
    graphiti_agent.enable_message(all_tools)
    graphiti_agent.enable_message(MemoryFormatterTool)

    graphiti_task = lr.Task(graphiti_agent, interactive=False)
    await graphiti_task.run_async()


if __name__ == "__main__":
    Fire(main)
