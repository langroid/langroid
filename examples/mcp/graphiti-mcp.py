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

# trying to connect to graphiti-mcp
URL = "http://0.0.0.0:8000/sse"


async def main(model: str = ""):
    SYSTEM_MESSAGE = """
        You are Graphiti, an intelligent memory agent managing a knowledge graph. 

        **General Instructions:**

        1.  **Always Search First:** Before adding new information, attempt to find existing data using `search_memory_nodes` or `search_memory_facts`.
        2.  **Optional Arguments:** It is perfectly acceptable to omit optional arguments if the information is not provided by the user or relevant to the current task. **Crucially, unless explicitly provided by the user, DO NOT GENERATE VALUES for optional arguments.** Simply exclude the argument from the tool call.
        3.  **Error Reporting:** If a tool call fails, inform the user about the error message.

        **Specific Tool Nuances:**
        *   ** add_memory 
        * You are intelligent to fill in name and source_description
        * Remember to format json source like where each quote inside curly is escaped by triple backslashes "episode_body": "{\\\"mission\\\": \\\"Apollo 11\\\", \\\"event_date\\\": \\\"1969-07-20\\\", \\\"first_step_by\\\": \\\"Neil Armstrong\\\"}",
        
        {
        "request": "add_memory",
        "name__": "Apollo 11 Moon Landing",
        "episode_body": "{\\\"mission\\\": \\\"Apollo 11\\\", \\\"event_date\\\": \\\"1969-07-20\\\", \\\"first_step_by\\\": \\\"Neil Armstrong\\\"}",
        "source": "json",
        "source_description": "Structured data of the first human moon landing"
        }

        {
        "request": "add_memory",
        "name__": "Fall of the Berlin Wall",
        "episode_body": "The Berlin Wall fell on November 9, 1989, marking a symbolic end to the Cold War and leading to German reunification.",
        "source": "text",
        "source_description": "Summary of the fall of the Berlin Wall"
        }

        {
        "request": "add_memory",
        "name__": "User Booked Doctor Appointment",
        "episode_body": "user: Can you schedule a doctor appointment for Tuesday?\nassistant: Sure, I’ve booked a doctor appointment for Tuesday at 3 PM.",
        "source": "message",
        "source_description": "Conversation about scheduling an appointment"
        }


        *   **`delete_entity_edge`, `delete_episode`, `get_entity_edge`:** These tools require a precise `uuid`.
        *   **`clear_graph`:** This action is irreversible and deletes ALL data. Only use if explicitly commanded and confirmed by the user.

        Your goal is to be helpful and accurate by leveraging your memory access capabilities.
            
    """
    transport = SSETransport(
        url=URL,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    all_tools = await get_tools_async(transport)

    graphiti_agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            system_message=SYSTEM_MESSAGE,
            # forward to user when LLM doesn't use a tool
            handle_llm_no_tool=NonToolAction.FORWARD_USER,
            llm=lm.OpenAIGPTConfig(
                # chat_model=model or "gpt-4.1-mini",
                chat_model="gemini/gemini-2.0-flash-lite",
                # chat_model="groq/llama-3.1-8b-instant",
                max_output_tokens=1000,
                async_stream_quiet=False,
            ),
        )
    )

    # enable the agent to use all tools
    graphiti_agent.enable_message(all_tools)
    # make task with interactive=False =>
    # waits for user only when LLM doesn't use a tool
    graphiti_task = lr.Task(graphiti_agent, interactive=False)

    await graphiti_task.run_async(
        "Based on the TOOLs available to you, greet the user and"
        "tell them what kinds of help you can provide."
    )


if __name__ == "__main__":
    Fire(main)
