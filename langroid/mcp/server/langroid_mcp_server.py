"""MCP server that exposes Langroid agents as tools."""

from typing import List, Optional

from fastmcp import Context, FastMCP

from langroid import ChatAgent, ChatAgentConfig
from langroid.agent.tools.duckduckgo_search_tool import DuckduckgoSearchTool
from langroid.language_models.client_lm import ClientLMConfig

# Create FastMCP server
server = FastMCP("Langroid Agent Server")  # type: ignore[var-annotated]


@server.tool()
async def langroid_chat(
    message: str,
    ctx: Context,
    enable_tools: Optional[List[str]] = None,
    agent_name: Optional[str] = None,
) -> str:
    """
    Chat with a Langroid agent that uses your LLM via MCP sampling.

    Args:
        message: User message to send to agent
        ctx: MCP context (automatically provided)
        enable_tools: List of tool names to enable (e.g., ["web_search"])
        agent_name: Optional name for the agent

    Returns:
        Agent's response as a string
    """
    # Create ClientLM without context in config
    llm_config = ClientLMConfig()

    # Create Langroid agent
    agent_config = ChatAgentConfig(
        name=agent_name or "LangroidAssistant",
        llm=llm_config,
    )
    agent = ChatAgent(agent_config)

    # Set context on the LLM instance after creation
    if agent.llm is not None and hasattr(agent.llm, "set_context"):
        agent.llm.set_context(ctx)

    # Enable requested tools
    if enable_tools:
        if "web_search" in enable_tools:
            agent.enable_message(DuckduckgoSearchTool)
        # More tools can be added here as needed

    # Get response from agent
    response = await agent.llm_response_async(message)
    return response.content if response else ""


@server.tool()
async def langroid_task(
    message: str,
    ctx: Context,
    enable_tools: Optional[List[str]] = None,
    agent_name: Optional[str] = None,
    max_turns: int = 10,
) -> str:
    """
    Run a Langroid task that allows multiple agent-user turns.

    Args:
        message: Initial user message
        ctx: MCP context (automatically provided)
        enable_tools: List of tool names to enable
        agent_name: Optional name for the agent
        max_turns: Maximum number of conversation turns

    Returns:
        Final result from the task
    """
    from langroid.agent.task import Task

    # Create ClientLM without context in config
    llm_config = ClientLMConfig()

    # Create Langroid agent
    agent_config = ChatAgentConfig(
        name=agent_name or "LangroidTaskAgent",
        llm=llm_config,
    )
    agent = ChatAgent(agent_config)

    # Set context on the LLM instance after creation
    if agent.llm is not None and hasattr(agent.llm, "set_context"):
        agent.llm.set_context(ctx)

    # Enable requested tools
    if enable_tools:
        if "web_search" in enable_tools:
            agent.enable_message(DuckduckgoSearchTool)

    # Create and run task
    task = Task(agent, max_turns=max_turns, interactive=False)
    result = await task.run_async(message)

    return result if isinstance(result, str) else str(result)


def main() -> None:
    """Entry point for UVX."""
    server.run()


if __name__ == "__main__":
    main()
