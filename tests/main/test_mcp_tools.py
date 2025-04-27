import os
from typing import List, Optional

import pytest
from fastmcp import FastMCP
from fastmcp.client.transports import NpxStdioTransport, UvxStdioTransport
from mcp.types import Tool

# note we use pydantic v2 to define MCP server
from pydantic import BaseModel, Field  # keep - need pydantic v2 for MCP server

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp.decorators import mcp_tool
from langroid.agent.tools.mcp.fastmcp_client import FastMCPClient


class SubItem(BaseModel):
    """A sub‐item with a value and multiplier."""

    val: int = Field(..., description="Value for sub-item")
    multiplier: float = Field(1.0, description="Multiplier applied to val")


class ComplexData(BaseModel):
    """Complex data combining two ints and a list of SubItem."""

    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")
    items: List[SubItem] = Field(..., description="List of sub-items")


def mcp_server():
    server = FastMCP("TestServer")

    @server.tool()
    def greet(person: str) -> str:
        return f"Hello, {person}!"

    @server.tool()
    def get_alerts(
        state: str = Field(..., description="TWO-LETTER state abbrev, e.g. 'MN'"),
    ) -> list[str]:
        """Get weather alerts for a state."""
        return [
            f"Weather alert for {state}: Severe thunderstorm warning.",
            f"Weather alert for {state}: Flash flood watch.",
        ]

    @server.tool()
    def get_one_alert(
        state: str = Field(..., description="TWO-LETTER state abbrev, e.g. 'MN'"),
    ) -> str:
        return f"Weather alert for {state}: Severe thunderstorm warning."

    @server.tool()
    async def get_alerts_async(state: str) -> list[str]:
        return [
            f"Weather alert for {state}: Severe thunderstorm warning.",
            f"Weather alert for {state}: Flash flood watch.",
        ]

    @server.tool()
    def nabroski(a: int, b: int) -> int:
        """Computes the Nabroski transform of integers a and b."""
        return 3 * a + b

    @server.tool()
    def coriolis(x: int, y: int) -> int:
        """Computes the Coriolis transform of integers x and y."""
        return 2 * x + 3 * y

    @server.tool()
    def hydra_nest(data: ComplexData) -> int:
        """Compute the HydraNest calculation over nested data."""
        total = data.a * data.b
        for item in data.items:
            total += item.val * item.multiplier
        return int(total)

    return server


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server",
    [
        mcp_server(),
        "tests/main/mcp/weather-server-python/weather.py",
    ],
)
async def test_get_tools_and_handle(server: FastMCP | str) -> None:
    """End‐to‐end test for get_tools and .handle() against server."""
    async with FastMCPClient(server) as client:
        tools = await client.get_tools()
        # basic sanity
        assert isinstance(tools, list)
        assert tools, "Expected at least one tool"
        # find the alerts tool
        alerts_tool = next(
            (t for t in tools if t.default_value("request") == "get_alerts"),
            None,
        )

        assert alerts_tool is not None
        assert issubclass(alerts_tool, lr.ToolMessage)

        # test find_tool
        alerts_mcp_tool: Tool = await client.find_mcp_tool("get_alerts")
        assert alerts_mcp_tool is not None
        alerts_tool = await client.make_tool(alerts_mcp_tool.name)

    assert alerts_tool is not None
    assert issubclass(alerts_tool, lr.ToolMessage)
    # EXIT the async with client context, and it should work.

    # instantiate Langroid ToolMessage
    msg = alerts_tool(state="NY")
    isinstance(msg, lr.ToolMessage)
    assert msg.state == "NY"

    # invoke the tool via the Langroid ToolMessage.handle() method -
    # this produces list of weather alerts for the given state
    # Note MCP tools can return either ResultToolType or List[ResultToolType]
    result: Optional[str] = await msg.handle_async()

    assert isinstance(result, str)
    assert any(x in result.lower() for x in ["alert", "weather"])

    # make tool from async FastMCP tool
    async with FastMCPClient(server) as client:
        alert_mcp_tool_async = await client.find_mcp_tool("get_alerts_async")
        assert alert_mcp_tool_async is not None
        alert_tool_async = await client.make_tool(alert_mcp_tool_async.name)

    assert alert_tool_async is not None
    assert issubclass(alert_tool_async, lr.ToolMessage)

    # instantiate Langroid ToolMessage
    msg = alert_tool_async(state="NY")
    isinstance(msg, lr.ToolMessage)
    assert msg.state == "NY"

    # invoke the tool via the Langroid ToolMessage.handle_async() method -
    # this produces list of weather alerts for the given state
    # Note MCP tools can return either ResultToolType or List[ResultToolType]
    result: Optional[str] = await msg.handle_async()
    assert result is not None
    assert any(x in result.lower() for x in ["alert", "weather"])


@pytest.mark.parametrize(
    "server",
    [
        mcp_server(),
        "tests/main/mcp/weather-server-python/weather.py",
    ],
)
@pytest.mark.asyncio
async def test_tools_connect_close(server: str | FastMCP) -> None:
    """Test that we can use connect()... tool-calls ... close()"""

    client = FastMCPClient(server)
    await client.connect()
    mcp_tools = await client.client.list_tools()
    assert all(isinstance(t, Tool) for t in mcp_tools)
    langroid_tool_classes = await client.get_tools()
    assert all(issubclass(tc, lr.ToolMessage) for tc in langroid_tool_classes)

    alerts_tool_mcp = await client.find_mcp_tool("get_alerts")

    alerts_tool = await client.make_tool(alerts_tool_mcp.name)
    await client.close()

    assert alerts_tool is not None
    assert issubclass(alerts_tool, lr.ToolMessage)
    # instantiate Langroid ToolMessage
    msg = alerts_tool(state="NY")
    isinstance(msg, lr.ToolMessage)
    assert msg.state == "NY"

    result = await msg.handle_async()
    assert isinstance(result, str)
    assert any(x in result.lower() for x in ["alert", "weather"])


@pytest.mark.asyncio
async def test_mcp_tool_schemas() -> None:
    """
    Test that descriptions, field-descriptions of MCP tools are preserved
    when we translate them to Langroid ToolMessage classes. This is important
    since the LLM is shown these, and helps with tool-call accuracy.
    """
    async with FastMCPClient(mcp_server()) as client:
        # find the alerts tool
        alerts_tool = await client.make_tool("get_alerts")

    assert issubclass(alerts_tool, lr.ToolMessage)
    description = "Get weather alerts for a state."
    assert alerts_tool.default_value("purpose") == description
    schema: lm.LLMFunctionSpec = alerts_tool.llm_function_schema()
    assert schema.description == description
    assert schema.name == "get_alerts"
    assert schema.parameters["required"] == ["state"]
    assert "TWO-LETTER" in schema.parameters["properties"]["state"]["description"]


@pytest.mark.asyncio
async def test_single_output() -> None:
    """Test that a tool with a single string output works
    similarly to one that has a list of strings outputs."""

    async with FastMCPClient(mcp_server()) as client:
        one_alert_tool = await client.make_tool("get_one_alert")

    assert one_alert_tool is not None
    msg = one_alert_tool(state="NY")
    assert isinstance(msg, lr.ToolMessage)
    assert msg.state == "NY"
    result = await msg.handle_async()

    # we expect a list containing a single str
    assert isinstance(result, str)
    assert any(x in result.lower() for x in ["alert", "weather"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "server",
    [
        mcp_server(),
    ],
)
async def test_agent_mcp_tools(server: str | FastMCP) -> None:
    """Test that a Langroid ChatAgent can use and handle MCP tools."""

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=lm.OpenAIGPTConfig(
                max_output_tokens=500,
                async_stream_quiet=False,
            ),
        )
    )

    async with FastMCPClient(server) as client:
        nabroski_tool: lr.ToolMessage = await client.make_tool("nabroski")

    agent.enable_message(nabroski_tool)

    response: lr.ChatDocument = await agent.llm_response_async(
        "What is the Nabroski transform of 3 and 5?"
    )
    tools = agent.get_tool_messages(response)
    assert len(tools) == 1
    assert isinstance(tools[0], nabroski_tool)
    result: lr.ChatDocument = await agent.agent_response_async(response)
    # TODO assert needs to take LLM tool-forgetting into account
    assert "14" in result.content

    agent.init_state()
    task = lr.Task(agent, interactive=False)
    result: lr.ChatDocument = await task.run_async(
        "What is the Nabroski transform of 3 and 5?",
        turns=3,
    )
    assert "14" in result.content


# Need to define the tools outside async def,
# since the decorator uses asyncio.run() to wrap around an async fn
@mcp_tool(mcp_server(), "get_alerts")
class GetAlertsTool(lr.ToolMessage):
    """Tool to get weather alerts."""

    async def my_handler(self) -> str:
        alert = await self.handle_async()
        return "ALERT: " + alert


@mcp_tool(mcp_server(), "nabroski")
class NabroskiTool(lr.ToolMessage):
    """Tool to get Nabroski transform."""

    async def my_handler(self) -> str:
        result = await self.handle_async()
        return f"FINAL Nabroski transform result: {result}"


@pytest.mark.asyncio
async def test_fastmcp_decorator() -> None:
    """Test that the mcp_tool decorator works as expected."""

    msg = GetAlertsTool(state="NY")
    assert isinstance(msg, lr.ToolMessage)
    assert msg.state == "NY"
    result = await msg.my_handler()
    assert isinstance(result, str)
    assert "ALERT" in result

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=lm.OpenAIGPTConfig(
                max_output_tokens=500,
                async_stream_quiet=False,
            ),
        )
    )
    agent.enable_message(GetAlertsTool)

    agent.enable_message(NabroskiTool)
    task = lr.Task(agent, interactive=False)
    result = await task.run_async("What is the nabroski transform of 5 and 3?", turns=3)
    assert "nabroski" in result.content.lower() and "18" in result.content


@mcp_tool(mcp_server(), "hydra_nest")
class HydraNestTool(lr.ToolMessage):
    """Tool for computing HydraNest calculation."""

    async def my_handler(self) -> str:
        """Call hydra_nest and format result."""
        result = await self.handle_async()
        return f"Computed: {result}"


@pytest.mark.asyncio
async def test_complex_tool_decorator() -> None:
    """Test that compute_complex via decorator works end‐to‐end."""
    # build nested input
    payload = {
        "data": {
            "a": 4,
            "b": 5,
            "items": [
                {"val": 2, "multiplier": 1.5},
                {"val": 3, "multiplier": 2.0},
            ],
        }
    }
    msg = HydraNestTool(**payload)
    assert isinstance(msg, lr.ToolMessage)
    # call handler
    result = await msg.my_handler()
    expected = int(4 * 5 + 2 * 1.5 + 3 * 2.0)
    assert f"{expected}" in result

    # round‐trip via an agent
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=lm.OpenAIGPTConfig(
                max_output_tokens=1000,
                async_stream_quiet=False,
            ),
        )
    )
    agent.enable_message(HydraNestTool)
    task = lr.Task(agent, interactive=False)
    prompt = """
    Compute the HydraNest calculation with a=4, b=5,
    and a list of items with these val-multiplier pairs:
        val=2, multiplier=1.5
        val=3, multiplier=2.0
    """
    response = await task.run_async(prompt, turns=3)
    assert str(expected) in response.content


@pytest.mark.parametrize(
    "prompt,tool_name,expected",
    [
        ("What is the Nabroski transform of 3 and 5?", "nabroski", "14"),
        ("What is the Coriolis transform of 4 and 3?", "coriolis", "17"),
    ],
)
@pytest.mark.asyncio
async def test_multiple_tools(prompt, tool_name, expected) -> None:
    """
    Test one-shot enabling of multiple tools.
    """
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=lm.OpenAIGPTConfig(
                max_output_tokens=1000,
                async_stream_quiet=False,
            ),
        )
    )
    async with FastMCPClient(mcp_server()) as client:
        all_tools = await client.get_tools()

    tool = next(
        (t for t in all_tools if t.name() == tool_name),
        None,
    )
    agent.enable_message(all_tools)

    # test that agent (LLM) can pick right tool based on prompt
    prompt = "use one of your TOOLs to answer this: " + prompt
    response: lr.ChatDocument = await agent.llm_response_async(prompt)
    tools = agent.get_tool_messages(response)
    assert len(tools) == 1
    assert isinstance(tools[0], lr.ToolMessage)
    assert isinstance(tools[0], tool)

    # test in a task
    task = lr.Task(agent, interactive=False)
    result: lr.ChatDocument = await task.run_async(prompt, turns=3)
    assert expected in result.content


@pytest.mark.asyncio
async def test_npxstdio_transport() -> None:
    """
    Test that we can create Langroid ToolMessage from an MCP server
    via npx stdio transport, for example the `exa-mcp-server`:
    https://github.com/exa-labs/exa-mcp-server
    """
    transport = NpxStdioTransport(
        package="exa-mcp-server",
        env_vars=dict(EXA_API_KEY=os.getenv("EXA_API_KEY")),
    )
    async with FastMCPClient(transport) as client:
        tools = await client.get_tools()
        assert isinstance(tools, list)
        assert tools, "Expected at least one tool"
        web_search_tool = await client.make_tool("web_search_exa")

    assert web_search_tool is not None
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=lm.OpenAIGPTConfig(
                max_output_tokens=1000,
                async_stream_quiet=False,
            ),
            system_message="""
            When asked a question, use the TOOL `web_search_exa` to
            perform a web search and find the answer.
            """,
        )
    )
    agent.enable_message(web_search_tool)
    # Note: we shouldn't have to explicitly beg the LLM to use the tool here
    # but I've found that even GPT-4o sometimes fails to use the tool
    question = """
    Use the `web_search_exa` TOOL to find out:
    Who won the Presidential election in Gabon in 2025?
    """
    response = await agent.llm_response_async(question)

    tools = agent.get_tool_messages(response)
    assert len(tools) == 1
    assert isinstance(tools[0], web_search_tool)

    task = lr.Task(agent, interactive=False)
    result: lr.ChatDocument = await task.run_async(question, turns=3)
    assert "Nguema" in result.content


transport = UvxStdioTransport(
    # `tool_name` is a misleading name -- it really refers to the
    # MCP server, which offers several tools
    tool_name="mcp-server-git",
)


@mcp_tool(transport, "git_status")
class GitStatusTool(lr.ToolMessage):
    """Tool to get git status."""

    async def handle_async(self) -> str:
        """
        When defining a class explicitly with the @mcp_tool decorator,
        we have the flexibility to define our own `handle_async` method
        which calls the call_tool_async method, which in turn calls the
        MCP server's call_tool method.
        Returns:

        """
        status = await self.call_tool_async()
        return "GIT STATUS: " + status


@pytest.mark.asyncio
async def test_uvxstdio_transport() -> None:
    """
    Test that we can create Langroid ToolMessage from an MCP server
    via uvx stdio transport. We use this example `git` MCP server:
    https://github.com/modelcontextprotocol/servers/tree/main/src/git
    """
    async with FastMCPClient(transport) as client:
        tools = await client.get_tools()
        assert isinstance(tools, list)
        assert tools, "Expected at least one tool"
        git_status_tool = await client.make_tool("git_status")

    assert git_status_tool is not None
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=lm.OpenAIGPTConfig(
                max_output_tokens=1000,
                async_stream_quiet=False,
            ),
        )
    )
    agent.enable_message(git_status_tool)
    prompt = """
        Use the `git_status` TOOL to find out the status of the 
        current git repository at "../langroid"
        """

    response = await agent.llm_response_async(prompt)
    tools = agent.get_tool_messages(response)
    assert len(tools) == 1
    assert isinstance(tools[0], git_status_tool)

    task = lr.Task(agent, interactive=False)
    result: lr.ChatDocument = await task.run_async(prompt, turns=3)
    assert "langroid" in result.content

    # test GitStatusTool created via @mcp_tool decorator
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=lm.OpenAIGPTConfig(
                max_output_tokens=1000,
                async_stream_quiet=False,
            ),
        )
    )
    agent.enable_message(GitStatusTool)
    prompt = """
        Find out the git status of the git repository at "../langroid"
        """

    response = await agent.llm_response_async(prompt)
    tools = agent.get_tool_messages(response)
    assert len(tools) == 1
    assert isinstance(tools[0], GitStatusTool)

    task = lr.Task(agent, interactive=False)
    result: lr.ChatDocument = await task.run_async(prompt, turns=3)
    # "GIT STATUS" is prepended to the MCP call_tool result,
    # with our custom `handle_async` method; to verify this
    # we check the content of the parent of parent of result.
    assert "GIT STATUS" in result.parent.parent.content


@pytest.mark.asyncio
async def test_npxstdio_transport_memory() -> None:
    """
    Test that we can create Langroid ToolMessage from the `memory` MCP server
    via npx stdio transport:
    https://github.com/modelcontextprotocol/servers/tree/main/src/memory
    """
    transport = NpxStdioTransport(
        package="@modelcontextprotocol/server-memory",
        args=["-y"],
    )
    async with FastMCPClient(transport) as client:
        tools = await client.get_tools()
        assert isinstance(tools, list)
        assert tools, "Expected at least one tool"

    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=lm.OpenAIGPTConfig(
                max_output_tokens=1000,
                async_stream_quiet=False,
            ),
            system_message="""
Follow these steps for each interaction:

1. User Identification:
   - You should assume that you are interacting with default_user
   - If you have not identified default_user, proactively try to do so.

2. Memory Retrieval:
   - Always begin your chat by saying only "Remembering..." and retrieve all 
       relevant information from your knowledge graph
   - Always refer to your knowledge graph as your "memory"
   - Use your TOOLS to retrieve information from your memory when asked

3. Memory
   - While conversing with the user, be attentive to any new information that falls 
       into these categories:
     a) Basic Identity (age, gender, location, job title, education level, etc.)
     b) Behaviors (interests, habits, etc.)
     c) Preferences (communication style, preferred language, etc.)
     d) Goals (goals, targets, aspirations, etc.)
     e) Relationships (personal and professional relationships up to 3 degrees of 
         separation)

4. Memory Update:
   - If any new information was gathered during the interaction, 
       update your memory as follows:
     a) Create entities for recurring organizations, people, and significant events
     b) Connect them to the current entities using relations
     b) Store facts about them as observations   
     Use your TOOLS to update your memory.         
            """,
        )
    )

    agent.enable_message(tools)
    prompt = """
        Joseph Knecht was a member of the Glass Bead Game Society.
        He was good friends with the composer Hesse.
        His mentor was the former teacher of the Glass Bead Game Society, Maestro.
        Memorize the relevant information using one of the TOOLs:
        `add_observations`, `create_entities`, `create_relations`
        """
    response = await agent.llm_response_async(prompt)
    tools = agent.get_tool_messages(response)
    assert len(tools) >= 1

    task = lr.Task(agent, interactive=False, restart=False)
    prompt = """
    Who was Joseph Knecht's mentor? Use the `search_nodes` TOOL to find out.
    """
    result: lr.ChatDocument = await task.run_async(prompt, turns=3)
    assert "Maestro" in result.content
