import os
from typing import List, Optional

import pytest
from fastmcp import Context, FastMCP
from fastmcp.client.sampling import (
    RequestContext,
    SamplingMessage,
    SamplingParams,
)
from fastmcp.client.transports import NpxStdioTransport, UvxStdioTransport
from mcp.types import (
    BlobResourceContents,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
    Tool,
)

# note we use pydantic v2 to define MCP server
from pydantic import BaseModel, Field  # keep - need pydantic v2 for MCP server

import langroid as lr
import langroid.language_models as lm
from langroid.agent.tools.mcp import (
    FastMCPClient,
    get_mcp_tool_async,
    get_tool_async,
    get_tools_async,
    mcp_tool,
)


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

    class Counter:
        def __init__(self) -> None:
            self.count: int = 0

        def get_num_beans(self) -> int:
            """Return current counter."""
            return self.count

        def add_beans(
            self,
            x: int = Field(..., description="Number of beans to add"),
        ) -> int:
            """Increment and return new counter."""
            self.count += x
            return self.count

    # create a stateful tool
    counter = Counter()
    # we can't directly use the tool decorator on instance methods,
    # so we use the server.add_tool() method to add the tool
    server.add_tool(counter.get_num_beans)
    server.add_tool(counter.add_beans)

    # example of tool that uses an arg of type Context, and
    # uses this arg to request client LLM sampling, and send logs
    @server.tool()
    async def prime_check(number: int, ctx: Context) -> str:
        """
        Determine if the given number is Prime or not.
        """
        result: TextContent | ImageContent = await ctx.sample(
            f"Is the number {number} prime?",
        )
        assert isinstance(result, TextContent), "Expected a text response"
        return result.text

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

    # example of MCP tool whose fields clash with Langroid ToolMessage,
    # and a `name` field which is reserved by pydantic
    @server.tool()
    def info_tool(
        request: str = Field(..., description="Requested information"),
        name: str = Field(..., description="Name of the info sought"),
        recipient: str = Field(..., description="Recipient of the information"),
        purpose: str = Field(..., description="Purpose of the information"),
        date: str = Field(..., description="Date of the request"),
    ) -> str:
        """Get information for a recipient."""
        return f"""
        Info for {recipient}: {request} {name} (Purpose: {purpose}), date: {date}
        """

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
    tools = await get_tools_async(server)
    # basic sanity
    assert isinstance(tools, list)
    assert tools, "Expected at least one tool"
    # find the alerts tool
    AlertsTool = next(
        (t for t in tools if t.default_value("request") == "get_alerts"),
        None,
    )

    assert AlertsTool is not None
    assert issubclass(AlertsTool, lr.ToolMessage)

    # test get_mcp_tool_async
    AlertsMCPTool: Tool = await get_mcp_tool_async(server, "get_alerts")

    assert AlertsMCPTool is not None
    AlertsTool = await get_tool_async(server, AlertsMCPTool.name)

    assert AlertsTool is not None
    assert issubclass(AlertsTool, lr.ToolMessage)

    # instantiate Langroid ToolMessage
    msg = AlertsTool(state="NY")
    isinstance(msg, lr.ToolMessage)
    assert msg.state == "NY"

    # invoke the tool via the Langroid ToolMessage.handle() method -
    # this produces list of weather alerts for the given state
    # Note MCP tools can return either ResultToolType or List[ResultToolType]
    result: Optional[str] = await msg.handle_async()
    print(result)
    assert isinstance(result, str)
    assert result is not None

    # make tool from async FastMCP tool
    AlertsMCPToolAsync = await get_mcp_tool_async(server, "get_alerts_async")
    assert AlertsMCPToolAsync is not None
    AlertsToolAsync = await get_tool_async(server, AlertsMCPToolAsync.name)

    assert AlertsToolAsync is not None
    assert issubclass(AlertsToolAsync, lr.ToolMessage)

    # instantiate Langroid ToolMessage
    msg = AlertsToolAsync(state="NY")
    isinstance(msg, lr.ToolMessage)
    assert msg.state == "NY"

    # invoke the tool via the Langroid ToolMessage.handle_async() method -
    # this produces list of weather alerts for the given state
    # Note MCP tools can return either ResultToolType or List[ResultToolType]
    result: Optional[str] = await msg.handle_async()
    assert result is not None
    print(result)
    assert isinstance(result, str)
    assert result is not None

    # test making tool with utility functions
    AlertsTool = await get_tool_async(server, "get_alerts")
    assert issubclass(AlertsTool, lr.ToolMessage)
    # instantiate Langroid ToolMessage
    msg = AlertsTool(state="NY")
    isinstance(msg, lr.ToolMessage)


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
    langroid_tool_classes = await client.get_tools_async()
    assert all(issubclass(tc, lr.ToolMessage) for tc in langroid_tool_classes)

    AlertsMCPTool = await get_mcp_tool_async(server, "get_alerts")

    AlertsTool = await get_tool_async(server, AlertsMCPTool.name)
    await client.close()

    assert AlertsTool is not None
    assert issubclass(AlertsTool, lr.ToolMessage)
    # instantiate Langroid ToolMessage
    msg = AlertsTool(state="NY")
    isinstance(msg, lr.ToolMessage)
    assert msg.state == "NY"

    result = await msg.handle_async()
    assert isinstance(result, str)


@pytest.mark.asyncio
async def test_stateful_tool() -> None:
    # instantiate the server
    server = mcp_server()

    # get tools from the SAME instance of the server
    AddBeansTool = await get_tool_async(server, "add_beans")
    assert issubclass(AddBeansTool, lr.ToolMessage)

    GetNumBeansTool = await get_tool_async(server, "get_num_beans")
    assert issubclass(GetNumBeansTool, lr.ToolMessage)

    add_beans_msg = AddBeansTool(x=5)
    assert isinstance(add_beans_msg, lr.ToolMessage)

    result = await add_beans_msg.handle_async()
    assert isinstance(result, str)
    assert "5" in result

    get_num_beans_msg = GetNumBeansTool()
    assert isinstance(get_num_beans_msg, lr.ToolMessage)
    result = await get_num_beans_msg.handle_async()
    assert isinstance(result, str)
    assert "5" in result


@pytest.mark.asyncio
async def test_tool_with_context_and_sampling() -> None:
    async def sampling_handler(
        messages: list[SamplingMessage],
        params: SamplingParams,
        context: RequestContext,
    ) -> str:
        """Handle a sampling request from server"""
        # simulate an LLM call
        return "Yes"

    PrimeCheckTool = await get_tool_async(
        mcp_server(),
        "prime_check",
        sampling_handler=sampling_handler,
    )
    assert issubclass(PrimeCheckTool, lr.ToolMessage)
    # assert that "ctx" is NOT a field in the tool
    assert "ctx" not in PrimeCheckTool.llm_function_schema().parameters["properties"]

    # instantiate Langroid ToolMessage
    prime_check_msg = PrimeCheckTool(number=7)
    assert isinstance(prime_check_msg, lr.ToolMessage)

    result = await prime_check_msg.handle_async()
    assert isinstance(result, str)
    assert "yes" in result.lower()


@pytest.mark.asyncio
async def test_mcp_tool_schemas() -> None:
    """
    Test that descriptions, field-descriptions of MCP tools are preserved
    when we translate them to Langroid ToolMessage classes. This is important
    since the LLM is shown these, and helps with tool-call accuracy.
    """
    # make a langroid AlertsTool from the corresponding MCP tool
    AlertsTool = await get_tool_async(mcp_server(), "get_alerts")

    assert issubclass(AlertsTool, lr.ToolMessage)
    description = "Get weather alerts for a state."
    assert AlertsTool.default_value("purpose") == description
    schema: lm.LLMFunctionSpec = AlertsTool.llm_function_schema()
    assert schema.description == description
    assert schema.name == "get_alerts"
    assert schema.parameters["required"] == ["state"]
    assert "TWO-LETTER" in schema.parameters["properties"]["state"]["description"]

    InfoTool = await get_tool_async(mcp_server(), "info_tool")
    assert issubclass(InfoTool, lr.ToolMessage)
    description = "Get information for a recipient."
    assert InfoTool.default_value("purpose") == description
    assert InfoTool.default_value("request") == "info_tool"

    # instantiate InfoTool
    msg = InfoTool(
        name__="InfoName",
        request__="address",
        recipient__="John Doe",
        purpose__="to know the address",
        date="2023-10-01",
    )
    assert isinstance(msg, lr.ToolMessage)
    assert msg.name__ == "InfoName"
    assert msg.request__ == "address"
    assert msg.recipient__ == "John Doe"
    assert msg.purpose__ == "to know the address"
    assert msg.date == "2023-10-01"
    # call the tool
    result = await msg.handle_async()
    assert isinstance(result, str)
    assert "address" in result.lower()


@pytest.mark.asyncio
async def test_single_output() -> None:
    """Test that a tool with a single string output works
    similarly to one that has a list of strings outputs."""

    OneAlertTool = await get_tool_async(mcp_server(), "get_one_alert")

    assert OneAlertTool is not None
    msg = OneAlertTool(state="NY")
    assert isinstance(msg, lr.ToolMessage)
    assert msg.state == "NY"
    result = await msg.handle_async()

    # we expect a list containing a single str
    assert isinstance(result, str)
    assert any(x in result.lower() for x in ["alert", "weather"])


@pytest.mark.asyncio
async def test_agent_mcp_tools() -> None:
    """Test that a Langroid ChatAgent can use and handle MCP tools."""

    server = mcp_server()
    agent = lr.ChatAgent(
        lr.ChatAgentConfig(
            llm=lm.OpenAIGPTConfig(
                max_output_tokens=500,
                async_stream_quiet=False,
            ),
        )
    )

    NabroskiTool: lr.ToolMessage = await get_tool_async(server, "nabroski")

    agent.enable_message(NabroskiTool)

    response: lr.ChatDocument = await agent.llm_response_async(
        "What is the Nabroski transform of 3 and 5?"
    )
    tools = agent.get_tool_messages(response)
    assert len(tools) == 1
    assert isinstance(tools[0], NabroskiTool)
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

    # test MCP tool with fields that clash with Langroid ToolMessage
    InfoTool = await get_tool_async(server, "info_tool")
    agent.init_state()
    agent.enable_message(InfoTool)
    result: lr.ChatDocument = await task.run_async(
        """
        Use the TOOL `info_tool` to find the address of the Municipal Building
        so you can send it to John Doe on 2023-10-01.
        """,
        turns=3,
    )
    assert "address" in result.content.lower()


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
    all_tools = await get_tools_async(mcp_server())

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
    tools = await get_tools_async(transport)
    assert isinstance(tools, list)
    assert tools, "Expected at least one tool"
    WebSearchTool = await get_tool_async(transport, "web_search_exa")

    assert WebSearchTool is not None
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
    agent.enable_message(WebSearchTool)
    # Note: we shouldn't have to explicitly beg the LLM to use the tool here
    # but I've found that even GPT-4o sometimes fails to use the tool
    question = """
    Use the `web_search_exa` TOOL to find out:
    Who won the Presidential election in Gabon in 2025?
    """
    response = await agent.llm_response_async(question)

    tools = agent.get_tool_messages(response)
    assert len(tools) == 1
    assert isinstance(tools[0], WebSearchTool)

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
@pytest.mark.skip(reason="Requires mcp-server-git to be installed via uvx")
async def test_uvxstdio_transport() -> None:
    """
    Test that we can create Langroid ToolMessage from an MCP server
    via uvx stdio transport. We use this example `git` MCP server:
    https://github.com/modelcontextprotocol/servers/tree/main/src/git
    """
    tools = await get_tools_async(transport)
    assert isinstance(tools, list)
    assert tools, "Expected at least one tool"
    GitStatusTool = await get_tool_async(transport, "git_status")

    assert GitStatusTool is not None
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
        Use the `git_status` TOOL to find out the status of the 
        current git repository at "../langroid"
        """

    response = await agent.llm_response_async(prompt)
    tools = agent.get_tool_messages(response)
    assert len(tools) == 1
    assert isinstance(tools[0], GitStatusTool)

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
    assert "status" in result.content.lower()


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
    tools = await get_tools_async(transport)
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


@pytest.mark.asyncio
async def test_persist_connection() -> None:
    """Test that persist_connection keeps the connection open between tool calls."""
    server = mcp_server()

    # Create client with persist_connection=True
    async with FastMCPClient(server, persist_connection=True) as client:
        # First tool call - this should create and keep the connection open
        tool1 = await client.get_tool_async("add_beans")
        assert tool1 is not None

        # Check that client connection is established
        assert client.client is not None
        initial_client = client.client

        # Second tool call - should reuse the same connection
        tool2 = await client.get_tool_async("get_num_beans")
        assert tool2 is not None

        # Verify the same client connection was reused
        assert client.client is initial_client

        # Call the tools to ensure they work
        add_msg = tool1(x=5)
        result1 = await add_msg.handle_async()
        assert result1 == "5"  # handle_async returns string for backward compatibility

        get_msg = tool2()
        result2 = await get_msg.handle_async()
        assert result2 == "5"  # handle_async returns string for backward compatibility


@pytest.mark.asyncio
async def test_handle_async_with_images() -> None:
    """Test that response_async returns ChatDocument with file attachments."""
    # Create a mock server that returns image content
    server = FastMCP("ImageServer")

    @server.tool()
    async def get_chart() -> List[TextContent | ImageContent]:
        """Get a chart with image."""
        return [
            TextContent(type="text", text="Here is your chart:"),
            ImageContent(
                type="image",
                mimeType="image/png",
                data="iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",  # noqa: E501
            ),
        ]

    # Get the tool and test response_async
    async with FastMCPClient(server, forward_images=True) as client:
        ChartTool = await client.get_tool_async("get_chart")

        # Create a mock agent
        agent = lr.ChatAgent(lr.ChatAgentConfig())

        # Test response_async method
        chart_msg = ChartTool()
        response = await chart_msg.handle_async(agent)

        # Verify we got a ChatDocument
        assert isinstance(response, lr.ChatDocument)
        assert "Here is your chart:" in response.content

        # Verify we have file attachments in the files attribute
        assert response.files is not None
        assert len(response.files) == 1

        # Verify the file is an image
        file = response.files[0]
        assert file.mime_type == "image/png"


@pytest.mark.asyncio
async def test_forward_text_resources() -> None:
    """Test that forward_text_resources setting works correctly."""
    from mcp.types import CallToolResult

    server = FastMCP("TextResourceServer")

    # Test the _convert_tool_result method directly with mocked data
    async with FastMCPClient(server, forward_text_resources=True) as client:
        # Create a mock CallToolResult with text and text resource
        mock_result = CallToolResult(
            content=[
                TextContent(type="text", text="Document content:"),
                EmbeddedResource(
                    type="resource",
                    resource=TextResourceContents(
                        uri="file:///example.txt",
                        mimeType="text/plain",
                        text="This is embedded text content from a resource.",
                    ),
                ),
            ],
            isError=False,
        )

        # Test with forward_text_resources=True
        result = client._convert_tool_result("test_tool", mock_result)
        content, files = result

        # Should include both the main text and the resource text
        assert "Document content:" in content
        assert "This is embedded text content from a resource." in content

    # Test with forward_text_resources=False
    async with FastMCPClient(server, forward_text_resources=False) as client:
        result = client._convert_tool_result("test_tool", mock_result)
        content, files = result

        # Should only include the main text, not the resource text
        assert "Document content:" in content
        assert "This is embedded text content from a resource." not in content


@pytest.mark.asyncio
async def test_forward_blob_resources() -> None:
    """Test that forward_blob_resources setting works correctly."""
    from mcp.types import CallToolResult

    server = FastMCP("BlobResourceServer")

    # Test the _convert_tool_result method directly with mocked data
    async with FastMCPClient(server, forward_blob_resources=True) as client:
        # Small PNG data (1x1 blue pixel)
        png_data = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAD0lEQVR42mNkYPhfz/ADAAKAA4RkT4UVAAAAAElFTkSuQmCC"  # noqa: E501

        # Create a mock CallToolResult with text and blob resource
        mock_result = CallToolResult(
            content=[
                TextContent(type="text", text="Document with blob:"),
                EmbeddedResource(
                    type="resource",
                    resource=BlobResourceContents(
                        uri="file:///example.png", mimeType="image/png", blob=png_data
                    ),
                ),
            ],
            isError=False,
        )

        # Test with forward_blob_resources=True
        result = client._convert_tool_result("test_tool", mock_result)
        content, files = result

        # Should have text content and file attachment
        assert "Document with blob:" in content
        assert len(files) == 1
        assert files[0].mime_type == "image/png"

    # Test with forward_blob_resources=False
    async with FastMCPClient(server, forward_blob_resources=False) as client:
        result = client._convert_tool_result("test_tool", mock_result)
        content, files = result

        # Should only have text content, no file attachments
        assert "Document with blob:" in content
        assert len(files) == 0
