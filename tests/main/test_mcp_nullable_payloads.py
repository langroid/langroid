"""MCP calls preserve explicit null arguments without changing defaults."""

import json

import pytest
from fastmcp import FastMCP
from pydantic import BaseModel

from langroid.agent.tools.mcp import FastMCPClient


@pytest.mark.asyncio
@pytest.mark.parametrize("persist_connection", [False, True])
@pytest.mark.parametrize("tool_name", ["required_value", "optional_value"])
async def test_explicit_null_reaches_mcp_tool(
    persist_connection: bool, tool_name: str
) -> None:
    server = FastMCP("NullableArguments")

    @server.tool()
    def required_value(value: int | None) -> str:
        return "null" if value is None else str(value)

    @server.tool()
    def optional_value(value: int | None = 7) -> str:
        return "null" if value is None else str(value)

    async with FastMCPClient(server, persist_connection=persist_connection) as client:
        tool = await client.get_tool_async(tool_name)
        assert (
            await tool.model_validate_json('{"value": null}').handle_async() == "null"
        )
        assert await tool(value=3).handle_async() == "3"
        if tool_name == "optional_value":
            assert await tool().handle_async() == "7"


@pytest.mark.asyncio
async def test_nested_null_arguments_reach_mcp_tool() -> None:
    class Entry(BaseModel):
        value: int | None
        limit: int = 7
        label: str = "fallback"

    server = FastMCP("NestedNullableArguments")

    @server.tool()
    def inspect_entries(entry: Entry, entries: list[Entry], raw: dict) -> str:
        return json.dumps(
            {
                "entry": entry.model_dump(),
                "entries": [item.model_dump() for item in entries],
                "raw": raw,
            }
        )

    async with FastMCPClient(server) as client:
        definition = await client.get_mcp_tool_async("inspect_entries")
        assert definition is not None
        # A server may omit default annotations for optional parameters.
        entry_schema = Entry.model_json_schema()
        del entry_schema["properties"]["label"]["default"]
        definition.inputSchema["properties"]["entry"] = entry_schema
        definition.inputSchema["properties"]["entries"]["items"] = entry_schema
        tool = client.tool_model_from_mcp_tool(definition)
        arguments = {
            "entry": {"value": None},
            "entries": [{"value": None}, {"value": 3}],
            "raw": {"value": None, "items": [None]},
        }
        result = await tool(**arguments).handle_async()
        assert json.loads(result) == {
            "entry": {"value": None, "limit": 7, "label": "fallback"},
            "entries": [
                {"value": None, "limit": 7, "label": "fallback"},
                {"value": 3, "limit": 7, "label": "fallback"},
            ],
            "raw": arguments["raw"],
        }


@pytest.mark.asyncio
async def test_null_for_non_nullable_optional_uses_the_default() -> None:
    """A null for a param the server does NOT declare nullable is dropped.

    Strict tool calling is the default for OpenAI models, and
    `format_schema_for_strict` removes every default and marks every property
    required -- so the model's only way to say "use the default" is to emit
    `null`. Forwarding that null makes the server reject the call.
    """
    server = FastMCP("NonNullableOptional")

    @server.tool()
    def repeat(word: str, times: int = 2) -> str:
        return " ".join([word] * times)

    async with FastMCPClient(server) as client:
        tool = await client.get_tool_async("repeat")
        # Precondition: the server declares `times` as a plain integer, so a
        # null for it is NOT a valid argument.
        definition = await client.get_mcp_tool_async("repeat")
        assert definition is not None
        assert definition.inputSchema["properties"]["times"]["type"] == "integer"
        assert await tool(word="hi", times=None).handle_async() == "hi hi"
        # Assignment is the other route to an explicitly-set None, since
        # ToolMessage sets validate_assignment=True.
        msg = tool(word="hi", times=3)
        msg.times = None
        assert await msg.handle_async() == "hi hi"


@pytest.mark.asyncio
async def test_null_for_non_nullable_nested_field_uses_the_default() -> None:
    """The same rule applies inside a nested model parameter."""

    class Entry(BaseModel):
        value: int | None
        limit: int = 7

    server = FastMCP("NestedNonNullable")

    @server.tool()
    def inspect_entry(entry: Entry) -> str:
        return json.dumps(entry.model_dump())

    async with FastMCPClient(server) as client:
        tool = await client.get_tool_async("inspect_entry")
        result = await tool(entry={"value": None, "limit": None}).handle_async()
        # `value` is nullable and survives; `limit` is not, so it falls back.
        assert json.loads(result) == {"value": None, "limit": 7}


@pytest.mark.asyncio
async def test_mcp_tool_subclass_keeps_non_null_default() -> None:
    server = FastMCP("CustomDefaults")

    @server.tool()
    def optional_value(value: int | None = 7) -> int | None:
        return value

    async with FastMCPClient(server) as client:
        tool = await client.get_tool_async("optional_value")

        class CustomTool(tool):  # type: ignore
            value: int | None = 11

        assert await CustomTool().handle_async() == "11"


@pytest.mark.asyncio
async def test_renamed_null_argument_reaches_mcp_tool() -> None:
    server = FastMCP("RenamedNullableArguments")

    @server.tool()
    def lookup(id: int | None) -> str:
        return "null" if id is None else str(id)

    async with FastMCPClient(server) as client:
        tool = await client.get_tool_async("lookup")
        assert await tool(id__=None).handle_async() == "null"
