"""Langroid reads MCP model fields under either MCP SDK's naming.

MCP SDK v2 -- the SDK `fastmcp>=4` is built on -- renamed every camelCase
field of the `mcp.types` models to snake_case (`Tool.inputSchema` became
`Tool.input_schema`, `CallToolResult.isError` became `is_error`, ...). SDK v1,
which this package's `fastmcp>=2.2.5` floor still admits, has only the old
names. These tests pin the accessor that bridges the two, and then check it
against whichever SDK is actually installed.
"""

import warnings
from typing import Any

import pytest
from fastmcp import FastMCP
from mcp.types import CallToolResult, ImageContent, TextContent, Tool

from langroid.agent.tools.mcp import FastMCPClient
from langroid.agent.tools.mcp.fastmcp_client import _mcp_field

# A 1x1 PNG, base64, so the image branch has real bytes to decode.
_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA"
    "60e6kgAAAABJRU5ErkJggg=="
)


class _V2Model:
    """Carries only the SDK v2 (snake_case) spelling."""

    input_schema = {"type": "object", "properties": {"v2": {}}}


class _V1Model:
    """Carries only the SDK v1 (camelCase) spelling."""

    inputSchema = {"type": "object", "properties": {"v1": {}}}


class _BothModel:
    """Carries both, as the SDK v2 deprecation shim effectively does."""

    input_schema = "modern"
    inputSchema = "legacy"


def test_mcp_field_prefers_the_modern_name() -> None:
    """Reading the v2 name first is what avoids v2's deprecation warning."""
    assert _mcp_field(_BothModel(), "input_schema", "inputSchema") == "modern"


def test_mcp_field_falls_back_to_the_legacy_name() -> None:
    """SDK v1 offers no snake_case name at all."""
    got = _mcp_field(_V1Model(), "input_schema", "inputSchema")
    assert got == {"type": "object", "properties": {"v1": {}}}


def test_mcp_field_reads_the_modern_name() -> None:
    got = _mcp_field(_V2Model(), "input_schema", "inputSchema")
    assert got == {"type": "object", "properties": {"v2": {}}}


def test_mcp_field_returns_the_default_when_absent() -> None:
    """A model carrying neither name must not raise -- callers pass defaults."""
    assert _mcp_field(object(), "input_schema", "inputSchema") is None
    assert _mcp_field(object(), "a", "b", default=7) == 7


def test_mcp_field_distinguishes_a_present_falsy_value_from_absence() -> None:
    """An empty schema is a value, not a miss: `or` would skip past it."""

    class Empty:
        input_schema: dict[str, Any] = {}

    assert _mcp_field(Empty(), "input_schema", "inputSchema", default={"x": 1}) == {}


def test_mcp_field_reads_a_real_tool_from_the_installed_sdk() -> None:
    """Whatever SDK is installed, the accessor finds the tool's input schema.

    The model is built from the WIRE format (camelCase keys), which both SDKs
    accept through `model_validate` -- SDK v2 keeps the old spellings as
    validation aliases even though its constructor no longer takes them.
    """
    tool = Tool.model_validate({"name": "increment", "inputSchema": {"type": "object"}})
    assert _mcp_field(tool, "input_schema", "inputSchema") == {"type": "object"}


def test_mcp_field_reads_a_real_call_result_from_the_installed_sdk() -> None:
    """Same for the three `CallToolResult` fields Langroid reads."""
    result = CallToolResult.model_validate(
        {
            "content": [TextContent(type="text", text="hi")],
            "structuredContent": {"count": 1},
            "isError": True,
        }
    )
    assert _mcp_field(result, "is_error", "isError") is True
    assert _mcp_field(result, "structured_content", "structuredContent") == {"count": 1}


def _renamed_field_warnings(recorded: list[warnings.WarningMessage]) -> list[str]:
    """The recorded warnings that complain about an SDK v1 field name."""
    return [str(w.message) for w in recorded if "renamed this field" in str(w.message)]


@pytest.mark.asyncio
async def test_a_tool_call_warns_about_no_renamed_field() -> None:
    """Reading the camelCase names is what `_mcp_field` exists to avoid.

    Without it the five call sites each trip MCP SDK v2's deprecation shim,
    so a user on fastmcp 4 gets a `FastMCPDeprecationWarning` for every tool
    call -- and FastMCP 5 removes the shim those reads depend on. This drives
    a real client against a real server so the warnings, if any, come from
    the production path rather than from a hand-built model.

    On SDK v1 there is no such shim and nothing to warn about, so this test
    is a no-op there by construction; it is the v2 run that has teeth.
    """
    server = FastMCP("CompatServer")

    @server.tool()
    async def chart() -> list[TextContent | ImageContent]:
        """Text plus an image, for the two `mimeType` reads."""
        return [
            TextContent(type="text", text="chart"),
            ImageContent(type="image", mimeType="image/png", data=_PNG),
        ]

    @server.tool()
    async def counts() -> dict[str, int]:
        """A dict return, for the `structuredContent` read."""
        return {"count": 1}

    @server.tool()
    async def explode() -> str:
        """A failing tool, for the `isError` read."""
        raise ValueError("boom")

    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        async with FastMCPClient(server, forward_images=True) as client:
            # `get_tool_async` goes through the `inputSchema` read.
            for name in ("chart", "counts", "explode"):
                tool = await client.get_tool_async(name)
                assert tool is not None
                await client.call_mcp_tool(name, {})

    assert _renamed_field_warnings(recorded) == []
