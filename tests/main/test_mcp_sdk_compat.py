"""Langroid reads MCP model fields under either MCP SDK's naming.

MCP SDK v2 -- the SDK `fastmcp>=4` is built on -- renamed every camelCase
field of the `mcp.types` models to snake_case (`Tool.inputSchema` became
`Tool.input_schema`, `CallToolResult.isError` became `is_error`, ...). SDK v1,
which this package's `fastmcp>=2.2.5` floor still admits, has only the old
names. These tests pin the accessor that bridges the two, and then check it
against whichever SDK is actually installed.
"""

import ast
import inspect
from types import ModuleType
from typing import Any

from mcp.types import CallToolResult, TextContent, Tool

import langroid.agent.tools.mcp as mcp_package
import langroid.agent.tools.mcp.decorators as decorators
import langroid.agent.tools.mcp.fastmcp_client as fastmcp_client
from langroid.agent.tools.mcp.fastmcp_client import FastMCPClient, _mcp_field


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


# --- the five production reads actually use the modern name -----------------
#
# A warning-based check cannot do this job: MCP SDK v2's compatibility
# properties carry a once-only "already warned" flag that
# `warnings.simplefilter("always")` cannot reset, so any earlier test in the
# same process silences them and the check passes while reading the
# deprecated name. These two guards are deterministic instead -- one reads
# the source, the other makes touching a camelCase name raise.


# The SDK v1 names of the `mcp.types` fields this package reads. SDK v2
# renames ~54 fields in all; this is deliberately the subset Langroid
# touches, so a read of some field nobody uses yet (`nextCursor`,
# `progressToken`, ...) would not be flagged until it is added here.
RENAMED_FIELDS = {
    "inputSchema",
    "outputSchema",
    "isError",
    "structuredContent",
    "mimeType",
}


def _renamed_field_reads(module: ModuleType) -> list[str]:
    """Every place `module`'s source names an SDK v1 field.

    Counts BOTH forms a read can take: `obj.isError` and
    `getattr(obj, "isError", ...)`. The second matters because it is the form
    the pre-fix code actually used for `inputSchema`, and because rewriting a
    dotted read as a `getattr` is a one-token change that would otherwise
    slip past.

    What this cannot see is a name reached indirectly -- through a variable,
    an f-string, or `operator.attrgetter`. `_mcp_field` itself is exactly
    that (`getattr(obj, camel, default)`, where `camel` is a parameter), so
    the rule "only `_mcp_field` may name these" and the check's blind spot
    are the same thing, by construction.
    """
    tree = ast.parse(inspect.getsource(module))
    where = f"{module.__name__.rsplit('.', 1)[-1]}.py"
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in RENAMED_FIELDS:
            offenders.append(f"{where}:{node.lineno} .{node.attr}")
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "getattr"
            and len(node.args) >= 2
            and isinstance(node.args[1], ast.Constant)
            and node.args[1].value in RENAMED_FIELDS
        ):
            offenders.append(
                f"{where}:{node.lineno} getattr(..., {node.args[1].value!r})"
            )
    return offenders


def test_no_production_read_uses_a_renamed_field_name() -> None:
    """`_mcp_field` is the ONLY place allowed to name an SDK v1 field.

    A direct `result.isError` works on both SDKs today -- v2 keeps a
    deprecating shim -- so nothing fails when a call site regresses to one.
    What breaks is the user's logs now (a warning per tool call) and the
    package later, when FastMCP 5 removes the shim. There is no runtime
    check for this: a pydantic subclass cannot shadow an inherited field with
    a raising property, so the deprecated name cannot be booby-trapped on a
    real `ImageContent`. The source is therefore where the rule is asserted,
    across the WHOLE package -- the image and blob `mimeType` reads have no
    other guard.
    """
    offenders: list[str] = []
    for module in (mcp_package, decorators, fastmcp_client):
        offenders += _renamed_field_reads(module)
    assert offenders == [], (
        "read these through _mcp_field(obj, '<snake>', '<camel>') instead: "
        + "; ".join(offenders)
    )


class _SnakeOnlyResult:
    """A `CallToolResult` stand-in on which the SDK v1 names explode.

    `_mcp_field` reads the snake_case name first and returns as soon as it
    finds it, so it never touches these. A call site that reads the
    camelCase name directly raises instead of quietly working.
    """

    def __init__(
        self,
        content: list[Any],
        is_error: bool = False,
        structured_content: Any = None,
    ) -> None:
        self.content = content
        self.is_error = is_error
        self.structured_content = structured_content

    @property
    def isError(self) -> bool:
        raise AssertionError("read `is_error`, not the deprecated `isError`")

    @property
    def structuredContent(self) -> Any:
        raise AssertionError(
            "read `structured_content`, not the deprecated `structuredContent`"
        )


def test_convert_tool_result_reads_the_modern_error_flag() -> None:
    """The `isError` branch, with the deprecated name booby-trapped."""
    client = FastMCPClient("unused")
    result = _SnakeOnlyResult(
        content=[TextContent(type="text", text="refused")], is_error=True
    )
    assert client._convert_tool_result("t", result) == (  # type: ignore[arg-type]
        "ERROR: Tool call failed - ['refused']"
    )


def test_convert_tool_result_reads_the_modern_structured_content() -> None:
    """The `structuredContent` branch, likewise.

    `content` must be EMPTY to get here: a result carrying any text returns
    on the text branch before the structured read, which is why driving a
    dict-returning tool through a real fastmcp server does not reach it --
    fastmcp sends the JSON as text alongside the structured payload.
    """
    client = FastMCPClient("unused")
    result = _SnakeOnlyResult(content=[], structured_content={"count": 1})
    assert client._convert_tool_result("t", result) == ("1", [])  # type: ignore[arg-type]


def test_tool_model_from_mcp_tool_reads_the_modern_input_schema() -> None:
    """The `inputSchema` read, via a `Tool` whose v1 name is booby-trapped."""

    class _SnakeOnlyTool:
        name = "increment"
        description = "Increment a counter"
        input_schema = {
            "type": "object",
            "properties": {"amount": {"type": "integer"}},
            "required": ["amount"],
        }

        @property
        def inputSchema(self) -> Any:
            raise AssertionError(
                "read `input_schema`, not the deprecated `inputSchema`"
            )

    client = FastMCPClient("unused")
    model = client.tool_model_from_mcp_tool(_SnakeOnlyTool())  # type: ignore[arg-type]
    assert "amount" in model.model_fields
