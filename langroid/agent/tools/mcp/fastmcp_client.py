import asyncio
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, cast

from dotenv import load_dotenv
from fastmcp.client import Client
from fastmcp.client.transports import ClientTransport
from fastmcp.server import FastMCP
from mcp.types import CallToolResult, TextContent, Tool

from langroid.agent.tool_message import ToolMessage
from langroid.pydantic_v1 import BaseModel, Field, create_model

load_dotenv()  # load environment variables from .env


class FastMCPClient:
    """A client for interacting with a FastMCP server.

    Provides async context manager functionality to safely manage resources.
    """

    logger = logging.getLogger(__name__)
    _cm: Optional[Client] = None
    client: Optional[Client] = None

    def __init__(self, server: str | FastMCP[Any] | ClientTransport) -> None:
        """Initialize the FastMCPClient.

        Args:
            server: FastMCP server or path to such a server
        """
        self.server = server
        self.client = None
        self._cm = None

    async def __aenter__(self) -> "FastMCPClient":
        """Enter the async context manager and connect inner client."""
        # create inner client context manager
        self._cm = Client(self.server)
        # actually enter it (opens the session)
        self.client = await self._cm.__aenter__()  # type: ignore
        return self

    async def connect(self) -> None:
        """Open the underlying session."""
        await self.__aenter__()

    async def close(self) -> None:
        """Close the underlying session."""
        await self.__aexit__(None, None, None)

    async def __aexit__(
        self,
        exc_type: Optional[type[Exception]],
        exc_val: Optional[Exception],
        exc_tb: Optional[Any],
    ) -> None:
        """Exit the async context manager and close inner client."""
        # exit and close the inner fastmcp.Client
        if hasattr(self, "_cm"):
            if self._cm is not None:
                await self._cm.__aexit__(exc_type, exc_val, exc_tb)  # type: ignore
        self.client = None
        self._cm = None

    def _schema_to_field(
        self, name: str, schema: Dict[str, Any], prefix: str
    ) -> Tuple[Any, Any]:
        """Convert a JSON Schema snippet into a (type, Field) tuple.

        Args:
            name: Name of the field.
            schema: JSON Schema for this field.
            prefix: Prefix to use for nested model names.

        Returns:
            A tuple of (python_type, Field(...)) for create_model.
        """
        t = schema.get("type")
        default = schema.get("default", ...)
        desc = schema.get("description")
        # Object → nested BaseModel
        if t == "object" and "properties" in schema:
            sub_name = f"{prefix}_{name.capitalize()}"
            sub_fields: Dict[str, Tuple[type, Any]] = {}
            for k, sub_s in schema["properties"].items():
                ftype, fld = self._schema_to_field(sub_name + k, sub_s, sub_name)
                sub_fields[k] = (ftype, fld)
            submodel = create_model(  # type: ignore
                sub_name,
                __base__=BaseModel,
                **sub_fields,
            )
            return submodel, Field(default=default, description=desc)  # type: ignore
        # Array → List of items
        if t == "array" and "items" in schema:
            item_type, _ = self._schema_to_field(name, schema["items"], prefix)
            return List[item_type], Field(default=default, description=desc)  # type: ignore
        # Primitive types
        if t == "string":
            return str, Field(default=default, description=desc)
        if t == "integer":
            return int, Field(default=default, description=desc)
        if t == "number":
            return float, Field(default=default, description=desc)
        if t == "boolean":
            return bool, Field(default=default, description=desc)
        # Fallback or unions
        if any(key in schema for key in ("oneOf", "anyOf", "allOf")):
            self.logger.warning("Unsupported union schema in field %s; using Any", name)
            return Any, Field(default=default, description=desc)
        # Default fallback
        return Any, Field(default=default, description=desc)

    async def get_tool_async(self, tool_name: str) -> Type[ToolMessage]:
        """
        Create a Langroid ToolMessage subclass from the MCP Tool
        with the given `tool_name`.
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Use async with FastMCPClient.")
        target = await self.get_mcp_tool_async(tool_name)
        if target is None:
            raise ValueError(f"No tool named {tool_name}")
        props = target.inputSchema.get("properties", {})
        fields: Dict[str, Tuple[type, Any]] = {}
        for fname, schema in props.items():
            ftype, fld = self._schema_to_field(fname, schema, target.name)
            fields[fname] = (ftype, fld)

        # Convert target.name to CamelCase and add Tool suffix
        parts = target.name.replace("-", "_").split("_")
        camel_case = "".join(part.capitalize() for part in parts)
        model_name = f"{camel_case}Tool"

        from langroid.agent.tool_message import ToolMessage as _BaseToolMessage

        # IMPORTANT: Avoid clashes with reserved field names in Langroid ToolMessage!
        # First figure out which field names are reserved
        reserved = set(_BaseToolMessage.__annotations__.keys())
        reserved.update(["recipient", "_handler"])
        renamed: Dict[str, str] = {}
        new_fields: Dict[str, Tuple[type, Any]] = {}
        for fname, (ftype, fld) in fields.items():
            if fname in reserved:
                new_name = fname + "__"
                renamed[fname] = new_name
                new_fields[new_name] = (ftype, fld)
            else:
                new_fields[fname] = (ftype, fld)
        # now replace fields with our renamed‐aware mapping
        fields = new_fields

        # create Langroid ToolMessage subclass, with expected fields.
        tool_model = cast(
            Type[ToolMessage],
            create_model(  # type: ignore[call-overload]
                model_name,
                request=(str, target.name),
                purpose=(str, target.description or f"Use the tool {target.name}"),
                __base__=ToolMessage,
                **fields,
            ),
        )
        tool_model._server = self.server  # type: ignore[attr-defined]
        tool_model._renamed_fields = renamed  # type: ignore[attr-defined]

        # 2) define an arg-free call_tool_async()
        async def call_tool_async(self: ToolMessage) -> Any:
            from langroid.agent.tools.mcp.fastmcp_client import FastMCPClient

            # pack up the payload
            payload = self.dict(exclude=self.Config.schema_extra["exclude"])

            # restore any renamed fields
            for orig, new in self.__class__._renamed_fields.items():  # type: ignore
                if new in payload:
                    payload[orig] = payload.pop(new)

            # open a fresh client, call the tool, then close
            async with FastMCPClient(self.__class__._server) as client:  # type: ignore
                return await client.call_mcp_tool(self.request, payload)

        tool_model.call_tool_async = call_tool_async  # type: ignore

        if not hasattr(tool_model, "handle_async"):
            # 3) define an arg-free handle_async() method
            # if the tool model doesn't already have one
            async def handle_async(self: ToolMessage) -> Any:
                return await self.call_tool_async()  # type: ignore[attr-defined]

            # add the handle_async() method to the tool model
            tool_model.handle_async = handle_async  # type: ignore

        return tool_model

    async def get_tools_async(self) -> List[Type[ToolMessage]]:
        """
        Get all available tools as Langroid ToolMessage classes,
        handling nested schemas, with `handle_async` methods
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Use async with FastMCPClient.")
        resp = await self.client.list_tools()
        tools: List[Type[ToolMessage]] = []
        for t in resp:
            tools.append(await self.get_tool_async(t.name))
        return tools

    async def get_mcp_tool_async(self, name: str) -> Optional[Tool]:
        """Find the "original" MCP Tool (i.e. of type mcp.types.Tool) on the server
         matching `name`, or None if missing. This contains the metadata for the tool:
         name, description, inputSchema, etc.

        Args:
            name: Name of the tool to look up.

        Returns:
            The raw Tool object from the server, or None.
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Use async with FastMCPClient.")
        resp: List[Tool] = await self.client.list_tools()
        return next((t for t in resp if t.name == name), None)

    def _convert_tool_result(
        self,
        tool_name: str,
        result: CallToolResult,
    ) -> List[str] | str | None:
        if result.isError:
            self.logger.error(f"Error calling MCP tool {tool_name}")
            return None
        has_nontext_results = any(
            not isinstance(item, TextContent) for item in result.content
        )
        if has_nontext_results:
            self.logger.warning(
                f"""
                MCP Tool {tool_name} returned non-text results,
                which will be skipped.
                """,
            )
        results = [
            item.text for item in result.content if isinstance(item, TextContent)
        ]
        if len(results) == 1:
            return results[0]
        return results

    async def call_mcp_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> str | List[str] | None:
        """Call an MCP tool with the given arguments.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            The result of the tool call.
        """
        if not self.client:
            raise RuntimeError("Client not initialized. Use async with FastMCPClient.")
        result: CallToolResult = await self.client.session.call_tool(
            tool_name,
            arguments,
        )
        return self._convert_tool_result(tool_name, result)


async def get_tool_async(
    server: str | ClientTransport,
    tool_name: str,
) -> Type[ToolMessage]:
    async with FastMCPClient(server) as client:
        return await client.get_tool_async(tool_name)


def get_tool(
    server: str | ClientTransport,
    tool_name: str,
) -> Type[ToolMessage]:
    return asyncio.run(get_tool_async(server, tool_name))


async def get_tools_async(
    server: str | ClientTransport,
) -> List[Type[ToolMessage]]:
    async with FastMCPClient(server) as client:
        return await client.get_tools_async()


def get_tools(
    server: str | ClientTransport,
) -> List[Type[ToolMessage]]:
    return asyncio.run(get_tools_async(server))


async def get_mcp_tool_async(
    server: str | ClientTransport,
    name: str,
) -> Optional[Tool]:
    async with FastMCPClient(server) as client:
        return await client.get_mcp_tool_async(name)


async def get_mcp_tools_async(
    server: str | ClientTransport,
) -> List[Tool]:
    async with FastMCPClient(server) as client:
        if not client.client:
            raise RuntimeError("Client not initialized. Use async with FastMCPClient.")
        return await client.client.list_tools()
