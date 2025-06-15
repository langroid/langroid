import asyncio
import datetime
import logging
from base64 import b64decode
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple, Type, TypeAlias, cast

from dotenv import load_dotenv
from fastmcp.client import Client
from fastmcp.client.roots import (
    RootsHandler,
    RootsList,
)
from fastmcp.client.sampling import SamplingHandler
from fastmcp.client.transports import ClientTransport
from fastmcp.server import FastMCP
from mcp.client.session import (
    LoggingFnT,
    MessageHandlerFnT,
)
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
    Tool,
)

from langroid.agent.base import Agent
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tool_message import ToolMessage
from langroid.parsing.file_attachment import FileAttachment
from langroid.pydantic_v1 import AnyUrl, BaseModel, Field, create_model

load_dotenv()  # load environment variables from .env

FastMCPServerSpec: TypeAlias = str | FastMCP[Any] | ClientTransport | AnyUrl


class FastMCPClient:
    """A client for interacting with a FastMCP server.

    Provides async context manager functionality to safely manage resources.
    """

    logger = logging.getLogger(__name__)
    _cm: Optional[Client] = None
    client: Optional[Client] = None

    def __init__(
        self,
        server: FastMCPServerSpec,
        persist_connection: bool = False,
        forward_images: bool = True,
        forward_text_resources: bool = False,
        forward_blob_resources: bool = False,
        sampling_handler: SamplingHandler | None = None,  # type: ignore
        roots: RootsList | RootsHandler | None = None,  # type: ignore
        log_handler: LoggingFnT | None = None,
        message_handler: MessageHandlerFnT | None = None,
        read_timeout_seconds: datetime.timedelta | None = None,
    ) -> None:
        """Initialize the FastMCPClient.

        Args:
            server: FastMCP server or path to such a server
        """
        self.server = server
        self.client = None
        self._cm = None
        self.sampling_handler = sampling_handler
        self.roots = roots
        self.log_handler = log_handler
        self.message_handler = message_handler
        self.read_timeout_seconds = read_timeout_seconds
        self.persist_connection = persist_connection
        self.forward_text_resources = forward_text_resources
        self.forward_blob_resources = forward_blob_resources
        self.forward_images = forward_images

    async def __aenter__(self) -> "FastMCPClient":
        """Enter the async context manager and connect inner client."""
        # create inner client context manager
        self._cm = Client(
            self.server,
            sampling_handler=self.sampling_handler,
            roots=self.roots,
            log_handler=self.log_handler,
            message_handler=self.message_handler,
            timeout=self.read_timeout_seconds,
        )
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

    def __del__(self) -> None:
        """Warn about unclosed persistent connections."""
        if self.client is not None and self.persist_connection:
            import warnings

            warnings.warn(
                f"FastMCPClient with persist_connection=True was not properly closed. "
                f"Connection to {self.server} may leak resources. "
                f"Use 'async with' or call await client.close()",
                ResourceWarning,
                stacklevel=2,
            )

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
            if self.persist_connection:
                await self.connect()
                assert self.client
            else:
                raise RuntimeError(
                    "Client not initialized. Use async with FastMCPClient."
                )
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
        reserved.update(["recipient", "_handler", "name"])
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
        # Store ALL client configuration needed to recreate a client
        client_config = {
            "server": self.server,
            "sampling_handler": self.sampling_handler,
            "roots": self.roots,
            "log_handler": self.log_handler,
            "message_handler": self.message_handler,
            "read_timeout_seconds": self.read_timeout_seconds,
        }

        tool_model._client_config = client_config  # type: ignore [attr-defined]
        tool_model._renamed_fields = renamed  # type: ignore[attr-defined]

        # 2) define an arg-free call_tool_async()
        async def call_tool_async(itself: ToolMessage) -> Any:
            from langroid.agent.tools.mcp.fastmcp_client import FastMCPClient

            # pack up the payload
            payload = itself.dict(
                exclude=itself.Config.schema_extra["exclude"].union(
                    ["request", "purpose"]
                ),
            )

            # restore any renamed fields
            for orig, new in itself.__class__._renamed_fields.items():  # type: ignore
                if new in payload:
                    payload[orig] = payload.pop(new)

            client_cfg = getattr(itself.__class__, "_client_config", None)  # type: ignore
            if not client_cfg:
                # Fallback or error - ideally _client_config should always exist
                raise RuntimeError(f"Client config missing on {itself.__class__}")

            # Connect the client if not yet connected and keep the connection open
            if self.persist_connection:
                if not self.client:
                    await self.connect()

                return await self.call_mcp_tool(itself.request, payload)

            # open a fresh client, call the tool, then close
            async with FastMCPClient(**client_cfg) as client:  # type: ignore
                return await client.call_mcp_tool(itself.request, payload)

        tool_model.call_tool_async = call_tool_async  # type: ignore

        if not hasattr(tool_model, "handle_async"):
            # 3) define handle_async() method with optional agent parameter
            from typing import Union

            async def handle_async(
                self: ToolMessage, agent: Optional[Agent] = None
            ) -> Union[str, Optional[ChatDocument]]:
                """
                Auto-generated handler for MCP tool. Returns ChatDocument with files
                if files are present and agent is provided, otherwise returns text.

                To override: define your own handle_async method with matching signature
                if you need file handling, or simpler signature if you only need text.
                """
                response = await self.call_tool_async()  # type: ignore[attr-defined]
                if response is None:
                    return None

                content, files = response

                # If we have files and an agent is provided, return a ChatDocument
                if files and agent is not None:
                    return agent.create_agent_response(
                        content=content,
                        files=files,
                    )
                else:
                    # Otherwise, just return the text content
                    return str(content) if content is not None else None

            # add the handle_async() method to the tool model
            tool_model.handle_async = handle_async  # type: ignore

        return tool_model

    async def get_tools_async(self) -> List[Type[ToolMessage]]:
        """
        Get all available tools as Langroid ToolMessage classes,
        handling nested schemas, with `handle_async` methods
        """
        if not self.client:
            if self.persist_connection:
                await self.connect()
                assert self.client
            else:
                raise RuntimeError(
                    "Client not initialized. Use async with FastMCPClient."
                )
        resp = await self.client.list_tools()
        return [await self.get_tool_async(t.name) for t in resp]

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
            if self.persist_connection:
                await self.connect()
                assert self.client
            else:
                raise RuntimeError(
                    "Client not initialized. Use async with FastMCPClient."
                )
        resp: List[Tool] = await self.client.list_tools()
        return next((t for t in resp if t.name == name), None)

    def _convert_tool_result(
        self,
        tool_name: str,
        result: CallToolResult,
    ) -> Optional[str | tuple[str, list[FileAttachment]]]:
        if result.isError:
            # Log more detailed error information
            error_content = None
            if result.content and len(result.content) > 0:
                try:
                    error_content = [
                        item.text if hasattr(item, "text") else str(item)
                        for item in result.content
                    ]
                except Exception as e:
                    error_content = [f"Could not extract error content: {str(e)}"]

            self.logger.error(
                f"Error calling MCP tool {tool_name}. Details: {error_content}"
            )
            return f"ERROR: Tool call failed - {error_content}"

        results_text = [
            item.text for item in result.content if isinstance(item, TextContent)
        ]
        results_file = []

        for item in result.content:
            if isinstance(item, ImageContent) and self.forward_images:
                results_file.append(
                    FileAttachment.from_bytes(
                        b64decode(item.data),
                        mime_type=item.mimeType,
                    )
                )
            elif isinstance(item, EmbeddedResource):
                if (
                    isinstance(item.resource, TextResourceContents)
                    and self.forward_text_resources
                ):
                    results_text.append(item.resource.text)
                elif (
                    isinstance(item.resource, BlobResourceContents)
                    and self.forward_blob_resources
                ):
                    results_file.append(
                        FileAttachment.from_io(
                            BytesIO(b64decode(item.resource.blob)),
                            mime_type=item.resource.mimeType,
                        )
                    )

        return "\n".join(results_text), results_file

    async def call_mcp_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Optional[tuple[str, list[FileAttachment]]]:
        """Call an MCP tool with the given arguments.

        Args:
            tool_name: Name of the tool to call.
            arguments: Arguments to pass to the tool.

        Returns:
            The result of the tool call.
        """
        if not self.client:
            if self.persist_connection:
                await self.connect()
                assert self.client
            else:
                raise RuntimeError(
                    "Client not initialized. Use async with FastMCPClient."
                )
        result: CallToolResult = await self.client.session.call_tool(
            tool_name,
            arguments,
        )
        results = self._convert_tool_result(tool_name, result)

        if isinstance(results, str):
            return results, []

        return results


# ==============================================================================
# Convenience functions (wrappers around FastMCPClient methods)
# These are useful for one-off calls without needing to manage the
# FastMCPClient context explicitly.
# ==============================================================================


async def get_tool_async(
    server: FastMCPServerSpec,
    tool_name: str,
    **client_kwargs: Any,
) -> Type[ToolMessage]:
    """Get a single Langroid ToolMessage subclass for a specific MCP tool name (async).

    This is a convenience wrapper that creates a temporary FastMCPClient.

    Args:
        server: Specification of the FastMCP server to connect to.
        tool_name: The name of the tool to retrieve.
        **client_kwargs: Additional keyword arguments to pass to the
            FastMCPClient constructor (e.g., sampling_handler, roots).

    Returns:
        A dynamically created Langroid ToolMessage subclass representing the
        requested tool.
    """
    async with FastMCPClient(server, **client_kwargs) as client:
        return await client.get_tool_async(tool_name)


def get_tool(
    server: FastMCPServerSpec,
    tool_name: str,
    **client_kwargs: Any,
) -> Type[ToolMessage]:
    """Get a single Langroid ToolMessage subclass
    for a specific MCP tool name (synchronous).

    This is a convenience wrapper that creates a temporary FastMCPClient and runs the
    async `get_tool_async` function using `asyncio.run()`.

    Args:
        server: Specification of the FastMCP server to connect to.
        tool_name: The name of the tool to retrieve.
        **client_kwargs: Additional keyword arguments to pass to the
            FastMCPClient constructor (e.g., sampling_handler, roots).

    Returns:
        A dynamically created Langroid ToolMessage subclass representing the
        requested tool.
    """
    return asyncio.run(get_tool_async(server, tool_name, **client_kwargs))


async def get_tools_async(
    server: FastMCPServerSpec,
    **client_kwargs: Any,
) -> List[Type[ToolMessage]]:
    """Get all available tools as Langroid ToolMessage subclasses (async).

    This is a convenience wrapper that creates a temporary FastMCPClient.

    Args:
        server: Specification of the FastMCP server to connect to.
        **client_kwargs: Additional keyword arguments to pass to the
            FastMCPClient constructor (e.g., sampling_handler, roots).

    Returns:
        A list of dynamically created Langroid ToolMessage subclasses
        representing all available tools on the server.
    """
    async with FastMCPClient(server, **client_kwargs) as client:
        return await client.get_tools_async()


def get_tools(
    server: FastMCPServerSpec,
    **client_kwargs: Any,
) -> List[Type[ToolMessage]]:
    """Get all available tools as Langroid ToolMessage subclasses (synchronous).

    This is a convenience wrapper that creates a temporary FastMCPClient and runs the
    async `get_tools_async` function using `asyncio.run()`.

    Args:
        server: Specification of the FastMCP server to connect to.
        **client_kwargs: Additional keyword arguments to pass to the
            FastMCPClient constructor (e.g., sampling_handler, roots).

    Returns:
        A list of dynamically created Langroid ToolMessage subclasses
        representing all available tools on the server.
    """
    return asyncio.run(get_tools_async(server, **client_kwargs))


async def get_mcp_tool_async(
    server: FastMCPServerSpec,
    name: str,
    **client_kwargs: Any,
) -> Optional[Tool]:
    """Get the raw MCP Tool object for a specific tool name (async).

    This is a convenience wrapper that creates a temporary FastMCPClient to
    retrieve the tool definition from the server.

    Args:
        server: Specification of the FastMCP server to connect to.
        name: The name of the tool to look up.
        **client_kwargs: Additional keyword arguments to pass to the
            FastMCPClient constructor.

    Returns:
        The raw `mcp.types.Tool` object from the server, or `None` if the tool
        is not found.
    """
    async with FastMCPClient(server, **client_kwargs) as client:
        return await client.get_mcp_tool_async(name)


async def get_mcp_tools_async(
    server: FastMCPServerSpec,
    **client_kwargs: Any,
) -> List[Tool]:
    """Get all available raw MCP Tool objects from the server (async).

    This is a convenience wrapper that creates a temporary FastMCPClient to
    retrieve the list of tool definitions from the server.

    Args:
        server: Specification of the FastMCP server to connect to.
        **client_kwargs: Additional keyword arguments to pass to the
            FastMCPClient constructor.

    Returns:
        A list of raw `mcp.types.Tool` objects available on the server.
    """
    async with FastMCPClient(server, **client_kwargs) as client:
        if not client.client:
            raise RuntimeError("Client not initialized. Use async with FastMCPClient.")
        return await client.client.list_tools()
