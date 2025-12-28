import asyncio
import datetime
import inspect
import logging
import os
from base64 import b64decode
from io import BytesIO
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeAlias, cast

from dotenv import load_dotenv
from fastmcp.client import Client
from fastmcp.client.roots import (
    RootsHandler,
    RootsList,
)
from fastmcp.client.sampling import SamplingHandler
from fastmcp.client.transports import ClientTransport, StdioTransport

try:
    # Optional transports; import guarded for environments without uvx/npx
    from fastmcp.client.transports import NpxStdioTransport, UvxStdioTransport
except Exception:  # pragma: no cover - optional
    NpxStdioTransport = tuple()  # type: ignore
    UvxStdioTransport = tuple()  # type: ignore
from anyio import ClosedResourceError
from fastmcp.server import FastMCP
from mcp.client.session import (
    LoggingFnT,
    MessageHandlerFnT,
)
from mcp.shared.exceptions import McpError
from mcp.types import (
    BlobResourceContents,
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    TextContent,
    TextResourceContents,
    Tool,
)
from pydantic import AnyUrl, BaseModel, Field, create_model

from langroid.agent.base import Agent
from langroid.agent.chat_document import ChatDocument
from langroid.agent.tool_message import ToolMessage
from langroid.parsing.file_attachment import FileAttachment

load_dotenv()  # load environment variables from .env

# Concrete server/transport spec accepted by fastmcp.Client
FastMCPServerConcrete: TypeAlias = str | FastMCP[Any] | ClientTransport | AnyUrl
# Public spec we accept: concrete spec or a zero-arg factory returning a spec
FastMCPServerSpec: TypeAlias = (
    FastMCPServerConcrete | Callable[[], FastMCPServerConcrete]
)


class FastMCPClient:
    """A client for interacting with a FastMCP server.

    Provides async context manager functionality to safely manage resources.
    """

    logger = logging.getLogger(__name__)
    _cm: Optional[Client[ClientTransport]] = None
    client: Optional[Client[ClientTransport]] = None
    read_timeout_seconds: datetime.timedelta | None = None

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
        # Default a slightly larger read timeout for stdio transports on first
        # connects. Allows flaky subprocess servers a bit more time to boot.
        if read_timeout_seconds is None:
            try:
                default_secs = int(os.getenv("LANGROID_MCP_READ_TIMEOUT", "15"))
                self.read_timeout_seconds = datetime.timedelta(seconds=default_secs)
            except Exception:
                self.read_timeout_seconds = None
        else:
            self.read_timeout_seconds = read_timeout_seconds
        self.persist_connection = persist_connection
        self.forward_text_resources = forward_text_resources
        self.forward_blob_resources = forward_blob_resources
        self.forward_images = forward_images

    async def __aenter__(self) -> "FastMCPClient":
        """Enter the async context manager and connect inner client.

        Always obtain a fresh transport/spec via a factory, then connect.
        If the session initialization fails due to a transient stdio issue
        (e.g., ClosedResourceError / connection closed), retry once with a
        new transport instance for better resilience across fastmcp/mcp
        versions and server launch timing.
        """
        # Always normalize to a server factory and create a fresh spec
        server_factory = self._as_server_factory(self.server)

        # Configurable retry/backoff for transient stdio startup races.
        max_retries = int(os.getenv("LANGROID_MCP_CONNECT_RETRIES", "6"))
        try:
            backoff_base = float(os.getenv("LANGROID_MCP_CONNECT_BACKOFF_BASE", "0.35"))
        except Exception:
            backoff_base = 0.35

        last_err: Optional[BaseException] = None
        for attempt in range(1, max_retries + 1):
            server_spec: FastMCPServerConcrete = server_factory()
            # create inner client context manager
            self._cm = Client(  # type: ignore[assignment]
                server_spec,
                sampling_handler=self.sampling_handler,
                roots=self.roots,
                log_handler=self.log_handler,
                message_handler=self.message_handler,
                timeout=self.read_timeout_seconds,
            )
            try:
                # actually enter it (opens the session)
                self.client = await self._cm.__aenter__()  # type: ignore
                return self
            except (ClosedResourceError, McpError) as e:
                # Common transient failures when a subprocess exits early or
                # closes during initialize. Retry once with a fresh transport.
                self.logger.warning(
                    "FastMCPClient connect attempt %s failed: %s. Retrying...",
                    attempt,
                    e,
                )
                last_err = e
                # ensure we reset _cm/client before retry
                try:
                    if self._cm is not None:
                        await self._cm.__aexit__(None, None, None)  # type: ignore
                except Exception:
                    pass
                self._cm = None
                self.client = None
                # brief backoff to allow server process to finish booting
                try:
                    await asyncio.sleep(min(backoff_base * (2 ** (attempt - 1)), 2.0))
                except Exception:
                    pass
                continue
            except RuntimeError as e:
                # fastmcp wraps ClosedResourceError into RuntimeError
                # "Server session was closed unexpectedly". Treat as transient.
                emsg = str(e)
                if (
                    "Server session was closed unexpectedly" in emsg
                    or "Client failed to connect" in emsg
                ):
                    self.logger.warning(
                        (
                            "FastMCPClient connect attempt %s failed (runtime): %s. "
                            "Retrying..."
                        ),
                        attempt,
                        e,
                    )
                    last_err = e
                    try:
                        if self._cm is not None:
                            await self._cm.__aexit__(None, None, None)  # type: ignore
                    except Exception:
                        pass
                    self._cm = None
                    self.client = None
                    try:
                        await asyncio.sleep(
                            min(backoff_base * (2 ** (attempt - 1)), 2.0)
                        )
                    except Exception:
                        pass
                    continue
                # otherwise re-raise
                raise

        # If we get here both attempts failed
        assert last_err is not None
        raise last_err

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
        self, name: str, schema: Dict[str, Any], prefix: str, is_required: bool = True
    ) -> Tuple[Any, Any]:
        """Convert a JSON Schema snippet into a (type, Field) tuple.

        Args:
            name: Name of the field.
            schema: JSON Schema for this field.
            prefix: Prefix to use for nested model names.
            is_required: Whether this field is required (from JSON Schema "required").

        Returns:
            A tuple of (python_type, Field(...)) for create_model.
        """
        t = schema.get("type")
        # Use schema default if present, otherwise:
        # ... for required fields, None for optional fields
        if "default" in schema:
            default = schema["default"]
        else:
            default = ... if is_required else None
        desc = schema.get("description")
        # Object → nested BaseModel
        if t == "object" and "properties" in schema:
            sub_name = f"{prefix}_{name.capitalize()}"
            sub_fields: Dict[str, Tuple[type, Any]] = {}
            # Get required fields for this nested object
            nested_required = set(schema.get("required", []))
            for k, sub_s in schema["properties"].items():
                ftype, fld = self._schema_to_field(
                    sub_name + k, sub_s, sub_name, is_required=k in nested_required
                )
                sub_fields[k] = (ftype, fld)
            submodel = create_model(  # type: ignore
                sub_name,
                __base__=BaseModel,
                **sub_fields,
            )
            # Wrap in Optional if not required
            model_type = submodel if is_required else Optional[submodel]
            return model_type, Field(default=default, description=desc)  # type: ignore
        # Array → List of items
        if t == "array" and "items" in schema:
            item_type, _ = self._schema_to_field(name, schema["items"], prefix)
            array_type = List[item_type]  # type: ignore
            if not is_required:
                array_type = Optional[array_type]  # type: ignore
            return array_type, Field(default=default, description=desc)  # type: ignore
        # Primitive types
        if t == "string":
            str_type = str if is_required else Optional[str]
            return str_type, Field(default=default, description=desc)
        if t == "integer":
            int_type = int if is_required else Optional[int]
            return int_type, Field(default=default, description=desc)
        if t == "number":
            float_type = float if is_required else Optional[float]
            return float_type, Field(default=default, description=desc)
        if t == "boolean":
            bool_type = bool if is_required else Optional[bool]
            return bool_type, Field(default=default, description=desc)
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
        # Get the list of required fields from JSON Schema
        required_fields = set(target.inputSchema.get("required", []))
        fields: Dict[str, Tuple[type, Any]] = {}
        for fname, schema in props.items():
            ftype, fld = self._schema_to_field(
                fname, schema, target.name, is_required=fname in required_fields
            )
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
            # Always store a SERVER FACTORY to ensure a fresh transport per call
            "server": self._as_server_factory(self.server),
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
            # Get exclude fields from model config with proper type checking
            exclude_fields = set()
            model_config = getattr(itself, "model_config", {})
            if (
                isinstance(model_config, dict)
                and "json_schema_extra" in model_config
                and model_config["json_schema_extra"] is not None
                and isinstance(model_config["json_schema_extra"], dict)
                and "exclude" in model_config["json_schema_extra"]
            ):
                exclude_list = model_config["json_schema_extra"]["exclude"]
                if isinstance(exclude_list, (list, set, tuple)):
                    exclude_fields = set(exclude_list)

            # Add standard excluded fields
            exclude_fields.update(["request", "purpose"])

            # Exclude None values - MCP servers don't expect None for optional params
            payload = itself.model_dump(exclude=exclude_fields, exclude_none=True)

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

    @staticmethod
    def _as_server_factory(
        server: FastMCPServerSpec,
    ) -> Callable[[], FastMCPServerConcrete]:
        """Normalize a server spec to a zero-arg factory.

        - If already callable, return as-is.
        - If a ClientTransport instance, return a factory that yields the SAME
          instance. This preserves state for keep-alive stdio transports (e.g.,
          npx/uvx servers) so multi-call workflows can share process state.
          Recreating a fresh transport each call would lose stateful servers
          like `@modelcontextprotocol/server-memory` and break tests.
        - Otherwise return a factory that yields the given spec.
        """
        if callable(server):  # type: ignore[arg-type]
            return server  # type: ignore[return-value]

        if isinstance(server, ClientTransport):
            # Reuse policy split:
            # - Npx/Uvx stdio transports: reuse the SAME instance to preserve
            #   keep-alive subprocess state (stateful MCP servers).
            # - Plain StdioTransport: CLONE a fresh transport to avoid reusing
            #   process/pipes across decorator-time schema fetch and runtime calls
            #   (some stdio servers close after first session, like CLI wrappers).
            try:
                if (
                    not isinstance(NpxStdioTransport, tuple)
                    and isinstance(server, NpxStdioTransport)
                ) or (  # type: ignore[arg-type]
                    not isinstance(UvxStdioTransport, tuple)
                    and isinstance(server, UvxStdioTransport)
                ):  # type: ignore[arg-type]
                    return lambda: server
            except Exception:
                # If optional classes are tuples (import failed), fall through
                pass

            if isinstance(server, StdioTransport):
                # Best‑effort clone with back‑compat: only pass kwargs supported
                # by this installed fastmcp version's StdioTransport.__init__.
                sig = inspect.signature(StdioTransport.__init__)
                params = sig.parameters

                def _pick(name: str, default: Any = None) -> Any:
                    return getattr(server, name, default) if name in params else None

                # Required in all known versions
                cmd = getattr(server, "command", None)
                args = list(getattr(server, "args", []) or [])

                # Optional, filter by signature presence
                env = _pick("env")
                cwd = _pick("cwd")
                keep_alive = _pick("keep_alive")
                log_file = _pick("log_file")

                def _factory() -> StdioTransport:
                    kwargs = {"command": cmd, "args": args}
                    if "env" in params and env is not None:
                        kwargs["env"] = env
                    if "cwd" in params and cwd is not None:
                        kwargs["cwd"] = cwd
                    if "keep_alive" in params and keep_alive is not None:
                        kwargs["keep_alive"] = keep_alive
                    if "log_file" in params and log_file is not None:
                        kwargs["log_file"] = log_file
                    return StdioTransport(**kwargs)  # type: ignore[arg-type]

                return _factory

            # Default for other ClientTransport types: reuse
            return lambda: server

        return lambda: server  # type: ignore[return-value]

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

        # 1) Collect any plain TextContent first. This preserves legacy behavior
        # for simple servers that return only text. If we have text, prefer it
        # over structuredContent to avoid surprising downstream code.
        results_text: list[str] = [
            item.text for item in result.content if isinstance(item, TextContent)
        ]
        results_file: list[FileAttachment] = []

        # Also collect resources alongside text; callers may want them.
        for item in result.content:
            if isinstance(item, ImageContent) and self.forward_images:
                results_file.append(
                    FileAttachment.from_bytes(
                        b64decode(item.data), mime_type=item.mimeType
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

        if results_text:
            return "\n".join(results_text), results_file

        # 2) No plain text — use structuredContent if available. To maintain
        # backwards compatibility, unwrap simple shapes like {"result": 5}
        # into "5"; otherwise serialize the full object as JSON for fidelity.
        if result.structuredContent is not None:
            sc = result.structuredContent
            try:
                # Unwrap primitives directly
                if isinstance(sc, (str, int, float, bool)):
                    return str(sc), results_file
                # Unwrap single-key primitive dicts commonly used by tools
                if (
                    isinstance(sc, dict)
                    and len(sc) == 1
                    and next(iter(sc.values())) is not None
                    and isinstance(next(iter(sc.values())), (str, int, float, bool))
                ):
                    return str(next(iter(sc.values()))), results_file

                # Otherwise, serialize to JSON for rich/structured tools
                import json

                return json.dumps(sc, ensure_ascii=False), results_file
            except Exception:
                return str(sc), results_file

        # 3) Nothing usable — return empty text and any files
        return "", results_file

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
        # Prefer validated call; if server fails to provide structured content
        # despite declaring a schema, fall back to a raw request to bypass
        # client-side validation and still surface the data.
        try:
            result: CallToolResult = await self.client.session.call_tool(
                tool_name,
                arguments,
            )
        except RuntimeError as e:
            msg = str(e)
            if "has an output schema but did not return structured content" not in msg:
                raise
            from mcp.types import (
                CallToolRequest,
                CallToolRequestParams,
                ClientRequest,
            )
            from mcp.types import (
                CallToolResult as _CallToolResult,
            )

            result = await self.client.session.send_request(  # type: ignore[assignment]
                ClientRequest(
                    CallToolRequest(
                        params=CallToolRequestParams(
                            name=tool_name, arguments=arguments
                        )
                    )
                ),
                _CallToolResult,
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
