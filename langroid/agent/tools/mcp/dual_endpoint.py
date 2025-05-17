"""
Dual Endpoint Transport for MCP servers with separate SSE and message posting endpoints.

This module provides a transport implementation that can handle MCP servers that use
different URLs for the SSE event stream and for posting messages back to the server.
"""

import contextlib
import datetime
import logging
from typing import Any, AsyncIterator, Dict, Optional

import anyio
from fastmcp.client.transports import ClientTransport
from mcp import ClientSession
from mcp.client.session import (
    ListRootsFnT,
    LoggingFnT,
    MessageHandlerFnT,
    SamplingFnT,
)
from mcp.client.sse import sse_client
from mcp.shared._httpx_utils import create_mcp_http_client

logger = logging.getLogger(__name__)


class DualEndpointTransport(ClientTransport):
    """
    Transport for MCP servers that use different endpoints for SSE and message posting.

    Some MCP servers use one URL for the SSE event stream connection and a different URL
    for posting messages. This transport handles that scenario by allowing the two
    endpoints to be specified separately.
    """

    def __init__(
        self,
        sse_url: str,
        message_url: Optional[str] = None,
        headers: Optional[Dict[str, str]] = None,
        sse_read_timeout: Optional[datetime.timedelta | float | int] = None,
    ):
        """
        Initialize a transport with separate SSE and message posting endpoints.

        Args:
            sse_url: URL for the SSE event stream connection
            message_url: URL for posting messages. If None, this will be determined
                         from the 'endpoint' event sent by the server.
            headers: Additional HTTP headers to include in requests
            sse_read_timeout: Timeout for the SSE read operations
        """
        if not isinstance(sse_url, str) or not sse_url.startswith("http"):
            raise ValueError("Invalid HTTP/S URL provided for SSE.")

        self.sse_url = sse_url
        self.message_url = message_url
        self.headers = headers or {}

        if isinstance(sse_read_timeout, (int, float)):
            sse_read_timeout = datetime.timedelta(seconds=sse_read_timeout)
        self.sse_read_timeout = sse_read_timeout

    @contextlib.asynccontextmanager
    async def connect_session(
        self,
        sampling_callback: Optional[SamplingFnT] = None,
        list_roots_callback: Optional[ListRootsFnT] = None,
        logging_callback: Optional[LoggingFnT] = None,
        message_handler: Optional[MessageHandlerFnT] = None,
        read_timeout_seconds: Optional[datetime.timedelta] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ClientSession]:
        """Create and yield a client session using custom dual-endpoint connections."""
        client_kwargs = {}

        # Handle timeout settings - add a default timeout if none is specified
        if self.sse_read_timeout is not None:
            client_kwargs["sse_read_timeout"] = self.sse_read_timeout.total_seconds()
        else:
            # Default 30 second timeout
            client_kwargs["sse_read_timeout"] = 30.0

        if read_timeout_seconds is not None:
            client_kwargs["timeout"] = read_timeout_seconds.total_seconds()
        else:
            # Default 30 second timeout
            client_kwargs["timeout"] = 30.0

        logger.info(
            f"Connecting to SSE URL: {self.sse_url} with timeouts: {client_kwargs}"
        )

        # Use the standard sse_client but potentially override the write stream
        async with sse_client(
            self.sse_url, headers=self.headers, **client_kwargs
        ) as transport:
            read_stream, original_write_stream = transport

            # If we have a predefined message URL, we need to create our own write
            # stream
            if self.message_url:
                # Create new memory streams for the write path
                write_stream, write_stream_reader = anyio.create_memory_object_stream[
                    Any
                ](0)

                # Start a task to handle our custom message posting
                async with anyio.create_task_group() as tg:

                    async def custom_post_writer() -> None:
                        try:
                            async with write_stream_reader:
                                async with create_mcp_http_client(
                                    headers=self.headers
                                ) as client:
                                    async for session_message in write_stream_reader:
                                        logger.debug(
                                            f"Sending message to {self.message_url}"
                                        )
                                        try:
                                            if self.message_url:
                                                response = await client.post(
                                                    self.message_url,
                                                    json=session_message.message.model_dump(
                                                        by_alias=True,
                                                        mode="json",
                                                        exclude_none=True,
                                                    ),
                                                )
                                            else:
                                                # This should never happen as we only
                                                # run custom writer when URL exists
                                                # but needed for type checking
                                                raise ValueError("Missing message URL")
                                            response.raise_for_status()
                                            logger.debug(
                                                "Client message sent successfully: "
                                                f"{response.status_code}"
                                            )
                                        except Exception as e:
                                            logger.error(f"Error sending message: {e}")
                                            raise
                        except Exception as exc:
                            logger.error(f"Error in custom_post_writer: {exc}")
                        finally:
                            await write_stream.aclose()

                    # Start our custom writer task
                    tg.start_soon(custom_post_writer)

                    # Use the custom write stream with the ClientSession
                    async with ClientSession(
                        read_stream,
                        write_stream,
                        read_timeout_seconds=read_timeout_seconds,
                        sampling_callback=sampling_callback,
                        list_roots_callback=list_roots_callback,
                        logging_callback=logging_callback,
                        message_handler=message_handler,
                    ) as session:
                        await session.initialize()
                        try:
                            yield session
                        finally:
                            tg.cancel_scope.cancel()
            else:
                # Use the standard streams if no custom message URL
                async with ClientSession(
                    read_stream,
                    original_write_stream,
                    read_timeout_seconds=read_timeout_seconds,
                    sampling_callback=sampling_callback,
                    list_roots_callback=list_roots_callback,
                    logging_callback=logging_callback,
                    message_handler=message_handler,
                ) as session:
                    await session.initialize()
                    yield session

    def __repr__(self) -> str:
        if self.message_url:
            return f"<DualEndpoint(sse='{self.sse_url}', msg='{self.message_url}')>"
        return f"<DualEndpoint(sse='{self.sse_url}')>"
