"""Tests for DualEndpointTransport for MCP connections."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastmcp.client.transports import ClientTransport

from langroid.agent.tools.mcp.dual_endpoint import DualEndpointTransport


@pytest.fixture
def dual_endpoint_transport():
    """Create a DualEndpointTransport for testing."""
    return DualEndpointTransport(
        sse_url="http://test-server.com/sse",
        message_url="http://test-server.com/api/message",
        headers={"Content-Type": "application/json"},
    )


@pytest.fixture
def standard_transport():
    """Create a DualEndpointTransport without custom message URL."""
    return DualEndpointTransport(
        sse_url="http://test-server.com/sse",
        headers={"Content-Type": "application/json"},
    )


class TestMCPDualEndpointTransport:
    """Test suite for DualEndpointTransport."""

    def test_init(self, dual_endpoint_transport):
        """Test initialization of DualEndpointTransport."""
        assert dual_endpoint_transport.sse_url == "http://test-server.com/sse"
        assert (
            dual_endpoint_transport.message_url == "http://test-server.com/api/message"
        )
        assert dual_endpoint_transport.headers == {"Content-Type": "application/json"}

    def test_init_standard(self, standard_transport):
        """Test initialization without message URL."""
        assert standard_transport.sse_url == "http://test-server.com/sse"
        assert standard_transport.message_url is None
        assert standard_transport.headers == {"Content-Type": "application/json"}

    def test_str_repr(self, dual_endpoint_transport, standard_transport):
        """Test string representation."""
        assert "sse" in str(dual_endpoint_transport)
        assert "msg" in str(dual_endpoint_transport)
        assert "sse" in str(standard_transport)
        assert "msg" not in str(standard_transport)

    def test_init_validation(self):
        """Test URL validation."""
        with pytest.raises(ValueError):
            DualEndpointTransport(sse_url="not-a-url")


@pytest.mark.asyncio
class TestMCPDualEndpointConnections:
    """Test connection handling in DualEndpointTransport."""

    @patch("langroid.agent.tools.mcp.dual_endpoint.sse_client")
    @patch("langroid.agent.tools.mcp.dual_endpoint.ClientSession")
    async def test_connect_with_message_url(
        self, mock_client_session, mock_sse_client, dual_endpoint_transport
    ):
        """Test connecting with custom message URL."""
        # Mock the read/write streams
        mock_read_stream = MagicMock()
        mock_write_stream = MagicMock()
        mock_sse_client.return_value.__aenter__.return_value = (
            mock_read_stream,
            mock_write_stream,
        )

        # Mock the session
        mock_session = AsyncMock()
        mock_client_session.return_value.__aenter__.return_value = mock_session

        # Test the connection
        async with dual_endpoint_transport.connect_session() as session:
            assert session is mock_session
            mock_session.initialize.assert_called_once()

        # Verify sse_client was called with correct args
        mock_sse_client.assert_called_once_with(
            "http://test-server.com/sse",
            headers={"Content-Type": "application/json"},
        )

    @patch("langroid.agent.tools.mcp.dual_endpoint.sse_client")
    @patch("langroid.agent.tools.mcp.dual_endpoint.ClientSession")
    async def test_connect_without_message_url(
        self, mock_client_session, mock_sse_client, standard_transport
    ):
        """Test connecting without custom message URL."""
        # Mock the read/write streams
        mock_read_stream = MagicMock()
        mock_write_stream = MagicMock()
        mock_sse_client.return_value.__aenter__.return_value = (
            mock_read_stream,
            mock_write_stream,
        )

        # Mock the session
        mock_session = AsyncMock()
        mock_client_session.return_value.__aenter__.return_value = mock_session

        # Test the connection
        async with standard_transport.connect_session() as session:
            assert session is mock_session
            mock_session.initialize.assert_called_once()

        # Verify sse_client was called with correct args
        mock_sse_client.assert_called_once_with(
            "http://test-server.com/sse",
            headers={"Content-Type": "application/json"},
        )


class TestMCPOpenMemoryTransport:
    """
    Test functions for creating an OpenMemory transport.

    These tests don't actually connect to a real OpenMemory server,
    but verify the transport is created correctly with the expected URLs.
    """

    def create_openmemory_transport(
        self, base_url, user_id, client_name="test-client", message_path=None
    ):
        """Helper function to create an OpenMemory transport for testing."""
        base_url = base_url.rstrip("/")
        sse_url = f"{base_url}/mcp/{client_name}/sse/{user_id}"

        # Allow customizing the message endpoint or use the default
        message_url = f"{base_url}/{message_path.lstrip('/')}" if message_path else None

        # Set default headers for OpenMemory
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

        return DualEndpointTransport(
            sse_url=sse_url,
            message_url=message_url,
            headers=headers,
        )

    def test_openmemory_transport_creation(self):
        """Test creating an OpenMemory transport with default settings."""
        transport = self.create_openmemory_transport(
            base_url="http://localhost:8765",
            user_id="test-user",
        )

        assert isinstance(transport, ClientTransport)
        assert (
            transport.sse_url == "http://localhost:8765/mcp/test-client/sse/test-user"
        )
        assert transport.message_url is None
        assert "Content-Type" in transport.headers
        assert "Accept" in transport.headers

    def test_openmemory_transport_with_message_path(self):
        """Test creating an OpenMemory transport with custom message path."""
        transport = self.create_openmemory_transport(
            base_url="http://localhost:8765",
            user_id="test-user",
            message_path="api/messages",
        )

        assert isinstance(transport, ClientTransport)
        assert (
            transport.sse_url == "http://localhost:8765/mcp/test-client/sse/test-user"
        )
        assert transport.message_url == "http://localhost:8765/api/messages"
