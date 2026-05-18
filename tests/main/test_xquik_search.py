"""
Tests for Xquik search integration.

Unit tests use mocking and do not require a XQUIK_API_KEY.
Integration tests require XQUIK_API_KEY to be set.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from langroid.agent.tools.xquik_search_tool import XquikSearchTool
from langroid.parsing.web_search import xquik_search


def mock_xquik_response() -> MagicMock:
    """Create a mock Xquik API response."""
    response = MagicMock()
    response.json.return_value = {
        "tweets": [
            {
                "id": "1234567890",
                "text": "First Xquik result about agent tools.",
                "url": "https://x.com/xquikcom/status/1234567890",
                "author": {"username": "xquikcom"},
            },
            {
                "id": "9876543210",
                "text": "Second Xquik result about public API search.",
                "url": "https://x.com/xquikcom/status/9876543210",
                "author": {"username": "xquikcom"},
            },
        ],
        "has_next_page": False,
        "next_cursor": "",
    }
    return response


class TestXquikSearchUnit:
    """Unit tests for xquik_search."""

    @patch.dict(os.environ, {"XQUIK_API_KEY": "test-key"})
    @patch("langroid.parsing.web_search.requests.get")
    def test_xquik_search_returns_results(self, mock_get: MagicMock) -> None:
        """Test that xquik_search returns formatted WebSearchResult objects."""
        mock_get.return_value = mock_xquik_response()

        results = xquik_search("from:xquikcom API", num_results=2)

        mock_get.assert_called_once()
        _, kwargs = mock_get.call_args
        assert kwargs["headers"]["x-api-key"] == "test-key"
        assert kwargs["headers"]["xquik-api-contract"] == "2026-04-29"
        assert kwargs["params"] == {
            "q": "from:xquikcom API",
            "queryType": "Latest",
            "limit": 2,
        }
        assert len(results) == 2
        assert results[0].title == "@xquikcom: 1234567890"
        assert results[0].link == "https://x.com/xquikcom/status/1234567890"
        assert results[0].full_content == "First Xquik result about agent tools."
        assert "public API search" in results[1].summary

    def test_xquik_search_missing_api_key(self) -> None:
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="XQUIK_API_KEY"):
                xquik_search("from:xquikcom API")


class TestXquikSearchToolUnit:
    """Unit tests for XquikSearchTool."""

    @patch.dict(os.environ, {"XQUIK_API_KEY": "test-key"})
    @patch("langroid.parsing.web_search.requests.get")
    def test_xquik_search_tool_handle(self, mock_get: MagicMock) -> None:
        """Test that XquikSearchTool.handle() returns formatted results."""
        mock_get.return_value = mock_xquik_response()

        tool = XquikSearchTool(query="from:xquikcom API", num_results=2)
        result = tool.handle()

        assert "BELOW ARE THE RESULTS FROM THE X POST SEARCH" in result
        assert "https://x.com/xquikcom/status/1234567890" in result
        assert "agent tools" in result

    def test_xquik_search_tool_examples(self) -> None:
        """Test that examples are properly defined."""
        examples = XquikSearchTool.examples()
        assert len(examples) == 1
        assert isinstance(examples[0], XquikSearchTool)
        assert examples[0].num_results == 3

    def test_xquik_search_tool_name(self) -> None:
        """Test the tool request name."""
        assert XquikSearchTool.name() == "xquik_search"


@pytest.mark.skipif(
    not os.environ.get("XQUIK_API_KEY"),
    reason="XQUIK_API_KEY not set",
)
class TestXquikSearchIntegration:
    """Integration tests requiring a real XQUIK_API_KEY."""

    def test_xquik_search_real_query(self) -> None:
        """Test a real Xquik search query."""
        results = xquik_search("from:xquikcom API", num_results=3)
        assert len(results) > 0
        assert all(r.link is not None for r in results)
        assert all(len(r.full_content) > 0 for r in results)

    def test_xquik_search_tool_real_query(self) -> None:
        """Test XquikSearchTool with a real query."""
        tool = XquikSearchTool(query="from:xquikcom API", num_results=3)
        result = tool.handle()
        assert "BELOW ARE THE RESULTS FROM THE X POST SEARCH" in result
        assert len(result) > 100
