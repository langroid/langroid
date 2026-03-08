"""
Tests for Seltz search integration.

Unit tests use mocking and do not require a SELTZ_API_KEY.
Integration tests require SELTZ_API_KEY to be set.
"""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from langroid.agent.tools.seltz_search_tool import SeltzSearchTool
from langroid.parsing.web_search import seltz_search


@pytest.fixture
def mock_seltz_response():
    """Create a mock Seltz API response."""
    doc1 = SimpleNamespace(
        url="https://example.com/page1",
        content="First result content about LK-99 superconductor material.",
    )
    doc2 = SimpleNamespace(
        url="https://example.com/page2",
        content="Second result content about LK-99 research findings.",
    )
    response = SimpleNamespace(documents=[doc1, doc2])
    return response


class TestSeltzSearchUnit:
    """Unit tests for seltz_search (mocked, no API key needed)."""

    @patch.dict(os.environ, {"SELTZ_API_KEY": "test-key"})
    @patch("langroid.parsing.web_search.Seltz", create=True)
    def test_seltz_search_returns_results(self, mock_seltz_cls, mock_seltz_response):
        """Test that seltz_search returns properly formatted WebSearchResult objects."""
        # Patch the import inside seltz_search
        with patch(
            "langroid.parsing.web_search.seltz_search.__code__",
        ):
            pass

        # We need to mock the import and client
        mock_client = MagicMock()
        mock_client.search.return_value = mock_seltz_response

        with patch.dict("sys.modules", {"seltz": MagicMock()}):
            import sys

            sys.modules["seltz"].Seltz.return_value = mock_client

            results = seltz_search("LK-99 superconductor", num_results=2)

        assert len(results) == 2
        assert results[0].link == "https://example.com/page1"
        assert "LK-99 superconductor" in results[0].full_content
        assert results[1].link == "https://example.com/page2"

    @patch.dict(os.environ, {"SELTZ_API_KEY": "test-key"})
    def test_seltz_search_content_assignment(self, mock_seltz_response):
        """Test that content is assigned directly without HTTP fetch."""
        mock_client = MagicMock()
        mock_client.search.return_value = mock_seltz_response

        with patch.dict("sys.modules", {"seltz": MagicMock()}):
            import sys

            sys.modules["seltz"].Seltz.return_value = mock_client

            results = seltz_search("test query", num_results=2)

        # Content should come from Seltz, not from HTTP fetch
        assert results[0].full_content == (
            "First result content about LK-99 superconductor material."
        )
        assert results[0].summary == (
            "First result content about LK-99 superconductor material."
        )

    def test_seltz_search_missing_api_key(self):
        """Test that missing API key raises ValueError."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("SELTZ_API_KEY", None)
            with pytest.raises(ValueError, match="SELTZ_API_KEY"):
                seltz_search("test query")


class TestSeltzSearchToolUnit:
    """Unit tests for SeltzSearchTool (mocked)."""

    @patch.dict(os.environ, {"SELTZ_API_KEY": "test-key"})
    def test_seltz_search_tool_handle(self, mock_seltz_response):
        """Test that SeltzSearchTool.handle() returns formatted results."""
        mock_client = MagicMock()
        mock_client.search.return_value = mock_seltz_response

        with patch.dict("sys.modules", {"seltz": MagicMock()}):
            import sys

            sys.modules["seltz"].Seltz.return_value = mock_client

            tool = SeltzSearchTool(query="LK-99", num_results=2)
            result = tool.handle()

        assert "BELOW ARE THE RESULTS FROM THE WEB SEARCH" in result
        assert "example.com/page1" in result

    def test_seltz_search_tool_examples(self):
        """Test that examples are properly defined."""
        examples = SeltzSearchTool.examples()
        assert len(examples) == 1
        assert isinstance(examples[0], SeltzSearchTool)
        assert examples[0].num_results == 3

    def test_seltz_search_tool_name(self):
        """Test the tool request name."""
        assert SeltzSearchTool.name() == "seltz_search"


@pytest.mark.skipif(
    not os.environ.get("SELTZ_API_KEY"),
    reason="SELTZ_API_KEY not set",
)
class TestSeltzSearchIntegration:
    """Integration tests requiring a real SELTZ_API_KEY."""

    def test_seltz_search_real_query(self):
        """Test a real Seltz search query."""
        results = seltz_search("Python programming language", num_results=3)
        assert len(results) > 0
        assert all(r.link is not None for r in results)
        assert all(len(r.full_content) > 0 for r in results)

    def test_seltz_search_tool_real_query(self):
        """Test SeltzSearchTool with a real query."""
        tool = SeltzSearchTool(query="Python programming language", num_results=3)
        result = tool.handle()
        assert "BELOW ARE THE RESULTS FROM THE WEB SEARCH" in result
        assert len(result) > 100
