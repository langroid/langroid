"""GoogleSearchTool is deprecated (Google Custom Search JSON API sunset,
2027-01-01). These tests check the deprecation is surfaced without making
any live API call.
"""

import warnings
from unittest.mock import MagicMock, patch

import pytest

from langroid.agent.tools.google_search_tool import GoogleSearchTool
from langroid.parsing import web_search
from langroid.parsing.web_search import GOOGLE_SEARCH_DEPRECATION, google_search


def _fake_service() -> MagicMock:
    """A stand-in for googleapiclient's customsearch service."""
    service = MagicMock()
    service.cse.return_value.list.return_value.execute.return_value = {
        "items": [{"title": "Example", "link": "https://example.com"}]
    }
    return service


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GOOGLE_API_KEY", "dummy")
    monkeypatch.setenv("GOOGLE_CSE_ID", "dummy")


def test_google_search_emits_deprecation_warning(no_network: None) -> None:
    with (
        patch.object(web_search, "build", return_value=_fake_service()),
        patch.object(web_search.WebSearchResult, "get_full_content", return_value=""),
        pytest.warns(DeprecationWarning, match="discontinued on January 1, 2027"),
    ):
        results = google_search("anything", num_results=1)
    assert len(results) == 1  # still works for existing-credential users


def test_google_search_tool_handle_emits_deprecation_warning(
    no_network: None,
) -> None:
    tool = GoogleSearchTool(query="anything", num_results=1)
    with (
        patch.object(web_search, "build", return_value=_fake_service()),
        patch.object(web_search.WebSearchResult, "get_full_content", return_value=""),
        warnings.catch_warnings(record=True) as caught,
    ):
        warnings.simplefilter("always")
        tool.handle()
    assert any(
        issubclass(w.category, DeprecationWarning)
        and str(w.message) == GOOGLE_SEARCH_DEPRECATION
        for w in caught
    )


def test_deprecation_message_names_alternatives() -> None:
    for alt in ("TavilySearchTool", "ExaSearchTool", "DuckduckgoSearchTool"):
        assert alt in GOOGLE_SEARCH_DEPRECATION
