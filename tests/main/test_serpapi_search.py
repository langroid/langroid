"""Tests for SerpApi search support."""

import os
from unittest.mock import MagicMock, patch

import pytest
import requests

from langroid.agent.tools.serpapi_search_tool import SerpApiSearchTool
from langroid.parsing.web_search import WebSearchResult, serpapi_search

ORGANIC_RESULTS = [
    {"title": "First result", "link": "https://example.com/first"},
    {"title": "Second result", "link": "https://example.com/second"},
    {"title": "Third result", "link": "https://example.com/third"},
]


def mock_search_response(results=ORGANIC_RESULTS):
    response = MagicMock()
    response.json.return_value = {"organic_results": results}
    return response


@patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"})
@patch("langroid.parsing.web_search.WebSearchResult")
@patch("langroid.parsing.web_search.requests.get")
def test_request_authentication(mock_get, mock_result):
    mock_get.return_value = mock_search_response()

    serpapi_search("test query", num_results=2)

    mock_get.assert_called_once_with(
        "https://serpapi.com/search.json",
        params={
            "engine": "google",
            "q": "test query",
            "num": 2,
            "api_key": "test-key",
        },
        timeout=30,
    )
    mock_get.return_value.raise_for_status.assert_called_once_with()


@patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"})
@patch("langroid.parsing.web_search.WebSearchResult")
@patch("langroid.parsing.web_search.requests.get")
def test_parses_organic_results_and_honors_num_results(mock_get, mock_result):
    mock_get.return_value = mock_search_response()

    results = serpapi_search("test", num_results=2)

    assert len(results) == 2
    assert mock_result.call_args_list == [
        (
            (),
            dict(
                title="First result",
                link="https://example.com/first",
                max_content_length=3500,
                max_summary_length=300,
            ),
        ),
        (
            (),
            dict(
                title="Second result",
                link="https://example.com/second",
                max_content_length=3500,
                max_summary_length=300,
            ),
        ),
    ]


@patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"})
@patch("langroid.parsing.web_search.WebSearchResult")
@patch("langroid.parsing.web_search.requests.get")
def test_skips_results_without_link(mock_get, mock_result):
    """A result missing `link` is skipped, and does not consume a slot."""
    mock_get.return_value = mock_search_response(
        [
            {"title": "No link here"},
            {"title": "Empty link", "link": ""},
            {"title": "First result", "link": "https://example.com/first"},
            {"title": "Second result", "link": "https://example.com/second"},
        ]
    )

    results = serpapi_search("test", num_results=2)

    assert len(results) == 2
    assert [call.kwargs["link"] for call in mock_result.call_args_list] == [
        "https://example.com/first",
        "https://example.com/second",
    ]


@patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"})
@patch("langroid.parsing.web_search.WebSearchResult")
@patch("langroid.parsing.web_search.requests.get")
def test_missing_title_defaults_to_empty_string(mock_get, mock_result):
    """A result missing `title` is kept, with an empty title (no KeyError)."""
    mock_get.return_value = mock_search_response(
        [{"link": "https://example.com/untitled"}]
    )

    results = serpapi_search("test", num_results=3)

    assert len(results) == 1
    assert mock_result.call_args_list[0].kwargs["title"] == ""


@patch("langroid.parsing.web_search.load_dotenv")
@patch("langroid.parsing.web_search.requests.get")
def test_missing_api_key(mock_get, mock_load_dotenv):
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="SERPAPI_API_KEY"):
            serpapi_search("test")
    mock_load_dotenv.assert_called_once_with()
    mock_get.assert_not_called()


@pytest.mark.parametrize("payload", [{}, {"organic_results": []}])
@patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"})
@patch("langroid.parsing.web_search.requests.get")
def test_empty_or_missing_organic_results(mock_get, payload):
    mock_get.return_value.json.return_value = payload
    assert serpapi_search("test") == []


@patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"})
@patch("langroid.parsing.web_search.requests.get")
def test_http_error_propagates_with_redacted_key(mock_get):
    """An HTTP failure still raises HTTPError, but without the API key."""
    leaky_url = "https://serpapi.com/search.json?engine=google&q=test&api_key=test-key"
    # requests attaches the request/response, whose .url carries the key
    failed_request = MagicMock()
    failed_request.url = leaky_url
    failed_response = MagicMock()
    failed_response.url = leaky_url
    failed_response.request = failed_request
    error = requests.HTTPError(
        f"403 Client Error: Forbidden for url: {leaky_url}",
        response=failed_response,
        request=failed_request,
    )
    mock_get.return_value.raise_for_status.side_effect = error

    with pytest.raises(requests.HTTPError) as excinfo:
        serpapi_search("test")

    message = str(excinfo.value)
    assert "test-key" not in message
    assert "***" in message
    assert "403 Client Error" in message
    # the leaking original must not be chained onto the redacted error
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None
    # nor reachable through the objects requests attaches, which hold the
    # same key-bearing URL
    assert excinfo.value.response is None
    assert excinfo.value.request is None
    assert "test-key" not in repr(excinfo.value)
    assert not any("test-key" in str(arg) for arg in excinfo.value.args)


@patch.dict(os.environ, {"SERPAPI_API_KEY": "a+b/c key"})
@patch("langroid.parsing.web_search.requests.get")
def test_url_encoded_key_is_redacted(mock_get):
    """A key that requests percent-encodes into the URL is still redacted."""
    mock_get.side_effect = requests.ConnectionError(
        "Failed to establish a new connection to "
        "https://serpapi.com/search.json?api_key=a%2Bb%2Fc+key"
    )

    with pytest.raises(requests.ConnectionError) as excinfo:
        serpapi_search("test")

    message = str(excinfo.value)
    assert "a%2Bb%2Fc+key" not in message
    assert "a+b/c key" not in message
    assert "***" in message


@patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"})
@patch("langroid.parsing.web_search.requests.get")
def test_error_subclass_without_generic_constructor(mock_get):
    """A subclass that cannot be rebuilt falls back without leaking the key."""
    mock_get.side_effect = requests.JSONDecodeError(
        "boom for url https://serpapi.com/search.json?api_key=test-key",
        "{}",
        0,
    )

    with pytest.raises(requests.RequestException) as excinfo:
        serpapi_search("test")

    assert "test-key" not in str(excinfo.value)
    assert "***" in str(excinfo.value)
    assert excinfo.value.__context__ is None


@patch.dict(os.environ, {"SERPAPI_API_KEY": "test-key"})
@patch("langroid.parsing.web_search.requests.get")
def test_connection_error_redacts_key(mock_get):
    """A transport-level failure is redacted too, keeping its error type."""
    mock_get.side_effect = requests.ConnectionError(
        "Failed to establish a new connection to "
        "https://serpapi.com/search.json?engine=google&api_key=test-key"
    )

    with pytest.raises(requests.ConnectionError) as excinfo:
        serpapi_search("test")

    assert "test-key" not in str(excinfo.value)
    assert "***" in str(excinfo.value)


@patch("langroid.agent.tools.serpapi_search_tool.serpapi_search")
def test_tool_handle(mock_search):
    result = MagicMock(spec=WebSearchResult)
    result.__str__.return_value = (
        "Title: Result\nLink: https://example.com\nSummary: Example summary"
    )
    mock_search.return_value = [result]

    output = SerpApiSearchTool(query="test", num_results=2).handle()

    mock_search.assert_called_once_with("test", 2)
    assert "BELOW ARE THE RESULTS FROM THE WEB SEARCH" in output
    assert "https://example.com" in output


def test_tool_examples():
    examples = SerpApiSearchTool.examples()
    assert len(examples) == 1
    assert isinstance(examples[0], SerpApiSearchTool)
    assert examples[0].num_results == 3


def test_tool_name_and_request():
    assert SerpApiSearchTool.name() == "serpapi_search"
    tool = SerpApiSearchTool(query="test", num_results=1)
    assert tool.request == "serpapi_search"


@pytest.mark.skipif(
    not os.environ.get("SERPAPI_API_KEY"),
    reason="SERPAPI_API_KEY not set",
)
def test_serpapi_real_query():
    results = serpapi_search("Python programming language", num_results=3)
    assert 0 < len(results) <= 3
    assert all(result.link for result in results)
