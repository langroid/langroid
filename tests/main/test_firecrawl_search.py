"""Tests for Firecrawl search support."""

import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import requests

from langroid.agent.tools.firecrawl_search_tool import FirecrawlSearchTool
from langroid.exceptions import LangroidImportError
from langroid.parsing.web_search import WebSearchResult, firecrawl_search

pytest.importorskip("firecrawl")

from firecrawl.v2.types import (  # noqa: E402
    Document,
    DocumentMetadata,
    ScrapeOptions,
    SearchData,
    SearchResultWeb,
)


def web(url: str, description: str | None = "A snippet", title: str = "Result"):
    return SearchResultWeb(url=url, title=title, description=description)


def scraped(url: str, markdown: str, **metadata) -> Document:
    metadata = dict(url=url, source_url=url, title=f"Title {url}") | metadata
    return Document(markdown=markdown, metadata=DocumentMetadata(**metadata))


def mock_client(items: list | None) -> MagicMock:
    client = MagicMock()
    client.search.return_value = SearchData(web=items)
    return client


@patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"})
@patch("firecrawl.Firecrawl")
def test_request_shape(mock_firecrawl):
    mock_firecrawl.return_value = mock_client([])

    firecrawl_search("test query", num_results=2)
    firecrawl_search("test query", num_results=2, scrape=True)

    mock_firecrawl.assert_called_with(api_key="test-key", origin="langroid", timeout=60)
    search = mock_firecrawl.return_value.search
    assert search.call_args_list[0].args == ("test query",)
    assert search.call_args_list[0].kwargs == dict(limit=2, scrape_options=None)
    assert search.call_args_list[1].kwargs == dict(
        limit=2, scrape_options=ScrapeOptions(formats=["markdown"])
    )


@patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"})
@patch("requests.post")
def test_real_client_payload(mock_post):
    """The real SDK client sends the expected search request."""
    response = MagicMock(spec=requests.Response)
    response.status_code = 200
    response.ok = True
    response.json.return_value = {
        "success": True,
        "data": {"web": [{"url": "https://example.com", "description": "Hi"}]},
    }
    mock_post.return_value = response

    results = firecrawl_search("test query", num_results=3)

    url, payload = mock_post.call_args.args[0], mock_post.call_args.kwargs["json"]
    assert url.endswith("/v2/search")
    assert payload["query"] == "test query"
    assert payload["limit"] == 3
    assert payload["origin"] == "langroid"
    assert "scrapeOptions" not in payload
    assert [r.link for r in results] == ["https://example.com"]


@patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"})
@patch("langroid.parsing.web_search.requests")
@patch("firecrawl.Firecrawl")
def test_maps_descriptions_without_fetching(mock_firecrawl, mock_requests):
    """The search description fills the result; the link is not fetched."""
    long_description = "## Highlights\n" + "x" * 5000
    mock_firecrawl.return_value = mock_client(
        [
            web("https://example.com/a", long_description, title=" Page A\n"),
            web("https://example.com/b", "Short snippet", title=""),
        ]
    )

    results = firecrawl_search("test", num_results=2)

    assert [r.link for r in results] == [
        "https://example.com/a",
        "https://example.com/b",
    ]
    assert results[0].title == "Page A"
    assert results[0].full_content == long_description[:3500]
    assert results[0].summary == " ".join(long_description.split())[:300]
    # no title: falls back to the link
    assert results[1].title == "https://example.com/b"
    assert results[1].summary == "Short snippet"
    mock_requests.head.assert_not_called()
    mock_requests.get.assert_not_called()


@patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"})
@patch("langroid.parsing.web_search.requests")
@patch("firecrawl.Firecrawl")
def test_scrape_maps_markdown(mock_firecrawl, mock_requests):
    long_markdown = "# Page\n" + "x" * 5000
    mock_firecrawl.return_value = mock_client(
        [
            scraped("https://example.com/a", long_markdown, description="About A"),
            scraped("https://example.com/b", "# B\nbody"),
        ]
    )

    results = firecrawl_search("test", num_results=2, scrape=True)

    assert results[0].full_content == long_markdown[:3500]
    assert results[0].summary == "About A"
    # no description: the summary comes from the markdown, on one line
    assert results[1].summary == "# B body"
    mock_requests.head.assert_not_called()
    mock_requests.get.assert_not_called()


@patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"})
@patch("firecrawl.Firecrawl")
def test_each_result_prints_as_one_block(mock_firecrawl):
    """Multi-line titles and snippets must not add blank lines to the output.

    The tool joins results with blank lines, so a blank line inside a
    result would read as an extra result.
    """
    mock_firecrawl.return_value = mock_client(
        [
            web("https://example.com/a", "## One\n\nTwo\n\n", title="\n  A\n\n"),
            web("https://example.com/b", "Line one\n\nLine two"),
        ]
    )

    output = FirecrawlSearchTool(query="test", num_results=2).handle()

    blocks = output.split("BELOW ARE THE RESULTS")[1].strip().split("\n\n")
    assert len(blocks) == 2
    assert "Title: A\n" in blocks[0]


@patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"})
@patch("firecrawl.Firecrawl")
def test_prefers_search_link_over_final_url(mock_firecrawl):
    mock_firecrawl.return_value = mock_client(
        [
            scraped(
                "https://example.com/final",
                "# Page",
                source_url="https://example.com/searched",
            )
        ]
    )

    results = firecrawl_search("test", scrape=True)

    assert results[0].link == "https://example.com/searched"


@patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"})
@patch.object(WebSearchResult, "get_full_content", return_value="fetched page")
@patch("firecrawl.Firecrawl")
def test_failed_or_empty_content_falls_back_to_fetch(mock_firecrawl, mock_fetch):
    mock_firecrawl.return_value = mock_client(
        [
            scraped(
                "https://example.com/blocked",
                "Error 401",
                status_code=401,
                description="Blocked page",
            ),
            scraped("https://example.com/blank", "   "),
            web("https://example.com/plain", description=None),
        ]
    )

    results = firecrawl_search("test", num_results=3, scrape=True)

    assert [r.full_content for r in results] == ["fetched page"] * 3
    assert results[0].summary == "Blocked page"
    assert mock_fetch.call_count == 3


@patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"})
@patch("firecrawl.Firecrawl")
def test_skips_results_without_link(mock_firecrawl):
    """A result missing a link is skipped and does not consume a slot."""
    mock_firecrawl.return_value = mock_client(
        [
            Document(markdown="# no metadata"),
            scraped("", "# empty url", source_url=""),
            web("https://example.com/first"),
        ]
    )

    results = firecrawl_search("test", num_results=3, scrape=True)

    assert [r.link for r in results] == ["https://example.com/first"]


@patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"})
@patch("firecrawl.Firecrawl")
def test_num_results_bounds(mock_firecrawl):
    mock_firecrawl.return_value = mock_client(
        [web(f"https://example.com/{i}") for i in range(4)]
    )

    assert firecrawl_search("test", num_results=0) == []
    mock_firecrawl.assert_not_called()

    assert len(firecrawl_search("test", num_results=2)) == 2
    firecrawl_search("test", num_results=500)
    assert mock_firecrawl.return_value.search.call_args.kwargs["limit"] == 100


@pytest.mark.parametrize("items", [None, []])
@patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"})
@patch("firecrawl.Firecrawl")
def test_no_web_results(mock_firecrawl, items):
    mock_firecrawl.return_value = mock_client(items)
    assert firecrawl_search("test") == []


@patch("langroid.parsing.web_search.load_dotenv")
@patch("firecrawl.Firecrawl")
def test_missing_api_key(mock_firecrawl, mock_load_dotenv):
    with patch.dict(os.environ, {}, clear=True):
        with pytest.raises(ValueError, match="FIRECRAWL_API_KEY"):
            firecrawl_search("test")
    mock_load_dotenv.assert_called_once_with()
    mock_firecrawl.assert_not_called()


@patch.dict(os.environ, {"FIRECRAWL_API_KEY": "test-key"})
def test_missing_dependency():
    with patch.dict(sys.modules, {"firecrawl": None}):
        with pytest.raises(LangroidImportError):
            firecrawl_search("test")


@patch("langroid.agent.tools.firecrawl_search_tool.firecrawl_search")
def test_tool_handle(mock_search):
    result = MagicMock(spec=WebSearchResult)
    result.__str__.return_value = (
        "Title: Result\nLink: https://example.com\nSummary: Example summary"
    )
    mock_search.return_value = [result]

    output = FirecrawlSearchTool(query="test", num_results=2).handle()

    mock_search.assert_called_once_with("test", 2)
    assert "BELOW ARE THE RESULTS FROM THE WEB SEARCH" in output
    assert "https://example.com" in output


def test_tool_examples():
    examples = FirecrawlSearchTool.examples()
    assert len(examples) == 1
    assert isinstance(examples[0], FirecrawlSearchTool)
    assert examples[0].num_results == 3


def test_tool_name_and_request():
    assert FirecrawlSearchTool.name() == "firecrawl_search"
    tool = FirecrawlSearchTool(query="test", num_results=1)
    assert tool.request == "firecrawl_search"


@pytest.mark.skipif(
    not os.environ.get("FIRECRAWL_API_KEY"),
    reason="FIRECRAWL_API_KEY not set",
)
def test_firecrawl_real_query():
    results = firecrawl_search("Python programming language", num_results=3)
    assert 0 < len(results) <= 3
    assert all(result.link for result in results)
    assert all(result.summary for result in results)
