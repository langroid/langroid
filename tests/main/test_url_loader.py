import http.server
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import requests

from langroid.parsing.parser import ParsingConfig
from langroid.parsing.url_loader import (
    Crawl4aiConfig,
    ExaCrawlerConfig,
    FirecrawlConfig,
    TrafilaturaConfig,
    URLLoader,
)

urls = [
    "https://pytorch.org",
    "https://arxiv.org/pdf/1706.03762",
]


@pytest.mark.xfail(
    condition=lambda crawler_config=None: isinstance(crawler_config, FirecrawlConfig),
    reason="Firecrawl may fail due to timeouts",
    run=True,
    strict=False,
)
@pytest.mark.parametrize(
    "crawler_config",
    [
        TrafilaturaConfig(),
        ExaCrawlerConfig(),
        FirecrawlConfig(timeout=60000),
    ],
)
def test_crawler(crawler_config):
    loader = URLLoader(urls=urls, crawler_config=crawler_config)

    docs = loader.load()

    # there are likely some chunked docs among these,
    # so we expect at least as many docs as urls
    assert len(docs) >= len(urls)
    for doc in docs:
        assert len(doc.content) > 0


@patch("crawl4ai.AsyncWebCrawler")
def test_crawl4ai_mocked(mock_crawler_class):
    """Test Crawl4aiCrawler with mocked dependencies."""
    # Create mock crawler instance
    mock_crawler = AsyncMock()
    mock_crawler_class.return_value.__aenter__.return_value = mock_crawler

    # Create mock result
    mock_result = MagicMock()
    mock_result.success = True
    mock_result.url = "https://example.com"
    mock_result.extracted_content = None
    mock_result.markdown = MagicMock()
    mock_result.markdown.fit_markdown = "# Test Content\nThis is test content."
    mock_result.metadata = {"title": "Test Page", "published_date": "2024-01-01"}

    # Set up async return value
    mock_crawler.arun.return_value = mock_result

    # Test with simple crawl mode
    config = Crawl4aiConfig(crawl_mode="simple")
    loader = URLLoader(urls=["https://example.com"], crawler_config=config)

    docs = loader.load()

    assert len(docs) == 1
    assert docs[0].content == "# Test Content\nThis is test content."
    assert docs[0].metadata.title == "Test Page"
    assert docs[0].metadata.source == "https://example.com"


@pytest.mark.skipif(
    os.getenv("CI") == "true",  # Skip on CI to avoid install of playwright
    reason="Crawl4ai integration test skipped by default. Set TEST_CRAWL4AI=1 to run.",
)
def test_crawl4ai_integration():
    """Integration test for real Crawl4ai functionality.
    
    Run with: TEST_CRAWL4AI=1 pytest \
        tests/main/test_url_loader.py::test_crawl4ai_integration
    """
    # Use a simple, fast-loading page
    test_urls = ["https://example.com"]

    config = Crawl4aiConfig(crawl_mode="simple")
    loader = URLLoader(urls=test_urls, crawler_config=config)

    docs = loader.load()

    assert len(docs) >= 1
    assert len(docs[0].content) > 0
    assert "Example Domain" in docs[0].content or "example" in docs[0].content.lower()


# ---------------------------------------------------------------------------
# Bounded fetching: URLLoader must apply the ParsingConfig URL limits when it
# downloads a document whose type is only known from its Content-Type header.
# ---------------------------------------------------------------------------

_STALL = 0.5  # server-side stall, longer than the timeout under test
_OVERSIZED_CHUNK = 64 * 1024
_OVERSIZED_BODY = 8 * 1024 * 1024


class _CrawlHandler(http.server.BaseHTTPRequestHandler):
    """Serves the transport edge cases the bounded-fetch tests need.

    Every path is extensionless, so `_is_document_url` is False and the
    crawler takes the HEAD-then-GET branch under test.
    """

    served_bytes = 0

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _send_pdf_headers(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "application/pdf")
        self.end_headers()

    def do_HEAD(self) -> None:
        if self.path == "/stalled-head":
            time.sleep(_STALL)  # slow loris: the headers never arrive
            return
        self._send_pdf_headers()

    def do_GET(self) -> None:
        if self.path == "/stalled-body":
            self._send_pdf_headers()
            time.sleep(_STALL)  # stalls after the headers
            return
        # /oversized: a body far larger than url_max_size, with no
        # Content-Length, so only streaming can bound it.
        self._send_pdf_headers()
        chunk = b"%PDF-1.4" + b"0" * (_OVERSIZED_CHUNK - 8)
        for _ in range(_OVERSIZED_BODY // _OVERSIZED_CHUNK):
            try:
                self.wfile.write(chunk)
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return
            type(self).served_bytes += len(chunk)


@pytest.fixture
def crawl_server_url() -> Iterator[str]:
    """Run the handler above on a local HTTP port; yield the base URL."""
    _CrawlHandler.served_bytes = 0
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _CrawlHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        thread.join()


@pytest.fixture
def root_log_messages() -> Iterator[list[str]]:
    """Collect root-logger messages via a plain handler.

    CI runs tests/main with `-p no:logging`, which disables pytest's
    `caplog` fixture, so capture with a temporary `logging.Handler` (as in
    `tests/main/test_vecstore_env_prefix.py`).
    """
    messages: list[str] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            messages.append(record.getMessage())

    root = logging.getLogger()
    handler = Collector(level=logging.WARNING)
    root.addHandler(handler)
    try:
        yield messages
    finally:
        root.removeHandler(handler)


@pytest.mark.parametrize("path", ["stalled-head", "stalled-body"])
def test_url_loader_document_fetch_honors_timeouts(
    crawl_server_url: str, path: str, root_log_messages: list[str]
) -> None:
    """A stalled server must not hang the crawler.

    Regression test: `_process_document` called `requests.head` and
    `requests.get` with no timeout, so either half of the download blocked
    forever, even though `ParsingConfig` already carries the limits and
    `parsing/document_url.py` already applies them.
    """
    loader = URLLoader(
        urls=[],
        parsing_config=ParsingConfig(url_connect_timeout=0.01, url_read_timeout=0.01),
    )
    start = time.monotonic()

    assert loader.crawler._process_document(f"{crawl_server_url}/{path}") == []

    assert time.monotonic() - start < _STALL / 2
    # The timeout must be reported with a pointer to the config knobs.
    assert any("url_read_timeout" in msg for msg in root_log_messages)


def test_url_loader_document_fetch_honors_max_size(
    crawl_server_url: str, root_log_messages: list[str]
) -> None:
    """An unbounded response body must not be buffered whole into memory.

    Regression test: `_process_document` read `requests.get(url).content`,
    ignoring the `url_max_size` its own `ParsingConfig` defines.
    """
    loader = URLLoader(urls=[], parsing_config=ParsingConfig(url_max_size=16))

    assert loader.crawler._process_document(f"{crawl_server_url}/oversized") == []

    # Streaming aborts within a chunk or two; only an unbounded read drains
    # the whole body.
    assert _CrawlHandler.served_bytes < _OVERSIZED_BODY // 2
    # The size rejection must tell the user which config field to raise.
    assert any("url_max_size" in msg for msg in root_log_messages)


def _fc_response(status_code: int, body: dict[str, Any]) -> Mock:
    response = Mock(spec=requests.Response)
    response.status_code = status_code
    response.ok = status_code < 400
    response.json.return_value = body
    response.text = str(body)
    response.headers = {}
    return response


def _fc_page(url: str, markdown: str, status_code: int = 200) -> dict[str, Any]:
    metadata = dict(url=url, sourceURL=url, title=f"Title {url}")
    return dict(markdown=markdown, metadata=metadata | {"statusCode": status_code})


def _fc_scraped(url: str, markdown: str, status_code: int = 200) -> Mock:
    return _fc_response(
        200, {"success": True, "data": _fc_page(url, markdown, status_code)}
    )


@pytest.fixture
def firecrawl_offline(monkeypatch: pytest.MonkeyPatch) -> dict[str, Mock]:
    """Run the real Firecrawl SDK client against mocked HTTP calls.

    The SDK sends requests via `requests.post` / `requests.get`; anything
    else that reaches the network fails the test.
    """
    pytest.importorskip("firecrawl")
    for name in ("API_KEY", "API_URL", "MODE", "PARAMS", "TIMEOUT"):
        monkeypatch.delenv(f"FIRECRAWL_{name}", raising=False)
    monkeypatch.setattr(
        requests.sessions.Session,
        "request",
        Mock(side_effect=AssertionError("Unexpected network request")),
    )
    mocks = {"post": Mock(), "get": Mock()}
    monkeypatch.setattr(requests, "post", mocks["post"])
    monkeypatch.setattr(requests, "get", mocks["get"])
    monkeypatch.setattr(time, "sleep", Mock())
    return mocks


def test_firecrawl_scrape(firecrawl_offline: dict[str, Mock]) -> None:
    """Scrape mode sends a v2 scrape request and maps the response.

    Regression test: the crawler passed the v1 `params=` keyword, which
    firecrawl-py v4 rejects with a TypeError that was logged and swallowed,
    so every URL returned no documents.
    """
    post = firecrawl_offline["post"]
    post.side_effect = [
        _fc_scraped("https://a.com", "# A"),
        _fc_scraped("https://b.com", "not found", status_code=404),
    ]
    config = FirecrawlConfig(
        api_key="fc-test", params={"onlyMainContent": False}, timeout=60000
    )

    docs = URLLoader(
        urls=["https://a.com", "https://b.com"], crawler_config=config
    ).load()

    payload = post.call_args_list[0].kwargs["json"]
    assert post.call_args_list[0].args[0].endswith("/v2/scrape")
    assert payload["url"] == "https://a.com"
    assert payload["formats"] == ["markdown"]
    assert payload["onlyMainContent"] is False
    assert payload["timeout"] == 60000
    assert payload["origin"] == "langroid"
    # the 404 page is dropped
    assert len(docs) == 1
    assert docs[0].content == "# A"
    assert docs[0].metadata.source == "https://a.com"
    assert docs[0].metadata.title == "Title https://a.com"


def test_firecrawl_scrape_skips_failed_url(
    firecrawl_offline: dict[str, Mock],
) -> None:
    """An API error on one URL is logged and skipped."""
    firecrawl_offline["post"].side_effect = [
        _fc_response(400, {"success": False, "error": "bad url"}),
        _fc_scraped("https://b.com", "# B"),
    ]
    docs = URLLoader(
        urls=["https://a.com", "https://b.com"],
        crawler_config=FirecrawlConfig(api_key="fc-test"),
    ).load()
    assert [d.content for d in docs] == ["# B"]


def test_firecrawl_scrape_raises_on_bad_key(
    firecrawl_offline: dict[str, Mock],
) -> None:
    """A rejected key fails loudly instead of returning no documents."""
    from firecrawl.v2.utils.error_handler import UnauthorizedError

    firecrawl_offline["post"].return_value = _fc_response(
        401, {"success": False, "error": "Unauthorized: Invalid token"}
    )
    with pytest.raises(UnauthorizedError):
        URLLoader(
            urls=["https://a.com", "https://b.com"],
            crawler_config=FirecrawlConfig(api_key="fc-bad"),
        ).load()
    assert firecrawl_offline["post"].call_count == 1


def test_firecrawl_scrape_rejects_unknown_option(
    firecrawl_offline: dict[str, Mock],
) -> None:
    """An option the SDK does not accept is reported before any request."""
    with pytest.raises(ValueError, match="bogus_option"):
        URLLoader(
            urls=["https://a.com"],
            crawler_config=FirecrawlConfig(
                api_key="fc-test", params={"bogusOption": 1}
            ),
        ).load()
    firecrawl_offline["post"].assert_not_called()


def test_firecrawl_crawl(
    firecrawl_offline: dict[str, Mock],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Crawl mode sends scrape options and polls until the crawl completes.

    `timeout` and other page options must reach `scrapeOptions`: the SDK
    ignores flat scrape options once `scrape_options` is set.
    """
    monkeypatch.chdir(tmp_path)
    home = _fc_page("https://site.com", "# Home")
    about = _fc_page("https://site.com/about", "# About")
    firecrawl_offline["post"].return_value = _fc_response(
        200, {"success": True, "id": "job-1", "url": "https://x"}
    )
    firecrawl_offline["get"].side_effect = [
        _fc_response(200, {"success": True, "status": "scraping", "data": [home]}),
        _fc_response(
            200, {"success": True, "status": "completed", "data": [home, about]}
        ),
    ]
    config = FirecrawlConfig(
        api_key="fc-test",
        mode="crawl",
        params={"limit": 5, "onlyMainContent": False},
        timeout=30000,
    )

    docs = URLLoader(urls=["https://site.com"], crawler_config=config).load()

    post = firecrawl_offline["post"]
    payload = post.call_args.kwargs["json"]
    assert post.call_args.args[0].endswith("/v2/crawl")
    assert payload["url"] == "https://site.com"
    assert payload["limit"] == 5
    assert payload["origin"] == "langroid"
    assert payload["scrapeOptions"]["formats"] == ["markdown"]
    assert payload["scrapeOptions"]["timeout"] == 30000
    assert payload["scrapeOptions"]["onlyMainContent"] is False
    assert firecrawl_offline["get"].call_args.args[0].endswith("/v2/crawl/job-1")
    assert [d.content for d in docs] == ["# Home", "# About"]
    assert [d.metadata.source for d in docs] == [
        "https://site.com",
        "https://site.com/about",
    ]
    assert (tmp_path / "firecrawl_output" / "full_results.json").exists()


def test_firecrawl_crawl_failed(
    firecrawl_offline: dict[str, Mock],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed crawl stops polling and returns the pages it saved."""
    monkeypatch.chdir(tmp_path)
    firecrawl_offline["post"].return_value = _fc_response(
        200, {"success": True, "id": "job-2", "url": "https://x"}
    )
    firecrawl_offline["get"].return_value = _fc_response(
        200,
        {
            "success": True,
            "status": "failed",
            "data": [_fc_page("https://site.com", "# Home")],
        },
    )

    docs = URLLoader(
        urls=["https://site.com"],
        crawler_config=FirecrawlConfig(api_key="fc-test", mode="crawl"),
    ).load()

    assert [d.content for d in docs] == ["# Home"]
    assert firecrawl_offline["get"].call_count == 1


def test_firecrawl_crawl_scrape_options_object(
    firecrawl_offline: dict[str, Mock],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A `ScrapeOptions` object in params keeps markdown and the timeout."""
    from firecrawl.v2.types import ScrapeOptions

    monkeypatch.chdir(tmp_path)
    firecrawl_offline["post"].return_value = _fc_response(
        200, {"success": True, "id": "job-3", "url": "https://x"}
    )
    firecrawl_offline["get"].return_value = _fc_response(
        200, {"success": True, "status": "completed", "data": []}
    )
    config = FirecrawlConfig(
        api_key="fc-test",
        mode="crawl",
        params={"scrape_options": ScrapeOptions(formats=["html"])},
        timeout=30000,
    )

    URLLoader(urls=["https://site.com"], crawler_config=config).load()

    scrape_options = firecrawl_offline["post"].call_args.kwargs["json"]["scrapeOptions"]
    assert scrape_options["formats"] == ["markdown", "html"]
    assert scrape_options["timeout"] == 30000


def test_firecrawl_crawl_rejects_unknown_option(
    firecrawl_offline: dict[str, Mock],
) -> None:
    """A v1 crawl option renamed in v2 is reported before any request."""
    with pytest.raises(ValueError, match="max_depth"):
        URLLoader(
            urls=["https://site.com"],
            crawler_config=FirecrawlConfig(
                api_key="fc-test", mode="crawl", params={"maxDepth": 2}
            ),
        ).load()
    firecrawl_offline["post"].assert_not_called()


def test_firecrawl_self_hosted_api_url(firecrawl_offline: dict[str, Mock]) -> None:
    """`api_url` points the client at a self-hosted Firecrawl instance."""
    firecrawl_offline["post"].return_value = _fc_scraped("https://a.com", "# A")
    config = FirecrawlConfig(api_key="fc-test", api_url="http://localhost:3002")

    docs = URLLoader(urls=["https://a.com"], crawler_config=config).load()

    url = firecrawl_offline["post"].call_args.args[0]
    assert url == "http://localhost:3002/v2/scrape"
    assert [d.content for d in docs] == ["# A"]


def test_firecrawl_nested_keys_are_left_alone(
    firecrawl_offline: dict[str, Mock],
) -> None:
    """camelCase conversion stops at the top level, and must keep doing so.

    A nested dict is as likely to hold user data as option names, so the
    keys of a JSON extraction schema and of a `headers` map have to reach
    Firecrawl exactly as written. Converting them -- an obvious-looking
    extension of the v1 back-compat shim -- would silently rewrite a
    user's schema properties.
    """
    firecrawl_offline["post"].return_value = _fc_scraped("https://a.com", "# A")
    schema = {
        "type": "object",
        "properties": {"firstName": {"type": "string"}},
        "additionalProperties": False,
    }
    config = FirecrawlConfig(
        api_key="fc-test",
        params={
            "headers": {"X-My-Header": "keepMe"},
            "formats": [{"type": "json", "schema": schema}],
        },
    )

    URLLoader(urls=["https://a.com"], crawler_config=config).load()

    payload = firecrawl_offline["post"].call_args.kwargs["json"]
    assert payload["headers"] == {"X-My-Header": "keepMe"}
    sent = next(
        f for f in payload["formats"] if isinstance(f, dict) and f.get("type") == "json"
    )
    assert sent["schema"] == schema


def test_firecrawl_scrape_skips_empty_markdown(
    firecrawl_offline: dict[str, Mock],
) -> None:
    """A 200 with no markdown is skipped, not turned into a blank doc."""
    firecrawl_offline["post"].side_effect = [
        _fc_response(
            200,
            {
                "success": True,
                "data": {"metadata": {"url": "https://a.com", "statusCode": 200}},
            },
        ),
        _fc_scraped("https://b.com", "# B"),
    ]

    docs = URLLoader(
        urls=["https://a.com", "https://b.com"],
        crawler_config=FirecrawlConfig(api_key="fc-test"),
    ).load()

    assert [d.content for d in docs] == ["# B"]


def test_firecrawl_crawl_rejects_unknown_scrape_option(
    firecrawl_offline: dict[str, Mock],
) -> None:
    """A typo inside `scrape_options` is reported, not silently dropped.

    `ScrapeOptions` does not forbid extra fields, so an unknown key would
    otherwise be dropped on construction.
    """
    with pytest.raises(ValueError, match="bogus_option"):
        URLLoader(
            urls=["https://site.com"],
            crawler_config=FirecrawlConfig(
                api_key="fc-test",
                mode="crawl",
                params={"scrape_options": {"bogusOption": 1}},
            ),
        ).load()
    firecrawl_offline["post"].assert_not_called()


def test_firecrawl_with_markdown_keeps_a_lone_format_object() -> None:
    """A single format *object* is wrapped, not iterated into key/value pairs.

    Pydantic models iterate as `(field, value)` tuples, so a bare format
    object used to be shredded into nonsense entries.
    """
    pytest.importorskip("firecrawl")
    from firecrawl.v2.types import JsonFormat

    from langroid.parsing.url_loader import _with_markdown

    fmt = JsonFormat(prompt="extract")
    assert _with_markdown(fmt) == ["markdown", fmt]


def test_firecrawl_scrape_formats_container(
    firecrawl_offline: dict[str, Mock],
) -> None:
    """The SDK's `ScrapeFormats` container gets markdown flipped on.

    It holds one boolean per format rather than a `type`, so appending to
    it as if it were a list of formats produced a payload the SDK
    rejected -- and scrape mode turned that into zero documents.
    """
    from firecrawl.v2.types import ScrapeFormats

    firecrawl_offline["post"].return_value = _fc_scraped("https://a.com", "# A")
    config = FirecrawlConfig(
        api_key="fc-test", params={"formats": ScrapeFormats(html=True)}
    )

    docs = URLLoader(urls=["https://a.com"], crawler_config=config).load()

    payload = firecrawl_offline["post"].call_args.kwargs["json"]
    assert sorted(payload["formats"]) == ["html", "markdown"]
    assert [d.content for d in docs] == ["# A"]
