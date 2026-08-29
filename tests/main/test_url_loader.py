import http.server
import logging
import os
import threading
import time
from typing import Any, Iterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
