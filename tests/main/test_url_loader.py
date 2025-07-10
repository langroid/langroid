import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
