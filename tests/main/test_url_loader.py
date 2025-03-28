import pytest

from langroid.parsing.url_loader import (
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
