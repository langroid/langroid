import re

from langroid.parsing.url_loader import FirecrawlConfig, TrafilaturaConfig, URLLoader

urls = [
    "https://pytorch.org",
    "https://www.tensorflow.org",
]


def test_firecrawl_crawler():
    loader = URLLoader(urls=urls, crawler_config=FirecrawlConfig())

    docs = loader.load()

    assert len(docs) == 2
    delimiters = re.compile(r"[:/?=&.]")
    for doc in docs:
        assert len(doc.content) > 0
        assert re.split(delimiters, doc.metadata.source)[-1] in doc.content.lower()


def test_trafilatura_crawler():
    loader = URLLoader(urls=urls, crawler_config=TrafilaturaConfig())

    docs = loader.load()

    assert len(docs) == 2
    delimiters = re.compile(r"[:/?=&.]")
    for doc in docs:
        assert len(doc.content) > 0
        assert re.split(delimiters, doc.metadata.source)[-1] in doc.content.lower()
