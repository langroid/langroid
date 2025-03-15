import re

from langroid.parsing.url_loader import FirecrawlCrawler, URLLoader


def test_url_loader():
    loader = URLLoader(
        urls=[
            "https://pytorch.org",
            "https://www.tensorflow.org",
        ],
        crawler=FirecrawlCrawler(),
    )

    docs = loader.load()

    assert len(docs) == 2
    delimiters = re.compile(r"[:/?=&.]")
    for doc in docs:
        assert len(doc.content) > 0
        assert re.split(delimiters, doc.metadata.source)[-1] in doc.content.lower()
