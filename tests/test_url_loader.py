from llmagent.parsing.url_loader import URLLoader


def test_url_loader():
    loader = URLLoader(urls=[
        "https://pytorch.org",
        "https://openai.com",
    ])

    docs = loader.load()

    assert len(docs) == 2

    assert "pytorch" in docs[0].content.lower()

    assert "openai" in docs[1].content.lower()