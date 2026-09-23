"""Offline tests for the opt-in Spider Cloud crawler."""

import logging
from typing import Any, Iterator
from unittest.mock import Mock

import pytest
import requests
from pydantic import ValidationError

from langroid.mytypes import DocMetaData, Document
from langroid.parsing.url_loader import (
    SpiderConfig,
    SpiderCrawler,
    TrafilaturaCrawler,
    URLLoader,
)

URL = "https://example.com"
KEY = "test-spider-key"


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Isolate settings and fail if a test accidentally reaches the network."""
    for name in ("API_KEY", "MODE", "LIMIT", "TIMEOUT"):
        monkeypatch.delenv(f"SPIDER_{name}", raising=False)
    monkeypatch.setattr(
        requests.sessions.Session,
        "request",
        Mock(side_effect=AssertionError("Unexpected network request")),
    )


@pytest.fixture
def post(monkeypatch: pytest.MonkeyPatch) -> Mock:
    response = Mock(spec=requests.Response)
    response.json.return_value = [page()]
    response.raise_for_status.return_value = None
    mock = Mock(return_value=response)
    monkeypatch.setattr(requests, "post", mock)
    return mock


@pytest.fixture
def messages() -> Iterator[list[str]]:
    """CI disables pytest logging, so use an ordinary logging handler."""
    result: list[str] = []

    class Collector(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            result.append(record.getMessage())

    handler = Collector()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        yield result
    finally:
        root.removeHandler(handler)


def page(**overrides: Any) -> dict[str, Any]:
    return dict(content="# Content", status=200, error=None, url=URL) | overrides


def loader(urls: list[str] | None = None, **options: Any) -> URLLoader:
    return URLLoader(
        urls=[URL] if urls is None else urls,
        crawler_config=SpiderConfig(api_key=KEY, **options),
    )


def test_spider_factory_and_default_loader() -> None:
    assert isinstance(loader().crawler, SpiderCrawler)
    assert isinstance(URLLoader([]).crawler, TrafilaturaCrawler)


@pytest.mark.parametrize("mode", ["scrape", "crawl"])
def test_spider_request_and_document(post: Mock, mode: str) -> None:
    post.return_value.json.return_value = [page(url=f"{URL}/final")]
    docs = loader(mode=mode, limit=3, timeout=12).load()
    payload: dict[str, Any] = {"url": URL, "return_format": "markdown"}
    if mode == "crawl":
        payload["limit"] = 3
    post.assert_called_once_with(
        f"https://api.spider.cloud/{mode}",
        headers={
            "Authorization": f"Bearer {KEY}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=12,
    )
    assert len(docs) == 1
    assert docs[0].content == "# Content"
    assert docs[0].metadata.source == f"{URL}/final"
    assert docs[0].metadata.is_chunk is False


def test_spider_default_crawl_limit(post: Mock) -> None:
    loader(mode="crawl").load()
    assert post.call_args.kwargs["json"]["limit"] == 1
    assert post.call_args.kwargs["timeout"] == 60


def test_spider_multiple_pages_and_seeds(post: Mock) -> None:
    post.return_value.json.side_effect = [
        [page(url=f"{URL}/a"), page(url=f"{URL}/b", content="B")],
        [page(url="https://other.example/c", content="C")],
    ]
    docs = loader([URL, "https://other.example"], mode="crawl", limit=2).load()
    assert [doc.metadata.source for doc in docs] == [
        f"{URL}/a",
        f"{URL}/b",
        "https://other.example/c",
    ]
    assert [call.kwargs["json"]["limit"] for call in post.call_args_list] == [2, 2]


def test_spider_empty_input_needs_no_key(post: Mock) -> None:
    assert URLLoader([], crawler_config=SpiderConfig()).load() == []
    post.assert_not_called()


@pytest.mark.parametrize("result", [[], [page(content="")], [page(content=" \n")]])
def test_spider_empty_results(post: Mock, result: Any) -> None:
    post.return_value.json.return_value = result
    assert loader().load() == []


@pytest.mark.parametrize("mode", ["scrape", "crawl"])
def test_spider_missing_source(post: Mock, mode: str, messages: list[str]) -> None:
    result = page()
    del result["url"]
    post.return_value.json.return_value = [result]
    docs = loader(mode=mode).load()
    if mode == "scrape":
        assert docs[0].metadata.source == URL
    else:
        assert docs == []
        assert any("URL" in message for message in messages)


@pytest.mark.parametrize("result", [None, {}, "invalid", 42])
def test_spider_malformed_response(
    post: Mock, result: Any, messages: list[str]
) -> None:
    post.return_value.json.return_value = result
    assert loader().load() == []
    assert messages


@pytest.mark.parametrize(
    "bad_page",
    [
        None,
        [],
        {},
        page(status=500),
        page(status="200"),
        page(status=None),
        page(error=KEY),
        page(content={"text": "invalid"}),
        page(url=123),
    ],
)
def test_spider_bad_page_preserves_success(
    post: Mock, bad_page: Any, messages: list[str]
) -> None:
    post.return_value.json.return_value = [page(), bad_page, page(content="Last")]
    docs = loader(mode="crawl", limit=3).load()
    assert [doc.content for doc in docs] == ["# Content", "Last"]
    assert messages
    assert KEY not in " ".join(messages)


@pytest.mark.parametrize(
    "failure", [requests.Timeout(KEY), requests.HTTPError(KEY), ValueError(KEY)]
)
def test_spider_failed_seed_preserves_other_results(
    post: Mock, failure: Exception, messages: list[str]
) -> None:
    good_response = post.return_value
    bad_response = Mock(spec=requests.Response)
    bad_response.raise_for_status.return_value = None
    if isinstance(failure, requests.HTTPError):
        bad_response.raise_for_status.side_effect = failure
        post.side_effect = [good_response, bad_response, good_response]
    elif isinstance(failure, ValueError):
        bad_response.json.side_effect = failure
        post.side_effect = [good_response, bad_response, good_response]
    else:
        post.side_effect = [good_response, failure, good_response]
    docs = loader([URL, f"{URL}/bad", f"{URL}/last"]).load()
    assert len(docs) == 2
    assert post.call_count == 3
    assert messages
    assert KEY not in " ".join(messages)


def test_spider_missing_key_fails_before_request(post: Mock) -> None:
    with pytest.raises(ValueError, match="SPIDER_API_KEY"):
        URLLoader([URL], crawler_config=SpiderConfig()).load()
    post.assert_not_called()


def test_spider_settings_precedence_and_isolation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPIDER_API_KEY", "env-key")
    from_env = SpiderConfig()
    explicit = SpiderConfig(api_key=KEY, limit=5)
    monkeypatch.setenv("SPIDER_API_KEY", "new-key")
    assert from_env.api_key == "env-key"
    assert explicit.api_key == KEY
    assert SpiderConfig().api_key == "new-key"
    assert from_env.limit == 1
    assert KEY not in repr(explicit)


def test_spider_numeric_settings_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SPIDER_LIMIT", "3")
    monkeypatch.setenv("SPIDER_TIMEOUT", "12.5")
    config = SpiderConfig()
    assert config.limit == 3
    assert config.timeout == 12.5


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "search"},
        {"limit": 0},
        {"limit": -1},
        {"limit": 1.5},
        {"limit": True},
        {"timeout": 0},
        {"timeout": -1},
        {"timeout": float("inf")},
        {"timeout": float("nan")},
    ],
)
def test_spider_invalid_configuration(options: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        SpiderConfig(**options)


@pytest.mark.parametrize("extension", ["pdf", "docx", "doc"])
@pytest.mark.parametrize("empty", [False, True])
def test_spider_direct_documents_do_not_reach_api(
    monkeypatch: pytest.MonkeyPatch, post: Mock, extension: str, empty: bool
) -> None:
    url = f"{URL}/file.{extension}"
    docs = (
        []
        if empty
        else [
            Document(content="Parsed", metadata=DocMetaData(source=url, is_chunk=True))
        ]
    )
    document_parser = Mock()
    document_parser.get_doc_chunks.return_value = docs
    create = Mock(return_value=document_parser)
    monkeypatch.setattr("langroid.parsing.url_loader.DocumentParser.create", create)
    monkeypatch.setattr(
        "langroid.parsing.url_loader.ImagePdfParser", Mock(return_value=document_parser)
    )
    assert loader([url]).load() == docs
    create.assert_called_once()
    post.assert_not_called()
