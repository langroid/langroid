from unittest.mock import patch

import pytest
from requests import Response

from langroid.parsing.urls import find_urls

START_URL = "https://example.test/start"
LOCAL_URL = "https://example.test/local"
REMOTE_URL = "https://other.test/remote"


def page_response(url: str, timeout: int) -> Response:
    assert timeout == 5
    pages = {
        START_URL: (
            '<a href="/local#section">Local</a>'
            '<a href="https://other.test/remote#section">Remote</a>'
        ),
        LOCAL_URL: '<a href="/start">Back</a>',
        REMOTE_URL: '<a href="https://example.test/start">Back</a>',
    }
    response = Response()
    response.status_code = 200
    response.url = url
    response._content = pages[url].encode("utf-8")
    response.encoding = "utf-8"
    return response


@pytest.mark.parametrize("max_links", [3, 4])
@pytest.mark.parametrize("match_domain", [True, False])
def test_find_urls_domain_option(match_domain: bool, max_links: int) -> None:
    with patch("langroid.parsing.urls.requests.get", side_effect=page_response):
        found = find_urls(START_URL, max_links=max_links, match_domain=match_domain)

    expected = {START_URL, LOCAL_URL}
    if not match_domain:
        expected.add(REMOTE_URL)
    assert found == expected


def test_find_urls_stays_on_domain_by_default() -> None:
    with patch("langroid.parsing.urls.requests.get", side_effect=page_response):
        found = find_urls(START_URL, max_links=4)

    assert found == {START_URL, LOCAL_URL}


def test_find_urls_cross_domain_crawl_still_obeys_depth() -> None:
    with patch("langroid.parsing.urls.requests.get", side_effect=page_response) as get:
        found = find_urls(START_URL, max_links=4, max_depth=0, match_domain=False)

    assert found == {START_URL}
    get.assert_called_once_with(START_URL, timeout=5)
