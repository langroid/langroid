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
    with patch("langroid.parsing.urls.requests.get", side_effect=page_response) as get:
        found = find_urls(START_URL, max_links=max_links, match_domain=match_domain)

    expected = {START_URL, LOCAL_URL}
    if not match_domain:
        expected.add(REMOTE_URL)
    assert found == expected

    # The fetch set, not just the result set, must respect the domain option:
    # `find_urls` swallows every request exception, so a stray fetch would
    # otherwise be invisible here.
    fetched = {call.args[0] for call in get.call_args_list}
    assert fetched <= expected
    if match_domain:
        assert REMOTE_URL not in fetched


def test_find_urls_stays_on_domain_by_default() -> None:
    with patch("langroid.parsing.urls.requests.get", side_effect=page_response):
        found = find_urls(START_URL, max_links=4)

    assert found == {START_URL, LOCAL_URL}


def test_find_urls_cross_domain_crawl_still_obeys_depth() -> None:
    with patch("langroid.parsing.urls.requests.get", side_effect=page_response) as get:
        found = find_urls(START_URL, max_links=4, max_depth=0, match_domain=False)

    assert found == {START_URL}
    get.assert_called_once_with(START_URL, timeout=5)


SCHEMES_URL = "https://example.test/schemes"
SCHEMES_HTML = (
    '<a href="mailto:someone@example.com">Mail</a>'
    '<a href="javascript:void(0)">JS</a>'
    '<a href="tel:+15551234">Tel</a>'
    '<a href="data:text/html,hello">Data</a>'
    '<a href="file:///etc/passwd">File</a>'
    '<a href="https://other.test/remote">Remote</a>'
)


def schemes_response(url: str, timeout: int) -> Response:
    assert timeout == 5
    pages = {
        SCHEMES_URL: SCHEMES_HTML,
        REMOTE_URL: "<a href='https://example.test/schemes'>Back</a>",
    }
    response = Response()
    response.status_code = 200
    response.url = url
    response._content = pages[url].encode("utf-8")
    response.encoding = "utf-8"
    return response


@pytest.mark.parametrize("match_domain", [True, False])
def test_find_urls_skips_non_web_schemes(match_domain: bool) -> None:
    """Only http/https links are crawled or returned, in BOTH domain modes.

    When `match_domain` is True the domain comparison rejects these schemes as a
    side effect, since they all have an empty netloc. With `match_domain` False
    that comparison is skipped, so the scheme filter is the only thing keeping
    `mailto:`/`javascript:`/`file:` links out of the results and out of the
    `max_links` budget.
    """
    with patch(
        "langroid.parsing.urls.requests.get", side_effect=schemes_response
    ) as get:
        found = find_urls(SCHEMES_URL, max_links=6, match_domain=match_domain)

    expected = {SCHEMES_URL}
    if not match_domain:
        expected.add(REMOTE_URL)
    assert found == expected

    for url in found | {call.args[0] for call in get.call_args_list}:
        assert url.startswith("https://"), f"non-web URL crawled or returned: {url}"


def test_find_urls_cross_domain_fan_out_is_bounded() -> None:
    """max_links bounds the FETCH count even when every page links new domains.

    With the domain filter honored, this is the only constraint left on how far
    the crawl spreads, so it is asserted on the number of requests issued rather
    than on the size of the returned set.
    """
    max_links = 5

    def fan_out_response(url: str, timeout: int) -> Response:
        assert timeout == 5
        # Every page links to three brand-new domains plus a cycle back home.
        depth = url.count("-")
        body = (
            "".join(
                f'<a href="https://host{depth}-{i}.test/page">Next</a>'
                for i in range(3)
            )
            + '<a href="https://seed.test/page">Home</a>'
        )
        response = Response()
        response.status_code = 200
        response.url = url
        response._content = body.encode("utf-8")
        response.encoding = "utf-8"
        return response

    with patch(
        "langroid.parsing.urls.requests.get", side_effect=fan_out_response
    ) as get:
        found = find_urls(
            "https://seed.test/page",
            max_links=max_links,
            max_depth=10,
            match_domain=False,
        )

    assert len(get.call_args_list) <= max_links
    assert len(found) <= max_links
    # The cycle back to the seed must not cause a refetch.
    fetched = [call.args[0] for call in get.call_args_list]
    assert len(fetched) == len(set(fetched))
