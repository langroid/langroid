from itertools import count
from unittest.mock import patch
from urllib.parse import urlparse

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
SCHEMES_OK_URL = "https://example.test/ok"
SCHEMES_HTML = (
    # Empty-netloc schemes: rejected by the domain comparison as a side effect
    # when match_domain is True, so only the scheme check excludes them when it
    # is False.
    '<a href="mailto:someone@example.com">Mail</a>'
    '<a href="javascript:void(0)">JS</a>'
    '<a href="tel:+15551234">Tel</a>'
    '<a href="data:text/html,hello">Data</a>'
    '<a href="file:///etc/passwd">File</a>'
    # Non-web schemes that DO carry a matching netloc: the domain comparison
    # admits these, so only the scheme check excludes them -- in either mode.
    '<a href="ftp://example.test/file.zip">FTP</a>'
    '<a href="ws://example.test/socket">WS</a>'
    # A real web link, so a page that was never fetched cannot be mistaken for
    # a page whose links were all filtered out.
    '<a href="https://example.test/ok">Ok</a>'
    '<a href="https://other.test/remote">Remote</a>'
)


def schemes_response(url: str, timeout: int) -> Response:
    assert timeout == 5
    pages = {
        SCHEMES_URL: SCHEMES_HTML,
        SCHEMES_OK_URL: '<a href="https://example.test/schemes">Back</a>',
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

    The scheme check is load-bearing in both modes, for different reasons.
    With `match_domain` False, the domain comparison is skipped, so nothing
    else keeps `mailto:`/`javascript:`/`file:` out of the results or out of the
    `max_links` budget. With `match_domain` True, those are rejected by the
    domain comparison anyway (empty netloc) -- but `ftp://example.test/...` and
    `ws://example.test/...` are not, since their netloc matches; before the
    scheme check they were returned.
    """
    with patch(
        "langroid.parsing.urls.requests.get", side_effect=schemes_response
    ) as get:
        found = find_urls(SCHEMES_URL, max_links=8, match_domain=match_domain)

    # SCHEMES_OK_URL can only appear if the seed page was fetched AND parsed,
    # so this also fails if the mock never served the page.
    expected = {SCHEMES_URL, SCHEMES_OK_URL}
    if not match_domain:
        expected.add(REMOTE_URL)
    assert found == expected

    for url in found | {call.args[0] for call in get.call_args_list}:
        assert url.startswith("https://"), f"non-web URL crawled or returned: {url}"


@pytest.mark.parametrize("max_links", [5, 20])
def test_find_urls_cross_domain_fan_out_is_bounded(max_links: int) -> None:
    """max_links bounds the crawl even on an endlessly branching web.

    With the domain filter honored, `max_links` is the only thing left limiting
    how far the crawl spreads, so this asserts on the requests actually issued
    as well as on the returned set. Every page serves three brand-new hostnames
    drawn from a shared counter, so the reachable set is unbounded and the crawl
    can only stop by hitting `max_links` -- never by running out of pages, which
    is what makes the bound the thing under test.
    """
    host_counter = count()

    def fan_out_response(url: str, timeout: int) -> Response:
        assert timeout == 5
        body = (
            "".join(
                f'<a href="https://host{next(host_counter)}.test/page">Next</a>'
                for _ in range(3)
            )
            # ...plus a cycle back to the seed, which must not be refetched.
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

    fetched = [call.args[0] for call in get.call_args_list]
    # Requests issued are bounded, and the result lands exactly on the cap --
    # on an infinite graph that can only be the bound stopping it, never
    # exhaustion of the link graph.
    assert len(fetched) <= max_links
    assert len(found) == max_links
    # The crawl really did leave the seed host, so the cross-domain path is
    # what was exercised.
    assert any(urlparse(u).netloc != "seed.test" for u in fetched)
    assert {u for u in found if urlparse(u).netloc != "seed.test"}
    # The cycle back to the seed must not cause a refetch.
    assert len(fetched) == len(set(fetched))
