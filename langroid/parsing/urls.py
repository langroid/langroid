import logging
import os
import tempfile
import urllib.parse
import urllib.robotparser
from typing import List, Optional, Set, Tuple
from urllib.parse import urljoin

import fire
import requests
from bs4 import BeautifulSoup
from pydantic import BaseModel, HttpUrl, ValidationError, parse_obj_as
from rich import print
from rich.prompt import Prompt
from trafilatura.spider import focused_crawler

from langroid.parsing.spider import scrapy_fetch_urls

logger = logging.getLogger(__name__)


def url_to_tempfile(url: str) -> str:
    """
    Fetch content from the given URL and save it to a temporary local file.

    Args:
        url (str): The URL of the content to fetch.

    Returns:
        str: The path to the temporary file where the content is saved.

    Raises:
        HTTPError: If there's any issue fetching the content.
    """

    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors

    # Create a temporary file and write the content
    with tempfile.NamedTemporaryFile(delete=False, suffix=".tmp") as temp_file:
        temp_file.write(response.content)
        return temp_file.name


def get_user_input(msg: str, color: str = "blue") -> str:
    """
    Prompt the user for input.
    Args:
        msg: printed prompt
        color: color of the prompt
    Returns:
        user input
    """
    color_str = f"[{color}]{msg} " if color else msg + " "
    print(color_str, end="")
    return input("")


def get_list_from_user(
    prompt: str = "Enter input (type 'done' or hit return to finish)",
    n: int | None = None,
) -> List[str]:
    """
    Prompt the user for inputs.
    Args:
        prompt: printed prompt
        n: how many inputs to prompt for. If None, then prompt until done, otherwise
            quit after n inputs.
    Returns:
        list of input strings
    """
    # Create an empty set to store the URLs.
    input_set = set()

    # Use a while loop to continuously ask the user for URLs.
    for _ in range(n or 1000):
        # Prompt the user for input.
        input_str = Prompt.ask(f"[blue]{prompt}")

        # Check if the user wants to exit the loop.
        if input_str.lower() == "done" or input_str == "":
            break

        # if it is a URL, ask how many to crawl
        if is_url(input_str):
            url = input_str
            input_str = Prompt.ask("[blue] How many new URLs to crawl?", default="0")
            max_urls = int(input_str) + 1
            tot_urls = scrapy_fetch_urls(url, k=max_urls)
            input_set.update(tot_urls)
        else:
            input_set.add(input_str.strip())

    return list(input_set)


class Url(BaseModel):
    url: HttpUrl


def is_url(s: str) -> bool:
    try:
        Url(url=parse_obj_as(HttpUrl, s))
        return True
    except ValidationError:
        return False


def get_urls_and_paths(inputs: List[str]) -> Tuple[List[str], List[str]]:
    """
    Given a list of inputs, return a list of URLs and a list of paths.
    Args:
        inputs: list of strings
    Returns:
        list of URLs, list of paths
    """
    urls = []
    paths = []
    for item in inputs:
        try:
            m = Url(url=parse_obj_as(HttpUrl, item))
            urls.append(str(m.url))
        except ValidationError:
            if os.path.exists(item):
                paths.append(item)
            else:
                logger.warning(f"{item} is neither a URL nor a path.")
    return urls, paths


def crawl_url(url: str, max_urls: int = 1) -> List[str]:
    """
    Crawl starting at the url and return a list of URLs to be parsed,
    up to a maximum of `max_urls`.
    """
    if max_urls == 1:
        # no need to crawl, just return the original list
        return [url]

    to_visit = None
    known_urls = None

    # Create a RobotFileParser object
    robots = urllib.robotparser.RobotFileParser()
    while True:
        if known_urls is not None and len(known_urls) >= max_urls:
            break
        # Set the RobotFileParser object to the website's robots.txt file
        robots.set_url(url + "/robots.txt")
        robots.read()

        if robots.can_fetch("*", url):
            # Start or resume the crawl
            to_visit, known_urls = focused_crawler(
                url,
                max_seen_urls=max_urls,
                max_known_urls=max_urls,
                todo=to_visit,
                known_links=known_urls,
                rules=robots,
            )
        if to_visit is None:
            break
    if known_urls is None:
        return [url]
    final_urls = [s.strip() for s in known_urls]
    return list(final_urls)[:max_urls]


def find_urls(
    url: str = "https://en.wikipedia.org/wiki/Generative_pre-trained_transformer",
    visited: Optional[Set[str]] = None,
    depth: int = 0,
    max_depth: int = 2,
) -> Set[str]:
    """
    Recursively find all URLs on a given page.
    Args:
        url:
        visited:
        depth:
        max_depth:

    Returns:

    """
    if visited is None:
        visited = set()
    visited.add(url)

    try:
        response = requests.get(url)
        response.raise_for_status()
    except (
        requests.exceptions.HTTPError,
        requests.exceptions.RequestException,
    ):
        print(f"Failed to fetch '{url}'")
        return visited

    soup = BeautifulSoup(response.content, "html.parser")
    links = soup.find_all("a", href=True)

    urls = [urljoin(url, link["href"]) for link in links]  # Construct full URLs

    if depth < max_depth:
        for link_url in urls:
            if link_url not in visited:
                find_urls(link_url, visited, depth + 1, max_depth)

    return visited


def org_user_from_github(url: str) -> str:
    parsed = urllib.parse.urlparse(url)
    org, user = parsed.path.lstrip("/").split("/")
    return f"{org}-{user}"


if __name__ == "__main__":
    # Example usage
    found_urls = set(fire.Fire(find_urls))
    for url in found_urls:
        print(url)
