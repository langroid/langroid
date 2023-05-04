import fire
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from rich import print


def get_urls_from_user(
    prompt="Enter a URL (type 'done' or hit return to finish): ",
    n=None,
) -> set:
    """
    Prompt the user for URLs.
    Args:
        prompt: printed prompt
        n: how many URLs to prompt for. If None, then prompt until done, otherwise
            quit after n URLs have been entered.
    Returns:
        set of URLs
    """
    # Create an empty set to store the URLs.
    url_set = set()

    # Use a while loop to continuously ask the user for URLs.
    i = 0
    while True:
        # Prompt the user for input.
        print(f"[blue]{prompt} ", end="")
        url = input("")

        # Check if the user wants to exit the loop.
        if url.lower() == "done" or url == "":
            break

        # Add the URL to the set.
        url_set.add(url.strip())
        i += 1
        if i == n:
            break

    return list(url_set)


def find_urls(
    url="https://en.wikipedia.org/wiki/Generative_pre-trained_transformer",
    visited=None,
    depth=0,
    max_depth=2,
):
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


if __name__ == "__main__":
    # Example usage
    found_urls = set(fire.Fire(find_urls))
    for url in found_urls:
        print(url)
