from _typeshed import Incomplete
from pydantic import BaseModel, HttpUrl

logger: Incomplete

def url_to_tempfile(url: str) -> str: ...
def get_user_input(msg: str, color: str = "blue") -> str: ...
def get_list_from_user(
    prompt: str = "Enter input (type 'done' or hit return to finish)",
    n: int | None = None,
) -> list[str]: ...

class Url(BaseModel):
    url: HttpUrl

def is_url(s: str) -> bool: ...
def get_urls_paths_bytes_indices(
    inputs: list[str | bytes],
) -> tuple[list[int], list[int], list[int]]: ...
def crawl_url(url: str, max_urls: int = 1) -> list[str]: ...
def find_urls(
    url: str = "https://en.wikipedia.org/wiki/Generative_pre-trained_transformer",
    max_links: int = 20,
    visited: set[str] | None = None,
    depth: int = 0,
    max_depth: int = 2,
    match_domain: bool = True,
) -> set[str]: ...
def org_user_from_github(url: str) -> str: ...
