from _typeshed import Incomplete
from googleapiclient.discovery import Resource as Resource

class WebSearchResult:
    title: Incomplete
    link: Incomplete
    max_content_length: Incomplete
    max_summary_length: Incomplete
    full_content: Incomplete
    summary: Incomplete
    def __init__(
        self,
        title: str,
        link: str,
        max_content_length: int = 3500,
        max_summary_length: int = 300,
    ) -> None: ...
    def get_summary(self) -> str: ...
    def get_full_content(self) -> str: ...
    def to_dict(self) -> dict[str, str]: ...

def google_search(query: str, num_results: int = 5) -> list[WebSearchResult]: ...
def metaphor_search(query: str, num_results: int = 5) -> list[WebSearchResult]: ...
def duckduckgo_search(query: str, num_results: int = 5) -> list[WebSearchResult]: ...
