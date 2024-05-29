from langroid.mytypes import Document as Document

from .utils import download_nltk_resource as download_nltk_resource

def find_fuzzy_matches_in_docs(
    query: str,
    docs: list[Document],
    docs_clean: list[Document],
    k: int,
    words_before: int | None = None,
    words_after: int | None = None,
) -> list[Document]: ...
def preprocess_text(text: str) -> str: ...
def find_closest_matches_with_bm25(
    docs: list[Document], docs_clean: list[Document], query: str, k: int = 5
) -> list[tuple[Document, float]]: ...
def get_context(
    query: str, text: str, words_before: int | None = 100, words_after: int | None = 100
) -> tuple[str, int, int]: ...
def eliminate_near_duplicates(
    passages: list[str], threshold: float = 0.8
) -> list[str]: ...
