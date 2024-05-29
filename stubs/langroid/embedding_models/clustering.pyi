from typing import Callable

import numpy as np

from langroid.mytypes import Document as Document

def find_optimal_clusters(
    X: np.ndarray, max_clusters: int, threshold: float = 0.1
) -> int: ...
def densest_clusters(
    embeddings: list[np.ndarray], k: int = 5
) -> list[tuple[np.ndarray, int]]: ...
def densest_clusters_DBSCAN(
    embeddings: np.ndarray, k: int = 10
) -> list[tuple[int, np.ndarray]]: ...
def densest_doc_clusters(
    docs: list[Document], k: int, embedding_fn: Callable[[str], np.ndarray]
) -> list[Document]: ...
