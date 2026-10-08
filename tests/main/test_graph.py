import numpy as np
import pytest

from langroid.utils.algorithms.graph import components
from langroid.vector_store.base import VectorStore


def test_components_merges_entire_groups() -> None:
    order = np.zeros((6, 6), dtype=int)
    # Node 4 joins two existing groups, including non-neighbors 0 and 1.
    for i, j in [(0, 2), (1, 3), (2, 4), (3, 4)]:
        order[i, j] = order[j, i] = 1

    assert components(order) == [[0, 1, 2, 3, 4], [5]]


@pytest.mark.parametrize(
    "indices",
    [(0, 1, 2, 3, 4), (4, 3, 2, 1, 0), (0, 2, 4, 3, 1)],
)
def test_remove_overlaps_merges_transitive_windows(indices: tuple[int, ...]) -> None:
    windows = [["a", "b"], ["e", "f"], ["b", "c"], ["d", "e"], ["c", "d"]]
    windows = [windows[i] for i in indices] + [["x", "y"]]

    assert VectorStore.remove_overlaps(windows) == [
        ["a", "b", "c", "d", "e", "f"],
        ["x", "y"],
    ]
