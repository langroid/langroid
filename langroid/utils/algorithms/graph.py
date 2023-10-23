"""
Graph algos.
"""

from typing import List, no_type_check

import numpy as np


@no_type_check
def topological_sort(order: np.array) -> List[int]:
    """
    Given a directed adjacency matrix, return a topological sort of the nodes.
    order[i,j] = -1 means there is an edge from i to j.
    order[i,j] = 0 means there is no edge from i to j.
    order[i,j] = 1 means there is an edge from j to i.

    Args:
        order (np.array): The adjacency matrix.

    Returns:
        List[int]: The topological sort of the nodes.

    """
    n = order.shape[0]

    # Calculate the in-degrees
    in_degree = [0] * n
    for i in range(n):
        for j in range(n):
            if order[i, j] == -1:
                in_degree[j] += 1

    # Initialize the queue with nodes of in-degree 0
    queue = [i for i in range(n) if in_degree[i] == 0]
    result = []

    while queue:
        node = queue.pop(0)
        result.append(node)

        for i in range(n):
            if order[node, i] == -1:
                in_degree[i] -= 1
                if in_degree[i] == 0:
                    queue.append(i)

    assert len(result) == n, "Cycle detected"
    return result
