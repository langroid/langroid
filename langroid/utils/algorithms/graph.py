"""
Graph algos.
"""

from typing import Dict, List, no_type_check

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


@no_type_check
def components(order: np.ndarray) -> List[List[int]]:
    """
    Find the connected components in an undirected graph represented by a matrix.

    Args:
        order (np.ndarray): A matrix with values 0 or 1 indicating
            undirected graph edges. `order[i][j] = 1` means an edge between `i`
            and `j`, and `0` means no edge.

    Returns:
        List[List[int]]: A list of List where each List contains the indices of
            nodes in the same connected component.

    Example:
        order = np.array([
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 0, 0, 1]
        ])
        components(order)
        # [[0, 1, 2], [3]]
    """

    i2g: Dict[int, int] = {}  # index to group mapping
    next_group = 0
    n = order.shape[0]
    for i in range(n):
        connected_groups = {i2g[j] for j in np.nonzero(order[i, :])[0] if j in i2g}

        # If the node is not part of any group
        # and is not connected to any groups, assign a new group
        if not connected_groups:
            i2g[i] = next_group
            next_group += 1
        else:
            # If the node is connected to multiple groups, we merge them
            main_group = min(connected_groups)
            for j in np.nonzero(order[i, :])[0]:
                if i2g.get(j) in connected_groups:
                    i2g[j] = main_group
            i2g[i] = main_group

    # Convert i2g to a list of Lists
    groups: Dict[int, List[int]] = {}
    for index, group in i2g.items():
        if group not in groups:
            groups[group] = []
        groups[group].append(index)

    return list(groups.values())
