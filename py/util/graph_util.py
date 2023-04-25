"""
Various graph theory algorithms.

Unless otherwise stated, graphs are represented as square numpy arrays of shape (n, n), where n is the number of nodes.
"""

import numpy as np


AdjMatrix = np.ndarray


def transitive_closure(G: AdjMatrix) -> AdjMatrix:
    """
    Computes the transitive closure of a graph.

    :param G: a graph
    :return: the transitive closure of G
    """
    n = G.shape[0]
    assert G.shape == (n, n)
    T = G.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                T[i, j] = T[i, j] or (T[i, k] and T[k, j])
    return T


def direct_children(G: AdjMatrix) -> AdjMatrix:
    """
    We say that v is a direct child of u if there is an edge from u to v and there is no other vertex w such that
    there is an edge from u to w and from w to v.

    Computes the direct children matrix of a graph.
    """
    n = G.shape[0]
    assert G.shape == (n, n)
    C = np.zeros((n, n), dtype=np.bool)

    for u, v in zip(*np.where(G)):
        C[u, v] = True
        for w in range(n):
            if G[u, w] and G[w, v]:
                C[u, v] = False
                break
    return C


def tarjans_algorithm(G: AdjMatrix) -> np.ndarray:
    """
    Computes the strongly connected components of a graph using Tarjan's algorithm.

    Outputs an array A such that A[i] is the index of the component containing node i.
    """
    n = G.shape[0]
    assert G.shape == (n, n)

    index = 0
    stack = []
    indices = np.full(n, -1)
    lowlinks = np.full(n, -1)
    on_stack = np.full(n, False)
    next_component_index = 0
    component_indices = np.full(n, -1)

    def strong_connect(v):
        nonlocal index, next_component_index
        indices[v] = index
        lowlinks[v] = index
        index += 1
        stack.append(v)
        on_stack[v] = True

        for w in np.where(G[v])[0]:
            if indices[w] == -1:
                strong_connect(w)
                lowlinks[v] = min(lowlinks[v], lowlinks[w])
            elif on_stack[w]:
                lowlinks[v] = min(lowlinks[v], indices[w])

        if lowlinks[v] == indices[v]:
            w = stack.pop()
            on_stack[w] = False
            component_indices[w] = next_component_index
            while w != v:
                w = stack.pop()
                on_stack[w] = False
                component_indices[w] = next_component_index
            next_component_index += 1

    for v in range(n):
        if indices[v] == -1:
            strong_connect(v)

    return component_indices
