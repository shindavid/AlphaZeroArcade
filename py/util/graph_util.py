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
