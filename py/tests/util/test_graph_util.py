import unittest
import numpy as np
from util.graph_util import topological_sort, transitive_closure, direct_children, tarjans_algorithm


def adj(*edges, n=None):
    """Helper: build a bool adjacency matrix from a list of (u, v) directed edges."""
    if n is None:
        n = max((max(u, v) for u, v in edges), default=-1) + 1
    G = np.zeros((n, n), dtype=bool)
    for u, v in edges:
        G[u, v] = True
    return G


class TestTopologicalSort(unittest.TestCase):
    def _check_order(self, G, order):
        """Verify that every edge u->v has u before v in order."""
        pos = {node: i for i, node in enumerate(order)}
        n = G.shape[0]
        for u in range(n):
            for v in np.where(G[u])[0]:
                self.assertLess(pos[u], pos[v], f'Edge {u}->{v} violated in order {order}')

    def test_single_node(self):
        G = adj(n=1)
        order = topological_sort(G)
        self.assertEqual(list(order), [0])

    def test_linear_chain(self):
        # 0 -> 1 -> 2 -> 3
        G = adj((0, 1), (1, 2), (2, 3))
        order = topological_sort(G)
        self._check_order(G, order)
        self.assertEqual(len(order), 4)

    def test_branching_dag(self):
        # 0 -> 1, 0 -> 2, 1 -> 3, 2 -> 3
        G = adj((0, 1), (0, 2), (1, 3), (2, 3))
        order = topological_sort(G)
        self._check_order(G, order)
        self.assertEqual(len(order), 4)

    def test_raises_on_cycle(self):
        # 0 -> 1 -> 0
        G = adj((0, 1), (1, 0))
        with self.assertRaises(ValueError):
            topological_sort(G)


class TestTransitiveClosure(unittest.TestCase):
    def test_chain_adds_transitive_edge(self):
        # 0 -> 1 -> 2  =>  closure should have 0 -> 2
        G = adj((0, 1), (1, 2))
        T = transitive_closure(G)
        self.assertTrue(T[0, 2])

    def test_existing_edges_preserved(self):
        G = adj((0, 1), (1, 2))
        T = transitive_closure(G)
        self.assertTrue(T[0, 1])
        self.assertTrue(T[1, 2])

    def test_no_self_loops_added(self):
        G = adj((0, 1), (1, 2))
        T = transitive_closure(G)
        for i in range(3):
            self.assertFalse(T[i, i])

    def test_already_closed(self):
        # If the transitive closure is already present, it should be unchanged
        G = adj((0, 1), (1, 2), (0, 2))
        T = transitive_closure(G)
        np.testing.assert_array_equal(T, G)

    def test_disconnected_nodes_not_connected(self):
        G = adj((0, 1), n=3)  # node 2 is isolated
        T = transitive_closure(G)
        self.assertFalse(T[0, 2])
        self.assertFalse(T[2, 0])


class TestDirectChildren(unittest.TestCase):
    def test_removes_transitive_edge(self):
        # 0 -> 1 -> 2, 0 -> 2 (transitive): direct children of 0 should only be {1}
        G = adj((0, 1), (1, 2), (0, 2))
        C = direct_children(G)
        self.assertTrue(C[0, 1])
        self.assertFalse(C[0, 2])
        self.assertTrue(C[1, 2])

    def test_keeps_direct_edges(self):
        G = adj((0, 1), (1, 2))
        C = direct_children(G)
        self.assertTrue(C[0, 1])
        self.assertTrue(C[1, 2])

    def test_no_extra_edges(self):
        G = adj((0, 1), (1, 2), (0, 2))
        C = direct_children(G)
        # Only 0->1 and 1->2 should remain
        self.assertEqual(int(C.sum()), 2)


class TestTarjansAlgorithm(unittest.TestCase):
    def test_all_singletons(self):
        # DAG: no cycles, every node is its own SCC
        G = adj((0, 1), (1, 2))
        components = tarjans_algorithm(G)
        # All components different
        self.assertEqual(len(set(components)), 3)

    def test_single_big_scc(self):
        # Complete cycle 0->1->2->0: all in same component
        G = adj((0, 1), (1, 2), (2, 0))
        components = tarjans_algorithm(G)
        self.assertEqual(components[0], components[1])
        self.assertEqual(components[1], components[2])

    def test_mixed_sccs(self):
        # 0->1->0 (SCC), 2 is isolated
        G = adj((0, 1), (1, 0), n=3)
        components = tarjans_algorithm(G)
        self.assertEqual(components[0], components[1])
        self.assertNotEqual(components[0], components[2])

    def test_output_length(self):
        G = adj((0, 1), (2, 3), n=4)
        components = tarjans_algorithm(G)
        self.assertEqual(len(components), 4)


if __name__ == '__main__':
    unittest.main()
