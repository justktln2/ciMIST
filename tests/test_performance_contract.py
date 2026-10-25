"""Portable checks for exact tree reconstruction and sparse export."""
from pathlib import Path
import tempfile
import unittest
import jax
jax.config.update("jax_enable_x64", True)
import h5py
import networkx as nx
import numpy as np
from cimist.mist import _PresortedDeletionMST
from cimist.sparse_probability import SparseProbability
from cimist.io import _write_probabilities

class PerformanceContract(unittest.TestCase):
    def test_deletion_forests_preserve_order_and_ties(self):
        # Includes every disconnected and tied simple graph on four nodes.
        pairs = [(i, j) for i in range(4) for j in range(i + 1, 4)]
        for bits in range(64):
            graph = nx.Graph()
            graph.add_nodes_from(range(4))
            graph.add_edges_from((u, v, {"I": 1.0}) for k, (u, v) in enumerate(pairs) if bits & (1 << k))
            prepared = _PresortedDeletionMST(graph)
            for removed in graph:
                expected = nx.maximum_spanning_tree(graph.subgraph([n for n in graph if n != removed]), weight="I")
                actual, _ = prepared.tree_without(removed)
                self.assertEqual(list(actual.edges), list(expected.edges))
                self.assertEqual([list(actual[n]) for n in actual], [list(expected[n]) for n in expected])

    def test_sparse_export_crosses_block_boundary(self):
        graph = nx.Graph()
        expected = np.zeros((133, 100, 100), dtype=np.float64)
        for i, row in enumerate(expected):
            row.flat[i] = 0.25
            row.flat[-1] = 0.75
            value = SparseProbability.encode(row)
            np.testing.assert_array_equal(value.dense_numpy(), row)
            graph.add_edge(0, i + 1, P=value)
        with tempfile.TemporaryDirectory() as folder:
            with h5py.File(Path(folder) / "output.h5", "w") as handle:
                _write_probabilities(handle, graph)
                np.testing.assert_array_equal(handle["P"][:], expected)

if __name__ == "__main__":
    unittest.main()
