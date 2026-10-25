"""CLI defaults select the measured memory configuration; dense remains opt-in."""
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from cimist.fit import parse_args
from cimist.mist import MIST
from cimist.io import dump_sparse_tree, load_sparse_tree
from cimist.sparse_probability import SparseProbability, dense_probability


class CLIDefaults(unittest.TestCase):
    def test_benchmarked_defaults_and_overrides(self):
        args = parse_args([])
        self.assertEqual(args.VMM_BATCH_SIZE, 32)
        self.assertEqual(args.PAIR_BATCH_SIZE, 128)
        self.assertEqual(args.probability_storage, "sparse")
        self.assertEqual(args.hdf5_compression, "gzip")
        override = parse_args(["--VMM_BATCH_SIZE", "100", "--PAIR_BATCH_SIZE", "999",
                               "--probability_storage", "dense", "--hdf5_compression", "none"])
        self.assertEqual((override.VMM_BATCH_SIZE, override.PAIR_BATCH_SIZE,
                          override.probability_storage, override.hdf5_compression),
                         (100, 999, "dense", "none"))

    def test_default_storage_preserves_dense_results_and_pickle(self):
        states = SimpleNamespace(states_traj=jnp.array([[0, 0, 1, 1, 0, 1, 0, 1],
                                                        [0, 1, 0, 1, 0, 1, 1, 0],
                                                        [1, 1, 0, 0, 1, 0, 1, 0]]),
                                 state_counts=jnp.full((3, 2), 4),
                                 n_states=jnp.full(3, 2))
        trees = []
        for argv in ([], ["--probability_storage", "dense"]):
            args = parse_args(argv)
            trees.append(MIST.from_residue_states(states, ["A", "B", "C"],
                entropy_estimator="mle", uncertainty=False,
                pair_batch_size=args.PAIR_BATCH_SIZE, probability_storage=args.probability_storage))
        sparse, dense = trees
        self.assertEqual(list(sparse.T.edges), list(dense.T.edges))
        self.assertEqual(float(sparse.entropy()), float(dense.entropy()))
        for edge in sparse.MI_graph.edges:
            actual = sparse.MI_graph.edges[edge]["P"]
            self.assertIsInstance(actual, SparseProbability)
            np.testing.assert_array_equal(dense_probability(actual), dense.MI_graph.edges[edge]["P"])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tree.pkl"
            with path.open("wb") as f:
                dump_sparse_tree(sparse, f)
            with path.open("rb") as f:
                restored = load_sparse_tree(f)
            self.assertEqual(list(restored.T.edges), list(sparse.T.edges))
            self.assertEqual(float(restored.entropy()), float(sparse.entropy()))


if __name__ == "__main__":
    unittest.main()
