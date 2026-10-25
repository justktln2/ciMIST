"""Lossless output and reader compatibility for both HDF5 compression policies."""
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import h5py
import networkx as nx
import numpy as np

from cimist.fit import parse_args
from cimist.io import (_create_h5_dataset, _write_probabilities, load_h5_tree,
                       save_tree_h5, save_coarse_graining_h5, save_VMM_h5)
from cimist.sparse_probability import SparseProbability


class HDF5Compression(unittest.TestCase):
    def test_cli_default_and_opt_out(self):
        self.assertEqual(parse_args([]).hdf5_compression, "gzip")
        self.assertEqual(parse_args(["--hdf5_compression", "none"]).hdf5_compression, "none")
        with self.assertRaises(SystemExit):
            parse_args(["--hdf5_compression", "lossy"])

    def test_special_values_and_unfiltered_types(self):
        arrays = {
            "float64": np.array([0., -0., np.inf, -np.inf, np.nan, 1e-300]),
            "float32": np.array([0., -0., np.inf, -np.inf, np.nan], dtype=np.float32),
            "integer": np.array([0, -1, 2**62], dtype=np.int64),
            "boolean": np.array([True, False]),
            "scalar": np.array(3.5),
            "empty": np.empty((0, 3)),
            "string": np.array([b"ALA", b"GLY"]),
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "values.h5"
            with h5py.File(path, "w") as f:
                for key, value in arrays.items():
                    _create_h5_dataset(f, key, data=value)
                with self.assertRaises(ValueError):
                    _create_h5_dataset(f, "invalid", data=[1], compression="lossy")
            with h5py.File(path, "r") as f:
                for key, value in arrays.items():
                    ds = f[key]
                    self.assertEqual(ds.dtype, value.dtype)
                    self.assertEqual(ds.shape, value.shape)
                    self.assertEqual(ds[()].tobytes(), value.tobytes())
                    compressible = key not in ("scalar", "empty", "string")
                    self.assertEqual(ds.compression, "gzip" if compressible else None)
                    if compressible:
                        self.assertEqual(ds.compression_opts, 1)
                        self.assertTrue(ds.shuffle)
                        self.assertIsNone(ds.scaleoffset)

    def test_sparse_chunk_boundaries_and_oversized_rows(self):
        for shape, count in [((100, 100), 133), ((400, 400), 3)]:
            with self.subTest(shape=shape), tempfile.TemporaryDirectory() as folder:
                graph = nx.Graph()
                expected = np.zeros((count, *shape))
                for i, row in enumerate(expected):
                    row.flat[i] = .25
                    row.flat[-1] = .75
                    # Mixed dense/sparse input also exercises the bounded writer.
                    graph.add_edge(0, i + 1, P=SparseProbability.encode(row) if i % 2 else row)
                paths = []
                for compression in (None, "gzip"):
                    path = Path(folder) / f"{compression}.h5"
                    paths.append(path)
                    with h5py.File(path, "w") as f:
                        _write_probabilities(f, graph, compression=compression)
                    with h5py.File(path, "r") as f:
                        ds = f["P"]
                        self.assertEqual(ds[()].tobytes(), expected.tobytes())
                        self.assertEqual(ds.compression, compression)
                        if compression:
                            self.assertLessEqual(np.prod(ds.chunks) * ds.dtype.itemsize, 1024**2)
                            if shape == (100, 100):
                                self.assertEqual(ds.chunks[1:], shape)
                self.assertLess(paths[1].stat().st_size, paths[0].stat().st_size)

    @staticmethod
    def tree(sparse):
        graph = nx.Graph()
        for name in ("A", "B", "C"):
            graph.add_node(name, S=1., S_pos_mean=1., S_se=.1, N_states=2,
                           p=np.array([.5, .5]), S_contribution_se=.2)
        for u, v in (("A", "B"), ("B", "C"), ("A", "C")):
            p = np.array([[.4, .1], [.1, .4]])
            graph.add_edge(u, v, P=SparseProbability.encode(p) if sparse else p,
                           I=.2, I_pos_mean=.19, I_se=.01, axes=(u, v))
        tree_graph = graph.copy()
        tree_graph.remove_edge("A", "C")
        return SimpleNamespace(T=tree_graph, MI_graph=graph, nodes=tree_graph.nodes,
                               N_obs=100, prior="haldane", entropy_estimator="bayes",
                               entropy=lambda: 2.62, sum_MIs=lambda: .38,
                               entropy_se=.1, entropy_mle_bias=.01,
                               counts_variance=.02, structural_variance=.03)

    def test_all_export_groups_and_unchanged_reader(self):
        States = namedtuple("States", "labels counts")
        VMM = namedtuple("VMM", "mu kappa n_iter r")
        for sparse in (False, True):
            with self.subTest(sparse=sparse), tempfile.TemporaryDirectory() as folder:
                paths = []
                for compression in (None, "gzip"):
                    path = Path(folder) / f"{compression}.h5"
                    paths.append(path)
                    save_tree_h5(self.tree(sparse), path, attrs_dict={"seed": 0}, compression=compression)
                    save_coarse_graining_h5(States(np.arange(20).reshape(10, 2), np.array([5, 5])),
                                           path, compression=compression)
                    save_VMM_h5(VMM(np.zeros((3, 2)), np.full((3, 2), 10.), np.array(4), np.ones((3, 2))),
                                path, compression=compression)
                with h5py.File(paths[0]) as a, h5py.File(paths[1]) as b:
                    names_a, names_b = [], []
                    a.visit(names_a.append)
                    b.visit(names_b.append)
                    self.assertEqual(names_a, names_b)
                    self.assertEqual(dict(a.attrs), dict(b.attrs))
                    self.assertNotIn("r", b["VMM"])
                    for name in names_a:
                        left, right = a[name], b[name]
                        self.assertEqual(dict(left.attrs), dict(right.attrs))
                        if isinstance(left, h5py.Dataset):
                            self.assertEqual(left.dtype, right.dtype)
                            self.assertEqual(left.shape, right.shape)
                            np.testing.assert_array_equal(left[()], right[()])
                            numeric = right.dtype.kind in "biufc" and right.ndim > 0 and right.size > 0
                            self.assertEqual(right.compression, "gzip" if numeric else None)
                            self.assertIsNone(left.compression)
                old, new = map(load_h5_tree, paths)
                self.assertEqual(list(old.T.edges), list(new.T.edges))
                self.assertEqual(float(old.entropy()), float(new.entropy()))
                for edge in old.MI_graph.edges:
                    np.testing.assert_array_equal(old.MI_graph.edges[edge]["P"], new.MI_graph.edges[edge]["P"])


if __name__ == "__main__":
    unittest.main()
