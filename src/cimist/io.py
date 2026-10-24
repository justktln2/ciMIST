"""
Input and output operations for molecular dynamics trajectories and ciMIST objects.
"""

import mdtraj as md
import os
import dill as pkl
from itertools import islice
from functools import partial
from typing import Any, NamedTuple, Optional
from . import utils

from jax import Array
import cimist as cst
import numpy as np
import jax.numpy as jnp
import networkx as nx
import h5py
from .sparse_probability import SparseProbability

class DataSet(NamedTuple):
    traj: md.Trajectory
    angles: Array
    cloud_mask: Array

def load_dataset(trajfile: str, topfile: str, stride: int=1) -> DataSet:
    """
    Load and preprocess molecular dynamics trajectory for analysis.
    
    Handles both single trajectory files and directories containing multiple
    trajectory files. Automatically filters to protein atoms and performs
    structural superposition for consistent alignment.
    
    Parameters
    ----------
    trajpath : str
        Path to trajectory file or directory containing trajectory files.
        Supported formats: XTC, DCD, H5, PDB, etc. (MDTraj compatible)
    top : str
        Path to topology file defining molecular structure and connectivity
    stride : int, default=1
        Frame sampling interval - loads every nth frame to reduce memory usage
    protein_only : bool, default=True
        If True, extract only protein atoms and discard water/ions
        
    Returns
    -------
    md.Trajectory
        Preprocessed trajectory with:
        - Protein atoms only (if protein_only=True)
        - All frames superposed to first frame for structural alignment
        - Reduced to every stride-th frame
        
    Raises
    ------
    IOError
        If trajpath does not exist as file or directory
        
    Examples
    --------
    >>> # Load every 10th frame from single file
    >>> traj = load_traj("sim.xtc", "protein.pdb", stride=10)
    >>> 
    >>> # Load from directory of files
    >>> traj = load_traj("trajectory_dir/", "system.pdb")
    """

    traj = load_traj(trajfile, topfile, stride=stride)
    return traj_to_dataset(traj)

def traj_to_dataset(traj: md.Trajectory):
    """
    Load and preprocess molecular dynamics trajectory for analysis.
    
    Handles both single trajectory files and directories containing multiple
    trajectory files. Automatically filters to protein atoms and performs
    structural superposition for consistent alignment.
    
    Parameters
    ----------
    trajpath : str
        Path to trajectory file or directory containing trajectory files.
        Supported formats: XTC, DCD, H5, PDB, etc. (MDTraj compatible)
    top : str
        Path to topology file defining molecular structure and connectivity
    stride : int, default=1
        Frame sampling interval - loads every nth frame to reduce memory usage
    protein_only : bool, default=True
        If True, extract only protein atoms and discard water/ions
        
    Returns
    -------
    md.Trajectory
        Preprocessed trajectory with:
        - Protein atoms only (if protein_only=True)
        - All frames superposed to first frame for structural alignment
        - Reduced to every stride-th frame
        
    Raises
    ------
    IOError
        If trajpath does not exist as file or directory
        
    Examples
    --------
    >>> # Load every 10th frame from single file
    >>> traj = load_traj("sim.xtc", "protein.pdb", stride=10)
    >>> 
    >>> # Load from directory of files
    >>> traj = load_traj("trajectory_dir/", "system.pdb")
    """

    internal = utils.traj_to_internal(traj)
    angles, cloud_mask = utils.flatten_masks(internal)
    return DataSet(traj, angles, cloud_mask)

def load_traj(trajpath: str, top: str, stride: int = 1, protein_only=True) -> md.Trajectory:
    """
    Load and preprocess molecular dynamics trajectory for analysis.
    
    Handles both single trajectory files and directories containing multiple
    trajectory files. Automatically filters to protein atoms and performs
    structural superposition for consistent alignment.
    
    Parameters
    ----------
    trajpath : str
        Path to trajectory file or directory containing trajectory files.
        Supported formats: XTC, DCD, H5, PDB, etc. (MDTraj compatible)
    top : str
        Path to topology file defining molecular structure and connectivity
    stride : int, default=1
        Frame sampling interval - loads every nth frame to reduce memory usage
    protein_only : bool, default=True
        If True, extract only protein atoms and discard water/ions
        
    Returns
    -------
    md.Trajectory
        Preprocessed trajectory with:
        - Protein atoms only (if protein_only=True)
        - All frames superposed to first frame for structural alignment
        - Reduced to every stride-th frame
        
    Raises
    ------
    IOError
        If trajpath does not exist as file or directory
        
    Examples
    --------
    >>> # Load every 10th frame from single file
    >>> traj = load_traj("sim.xtc", "protein.pdb", stride=10)
    >>> 
    >>> # Load from directory of files
    >>> traj = load_traj("trajectory_dir/", "system.pdb")
    """

    if os.path.isfile(trajpath):
        traj = md.load(trajpath, top=top, stride=stride)
    elif os.path.isdir(trajpath):
        files = [trajpath + os.sep + f for f in sorted(os.listdir(trajpath))]
        traj = md.load(files, top=top, stride=stride)
    else:
        raise IOError(f"No such file or path \'{trajpath}\'")
    if protein_only:
        traj = traj.atom_slice(traj.topology.select('protein'))
    traj.superpose(traj)
    return traj

def to_pickle(obj: Any, path: str) -> None:
    """
    Serialize object using dill
    """
    with open(path, "wb") as f:
        pkl.dump(obj, f)


SPARSE_PICKLE_FORMAT = "cimist.sparse-probabilities.v1"


def dump_sparse_tree(tree, file):
    """Versioned opt-in representation; retains every tree field, no dense restore."""
    pkl.dump({"format": SPARSE_PICKLE_FORMAT, "tree": tree}, file)


def load_sparse_tree(file):
    """Read only trusted pickles. This returns sparse edge objects, not arrays."""
    payload = pkl.load(file)
    if not isinstance(payload, dict) or payload.get("format") != SPARSE_PICKLE_FORMAT:
        raise ValueError("Not a supported sparse tree pickle")
    return payload["tree"]


def _h5_compression_options(shape, dtype, compression):
    """Lossless filters with row-oriented chunks targeting at most 1 MiB."""
    if compression not in (None, "gzip"):
        raise ValueError("HDF5 compression must be 'gzip' or None")
    dtype = np.dtype(dtype)
    if compression is None or not shape or 0 in shape or dtype.kind not in "biufc":
        return {}
    row_bytes = int(np.prod(shape[1:], dtype=object)) * dtype.itemsize
    rows = max(1, min(shape[0], 128, (1024**2) // max(1, row_bytes)))
    chunks = [rows, *shape[1:]]
    # Oversized individual rows must be split to keep codec buffers bounded.
    while int(np.prod(chunks, dtype=object)) * dtype.itemsize > 1024**2:
        axis = max(range(len(chunks)), key=chunks.__getitem__)
        chunks[axis] = max(1, (chunks[axis] + 1) // 2)
    return dict(compression="gzip", compression_opts=1, shuffle=True,
                chunks=tuple(chunks))


def _create_h5_dataset(group, name, data=None, *, shape=None, dtype=None,
                       compression="gzip"):
    """Write numeric arrays losslessly; leave scalars/empty/string data unfiltered."""
    options = {}
    if data is not None:
        array = np.asarray(data)
        shape = array.shape if shape is None else shape
        storage_dtype = array.dtype if dtype is None else np.dtype(dtype)
        options["data"] = data
    else:
        storage_dtype = np.dtype(dtype)
    if dtype is not None:
        options["dtype"] = dtype
    options.update(_h5_compression_options(shape, storage_dtype, compression))
    return group.create_dataset(name, shape=shape, **options)


def _write_probabilities(group, graph, compression="gzip"):
    values = graph.edges(data="P")
    if not any(isinstance(value, SparseProbability) for _, _, value in values):
        # Retain the dense allocation path; apply the selected lossless filters.
        _create_h5_dataset(group, "P", data=np.stack([value for _, _, value in values]),
                           compression=compression)
        return
    first = next(iter(values))[2]
    shape = first.shape
    dtype = np.dtype(first.dtype)
    # At most 8 MiB or 128 edges per explicit dense block, unless one row
    # itself is larger. HDF5/library buffers are additional to this bound.
    row_bytes = int(np.prod(shape)) * dtype.itemsize
    rows = max(1, min(128, (8 * 1024**2) // max(row_bytes, 1)))
    dataset = _create_h5_dataset(group, "P", shape=(len(values), *shape),
                                 dtype=dtype, compression=compression)
    if dataset.chunks is not None:
        rows = max(dataset.chunks[0], (rows // dataset.chunks[0]) * dataset.chunks[0])
    iterator = iter(values)
    offset = 0
    while records := list(islice(iterator, rows)):
        block = np.zeros((len(records), *shape), dtype=dtype)
        for index, (_, _, value) in enumerate(records):
            if value.shape != shape or np.dtype(value.dtype) != dtype:
                raise ValueError("Inconsistent edge probability shape/dtype")
            if isinstance(value, SparseProbability):
                block[index].reshape(-1)[value.positions] = value.values
            else:
                block[index] = np.asarray(value)
        dataset[offset:offset + len(records)] = block
        offset += len(records)
        del block

def save_tree_h5(tree, path, mode="w", attrs_dict={}, compression="gzip"):
    """Save with gzip-1/shuffle by default; pass compression=None to disable."""
    # Saving, minimal
    create_dataset = partial(_create_h5_dataset, compression=compression)
    with h5py.File(path, mode) as f:
        for k,v in attrs_dict.items():
            f.attrs[k] = v

        T = f.create_group("tree")
        create_dataset(T, "N_obs", data=np.array([tree.N_obs]))
        T.attrs["prior"] = tree.prior
        T.attrs["entropy_estimator"] = tree.entropy_estimator

        create_dataset(T, "entropy", data=np.array([tree.entropy()]))
        create_dataset(T, "sum_MI", data=np.array([tree.sum_MIs()]))

        create_dataset(T, "entropy_se", data=np.array([tree.entropy_se]))
        create_dataset(T, "entropy_mle_bias", data=np.array([tree.entropy_mle_bias]))
        create_dataset(T, "counts_variance", data=np.array([tree.counts_variance]))
        create_dataset(T, "structural_variance", data=np.array([tree.structural_variance]))
        
        E_T = T.create_group("edges")
        V_T = T.create_group("vertices")

        G = f.create_group("MI_graph")
        E_G = G.create_group("edges")


        create_dataset(E_T, "name",
                           data=[(u,v) for u,v in tree.T.edges()],
                           dtype=h5py.string_dtype(encoding="utf-8"))
        _write_probabilities(E_T, tree.T, compression=compression)
        I_pm = np.array([I for (_,_,I) in tree.T.edges(data="I_pos_mean")])
        I_mle = np.array([I for (_,_,I) in tree.T.edges(data="I")])
        create_dataset(E_T, "I_pos_mean", data=I_pm)
        create_dataset(E_T, "I_bias_mle", data=I_mle-I_pm)

        create_dataset(E_T, "I_mle", data=np.array([I for (_,_,I) in tree.T.edges(data="I")]))
        create_dataset(E_T, "I_se", data=np.array([s for (_,_,s) in tree.T.edges(data="I_se")]))
        create_dataset(E_T, "axes", data=[ax for (_,_,ax) in tree.T.edges(data="axes")])
        
        
        create_dataset(V_T, "name", data=[n for n in tree.nodes()],
                           dtype=h5py.string_dtype(encoding="utf-8"))
        S_mle = data=np.array([S for (_, S) in tree.nodes(data="S")])
        S_pos_mean = np.array([S for (_, S) in tree.nodes(data="S_pos_mean")])
        N_res_states = np.array([K for (_, K) in tree.nodes(data="N_states")])

        create_dataset(V_T, "N_states", data=N_res_states)
        create_dataset(V_T, "S_pos_mean", data=S_pos_mean)
        create_dataset(V_T, "S_mle", data=S_mle)
        create_dataset(V_T, "S_bias_mle", data=S_mle - S_pos_mean)
        create_dataset(V_T, "S_se", data=np.array([S for (_, S) in tree.nodes(data="S_se")]))
        create_dataset(V_T, "p", data=np.stack([p for (_, p) in tree.nodes(data="p")]))
        create_dataset(V_T, "S_contribution_se", data=np.array([S for (_, S) in tree.nodes(data="S_contribution_se")]))

        create_dataset(E_G, "name", data=[(u,v) for u,v in tree.MI_graph.edges()])
        _write_probabilities(E_G, tree.MI_graph, compression=compression)
        create_dataset(E_G, "I_mle", data=np.array([I for (_,_,I) in tree.MI_graph.edges(data="I")]))
        #create_dataset(E_G, "I_pos_mean", data=np.array([I for (_,_,I) in tree.MI_graph.edges(data="I_pos_mean")]))
        #create_dataset(E_G, "I_se", data=np.array([s for (_,_,s) in tree.MI_graph.edges(data="I_se")]))
        create_dataset(E_G, "axes", data=[ax for (_,_,ax) in tree.MI_graph.edges(data="axes")])

    # Loading, minimal

def save_coarse_graining_h5(states, path, mode="a", compression="gzip"):
    create_dataset = partial(_create_h5_dataset, compression=compression)
    with h5py.File(path, mode) as f:
        States = f.create_group("states")
        for field, value in zip(states._fields, states):
            create_dataset(States, field, data=np.array(value))
            
def save_VMM_h5(mixture, path, mode="a", compression="gzip"):
    create_dataset = partial(_create_h5_dataset, compression=compression)
    with h5py.File(path, mode) as f:
        try:
            Mixture = f.create_group("VMM")
        except ValueError:
            Mixture = f["VMM"]
        for field, value in zip(mixture._fields, mixture):
            if field != "r":
                create_dataset(Mixture, field, data=np.array(value))
    
def load_h5_tree(path):
    """
    Load a tree saved as an HDF5 file.

    Parameters
    ----------
    path : Path to the HDF5 file.

    prior, Optional, default None : The prior to use when constructing the tree,
            if different than the one specified in the file. If no argument is given
            the prior saved in the file is used. If an argument is passed, it must
            be one of 'haldane', 'percs', 'jeffreys', or 'laplace'.
    """
    with h5py.File(path, "r") as file:
        nodes = file["tree/vertices/name"].asstr()[:]
        S_pm = file["tree/vertices/S_pos_mean"][:]
        S_se = file["tree/vertices/S_se"][:]
        p = jnp.array(file["tree/vertices/p"][:])
        edges = file["MI_graph/edges/name"].asstr()[:]
        I = file["MI_graph/edges/I_mle"][:]

        tree_edges = file["tree/edges/name"].asstr()[:]
        I_pm = file["tree/edges/I_pos_mean"][:]
        I_ses = file["tree/edges/I_se"][:]

        S = file["tree/vertices/S_mle"][:]
        N_states = file["tree/vertices/N_states"][:]
        P = file["MI_graph/edges/P"][:]
        N_obs = file["tree/N_obs"][:][0]
        

        prior = file["tree"].attrs["prior"]
        entropy_estimator = file["tree"].attrs["entropy_estimator"]

        entropy_se = file["tree/entropy_se"][:][0]
        entropy_mle_bias = file["tree/entropy_mle_bias"][:][0]
        
    G = nx.Graph()
    for n, S_res,  p_, K in zip(nodes, S, p, N_states):
        G.add_node(n.replace("_A", ""), S=S_res, p=jnp.array(p_), N_states=K)
    
    for (u,v), I_uv, P_uv in zip(edges, I, P,):
        u_, v_ = u.replace("_A", ""), v.replace("_A", "")
        p_u = G.nodes[u_]["p"]
        p_v = G.nodes[v_]["p"]
        G.add_edge(u_, v_,
                I=I_uv,
                P=jnp.array(P_uv), axes=(u_,v_))

    for (u,v), I_pos_mean, I_se in zip(tree_edges, I_pm, I_ses):
        u_, v_ = u.replace("_A", ""), v.replace("_A", "")
        G.add_edge(u_, v_,
                I_pos_mean=I_pos_mean,
                I_se=I_se,
                )
        
    tree = cst.MIST(G, N_obs=N_obs,
    uncertainty=False,
    prior=prior,
    entropy_estimator=entropy_estimator
    )

    tree.entropy_se = entropy_se
    tree.entropy_mle_bias = entropy_mle_bias
    return tree
