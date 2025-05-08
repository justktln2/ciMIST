"""
Input and output operations for molecular dynamics trajectories and cimist objects.
"""

import mdtraj as md  # type: ignore
import os
import dill as pkl  # type: ignore
from typing import Any, NamedTuple
from . import utils

from jax import Array
import cimist as cst
import numpy as np
import jax.numpy as jnp
import networkx as nx
import h5py  # type: ignore

from .mist import MIST

class DataSet(NamedTuple):
    traj: md.Trajectory
    angles: Array
    cloud_mask: Array


def load_dataset(trajfile: str, topfile: str, stride: int = 1) -> DataSet:
    traj = load_traj(trajfile, topfile, stride=stride)
    return traj_to_dataset(traj)


def traj_to_dataset(traj: md.Trajectory):
    internal = utils.traj_to_internal(traj)
    angles, cloud_mask = utils.flatten_masks(internal)
    return DataSet(traj, angles, cloud_mask)


def load_traj(trajpath: str, top: str, stride: int = 1, protein_only=True) -> md.Trajectory:
    if os.path.isfile(trajpath):
        traj = md.load(trajpath, top=top, stride=stride)
    elif os.path.isdir(trajpath):
        files = [trajpath + os.sep + f for f in sorted(os.listdir(trajpath))]
        traj = md.load(files, top=top, stride=stride)
    else:
        raise IOError(f"No such file or path '{trajpath}'")
    if protein_only:
        traj = traj.atom_slice(traj.topology.select("protein"))
    traj.superpose(traj)
    return traj


def to_pickle(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        pkl.dump(obj, f)


def save_tree_h5(tree, path, mode="w", attrs_dict={}):
    # Saving, minimal
    with h5py.File(path, mode) as f:
        for k, v in attrs_dict.items():
            f.attrs[k] = v

        T = f.create_group("tree")
        T.create_dataset("N_obs", data=np.array([tree.N_obs]))
        T.attrs["prior"] = tree.prior

        T.create_dataset("entropy", data=np.array([tree.entropy()]))
        T.create_dataset("sum_MI", data=np.array([tree.sum_MIs()]))

        T.create_dataset("entropy_se", data=np.array([tree.entropy_se]))
        T.create_dataset("entropy_mle_bias", data=np.array([tree.entropy_mle_bias]))
        T.create_dataset("counts_variance", data=np.array([tree.counts_variance]))
        T.create_dataset("structural_variance", data=np.array([tree.structural_variance]))

        E_T = T.create_group("edges")
        V_T = T.create_group("vertices")

        G = f.create_group("MI_graph")
        E_G = G.create_group("edges")

        E_T.create_dataset(
            "name",
            data=[(u, v) for u, v in tree.T.edges()],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        E_T.create_dataset("P", data=np.stack([P for (_, _, P) in tree.T.edges(data="P")]))
        I_pm = np.array([I for (_, _, I) in tree.T.edges(data="I_pos_mean")])
        I_mle = np.array([I for (_, _, I) in tree.T.edges(data="I")])
        E_T.create_dataset("I_pos_mean", data=I_pm)
        E_T.create_dataset("I_bias_mle", data=I_mle - I_pm)

        E_T.create_dataset("I_mle", data=np.array([I for (_, _, I) in tree.T.edges(data="I")]))
        E_T.create_dataset("I_se", data=np.array([s for (_, _, s) in tree.T.edges(data="I_se")]))
        E_T.create_dataset("axes", data=[ax for (_, _, ax) in tree.T.edges(data="axes")])

        V_T.create_dataset(
            "name",
            data=[n for n in tree.nodes()],
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        S_mle = np.array([S for (_, S) in tree.nodes(data="S")])
        S_pos_mean = np.array([S for (_, S) in tree.nodes(data="S_pos_mean")])
        N_res_states = np.array([K for (_, K) in tree.nodes(data="N_states")])

        V_T.create_dataset("N_states", data=N_res_states)
        V_T.create_dataset("S_pos_mean", data=S_pos_mean)
        V_T.create_dataset("S_mle", data=S_mle)
        V_T.create_dataset("S_bias_mle", data=S_mle - S_pos_mean)
        V_T.create_dataset("S_se", data=np.array([S for (_, S) in tree.nodes(data="S_se")]))
        V_T.create_dataset("p", data=np.stack([p for (_, p) in tree.nodes(data="p")]))
        V_T.create_dataset(
            "S_contribution_se",
            data=np.array([S for (_, S) in tree.nodes(data="S_contribution_se")]),
        )

        E_G.create_dataset("name", data=[(u, v) for u, v in tree.MI_graph.edges()])
        E_G.create_dataset("P", data=np.stack([P for (_, _, P) in tree.MI_graph.edges(data="P")]))
        E_G.create_dataset("I_mle", data=np.array([I for (_, _, I) in tree.MI_graph.edges(data="I")]))
        # E_G.create_dataset("I_pos_mean", data=np.array([I for (_,_,I) in tree.MI_graph.edges(data="I_pos_mean")]))
        # E_G.create_dataset("I_se", data=np.array([s for (_,_,s) in tree.MI_graph.edges(data="I_se")]))
        E_G.create_dataset("axes", data=[ax for (_, _, ax) in tree.MI_graph.edges(data="axes")])

    # Loading, minimal


def save_coarse_graining_h5(states, path, mode="a"):
    with h5py.File(path, mode) as f:
        States = f.create_group("states")
        for field, value in zip(states._fields, states):
            States.create_dataset(field, data=np.array(value))


def save_VMM_h5(mixture, path, mode="a"):
    with h5py.File(path, mode) as f:
        try:
            Mixture = f.create_group("VMM")
        except ValueError:
            Mixture = f["VMM"]
        for field, value in zip(mixture._fields, mixture):
            if field != "r":
                Mixture.create_dataset(field, data=np.array(value))  # type: ignore


def load_h5_tree(path: str) -> MIST:
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
        # Check if the file is a valid HDF5 file
        assert isinstance(file, h5py.Dataset), "File is not a valid HDF5 file."
        if isinstance(file, h5py.Group):
            raise ValueError("File is not a valid HDF5 file.")
        if isinstance(file, h5py.Datatype):
            raise ValueError("File is not a valid HDF5 file.")
        nodes = file["tree/vertices/name"].asstr()[:]
        #S_pm = file["tree/vertices/S_pos_mean"][:]
        #S_se = file["tree/vertices/S_se"][:]
        p = jnp.array(file["tree/vertices/p"][:])
        edges = file["MI_graph/edges/name"].asstr()[:]
        I = file["MI_graph/edges/I_mle"][:]  # type: ignore

        tree_edges = file["tree/edges/name"].asstr()[:]
        I_pm = file["tree/edges/I_pos_mean"][:]
        I_ses = file["tree/edges/I_se"][:]

        S = file["tree/vertices/S_mle"][:]
        N_states = file["tree/vertices/N_states"][:]
        P = file["MI_graph/edges/P"][:]
        N_obs = file["tree/N_obs"][:][0]

        prior = file["tree"].attrs["prior"]

        entropy_se = file["tree/entropy_se"][:][0]
        entropy_mle_bias = file["tree/entropy_mle_bias"][:][0]

    G: nx.Graph = nx.Graph()
    for n, S_res, p_, K in zip(nodes, S, p, N_states):
        G.add_node(n.replace("_A", ""), S=S_res, p=jnp.array(p_), N_states=K)

    for (u, v), I_uv, P_uv in zip(
        edges,
        I,
        P,
    ):
        u_, v_ = u.replace("_A", ""), v.replace("_A", "")
        #p_u = G.nodes[u_]["p"]
        #p_v = G.nodes[v_]["p"]
        G.add_edge(u_, v_, I=I_uv, P=jnp.array(P_uv), axes=(u_, v_))

    for (u, v), I_pos_mean, I_se in zip(tree_edges, I_pm, I_ses):
        u_, v_ = u.replace("_A", ""), v.replace("_A", "")
        G.add_edge(
            u_,
            v_,
            I_pos_mean=I_pos_mean,
            I_se=I_se,
        )

    tree = cst.MIST(G, N_obs=N_obs, uncertainty=False, prior=prior, update_posterior=False)
    tree.entropy_se = entropy_se
    tree.entropy_mle_bias = entropy_mle_bias
    return tree
